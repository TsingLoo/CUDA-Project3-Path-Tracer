#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/partition.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created

    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(0.0f);

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
        segment.throughput = glm::vec3(1.0f);

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, traceDepth);
        thrust::uniform_real_distribution<float> u01(0, 1);

        float jitterX = u01(rng);
		float jitterY = u01(rng);


        // TODO: implement antialiasing by jittering the ray
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * (((float)x + jitterX) - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * (((float)y + jitterY) - (float)cam.resolution.y * 0.5f)
        );

        pathSegments[index] = segment;
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth,
    int num_active_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_active_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SQUARE_PLANE)
            {
                t = squarePlaneIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            // TODO: add more intersection tests here... triangle? metaball? CSG?

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
        }
    }
}

__device__ glm::vec2 squareToDiskConcentric(glm::vec2 xi)
{
    float a = 2.0 * xi.x - 1.0;
    float b = 2.0 * xi.y - 1.0;

    if (a == 0.0 && b == 0.0)
    {
        return glm::vec2(0.0);
    }

    float r, phi;

    if (abs(a) > abs(b))
    {
        r = a;
        phi = (PI / 4.0) * (b / a);
    }
    else
    {
        r = b;
        phi = (PI / 2.0) - (a / b) * (PI / 4.0);
    }

    float x = r * cos(phi);
    float y = r * sin(phi);
    return glm::vec2(x, y);
}

__device__ glm::vec3 squareToHemisphereCosine(glm::vec2 xi)
{
    glm::vec2 disk = squareToDiskConcentric(xi);
    float z = glm::sqrt(glm::max(0.f, 1.0f - (disk.x * disk.x) - (disk.y * disk.y)));
    return glm::vec3(disk.x, disk.y, z);
}

/// <summary>
/// build a tangent space given a normal vector
/// </summary>
/// <param name="nor">the z-axis of the space</param>
/// <param name="v2">the tangent of the space, x-axis</param>
/// <param name="v3"></param>
/// <returns></returns>
__device__ void buildTangentSpace(const glm::vec3 nor, glm::vec3& v2, glm::vec3& v3)
{
    if (abs(nor.x) > abs(nor.y))
        //cross(v1, vec3(0, 1, 0)) = vec3(-v1.z, 0, v1.x)
        v2 = glm::vec3(-nor.z, 0, nor.x) / sqrt(nor.x * nor.x + nor.z * nor.z);
    else
        v2 = glm::vec3(0, nor.z, -nor.y) / sqrt(nor.y * nor.y + nor.z * nor.z);
    v3 = glm::cross(nor, v2);
}

__device__ glm::mat3 TangentSpaceToWorld(const glm::vec3 nor) {
    glm::vec3 tan, bit;
    buildTangentSpace(nor, tan, bit);
    return glm::mat3(tan, bit, nor);
}

__device__ glm::mat3 WorldToTangentSpace(const glm::vec3 nor) {
    return glm::transpose(TangentSpaceToWorld(nor));
}

__device__ glm::vec3 f_diffuse(const glm::vec3 albedo) {
    return albedo / PI;
}

__device__ glm::vec3 Sample_f_diffuse(const glm::vec3 albedo, const glm::vec2 xi, const glm::vec3 nor,
    glm::vec3& wiW, float& pdf) {
    glm::vec3 wiLocal = squareToHemisphereCosine(xi);
    pdf = wiLocal.z / PI;
    glm::mat3 mat = TangentSpaceToWorld(nor);
    wiW = mat * wiLocal;
    return f_diffuse(albedo);
}

__device__ glm::vec3 Sample_f_specular_refl(const glm::vec3 albedo, const glm::vec3 nor, const glm::vec3 wo,
    glm::vec3& wiW, int& sampledType)
{
    glm::vec3 wi = glm:: vec3(-wo.x, -wo.y, wo.z);

    wiW = TangentSpaceToWorld(nor) * wi;

	sampledType = MaterialType::SPEC_REFL;

    float cosThetaT = glm::abs(glm::dot(glm::vec3(0, 0, 1), wi));

    return albedo / cosThetaT;
}

__device__ bool Refract(glm::vec3 wi, glm::vec3 n, float eta, glm::vec3& wt) {
    // Compute cos theta using Snell's law
    float cosThetaI = dot(n, wi);
    float sin2ThetaI = glm::max(float(0), float(1 - cosThetaI * cosThetaI));
    float sin2ThetaT = eta * eta * sin2ThetaI;

    // Handle total internal reflection for transmission
    if (sin2ThetaT >= 1) return false;
    float cosThetaT = sqrt(1 - sin2ThetaT);
    wt = eta * -wi + (eta * cosThetaI - cosThetaT) * n;
    return true;
}

__device__ glm::vec3 Sample_f_specular_trans(glm::vec3 albedo, glm::vec3 nor, glm::vec3 wo,
    glm::vec3& wiW, int& sampledType)
{
    sampledType = MaterialType::SPEC_TRANS;
    float etaA = 1.0;
    float etaB = 1.55;
    bool entering = (wo.z > 0.0);

    float eta = entering ? (etaA / etaB) : (etaB / etaA);
    glm::vec3 n = entering ? glm::vec3(0, 0, 1) : glm::vec3(0, 0, -1);
    glm::vec3 wi;
    bool didRefract = Refract(wo, n, eta, wi);

    if (!didRefract)
    {
        wiW = glm::vec3(0);
        sampledType = SPEC_TRANS;
        return glm::vec3(0);
    }

    wiW = TangentSpaceToWorld(nor) * wi;

    float cosThetaT = glm::abs(glm::dot(glm::vec3(0, 0, 1), wi));
    return albedo / cosThetaT;
}

__device__ glm::vec3 FresnelDielectricEval(float cosThetaI) {
    float etaA = 1.0f;
    float etaB = 1.55f;
    bool entering = cosThetaI > 0.0f;

    float eta = entering ? (etaA / etaB) : (etaB / etaA);
    cosThetaI = glm::clamp(cosThetaI, -1.f, 1.f);

    float sin2ThetaT = eta * eta * glm::max(0.0f, 1.0f - cosThetaI * cosThetaI);

    // Total Internal Reflection
    if (sin2ThetaT > 1.0f) {
        return glm::vec3(1.0f);
    }

    float cosThetaT = sqrt(1.0f - sin2ThetaT);

    //return glm::vec3(cosThetaT);

    float Rs = ((etaA * abs(cosThetaI)) - (etaB * cosThetaT)) /
        ((etaA * abs(cosThetaI)) + (etaB * cosThetaT));
    float Rp = ((etaB * abs(cosThetaI)) - (etaA * cosThetaT)) /
        ((etaB * abs(cosThetaI)) + (etaA * cosThetaT));

    Rs = Rs * Rs;
    Rp = Rp * Rp;

    float F = 0.5f * (Rs + Rp);

    return glm::vec3(F);
}


__device__ glm::vec3 Sample_f_glass(glm::vec3 albedo, glm::vec3 nor, glm::vec2 xi, glm::vec3 wo, glm::vec3& wiW, int& sampledType) {
    float random = xi.x;
    if (random < 0.5) {
        // Have to double contribution b/c we only sample
        // reflection BxDF half the time
        glm::vec3 R = Sample_f_specular_refl(albedo, nor, wo, wiW, sampledType);
		sampledType = MaterialType::SPEC_REFL;
        return 2.0f * FresnelDielectricEval(glm::dot(nor, normalize(wiW))) * R;
    }
    else {
        // Have to double contribution b/c we only sample
        // transmit BxDF half the time
        glm::vec3 T = Sample_f_specular_trans(albedo, nor, wo, wiW, sampledType);
        sampledType = MaterialType::SPEC_TRANS;
        return 2.0f * (glm::vec3(1.) - FresnelDielectricEval(dot(nor, normalize(wiW)))) * T;
    }
}

/// <summary>
/// 
/// </summary>
/// <param name="material"></param>
/// <param name="normal">In world space</param>
/// <param name="woW">In world space</param>
/// <param name="wiW">In world space</param>
/// <param name="pdf"></param>
/// <param name="sampleType"></param>
/// <param name="xi"></param>
/// <returns>the light energy</returns>
__device__ glm::vec3 Sample_f(
    Material material,
    glm::vec3 normal,
    glm::vec3 woW, // outgoing direction in world space
    glm::vec2 xi,
    glm::vec3& wiW, // incoming direction in world space
    float& pdf,
    int& sampledType) // 2 random numbers in [0,1)
{
    glm::vec3 wo = WorldToTangentSpace(normal) * woW;

    if (material.type == MaterialType::DIFFUSE_REFL) {
        return Sample_f_diffuse(material.albedo, xi, normal, wiW, pdf);
    }
    else if (material.type == MaterialType::EMITTIVE) {
        return material.albedo;
    }
    else if (material.type == MaterialType::SPEC_REFL){
		pdf = 1.0f;
        return Sample_f_specular_refl(material.albedo, normal, wo, wiW, sampledType);
    }
    else if (material.type == MaterialType::SPEC_TRANS) {
        pdf = 1.0f;
        return normal;
        return Sample_f_specular_trans(material.albedo, normal, wo, wiW, sampledType);
    }
    else if (material.type == MaterialType::SPEC_GLASS) {
        pdf = 1.0f;
		return Sample_f_glass(material.albedo, normal, xi, wo, wiW, sampledType);
    }
    else {
        return DEBUG_EMPTY_COLOR;
    }
}


/// <summary>
/// Performs shading calculations for a set of path segments based on their intersections and material properties in a CUDA kernel.
/// </summary>
/// <param name="iter">The current iteration number, used for random number generation seeding.</param>
/// <param name="num_paths">The total number of path segments to process.</param>
/// <param name="shadeableIntersections">Pointer to an array of intersection data for each path segment.</param>
/// <param name="pathSegments">Pointer to an array of path segments to be shaded and updated.</param>
/// <param name="materials">Pointer to an array of material properties used for shading. From the shadeableIntersection we only konw the index of the material</param>
/// <returns>This function does not return a value; it updates the pathSegments array in place with new color and bounce information.</returns>
__global__ void shadeMaterial(
    int iter,
    int depth,
    int num_active_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    glm::vec3* dev_img)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_active_paths)
    {

        PathSegment pathSegment = pathSegments[idx];
        if (pathSegment.remainingBounces < 1) {
            return;
        }

        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
            // Set up the RNG
            // LOOK: this is how you use thrust's RNG! Please look at
            // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, pathSegment.pixelIndex, depth);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.albedo;

            //if the ray hit something, should be pink
            //pathSegment.color = DEBUG_PINK_COLOR;

            // If the material indicates that the object was a light, "light" the way and stop
			// A ray is meaningful only if it finally hits a light source
            if (material.emittance > 0.0f) {

                pathSegment.color += pathSegment.throughput * material.emittance;
                pathSegment.remainingBounces = 0;
            }
            else {
                //glm::vec3 intersectPos = pathSegments[idx].ray.origin + glm::normalize(pathSegments[idx].ray.direction) * (intersection.t);

                float pdf;
                int sampleType;
                glm::vec3 wiW;
				//woW is the outgoing direction in world space, which is the negative of the ray direction
				//the out direction of a light is from the surface to the camera, while a ray is from the camera to the surface
                glm::vec3 woW = - pathSegment.ray.direction;

                glm::vec2 xi = glm::vec2(u01(rng), u01(rng));

                glm::vec3 bsdf = Sample_f(material, intersection.surfaceNormal, woW, xi, wiW, pdf, sampleType);

                if (pdf < 1e-6) {
                    pathSegment.remainingBounces = 0;
                    pathSegment.color = glm::vec3(0.0f);

                    return;
                }

                glm::vec3 this_iter_throughput = bsdf * glm::abs(glm::dot(wiW, intersection.surfaceNormal)) / pdf;
                pathSegment.throughput *= this_iter_throughput;
                glm::vec3 intersectPos = pathSegment.ray.origin + pathSegment.ray.direction * intersection.t;
                pathSegment.ray.origin = intersectPos + intersection.surfaceNormal * 1e-4f; // Offset to avoid self-intersection
                pathSegment.ray.direction = wiW;

                pathSegment.remainingBounces--;
                //pathSegment.color = bsdf
                //pathSegment.color = DEBUG_PINK_COLOR;
            }
        }
        // If there was no intersection, color the PINK
        // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
        // used for opacity, in which case they can indicate "no opacity".
        // This can be useful for post-processing and image compositing.
        else {
            pathSegment.remainingBounces = 0;
            //pathSegment.color = DEBUG_PINK_COLOR;
            pathSegment.color = DEBUG_EMPTY_COLOR;
        }
        pathSegments[idx] = pathSegment;
    }
}

// Add the current iteration's output to the overall image


/// <summary>
/// 
/// </summary>
/// <param name="nPaths"></param>
/// <param name="image"></param>
/// <param name="iterationPaths"></param>
/// <returns></returns>
__global__ void finalGather(int pixelCount, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < pixelCount)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

struct is_ray_alive
{
    __host__ __device__
    bool operator()(const PathSegment& path)
    {
        return (path.remainingBounces > 0);
    }
};

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 * This is called every iteration from main.cpp
 * Everytime the camera changes in a frame, the iteration count should be reset to 0
 * And iteration goes to renderState->iterations
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

	//dev_paths keeps track of all the path segments in the iteration
    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_active_paths = pixelcount;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks
    for (int depth = 0; depth < traceDepth; ++depth)
    {
        if (num_active_paths == 0) { break; }

        // clean shading chunks
		// dev_intersections keeps track of all the intersections in the iteration
		// it is the same size as dev_paths, as each path segment has one intersection at most
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_active_paths + blockSize1d - 1) / blockSize1d;
		// dev_intersections is the result, where we store the intersections for each path segment
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_active_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_intersections
        );


        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        //depth++;

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

		// After we know the intersections, we can do the shading
		// An intersection means we know the intersectPos(t), normal, and 
        shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            depth,
            num_active_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_image
        );

        PathSegment* new_end = thrust::stable_partition(
            thrust::device,      // Execute this algorithm on the GPU
            dev_paths,           // The start of the array to process
            dev_paths + num_active_paths, // The end of the array to process
            is_ray_alive()        // The predicate functor to identify dead rays
        );

        num_active_paths = new_end - dev_paths;

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
    pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
