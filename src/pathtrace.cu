#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <curand_kernel.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_vector.h>
#include <thrust/partition.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "dispersion.h"

#if ENABLE_OPTIX
#include "optix_renderer.h"
#include "optix_params.h"

// Forward declarations for shadow ray kernels (defined in interactions.cu)
#if ENABLE_MIS
__global__ void kernPrepareShadowRays(
    int num_hit, LambertianHitWorkItem* queue, PathSegment* paths,
    Material* materials, curandState* rand_states,
    Geom* geoms, int geoms_size, glm::vec3* positions,
    int* light_indices, int num_lights,
    Geom* nonTriGeoms, int numNonTriGeoms,
    ShadowRayRequest* shadowRays,
    cudaTextureObject_t* textureObjects, int numTextures);

__global__ void kernPrepareEnvMapShadowRays(
    int num_hit, LambertianHitWorkItem* queue, PathSegment* paths,
    Material* materials, curandState* rand_states,
    Geom* nonTriGeoms, int numNonTriGeoms,
    ShadowRayRequest* shadowRays,
    cudaTextureObject_t* textureObjects, int numTextures,
    cudaTextureObject_t envMap,
    const float* marginalCDF, const float* conditionalCDF,
    int envW, int envH, float envTotalPower);

__global__ void kernApplyShadowResults(
    int num_rays, ShadowRayRequest* shadowRays, int* shadowOccluded, glm::vec3* dev_img);

__global__ void kernPrepareShadowRaysDisneyGGX(
    int num_hit, DisneyGGXHitWorkItem* queue, PathSegment* paths,
    Material* materials, curandState* rand_states,
    Geom* geoms, int geoms_size, glm::vec3* positions,
    int* light_indices, int num_lights,
    Geom* nonTriGeoms, int numNonTriGeoms,
    ShadowRayRequest* shadowRays,
    cudaTextureObject_t* textureObjects, int numTextures);

__global__ void kernPrepareEnvMapShadowRaysDisneyGGX(
    int num_hit, DisneyGGXHitWorkItem* queue, PathSegment* paths,
    Material* materials, curandState* rand_states,
    Geom* nonTriGeoms, int numNonTriGeoms,
    ShadowRayRequest* shadowRays,
    cudaTextureObject_t* textureObjects, int numTextures,
    cudaTextureObject_t envMap,
    const float* marginalCDF, const float* conditionalCDF,
    int envW, int envH, float envTotalPower);
#endif
#endif

#define ERRORCHECK 0  // Set to 1 only for debugging -- cudaDeviceSynchronize() kills perf
#define RUSSIAN_ROULETTE_DEPTH 3

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) return;
    fprintf(stderr, "CUDA error");
    if (file) fprintf(stderr, " (%s:%d)", file, line);
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif
    exit(EXIT_FAILURE);
#endif
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

struct is_ray_alive {
    __host__ __device__ bool operator()(const PathSegment& path) {
        return (path.remainingBounces > 0);
    }
};

struct CompareMaterial {
    __host__ __device__ bool operator()(const ShadeableIntersection& a, const ShadeableIntersection& b) {
        return a.materialId < b.materialId;
    }
};

// sRGB linear-to-gamma transfer function (IEC 61966-2-1)
__device__ inline float linearToSRGB(float c) {
    if (c <= 0.0031308f)
        return 12.92f * c;
    else
        return 1.055f * powf(c, 1.0f / 2.4f) - 0.055f;
}

__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];
#if ENABLE_SPECTRAL_RENDERING
        float divisor = (float)(iter * SPECTRAL_N);
#else
        float divisor = (float)iter;
#endif
        // Average, then apply sRGB gamma correction
        glm::vec3 linear = pix / divisor;
        glm::ivec3 color;
        color.x = glm::clamp((int)(linearToSRGB(linear.x) * 255.0f + 0.5f), 0, 255);
        color.y = glm::clamp((int)(linearToSRGB(linear.y) * 255.0f + 0.5f), 0, 255);
        color.z = glm::clamp((int)(linearToSRGB(linear.z) * 255.0f + 0.5f), 0, 255);
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

// Denoised version: input is already normalized float3 (not accumulated)
__global__ void sendDenoisedImageToPBO(uchar4* pbo, glm::ivec2 resolution, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];
        glm::ivec3 color;
        color.x = glm::clamp((int)(linearToSRGB(pix.x) * 255.0f + 0.5f), 0, 255);
        color.y = glm::clamp((int)(linearToSRGB(pix.y) * 255.0f + 0.5f), 0, 255);
        color.z = glm::clamp((int)(linearToSRGB(pix.z) * 255.0f + 0.5f), 0, 255);
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

static glm::vec3* dev_positions = NULL;
static glm::vec3* dev_normals = NULL;
static glm::vec2* dev_texcoords = NULL;

static cudaTextureObject_t* dev_texture_objects = NULL;
static cudaTextureObject_t* hst_texture_objects = NULL;
static cudaArray** dev_texture_arrays = NULL;
static int num_textures = 0;

#if ENABLE_OPTIX
static OptixRenderer* optixRenderer = NULL;
static Geom* dev_nonTriangleGeoms = NULL;  // separate buffer of box/sphere geoms only
static int hst_num_non_triangle_geoms = 0;
static ShadowRayRequest* dev_shadowRays = NULL;
static int* dev_shadowOccluded = NULL;
#endif

// Denoiser AOV buffers
#if ENABLE_DENOISER
static glm::vec3* dev_albedo_buffer = NULL;
static glm::vec3* dev_normal_buffer = NULL;
static glm::vec3* dev_denoised_image = NULL;
#endif

// HDRI environment map
static cudaTextureObject_t hst_envMapTexObj = 0;
static cudaArray* dev_envMapArray = NULL;
static bool hst_hasEnvMap = false;

// Env map importance sampling CDF
static float* dev_envCDF_marginal = NULL;
static float* dev_envCDF_conditional = NULL;
static int hst_envMapWidth = 0;
static int hst_envMapHeight = 0;
static float hst_envTotalPower = 0.0f;

static MissWorkItem* miss_queue = NULL;
static HitLightWorkItem* hit_light_queue = NULL;
static LambertianHitWorkItem* lambertian_queue = NULL;
static SpecularHitWorkItem* specular_queue = NULL;
static GlassHitWorkItem* glass_queue = NULL;
static DisneyGGXHitWorkItem* disney_ggx_queue = NULL;

static curandState* dev_rand_states = NULL;

static int* miss_queue_counter = NULL;
static int* hit_light_queue_counter = NULL;
static int* lambertian_queue_counter = NULL;
static int* specular_queue_counter = NULL;
static int* glass_queue_counter = NULL;
static int* disney_ggx_queue_counter = NULL;

static int* dev_light_indices = NULL;
static int hst_num_lights = 0;

void InitDataContainer(GuiDataContainer* imGuiData) { guiData = imGuiData; }

__global__ void initCurand_kernel(int seed, int num_pixels, curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pixels) return;
    curand_init(seed, idx, 0, &states[idx]);
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

    cudaMalloc(&dev_positions, scene->positions.size() * sizeof(glm::vec3));
    cudaMemcpy(dev_positions, scene->positions.data(), scene->positions.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_normals, scene->normals.size() * sizeof(glm::vec3));
    cudaMemcpy(dev_normals, scene->normals.data(), scene->normals.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);

    if (!scene->texcoords.empty()) {
        cudaMalloc(&dev_texcoords, scene->texcoords.size() * sizeof(glm::vec2));
        cudaMemcpy(dev_texcoords, scene->texcoords.data(), scene->texcoords.size() * sizeof(glm::vec2), cudaMemcpyHostToDevice);
    }

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&miss_queue, pixelcount * sizeof(MissWorkItem));
    cudaMalloc(&hit_light_queue, pixelcount * sizeof(HitLightWorkItem));
    cudaMalloc(&lambertian_queue, pixelcount * sizeof(LambertianHitWorkItem));
    cudaMalloc(&specular_queue, pixelcount * sizeof(SpecularHitWorkItem));
    cudaMalloc(&glass_queue, pixelcount * sizeof(GlassHitWorkItem));
    cudaMalloc(&disney_ggx_queue, pixelcount * sizeof(DisneyGGXHitWorkItem));

    cudaMalloc(&dev_rand_states, pixelcount * sizeof(curandState));

    cudaMalloc(&miss_queue_counter, sizeof(int));
    cudaMalloc(&hit_light_queue_counter, sizeof(int));
    cudaMalloc(&lambertian_queue_counter, sizeof(int));
    cudaMalloc(&specular_queue_counter, sizeof(int));
    cudaMalloc(&glass_queue_counter, sizeof(int));
    cudaMalloc(&disney_ggx_queue_counter, sizeof(int));

    {
        dim3 curandBlocks = (pixelcount + BLOCKSIZE1d - 1) / BLOCKSIZE1d;
        initCurand_kernel<<<curandBlocks, BLOCKSIZE1d>>>(42, pixelcount, dev_rand_states);
        checkCUDAError("curand init");
    }

    // Textures
    num_textures = (int)scene->textures.size();
    if (num_textures > 0) {
        hst_texture_objects = new cudaTextureObject_t[num_textures];
        dev_texture_arrays = new cudaArray*[num_textures];
        for (int i = 0; i < num_textures; i++) {
            const TextureData& tex = scene->textures[i];
            cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
            cudaMallocArray(&dev_texture_arrays[i], &channelDesc, tex.width, tex.height);
            if (tex.channels == 4) {
                cudaMemcpy2DToArray(dev_texture_arrays[i], 0, 0, tex.pixels.data(),
                    tex.width * 4, tex.width * 4, tex.height, cudaMemcpyHostToDevice);
            } else {
                std::vector<unsigned char> rgba(tex.width * tex.height * 4);
                for (int p = 0; p < tex.width * tex.height; p++) {
                    rgba[p*4+0] = (tex.channels > 0) ? tex.pixels[p*tex.channels+0] : 0;
                    rgba[p*4+1] = (tex.channels > 1) ? tex.pixels[p*tex.channels+1] : 0;
                    rgba[p*4+2] = (tex.channels > 2) ? tex.pixels[p*tex.channels+2] : 0;
                    rgba[p*4+3] = 255;
                }
                cudaMemcpy2DToArray(dev_texture_arrays[i], 0, 0, rgba.data(),
                    tex.width * 4, tex.width * 4, tex.height, cudaMemcpyHostToDevice);
            }
            cudaResourceDesc resDesc = {};
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = dev_texture_arrays[i];
            cudaTextureDesc texDesc = {};
            texDesc.addressMode[0] = cudaAddressModeWrap;
            texDesc.addressMode[1] = cudaAddressModeWrap;
            texDesc.filterMode = cudaFilterModeLinear;
            texDesc.readMode = cudaReadModeNormalizedFloat;
            texDesc.normalizedCoords = 1;
            cudaCreateTextureObject(&hst_texture_objects[i], &resDesc, &texDesc, NULL);
            printf("Created CUDA texture object %d (%dx%d)\n", i, tex.width, tex.height);
        }
        cudaMalloc(&dev_texture_objects, num_textures * sizeof(cudaTextureObject_t));
        cudaMemcpy(dev_texture_objects, hst_texture_objects, num_textures * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);
    }

    // Light indices for MIS
    {
        std::vector<int> lightIndices;
        for (int i = 0; i < (int)scene->geoms.size(); i++) {
            int matId = scene->geoms[i].materialid;
            if (matId >= 0 && matId < (int)scene->materials.size() && scene->materials[matId].emittance > 0.0f)
                lightIndices.push_back(i);
        }
        hst_num_lights = (int)lightIndices.size();
        if (hst_num_lights > 0) {
            cudaMalloc(&dev_light_indices, hst_num_lights * sizeof(int));
            cudaMemcpy(dev_light_indices, lightIndices.data(), hst_num_lights * sizeof(int), cudaMemcpyHostToDevice);
        }
        printf("MIS: Found %d light geometries\n", hst_num_lights);
    }

#if ENABLE_OPTIX
    optixRenderer = new OptixRenderer();
    optixRenderer->init();
    optixRenderer->buildAccel(scene);
    optixRenderer->setNormals(dev_normals);

    // Build separate buffer of ONLY non-triangle geoms (box/sphere)
    {
        std::vector<Geom> nonTriGeoms;
        for (const auto& g : scene->geoms) {
            if (g.type != TRIANGLE) nonTriGeoms.push_back(g);
        }
        hst_num_non_triangle_geoms = (int)nonTriGeoms.size();
        if (hst_num_non_triangle_geoms > 0) {
            cudaMalloc(&dev_nonTriangleGeoms, hst_num_non_triangle_geoms * sizeof(Geom));
            cudaMemcpy(dev_nonTriangleGeoms, nonTriGeoms.data(),
                hst_num_non_triangle_geoms * sizeof(Geom), cudaMemcpyHostToDevice);
        }
    }
    printf("OptiX hybrid mode: %d non-triangle primitives for CUDA intersection\n", hst_num_non_triangle_geoms);

    // Shadow ray buffers for MIS
    cudaMalloc(&dev_shadowRays, pixelcount * sizeof(ShadowRayRequest));
    cudaMalloc(&dev_shadowOccluded, pixelcount * sizeof(int));
#endif

#if ENABLE_DENOISER
    // Denoiser AOV buffers
    cudaMalloc(&dev_albedo_buffer, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_albedo_buffer, 0, pixelcount * sizeof(glm::vec3));
    cudaMalloc(&dev_normal_buffer, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_normal_buffer, 0, pixelcount * sizeof(glm::vec3));
    cudaMalloc(&dev_denoised_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_denoised_image, 0, pixelcount * sizeof(glm::vec3));

    // Initialize denoiser
    optixRenderer->initDenoiser(cam.resolution.x, cam.resolution.y);
#endif

    // HDRI Environment Map
    if (scene->envMap.loaded) {
        int envW = scene->envMap.width;
        int envH = scene->envMap.height;
        // Convert RGB float to RGBA float (CUDA textures need 1/2/4 channel)
        std::vector<float4> envRGBA(envW * envH);
        for (int i = 0; i < envW * envH; i++) {
            envRGBA[i] = make_float4(
                scene->envMap.pixels[i * 3 + 0],
                scene->envMap.pixels[i * 3 + 1],
                scene->envMap.pixels[i * 3 + 2],
                1.0f);
        }
        // Create CUDA array
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
        cudaMallocArray(&dev_envMapArray, &channelDesc, envW, envH);
        cudaMemcpy2DToArray(dev_envMapArray, 0, 0,
            envRGBA.data(), envW * sizeof(float4),
            envW * sizeof(float4), envH,
            cudaMemcpyHostToDevice);
        // Create texture object
        cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = dev_envMapArray;
        cudaTextureDesc texDesc = {};
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeElementType;  // float read
        texDesc.normalizedCoords = 1;
        cudaCreateTextureObject(&hst_envMapTexObj, &resDesc, &texDesc, NULL);
        hst_hasEnvMap = true;
        printf("Created HDRI environment map texture (%dx%d)\n", envW, envH);

        // Upload CDF for importance sampling
        if (!scene->envMap.marginalCDF.empty()) {
            hst_envMapWidth = envW;
            hst_envMapHeight = envH;
            hst_envTotalPower = scene->envMap.totalPower;
            size_t margSize = scene->envMap.marginalCDF.size() * sizeof(float);
            size_t condSize = scene->envMap.conditionalCDF.size() * sizeof(float);
            cudaMalloc(&dev_envCDF_marginal, margSize);
            cudaMemcpy(dev_envCDF_marginal, scene->envMap.marginalCDF.data(), margSize, cudaMemcpyHostToDevice);
            cudaMalloc(&dev_envCDF_conditional, condSize);
            cudaMemcpy(dev_envCDF_conditional, scene->envMap.conditionalCDF.data(), condSize, cudaMemcpyHostToDevice);
            printf("Uploaded env map importance sampling CDF (%zu + %zu bytes)\n", margSize, condSize);
        }
    }

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    cudaFree(dev_positions);
    cudaFree(dev_normals);
    cudaFree(dev_texcoords);

    for (int i = 0; i < num_textures; i++) {
        cudaDestroyTextureObject(hst_texture_objects[i]);
        cudaFreeArray(dev_texture_arrays[i]);
    }
    cudaFree(dev_texture_objects);
    delete[] hst_texture_objects;
    delete[] dev_texture_arrays;
    hst_texture_objects = NULL; dev_texture_arrays = NULL; dev_texture_objects = NULL;
    num_textures = 0;

    cudaFree(dev_light_indices);
    cudaFree(miss_queue); cudaFree(hit_light_queue);
    cudaFree(lambertian_queue); cudaFree(specular_queue); cudaFree(glass_queue); cudaFree(disney_ggx_queue);
    cudaFree(dev_rand_states);
    cudaFree(miss_queue_counter); cudaFree(hit_light_queue_counter);
    cudaFree(lambertian_queue_counter); cudaFree(specular_queue_counter); cudaFree(glass_queue_counter); cudaFree(disney_ggx_queue_counter);

#if ENABLE_OPTIX
    if (optixRenderer) { optixRenderer->cleanup(); delete optixRenderer; optixRenderer = NULL; }
    if (dev_nonTriangleGeoms) { cudaFree(dev_nonTriangleGeoms); dev_nonTriangleGeoms = NULL; }
    if (dev_shadowRays) { cudaFree(dev_shadowRays); dev_shadowRays = NULL; }
    if (dev_shadowOccluded) { cudaFree(dev_shadowOccluded); dev_shadowOccluded = NULL; }
#endif

#if ENABLE_DENOISER
    if (dev_albedo_buffer) { cudaFree(dev_albedo_buffer); dev_albedo_buffer = NULL; }
    if (dev_normal_buffer) { cudaFree(dev_normal_buffer); dev_normal_buffer = NULL; }
    if (dev_denoised_image) { cudaFree(dev_denoised_image); dev_denoised_image = NULL; }
#endif

    // HDRI environment map
    if (hst_hasEnvMap) {
        cudaDestroyTextureObject(hst_envMapTexObj);
        cudaFreeArray(dev_envMapArray);
        hst_envMapTexObj = 0;
        dev_envMapArray = NULL;
        hst_hasEnvMap = false;
    }
    if (dev_envCDF_marginal) { cudaFree(dev_envCDF_marginal); dev_envCDF_marginal = NULL; }
    if (dev_envCDF_conditional) { cudaFree(dev_envCDF_conditional); dev_envCDF_conditional = NULL; }

    checkCUDAError("pathtraceFree");
}

// ============================================================================
// Camera ray generation
// ============================================================================

__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments, curandState* rand_states)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];
        curandState local_rand_state = rand_states[index];
        glm::vec3 pinhole_origin = cam.position;

#if ENABLE_STOCHASTIC_ANTIALIASING
        float jitterX = curand_uniform(&local_rand_state);
        float jitterY = curand_uniform(&local_rand_state);
        glm::vec3 pinhole_direction = glm::normalize(
            cam.view
            - cam.right * cam.pixelLength.x * ((float)x + jitterX - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + jitterY - (float)cam.resolution.y * 0.5f));
#else
        glm::vec3 pinhole_direction = glm::normalize(
            cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f));
#endif

#if ENABLE_DEPTH_OF_FIELD
        glm::vec3 focusPoint = pinhole_origin + pinhole_direction * cam.focusDistance;
        float aperture_radius = (cam.focalLength / 100.0f / (cam.fAperture * 10.0f)) / 2.0f;
        glm::vec2 lens_uv = concentricSampleDisk(&local_rand_state) * aperture_radius;
        glm::vec3 rayOrigin = cam.position + cam.right * lens_uv.x + cam.up * lens_uv.y;
        glm::vec3 rayDir = glm::normalize(focusPoint - rayOrigin);
        segment.ray.origin = rayOrigin;
        segment.ray.direction = rayDir;
#else
        segment.ray.origin = cam.position;
        segment.ray.direction = pinhole_direction;
#endif

#if ENABLE_SPECTRAL_RENDERING
        float u_hero = curand_uniform(&local_rand_state);
        for (int i = 0; i < SPECTRAL_N; i++) {
            float u_i = u_hero + (float)i / (float)SPECTRAL_N;
            if (u_i >= 1.0f) u_i -= 1.0f;
            float pdf_i;
            segment.wavelengths[i] = sample_cie_wavelength(u_i, &pdf_i);
            segment.throughputs[i] = 1.0f;
            segment.pdfs[i] = pdf_i;
        }
#else
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);
#endif

        segment.lastBrdfPdf = -1.0f;
        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
        rand_states[index] = local_rand_state;
    }
}

// ============================================================================
// OptiX hybrid: merge OptiX triangle results with box/sphere CUDA intersection
// ============================================================================

#if ENABLE_OPTIX
__global__ void kernMergeOptixAndPrimitives(
    int num_paths,
    PathSegment* pathSegments,
    OptiXHitResult* optixResults,
    Geom* nonTriGeoms,        // ONLY box/sphere geoms (not triangles!)
    int numNonTriGeoms,       // typically ~6
    Material* materials,
    glm::vec3* positions,
    glm::vec3* normals,

    MissWorkItem* miss_queue, int* miss_queue_counter,
    HitLightWorkItem* hit_light_queue, int* hit_light_queue_counter,
    LambertianHitWorkItem* lambertian_hit_queue, int* lambertian_hit_queue_counter,
    SpecularHitWorkItem* specular_hit_queue, int* specular_hit_queue_counter,
    GlassHitWorkItem* glass_hit_queue, int* glass_hit_queue_counter,
    DisneyGGXHitWorkItem* disney_ggx_hit_queue, int* disney_ggx_hit_queue_counter
)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (path_index >= num_paths) return;

    PathSegment& pathSegment = pathSegments[path_index];
    if (pathSegment.remainingBounces <= 0) return;

    // Start with OptiX triangle result
    OptiXHitResult optixHit = optixResults[path_index];
    float best_t = (optixHit.t > 0.0f) ? optixHit.t : FLT_MAX;
    glm::vec3 best_normal = optixHit.normal;
    glm::vec2 best_uv = optixHit.uv;
    int best_materialId = optixHit.materialId;
    int best_geomIdx = -1;
    bool found_hit = (optixHit.t > 0.0f);

    // Test ONLY box/sphere primitives (typically 6 -- very fast)
    for (int i = 0; i < numNonTriGeoms; i++) {
        Geom& geom = nonTriGeoms[i];

        float t = -1.0f;
        glm::vec3 tmp_intersect, tmp_normal;
        bool outside = true;

        if (geom.type == CUBE) {
            t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
        } else if (geom.type == SPHERE) {
            t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
        }

        if (t > 0.0f && t < best_t) {
            best_t = t;
            best_normal = tmp_normal;
            best_uv = glm::vec2(0.0f);
            best_materialId = geom.materialid;
            best_geomIdx = i;
            found_hit = true;
        }
    }

    // Compute intersection point
    glm::vec3 intersect_point = pathSegment.ray.origin + pathSegment.ray.direction * best_t;

    // Route to wavefront queues
    if (!found_hit) {
        MissWorkItem item = { path_index };
        int slot = atomicAdd(miss_queue_counter, 1);
        if (slot < num_paths) miss_queue[slot] = item;
    } else {
        Material mat = materials[best_materialId];
        if (mat.emittance > 0.0f) {
            HitLightWorkItem item;
            item.path_idx = path_index;
            item.material_id = best_materialId;
            item.geom_idx = (best_geomIdx >= 0) ? best_geomIdx : 0;
            item.hit_point = intersect_point;
            item.hit_normal = best_normal;
            int slot = atomicAdd(hit_light_queue_counter, 1);
            if (slot < num_paths) hit_light_queue[slot] = item;
        } else {
            switch (mat.type) {
                case LAMBERTIAN: {
                    LambertianHitWorkItem item = { path_index, best_materialId, intersect_point, best_normal, best_uv };
                    int slot = atomicAdd(lambertian_hit_queue_counter, 1);
                    if (slot < num_paths) lambertian_hit_queue[slot] = item;
                    break;
                }
                case SPECULAR: {
                    SpecularHitWorkItem item = { path_index, best_materialId, intersect_point, best_normal,
                        pathSegments[path_index].ray.direction, best_uv };
                    int slot = atomicAdd(specular_hit_queue_counter, 1);
                    if (slot < num_paths) specular_hit_queue[slot] = item;
                    break;
                }
                case GLASS: {
                    GlassHitWorkItem item = { path_index, best_materialId, mat.indexOfRefraction,
                        intersect_point, best_normal, pathSegments[path_index].ray.direction, best_uv };
                    int slot = atomicAdd(glass_hit_queue_counter, 1);
                    if (slot < num_paths) glass_hit_queue[slot] = item;
                    break;
                }
                case DISNEY_GGX: {
                    DisneyGGXHitWorkItem item = { path_index, best_materialId,
                        intersect_point, best_normal, pathSegments[path_index].ray.direction, best_uv };
                    int slot = atomicAdd(disney_ggx_hit_queue_counter, 1);
                    if (slot < num_paths) disney_ggx_hit_queue[slot] = item;
                    break;
                }
            }
        }
    }
}
#endif

// ============================================================================
// Brute-force intersection kernels (fallback when OptiX disabled)
// ============================================================================

#if !ENABLE_OPTIX && ENABLE_WAVEFRONT
__global__ void kernComputerIntersectionAndPartition(
    int num_paths, PathSegment* pathSegments, Geom* geoms, int geoms_size, Material* materials,
    glm::vec3* positions, glm::vec3* normals, glm::vec2* texCoords,
    cudaTextureObject_t* textureObjects, int numTextures,
    MissWorkItem* miss_queue, int* miss_queue_counter,
    HitLightWorkItem* hit_light_queue, int* hit_light_queue_counter,
    LambertianHitWorkItem* lambertian_hit_queue, int* lambertian_hit_queue_counter,
    SpecularHitWorkItem* specular_hit_queue, int* specular_hit_queue_counter,
    GlassHitWorkItem* glass_hit_queue, int* glass_hit_queue_counter,
    DisneyGGXHitWorkItem* disney_ggx_hit_queue, int* disney_ggx_hit_queue_counter)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (path_index >= num_paths) return;

    PathSegment pathSegment = pathSegments[path_index];
    float t_min = FLT_MAX;
    int hit_geom_index = -1;
    float t;
    glm::vec3 intersect_point, normal;
    bool outside = true;
    float hit_baryU = 0.0f, hit_baryV = 0.0f;
    glm::vec3 tmp_intersect, tmp_normal;
    float tmpU, tmpV;

    for (int i = 0; i < geoms_size; i++) {
        Geom& geom = geoms[i];
        if (geom.type == CUBE)
            t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
        else if (geom.type == SPHERE)
            t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
        else if (geom.type == TRIANGLE) {
            const glm::vec3& v0 = positions[geom.v0];
            const glm::vec3& v1 = positions[geom.v1];
            const glm::vec3& v2 = positions[geom.v2];
            t = triangleIntersectionTest(v0, v1, v2, geom, pathSegment.ray, tmp_intersect, tmp_normal, outside, tmpU, tmpV);
        }
        if (t > 0.0f && t_min > t) {
            t_min = t; hit_geom_index = i; intersect_point = tmp_intersect; normal = tmp_normal;
            if (geom.type == TRIANGLE) { hit_baryU = tmpU; hit_baryV = tmpV; }
        }
    }

    glm::vec2 hit_uv(0.0f);
    if (hit_geom_index >= 0 && geoms[hit_geom_index].type == TRIANGLE) {
        Geom& hitGeom = geoms[hit_geom_index];
        float w = 1.0f - hit_baryU - hit_baryV;
        glm::vec3 n0 = normals[hitGeom.v0], n1 = normals[hitGeom.v1], n2 = normals[hitGeom.v2];
        glm::vec3 interpolatedNormal = glm::normalize(w * n0 + hit_baryU * n1 + hit_baryV * n2);
        if (glm::dot(interpolatedNormal, normal) < 0.0f) interpolatedNormal = -interpolatedNormal;
        normal = interpolatedNormal;
        hit_uv = w * hitGeom.uv0 + hit_baryU * hitGeom.uv1 + hit_baryV * hitGeom.uv2;
    }

    if (hit_geom_index == -1) {
        MissWorkItem item = { path_index };
        int idx = atomicAdd(miss_queue_counter, 1);
        if (idx < num_paths) miss_queue[idx] = item;
    } else {
        int material_id = geoms[hit_geom_index].materialid;
        Material mat = materials[material_id];
        if (mat.emittance > 0.0f) {
            HitLightWorkItem item; item.path_idx = path_index; item.material_id = material_id;
            item.geom_idx = hit_geom_index; item.hit_point = intersect_point; item.hit_normal = normal;
            int idx = atomicAdd(hit_light_queue_counter, 1);
            if (idx < num_paths) hit_light_queue[idx] = item;
        } else {
            switch (mat.type) {
                case LAMBERTIAN: {
                    LambertianHitWorkItem item = { path_index, material_id, intersect_point, normal, hit_uv };
                    int idx = atomicAdd(lambertian_hit_queue_counter, 1);
                    if (idx < num_paths) lambertian_hit_queue[idx] = item;
                    break;
                }
                case SPECULAR: {
                    SpecularHitWorkItem item = { path_index, material_id, intersect_point, normal, pathSegments[path_index].ray.direction, hit_uv };
                    int idx = atomicAdd(specular_hit_queue_counter, 1);
                    if (idx < num_paths) specular_hit_queue[idx] = item;
                    break;
                }
                case GLASS: {
                    GlassHitWorkItem item = { path_index, material_id, mat.indexOfRefraction, intersect_point, normal, pathSegments[path_index].ray.direction, hit_uv };
                    int idx = atomicAdd(glass_hit_queue_counter, 1);
                    if (idx < num_paths) glass_hit_queue[idx] = item;
                    break;
                }
                case DISNEY_GGX: {
                    DisneyGGXHitWorkItem item = { path_index, material_id,
                        intersect_point, normal, pathSegments[path_index].ray.direction, hit_uv };
                    int idx = atomicAdd(disney_ggx_hit_queue_counter, 1);
                    if (idx < num_paths) disney_ggx_hit_queue[idx] = item;
                    break;
                }
            }
        }
    }
}
#endif

#if !ENABLE_WAVEFRONT && !ENABLE_OPTIX
__global__ void computeIntersections(
    int depth, int num_paths, PathSegment* pathSegments,
    Geom* geoms, int geoms_size, ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (path_index < num_paths) {
        PathSegment pathSegment = pathSegments[path_index];
        float t, t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;
        glm::vec3 intersect_point, normal, tmp_intersect, tmp_normal;
        for (int i = 0; i < geoms_size; i++) {
            Geom& geom = geoms[i];
            if (geom.type == CUBE) t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            else if (geom.type == SPHERE) t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            if (t > 0.0f && t_min > t) { t_min = t; hit_geom_index = i; intersect_point = tmp_intersect; normal = tmp_normal; }
        }
        if (hit_geom_index == -1) { intersections[path_index].t = -1.0f; }
        else { intersections[path_index].t = t_min; intersections[path_index].materialId = geoms[hit_geom_index].materialid; intersections[path_index].surfaceNormal = normal; }
    }
}

__global__ void kernShadeMaterial(int iter, int num_paths, ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments, Material* materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths) {
        if (pathSegments[idx].remainingBounces <= 0) return;
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) {
            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
                pathSegments[idx].remainingBounces = 0;
            } else {
                thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
                thrust::uniform_real_distribution<float> u01(0, 1);
                Ray& ray = pathSegments[idx].ray;
                glm::vec3 intersect = ray.origin + ray.direction * intersection.t;
                scatterRay(pathSegments[idx], intersect, intersection.surfaceNormal, material, rng);
                float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
                pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
                pathSegments[idx].color *= u01(rng);
            }
        } else { pathSegments[idx].color = glm::vec3(0.0f); pathSegments[idx].remainingBounces = 0; }
    }
}
#endif

// ============================================================================
// Main path tracing function
// ============================================================================

void pathtrace(uchar4* pbo, int frame, int iter, bool denoiserEnabled)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths, dev_rand_states);
    checkCUDAError("generate camera ray");

    int num_paths = pixelcount;
    int num_active_paths = num_paths;

    for (int depth = 0; depth < traceDepth; ++depth)
    {
        if (num_active_paths == 0) break;

        cudaMemset(miss_queue_counter, 0, sizeof(int));
        cudaMemset(hit_light_queue_counter, 0, sizeof(int));
        cudaMemset(lambertian_queue_counter, 0, sizeof(int));
        cudaMemset(specular_queue_counter, 0, sizeof(int));
        cudaMemset(glass_queue_counter, 0, sizeof(int));
        cudaMemset(disney_ggx_queue_counter, 0, sizeof(int));

        dim3 numblocksPathSegmentTracing = (num_active_paths + BLOCKSIZE1d - 1) / BLOCKSIZE1d;

        // ===== INTERSECTION =====
#if ENABLE_OPTIX
        // Step 1: OptiX traces triangles (RT Core accelerated)
        optixRenderer->trace(dev_paths, num_active_paths);
        cudaDeviceSynchronize();  // wait for OptiX before merge

        // Step 2: CUDA merge -- tests only box/sphere (6 items), fills queues
        kernMergeOptixAndPrimitives<<<numblocksPathSegmentTracing, BLOCKSIZE1d>>>(
            num_active_paths,
            dev_paths,
            optixRenderer->dev_hitResults,
            dev_nonTriangleGeoms,
            hst_num_non_triangle_geoms,
            dev_materials,
            dev_positions,
            dev_normals,
            miss_queue, miss_queue_counter,
            hit_light_queue, hit_light_queue_counter,
            lambertian_queue, lambertian_queue_counter,
            specular_queue, specular_queue_counter,
            glass_queue, glass_queue_counter,
            disney_ggx_queue, disney_ggx_queue_counter
        );

#elif ENABLE_WAVEFRONT
        kernComputerIntersectionAndPartition<<<numblocksPathSegmentTracing, BLOCKSIZE1d>>>(
            num_active_paths, dev_paths, dev_geoms, hst_scene->geoms.size(), dev_materials,
            dev_positions, dev_normals, dev_texcoords, dev_texture_objects, num_textures,
            miss_queue, miss_queue_counter, hit_light_queue, hit_light_queue_counter,
            lambertian_queue, lambertian_queue_counter, specular_queue, specular_queue_counter,
            glass_queue, glass_queue_counter,
            disney_ggx_queue, disney_ggx_queue_counter);
#else
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
        computeIntersections<<<numblocksPathSegmentTracing, BLOCKSIZE1d>>>(
            depth, num_active_paths, dev_paths, dev_geoms, hst_scene->geoms.size(), dev_intersections);
#endif
        // Sync after intersection (needed for queue counters)
        cudaDeviceSynchronize();

#if ENABLE_MATERIAL_SORTING
        thrust::device_ptr<ShadeableIntersection> devPtr_intersections(dev_intersections);
        thrust::device_ptr<PathSegment> devPtr_paths(dev_paths);
        thrust::stable_sort_by_key(devPtr_intersections, devPtr_intersections + num_paths, devPtr_paths, CompareMaterial());
#endif

        // ===== SHADING =====
#if ENABLE_WAVEFRONT || ENABLE_OPTIX
        int num_miss = getQueueCount(miss_queue_counter);
        if (num_miss > 0) {
            dim3 nb = (num_miss + BLOCKSIZE1d - 1) / BLOCKSIZE1d;
            kernShadeMiss<<<nb, BLOCKSIZE1d>>>(num_miss, miss_queue, dev_paths, dev_image,
#if ENABLE_DENOISER
                dev_albedo_buffer, dev_normal_buffer, depth,
#else
                nullptr, nullptr, depth,
#endif
                hst_envMapTexObj, hst_hasEnvMap,
                dev_envCDF_marginal, dev_envCDF_conditional,
                hst_envMapWidth, hst_envMapHeight
            );
        }
        checkCUDAError("Miss Done");

        int num_hitLight = getQueueCount(hit_light_queue_counter);
        if (num_hitLight > 0) {
            dim3 nb = (num_hitLight + BLOCKSIZE1d - 1) / BLOCKSIZE1d;
            kernShadeHitLight<<<nb, BLOCKSIZE1d>>>(num_hitLight, hit_light_queue, dev_paths, dev_materials, dev_image,
                dev_geoms, dev_positions, dev_light_indices, hst_num_lights,
#if ENABLE_DENOISER
                dev_albedo_buffer, dev_normal_buffer, depth
#else
                nullptr, nullptr, depth
#endif
            );
        }
        checkCUDAError("Hit Done");

        int num_lambertian = getQueueCount(lambertian_queue_counter);
        if (num_lambertian > 0) {
#if ENABLE_OPTIX && ENABLE_MIS
            // OptiX-accelerated MIS: prepare -> trace -> apply shadow rays
            dim3 nb_shadow = (num_lambertian + BLOCKSIZE1d - 1) / BLOCKSIZE1d;
            kernPrepareShadowRays<<<nb_shadow, BLOCKSIZE1d>>>(
                num_lambertian, lambertian_queue, dev_paths, dev_materials,
                dev_rand_states, dev_geoms, hst_scene->geoms.size(), dev_positions,
                dev_light_indices, hst_num_lights,
                dev_nonTriangleGeoms, hst_num_non_triangle_geoms,
                dev_shadowRays, dev_texture_objects, num_textures);
            cudaDeviceSynchronize();

            // Batch trace shadow rays via RT Core
            optixRenderer->traceShadowRays(dev_shadowRays, num_lambertian, dev_shadowOccluded);
            cudaDeviceSynchronize();

            // Apply visible contributions to image
            kernApplyShadowResults<<<nb_shadow, BLOCKSIZE1d>>>(
                num_lambertian, dev_shadowRays, dev_shadowOccluded, dev_image);

            // Env map NEE: sample env map, trace shadow rays, apply
            if (hst_hasEnvMap && dev_envCDF_marginal) {
                kernPrepareEnvMapShadowRays<<<nb_shadow, BLOCKSIZE1d>>>(
                    num_lambertian, lambertian_queue, dev_paths, dev_materials,
                    dev_rand_states,
                    dev_nonTriangleGeoms, hst_num_non_triangle_geoms,
                    dev_shadowRays, dev_texture_objects, num_textures,
                    hst_envMapTexObj,
                    dev_envCDF_marginal, dev_envCDF_conditional,
                    hst_envMapWidth, hst_envMapHeight, hst_envTotalPower);
                cudaDeviceSynchronize();

                optixRenderer->traceShadowRays(dev_shadowRays, num_lambertian, dev_shadowOccluded);
                cudaDeviceSynchronize();

                kernApplyShadowResults<<<nb_shadow, BLOCKSIZE1d>>>(
                    num_lambertian, dev_shadowRays, dev_shadowOccluded, dev_image);
            }
#endif
            // BRDF sampling (indirect bounce) -- always runs
            dim3 nb = (num_lambertian + BLOCKSIZE1d - 1) / BLOCKSIZE1d;
            kernShadeLambertian<<<nb, BLOCKSIZE1d>>>(num_lambertian, lambertian_queue, dev_paths, dev_materials,
                dev_rand_states, dev_image, dev_geoms, hst_scene->geoms.size(), dev_positions,
                dev_light_indices, hst_num_lights, dev_texture_objects, num_textures,
#if ENABLE_DENOISER
                dev_albedo_buffer, dev_normal_buffer, depth
#else
                nullptr, nullptr, depth
#endif
            );
        }
        checkCUDAError("Lambertian Done");

#if ENABLE_SPECULAR
        int num_specular = getQueueCount(specular_queue_counter);
        if (num_specular > 0) {
            dim3 nb = (num_specular + BLOCKSIZE1d - 1) / BLOCKSIZE1d;
            kernShadeSpecular<<<nb, BLOCKSIZE1d>>>(num_specular, specular_queue, dev_paths, dev_materials,
#if ENABLE_DENOISER
                dev_albedo_buffer, dev_normal_buffer, depth
#else
                nullptr, nullptr, depth
#endif
            );
        }
        checkCUDAError("Specular Done");
#endif

#if ENABLE_GLASS
        int num_glass = getQueueCount(glass_queue_counter);
        if (num_glass > 0) {
            dim3 nb = (num_glass + BLOCKSIZE1d - 1) / BLOCKSIZE1d;
            kernShadeGlass<<<nb, BLOCKSIZE1d>>>(num_glass, glass_queue, dev_paths, dev_materials, dev_rand_states,
#if ENABLE_DENOISER
                dev_albedo_buffer, dev_normal_buffer, depth
#else
                nullptr, nullptr, depth
#endif
            );
        }
        checkCUDAError("Glass Done");
#endif

#if ENABLE_DISNEY_GGX
        int num_disney_ggx = getQueueCount(disney_ggx_queue_counter);
        if (num_disney_ggx > 0) {
#if ENABLE_OPTIX && ENABLE_MIS
            // OptiX-accelerated MIS: prepare -> trace -> apply shadow rays for Disney GGX
            dim3 nb_shadow_ggx = (num_disney_ggx + BLOCKSIZE1d - 1) / BLOCKSIZE1d;
            kernPrepareShadowRaysDisneyGGX<<<nb_shadow_ggx, BLOCKSIZE1d>>>(
                num_disney_ggx, disney_ggx_queue, dev_paths, dev_materials,
                dev_rand_states, dev_geoms, hst_scene->geoms.size(), dev_positions,
                dev_light_indices, hst_num_lights,
                dev_nonTriangleGeoms, hst_num_non_triangle_geoms,
                dev_shadowRays, dev_texture_objects, num_textures);
            cudaDeviceSynchronize();

            optixRenderer->traceShadowRays(dev_shadowRays, num_disney_ggx, dev_shadowOccluded);
            cudaDeviceSynchronize();

            kernApplyShadowResults<<<nb_shadow_ggx, BLOCKSIZE1d>>>(
                num_disney_ggx, dev_shadowRays, dev_shadowOccluded, dev_image);

            // Env map NEE for Disney GGX
            if (hst_hasEnvMap && dev_envCDF_marginal) {
                kernPrepareEnvMapShadowRaysDisneyGGX<<<nb_shadow_ggx, BLOCKSIZE1d>>>(
                    num_disney_ggx, disney_ggx_queue, dev_paths, dev_materials,
                    dev_rand_states,
                    dev_nonTriangleGeoms, hst_num_non_triangle_geoms,
                    dev_shadowRays, dev_texture_objects, num_textures,
                    hst_envMapTexObj,
                    dev_envCDF_marginal, dev_envCDF_conditional,
                    hst_envMapWidth, hst_envMapHeight, hst_envTotalPower);
                cudaDeviceSynchronize();

                optixRenderer->traceShadowRays(dev_shadowRays, num_disney_ggx, dev_shadowOccluded);
                cudaDeviceSynchronize();

                kernApplyShadowResults<<<nb_shadow_ggx, BLOCKSIZE1d>>>(
                    num_disney_ggx, dev_shadowRays, dev_shadowOccluded, dev_image);
            }
#endif
            // BRDF sampling (indirect bounce)
            dim3 nb = (num_disney_ggx + BLOCKSIZE1d - 1) / BLOCKSIZE1d;
            kernShadeDisneyGGX<<<nb, BLOCKSIZE1d>>>(num_disney_ggx, disney_ggx_queue, dev_paths, dev_materials,
                dev_rand_states, dev_image, dev_geoms, hst_scene->geoms.size(), dev_positions,
                dev_light_indices, hst_num_lights, dev_texture_objects, num_textures,
#if ENABLE_DENOISER
                dev_albedo_buffer, dev_normal_buffer, depth
#else
                nullptr, nullptr, depth
#endif
            );
        }
        checkCUDAError("DisneyGGX Done");
#endif

#else
        kernShadeMaterial<<<numblocksPathSegmentTracing, BLOCKSIZE1d>>>(iter, num_active_paths, dev_intersections, dev_paths, dev_materials);
#endif
        checkCUDAError("Shading Done");
        cudaDeviceSynchronize();

#if ENABLE_TERMINATE_DEAD_RAYS
        PathSegment* new_end = thrust::partition(thrust::device, dev_paths, dev_paths + num_active_paths, is_ray_alive());
        num_active_paths = new_end - dev_paths;
#endif

        if (guiData != NULL) { guiData->TracedDepth = depth; guiData->CamPos = cam.position; }
    }

    // ===== DISPLAY =====
#if ENABLE_DENOISER
    if (denoiserEnabled) {
        // Run denoiser and display denoised result
        optixRenderer->denoise(dev_image, dev_albedo_buffer, dev_normal_buffer,
                               dev_denoised_image, cam.resolution.x, cam.resolution.y, iter);
        cudaDeviceSynchronize();
        sendDenoisedImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, dev_denoised_image);
    } else {
        sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);
    }
#else
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);
#endif
    cudaMemcpy(hst_scene->state.image.data(), dev_image, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    checkCUDAError("pathtrace");
}

// ============================================================================
// Runtime Reload Functions
// ============================================================================

void pathtraceReloadEnvMap(const EnvironmentMap& envMap) {
    // Destroy old env map texture + CDF
    if (hst_hasEnvMap) {
        cudaDestroyTextureObject(hst_envMapTexObj);
        cudaFreeArray(dev_envMapArray);
        hst_envMapTexObj = 0;
        dev_envMapArray = NULL;
        hst_hasEnvMap = false;
    }
    if (dev_envCDF_marginal) { cudaFree(dev_envCDF_marginal); dev_envCDF_marginal = NULL; }
    if (dev_envCDF_conditional) { cudaFree(dev_envCDF_conditional); dev_envCDF_conditional = NULL; }

    if (!envMap.loaded) return;

    int envW = envMap.width;
    int envH = envMap.height;
    // Convert RGB float to RGBA float
    std::vector<float4> envRGBA(envW * envH);
    for (int i = 0; i < envW * envH; i++) {
        envRGBA[i] = make_float4(
            envMap.pixels[i * 3 + 0],
            envMap.pixels[i * 3 + 1],
            envMap.pixels[i * 3 + 2],
            1.0f);
    }
    // Create CUDA array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    cudaMallocArray(&dev_envMapArray, &channelDesc, envW, envH);
    cudaMemcpy2DToArray(dev_envMapArray, 0, 0,
        envRGBA.data(), envW * sizeof(float4),
        envW * sizeof(float4), envH, cudaMemcpyHostToDevice);
    // Create texture
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = dev_envMapArray;
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;
    cudaCreateTextureObject(&hst_envMapTexObj, &resDesc, &texDesc, NULL);
    hst_hasEnvMap = true;
    printf("Reloaded HDRI environment map texture (%dx%d)\n", envW, envH);

    // Upload CDF for importance sampling
    if (!envMap.marginalCDF.empty()) {
        hst_envMapWidth = envW;
        hst_envMapHeight = envH;
        hst_envTotalPower = envMap.totalPower;
        size_t margSize = envMap.marginalCDF.size() * sizeof(float);
        size_t condSize = envMap.conditionalCDF.size() * sizeof(float);
        cudaMalloc(&dev_envCDF_marginal, margSize);
        cudaMemcpy(dev_envCDF_marginal, envMap.marginalCDF.data(), margSize, cudaMemcpyHostToDevice);
        cudaMalloc(&dev_envCDF_conditional, condSize);
        cudaMemcpy(dev_envCDF_conditional, envMap.conditionalCDF.data(), condSize, cudaMemcpyHostToDevice);
        printf("Uploaded env map importance sampling CDF\n");
    }

    // Clear accumulated image to restart rendering with new env map
    int pixelcount = hst_scene->state.camera.resolution.x * hst_scene->state.camera.resolution.y;
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));
    checkCUDAError("pathtraceReloadEnvMap");
}

void pathtraceReloadScene(Scene* newScene) {
    pathtraceFree();
    pathtraceInit(newScene);
}
