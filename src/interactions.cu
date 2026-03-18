#include "interactions.h"

#include "utilities.h"

#include "dispersion.h"

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
}

__global__ void kernShadeMiss(int num_hit, MissWorkItem* queue, PathSegment* paths, glm::vec3* dev_img) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_hit) return;

    MissWorkItem item = queue[idx];
    PathSegment& path = paths[item.path_idx];

    path.remainingBounces = 0;
    //path.color = DEBUG_EMPTY_COLOR;
}

__global__ void kernShadeHitLight(int num_hit, HitLightWorkItem* queue, PathSegment* paths, Material* materials, glm::vec3* dev_img) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_hit) return;

    HitLightWorkItem item = queue[idx];
    PathSegment& path = paths[item.path_idx];
    Material material = materials[item.material_id];

    glm::vec3 contribution;

#if ENABLE_SPECTRAL_RENDERING
    float range = (float)(MAX_SAMPLE_WAVELENGTH - MIN_SAMPLE_WAVELENGTH);
    for (int i = 0; i < SPECTRAL_N; i++) {
        float color_at_wl = spectral_reflectance_from_rgb(material.color, path.wavelengths[i]);
        // IS weight: divide by (range * pdf) to compensate for importance sampling
        float is_weight = 1.0f / (range * path.pdfs[i]);
        float radiance = path.throughputs[i] * material.emittance * color_at_wl * is_weight;
        glm::vec3 c = wavelength_to_RGB(path.wavelengths[i]) * radiance;
        atomicAdd(&dev_img[path.pixelIndex].x, c.x);
        atomicAdd(&dev_img[path.pixelIndex].y, c.y);
        atomicAdd(&dev_img[path.pixelIndex].z, c.z);
    }
#else
    contribution = path.color * material.color * material.emittance;
    atomicAdd(&dev_img[path.pixelIndex].x, contribution.x);
    atomicAdd(&dev_img[path.pixelIndex].y, contribution.y);
    atomicAdd(&dev_img[path.pixelIndex].z, contribution.z);
#endif

    path.remainingBounces = 0;
}

__global__ void kernShadeLambertian(int num_hit, LambertianHitWorkItem* queue, PathSegment* paths, Material* materials, curandState* rand_states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_hit) return;

    LambertianHitWorkItem item = queue[idx];
    PathSegment& path = paths[item.path_idx];
    Material material = materials[item.material_id];

    curandState local_rand_state = rand_states[path.pixelIndex];

    const glm::vec2 xi = glm::vec2(curand_uniform(&local_rand_state), curand_uniform(&local_rand_state));
    float rand_for_rr = curand_uniform(&local_rand_state);


    glm::vec3 nor = item.surface_normal;
    glm::vec3 wiLocal = squareToHemisphereCosine(xi);
    glm::mat3 mat = TangentSpaceToWorld(nor);
    glm::vec3 wiWorld = mat * wiLocal;

#if ENABLE_SPECTRAL_RENDERING
    for (int i = 0; i < SPECTRAL_N; i++) {
        path.throughputs[i] *= spectral_reflectance_from_rgb(material.color, path.wavelengths[i]);
    }
#else
    path.color *= material.color;
#endif

    //float survival_prob = glm::max(path.color.r, glm::max(path.color.g, path.color.b));
    //survival_prob = glm::min(survival_prob, 1.0f);

    //if (rand > survival_prob){
    //    path.remainingBounces = 0;
    //}
    //else {
    //    path.color /= survival_prob;
    //}

    path.remainingBounces--;
    rand_states[path.pixelIndex] = local_rand_state;
    path.ray.origin = item.intersect_point + nor * EPSILON;
    path.ray.direction = wiWorld;
}

__global__ void kernShadeSpecular(int num_hit, SpecularHitWorkItem* queue, PathSegment* paths, Material* materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_hit) return;

    SpecularHitWorkItem item = queue[idx];
    PathSegment& path = paths[item.path_idx];
    Material material = materials[item.material_id];

    glm::vec3 reflected_dir = glm::reflect(item.incident_ray_dir, item.surface_normal);

#if ENABLE_SPECTRAL_RENDERING
    for (int i = 0; i < SPECTRAL_N; i++) {
        path.throughputs[i] *= spectral_reflectance_from_rgb(material.color, path.wavelengths[i]);
    }
#else
    // --- RGB Mode ---
    // Attenuate the RGB throughput by the full specular color.
    path.color *= material.color;
#endif

    path.ray.origin = item.intersect_point + item.surface_normal * EPSILON;
    path.ray.direction = reflected_dir;

    path.remainingBounces--;
}

__device__ float FresnelDielectricEval(float cosThetaI, float IOR) {
    float etaI = 1.f;
    float etaT = IOR;
    cosThetaI = glm::clamp(cosThetaI, -1.f, 1.f);

    if (cosThetaI > 0.f) {
        float temp = etaI;
        etaI = etaT;
        etaT = temp;
    }
    cosThetaI = glm::abs(cosThetaI);

    float sinThetaI = glm::sqrt(glm::max(0.f, 1.f - cosThetaI * cosThetaI));
    float sinThetaT = etaI / etaT * sinThetaI;
    float cosThetaT = glm::sqrt(glm::max(0.f, 1.f - sinThetaT * sinThetaT));
    float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
        ((etaT * cosThetaI) + (etaI * cosThetaT));
    float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
        ((etaI * cosThetaI) + (etaT * cosThetaT));

    return (Rparl * Rparl + Rperp * Rperp) * 0.5f;
}

__host__ __device__ float schlickFresnel(float cosTheta, float ior) {
    float r0 = (1.0f - ior) / (1.0f + ior);
    r0 = r0 * r0;
    float x = 1.0f - cosTheta;
    float x2 = x * x;
    return r0 + (1.0f - r0) * x2 * x2 * x;
}

__global__ void kernShadeGlass(int num_hit, GlassHitWorkItem* queue, PathSegment* paths, Material* materials, curandState* rand_states)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_hit) return;

    GlassHitWorkItem item = queue[idx];
    PathSegment& path = paths[item.path_idx];
    Material material = materials[item.material_id];
    curandState local_rand_state = rand_states[path.pixelIndex];

    glm::vec3 normal = glm::normalize(item.surface_normal);
    path.ray.origin = item.intersect_point;

    float cos_theta = glm::dot(glm::normalize(path.ray.direction), normal);

    float ior_for_this_ray;
    float base_IOR = material.indexOfRefraction;
#if ENABLE_SPECTRAL_RENDERING
    // --- Spectral Mode: Hero wavelength (index 0) determines direction ---
    float dispersion_abbe = material.abbe;
    ior_for_this_ray = compute_dispersion_ior(dispersion_abbe, base_IOR, path.wavelengths[0]);
#else
    // --- RGB Mode: Use the single, base IOR ---
    ior_for_this_ray = base_IOR;
#endif

    float eta1, eta2_val;
    glm::vec3 oriented_normal = normal;
    float abs_cos_theta;
    if (cos_theta < 0.0f) {
        eta1 = 1.0f;
        eta2_val = ior_for_this_ray;
        abs_cos_theta = -cos_theta;
    }
    else {
        eta1 = ior_for_this_ray;
        eta2_val = 1.0f;
        oriented_normal = -normal;
        abs_cos_theta = cos_theta;
    }

    float F_hero = FresnelDielectricEval(abs_cos_theta, eta1 / eta2_val);

    float rand = curand_uniform(&local_rand_state);
    bool did_reflect = (rand <= F_hero);

    glm::vec3 new_direction;
    if (did_reflect) {
        new_direction = glm::reflect(path.ray.direction, oriented_normal);
    }
    else {
        path.ray.origin += 0.0002f * glm::normalize(path.ray.direction);
        new_direction = glm::refract(glm::normalize(path.ray.direction), oriented_normal, eta1 / eta2_val);
    }

#if ENABLE_SPECTRAL_RENDERING
    // Update throughputs for all N wavelengths with importance weighting
    for (int i = 0; i < SPECTRAL_N; i++) {
        float ior_i = compute_dispersion_ior(dispersion_abbe, base_IOR, path.wavelengths[i]);
        float eta1_i, eta2_i;
        if (cos_theta < 0.0f) { eta1_i = 1.0f; eta2_i = ior_i; }
        else                  { eta1_i = ior_i; eta2_i = 1.0f; }
        float F_i = FresnelDielectricEval(abs_cos_theta, eta1_i / eta2_i);

        // Importance weight: ratio of this wavelength's probability to hero's
        if (did_reflect) {
            path.throughputs[i] *= (F_hero > 1e-6f) ? (F_i / F_hero) : 0.0f;
        } else {
            path.throughputs[i] *= ((1.0f - F_hero) > 1e-6f) ? ((1.0f - F_i) / (1.0f - F_hero)) : 0.0f;
        }
        path.throughputs[i] *= spectral_reflectance_from_rgb(material.color, path.wavelengths[i]);
    }
#else
    path.color *= material.color;
#endif

    path.ray.direction = glm::normalize(new_direction);
    path.remainingBounces--;

    rand_states[path.pixelIndex] = local_rand_state;
}