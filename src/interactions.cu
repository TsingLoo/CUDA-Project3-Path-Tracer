#include "interactions.h"

#include "utilities.h"

#include "dispersion.h"
#include "intersections.h"

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

// ============================================================================
// MIS Utility Functions
// ============================================================================

__device__ inline float powerHeuristic(float pdf_a, float pdf_b) {
    float a2 = pdf_a * pdf_a;
    return a2 / (a2 + pdf_b * pdf_b);
}

__device__ float computeGeomArea(const Geom& geom, const glm::vec3* positions) {
    if (geom.type == SPHERE) {
        float r = (geom.scale.x + geom.scale.y + geom.scale.z) / 6.0f;
        return 4.0f * PI * r * r;
    } else if (geom.type == CUBE) {
        float sx = geom.scale.x, sy = geom.scale.y, sz = geom.scale.z;
        return 2.0f * (sx*sy + sy*sz + sx*sz);
    } else if (geom.type == TRIANGLE) {
        glm::vec3 e1 = positions[geom.v1] - positions[geom.v0];
        glm::vec3 e2 = positions[geom.v2] - positions[geom.v0];
        return 0.5f * glm::length(glm::cross(e1, e2));
    }
    return 1.0f;
}

struct LightSample {
    glm::vec3 position;
    glm::vec3 normal;
    float area;
};

__device__ LightSample sampleGeomLight(const Geom& geom, const glm::vec3* positions, curandState* rng) {
    LightSample ls;
    if (geom.type == SPHERE) {
        float z = 2.0f * curand_uniform(rng) - 1.0f;
        float r = sqrtf(fmaxf(0.0f, 1.0f - z*z));
        float phi = TWO_PI * curand_uniform(rng);
        glm::vec3 localPt(r * cosf(phi), r * sinf(phi), z);
        ls.position = glm::vec3(geom.transform * glm::vec4(localPt * 0.5f, 1.0f));
        ls.normal = glm::normalize(glm::vec3(geom.invTranspose * glm::vec4(localPt, 0.0f)));
        float ar = (geom.scale.x + geom.scale.y + geom.scale.z) / 6.0f;
        ls.area = 4.0f * PI * ar * ar;
    } else if (geom.type == CUBE) {
        float sx = geom.scale.x, sy = geom.scale.y, sz = geom.scale.z;
        // Face areas: +-X faces = sy*sz, +-Y faces = sx*sz, +-Z faces = sx*sy
        float aYZ = sy * sz;  // faces 0,1
        float aXZ = sx * sz;  // faces 2,3
        float aXY = sx * sy;  // faces 4,5
        float totalArea = 2.0f * (aXY + aYZ + aXZ);

        // Area-weighted face selection (CDF)
        float r = curand_uniform(rng) * totalArea;
        int face;
        if      (r < aYZ)                   face = 0;
        else if (r < 2.0f * aYZ)            face = 1;
        else if (r < 2.0f * aYZ + aXZ)      face = 2;
        else if (r < 2.0f * aYZ + 2.0f*aXZ) face = 3;
        else if (r < 2.0f*aYZ + 2.0f*aXZ + aXY) face = 4;
        else                                 face = 5;

        float u1 = curand_uniform(rng) - 0.5f;
        float u2 = curand_uniform(rng) - 0.5f;
        glm::vec3 lp, ln;
        switch (face) {
            case 0: lp = glm::vec3( 0.5f, u1, u2); ln = glm::vec3(1,0,0); break;
            case 1: lp = glm::vec3(-0.5f, u1, u2); ln = glm::vec3(-1,0,0); break;
            case 2: lp = glm::vec3(u1, 0.5f, u2); ln = glm::vec3(0,1,0); break;
            case 3: lp = glm::vec3(u1,-0.5f, u2); ln = glm::vec3(0,-1,0); break;
            case 4: lp = glm::vec3(u1, u2, 0.5f); ln = glm::vec3(0,0,1); break;
            default:lp = glm::vec3(u1, u2,-0.5f); ln = glm::vec3(0,0,-1); break;
        }
        ls.position = glm::vec3(geom.transform * glm::vec4(lp, 1.0f));
        ls.normal = glm::normalize(glm::vec3(geom.invTranspose * glm::vec4(ln, 0.0f)));
        ls.area = totalArea;
    } else { // TRIANGLE
        glm::vec3 v0 = positions[geom.v0], v1 = positions[geom.v1], v2 = positions[geom.v2];
        float u1 = curand_uniform(rng), u2 = curand_uniform(rng);
        if (u1 + u2 > 1.0f) { u1 = 1.0f - u1; u2 = 1.0f - u2; }
        ls.position = (1.0f - u1 - u2) * v0 + u1 * v1 + u2 * v2;
        glm::vec3 e1 = v1 - v0, e2 = v2 - v0;
        glm::vec3 cr = glm::cross(e1, e2);
        ls.normal = glm::normalize(cr);
        ls.area = 0.5f * glm::length(cr);
    }
    return ls;
}

__device__ bool shadowRayOccluded(
    glm::vec3 origin, glm::vec3 target, int skipGeomIdx,
    Geom* geoms, int geoms_size, glm::vec3* positions)
{
    glm::vec3 dir = target - origin;
    float dist = glm::length(dir);
    dir /= dist;
    Ray shadowRay = { origin + dir * EPSILON * 10.0f, dir };
    float maxDist = dist - EPSILON * 20.0f;
    glm::vec3 tmp_p, tmp_n;
    bool tmp_o;
    float tmp_bu, tmp_bv;  // unused barycentrics for shadow ray
    for (int i = 0; i < geoms_size; i++) {
        if (i == skipGeomIdx) continue;
        float t = -1.0f;
        if (geoms[i].type == CUBE)
            t = boxIntersectionTest(geoms[i], shadowRay, tmp_p, tmp_n, tmp_o);
        else if (geoms[i].type == SPHERE)
            t = sphereIntersectionTest(geoms[i], shadowRay, tmp_p, tmp_n, tmp_o);
        else if (geoms[i].type == TRIANGLE)
            t = triangleIntersectionTest(positions[geoms[i].v0], positions[geoms[i].v1],
                positions[geoms[i].v2], geoms[i], shadowRay, tmp_p, tmp_n, tmp_o, tmp_bu, tmp_bv);
        if (t > 0.0f && t < maxDist) return true;
    }
    return false;
}

// ============================================================================
// Shading Kernels
// ============================================================================

__global__ void kernShadeMiss(int num_hit, MissWorkItem* queue, PathSegment* paths, glm::vec3* dev_img,
    glm::vec3* dev_albedo, glm::vec3* dev_normal, int depth,
    cudaTextureObject_t envMap, bool hasEnvMap) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_hit) return;

    MissWorkItem item = queue[idx];
    PathSegment& path = paths[item.path_idx];

    // Sample HDRI environment map if available
    if (hasEnvMap) {
        glm::vec3 dir = glm::normalize(path.ray.direction);
        // Equirectangular mapping: direction -> UV
        float u = 0.5f + atan2f(dir.z, dir.x) / (2.0f * PI);
        float v = 0.5f - asinf(glm::clamp(dir.y, -1.0f, 1.0f)) / PI;

        float4 envColor = tex2D<float4>(envMap, u, v);
        glm::vec3 Le(envColor.x, envColor.y, envColor.z);

#if ENABLE_SPECTRAL_RENDERING
        float range = (float)(MAX_SAMPLE_WAVELENGTH - MIN_SAMPLE_WAVELENGTH);
        for (int i = 0; i < SPECTRAL_N; i++) {
            float le_wl = spectral_reflectance_from_rgb(Le, path.wavelengths[i]);
            float is_weight = 1.0f / (range * path.pdfs[i]);
            float radiance = path.throughputs[i] * le_wl * is_weight;
            glm::vec3 c = wavelength_to_RGB(path.wavelengths[i]) * radiance;
            atomicAdd(&dev_img[path.pixelIndex].x, c.x);
            atomicAdd(&dev_img[path.pixelIndex].y, c.y);
            atomicAdd(&dev_img[path.pixelIndex].z, c.z);
        }
#else
        glm::vec3 contribution = path.color * Le;
        atomicAdd(&dev_img[path.pixelIndex].x, contribution.x);
        atomicAdd(&dev_img[path.pixelIndex].y, contribution.y);
        atomicAdd(&dev_img[path.pixelIndex].z, contribution.z);
#endif

        // AOV: first bounce miss with env map -> env color as albedo
        if (depth == 0 && dev_albedo && dev_normal) {
            dev_albedo[path.pixelIndex] = glm::vec3(1.0f);
            dev_normal[path.pixelIndex] = glm::vec3(0.0f);
        }
    } else {
        // No env map -> black background
        if (depth == 0 && dev_albedo && dev_normal) {
            dev_albedo[path.pixelIndex] = glm::vec3(0.0f);
            dev_normal[path.pixelIndex] = glm::vec3(0.0f);
        }
    }

    path.remainingBounces = 0;
}

__global__ void kernShadeHitLight(
    int num_hit, HitLightWorkItem* queue, PathSegment* paths,
    Material* materials, glm::vec3* dev_img,
    Geom* geoms, glm::vec3* positions,
    int* light_indices, int num_lights,
    glm::vec3* dev_albedo, glm::vec3* dev_normal, int depth)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_hit) return;

    HitLightWorkItem item = queue[idx];
    PathSegment& path = paths[item.path_idx];
    Material material = materials[item.material_id];

    // Compute MIS weight for BRDF-sampled light hits
    float mis_weight = 1.0f;
#if ENABLE_MIS
    if (path.lastBrdfPdf > 0.0f && num_lights > 0 && item.geom_idx >= 0) {
        float area = computeGeomArea(geoms[item.geom_idx], positions);
        glm::vec3 toLight = item.hit_point - path.ray.origin;
        float d2 = glm::dot(toLight, toLight);
        float cos_light = fabs(glm::dot(glm::normalize(path.ray.direction), item.hit_normal));
        cos_light = fmaxf(cos_light, 1e-6f);
        float pdf_light = (d2 / (area * cos_light)) / (float)num_lights;
        mis_weight = powerHeuristic(path.lastBrdfPdf, pdf_light);
    }
#endif

    glm::vec3 contribution;

#if ENABLE_SPECTRAL_RENDERING
    float range = (float)(MAX_SAMPLE_WAVELENGTH - MIN_SAMPLE_WAVELENGTH);
    for (int i = 0; i < SPECTRAL_N; i++) {
        float color_at_wl = spectral_reflectance_from_rgb(material.color, path.wavelengths[i]);
        float is_weight = 1.0f / (range * path.pdfs[i]);
        float radiance = path.throughputs[i] * material.emittance * color_at_wl * is_weight * mis_weight;
        glm::vec3 c = wavelength_to_RGB(path.wavelengths[i]) * radiance;
        atomicAdd(&dev_img[path.pixelIndex].x, c.x);
        atomicAdd(&dev_img[path.pixelIndex].y, c.y);
        atomicAdd(&dev_img[path.pixelIndex].z, c.z);
    }
#else
    contribution = path.color * material.color * material.emittance * mis_weight;
    atomicAdd(&dev_img[path.pixelIndex].x, contribution.x);
    atomicAdd(&dev_img[path.pixelIndex].y, contribution.y);
    atomicAdd(&dev_img[path.pixelIndex].z, contribution.z);
#endif

    // AOV: first bounce hits light -> white albedo, surface normal
    if (depth == 0 && dev_albedo && dev_normal) {
        dev_albedo[path.pixelIndex] = glm::vec3(1.0f);
        dev_normal[path.pixelIndex] = glm::normalize(item.hit_normal);
    }

    path.remainingBounces = 0;
}

__global__ void kernShadeLambertian(
    int num_hit, LambertianHitWorkItem* queue, PathSegment* paths,
    Material* materials, curandState* rand_states, glm::vec3* dev_img,
    Geom* geoms, int geoms_size, glm::vec3* positions,
    int* light_indices, int num_lights,
    cudaTextureObject_t* textureObjects, int numTextures,
    glm::vec3* dev_albedo, glm::vec3* dev_normal, int depth)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_hit) return;

    LambertianHitWorkItem item = queue[idx];
    PathSegment& path = paths[item.path_idx];
    Material material = materials[item.material_id];

    // Sample base color texture if available
    if (material.textureId >= 0 && material.textureId < numTextures && textureObjects != nullptr) {
        float4 texColor = tex2D<float4>(textureObjects[material.textureId], item.uv.x, item.uv.y);
        material.color *= glm::vec3(texColor.x, texColor.y, texColor.z);
    }

    curandState local_rand_state = rand_states[path.pixelIndex];

    glm::vec3 nor = item.surface_normal;
    glm::vec3 hitPt = item.intersect_point;

    // AOV: first bounce Lambertian -> material color albedo, surface normal
    if (depth == 0 && dev_albedo && dev_normal) {
        dev_albedo[path.pixelIndex] = material.color;
        dev_normal[path.pixelIndex] = glm::normalize(nor);
    }

#if ENABLE_MIS && !ENABLE_OPTIX
    // === NEE: Direct Light Sampling (brute-force -- only used without OptiX) ===
    // When OptiX is enabled, shadow rays are handled externally via kernPrepareShadowRays
    if (num_lights > 0) {
        int li = (int)(curand_uniform(&local_rand_state) * (float)num_lights);
        if (li >= num_lights) li = num_lights - 1;
        int lightGeomIdx = light_indices[li];
        Geom lightGeom = geoms[lightGeomIdx];
        Material lightMat = materials[lightGeom.materialid];

        LightSample ls = sampleGeomLight(lightGeom, positions, &local_rand_state);

        glm::vec3 toLight = ls.position - hitPt;
        float d2 = glm::dot(toLight, toLight);
        float d = sqrtf(d2);
        glm::vec3 wi = toLight / d;

        float cos_surface = glm::dot(wi, nor);
        float cos_light = glm::dot(-wi, ls.normal);

        if (cos_surface > 0.0f && cos_light > 0.0f) {
            if (!shadowRayOccluded(hitPt + nor * EPSILON, ls.position,
                                  lightGeomIdx, geoms, geoms_size, positions)) {
                float pdf_light = (d2 / (ls.area * cos_light)) / (float)num_lights;
                float pdf_brdf = cos_surface / PI;
                float mis_w = powerHeuristic(pdf_light, pdf_brdf);

#if ENABLE_SPECTRAL_RENDERING
                float range = (float)(MAX_SAMPLE_WAVELENGTH - MIN_SAMPLE_WAVELENGTH);
                for (int i = 0; i < SPECTRAL_N; i++) {
                    float refl = spectral_reflectance_from_rgb(material.color, path.wavelengths[i]);
                    float le = spectral_reflectance_from_rgb(lightMat.color, path.wavelengths[i])
                             * lightMat.emittance;
                    float is_w = 1.0f / (range * path.pdfs[i]);
                    float contrib = path.throughputs[i] * (refl / PI) * le
                                  * cos_surface * mis_w / pdf_light * is_w;
                    glm::vec3 c = wavelength_to_RGB(path.wavelengths[i]) * contrib;
                    atomicAdd(&dev_img[path.pixelIndex].x, c.x);
                    atomicAdd(&dev_img[path.pixelIndex].y, c.y);
                    atomicAdd(&dev_img[path.pixelIndex].z, c.z);
                }
#else
                glm::vec3 f_brdf = material.color / PI;
                glm::vec3 Le = lightMat.color * lightMat.emittance;
                glm::vec3 contrib = path.color * f_brdf * Le * cos_surface * mis_w / pdf_light;
                atomicAdd(&dev_img[path.pixelIndex].x, contrib.x);
                atomicAdd(&dev_img[path.pixelIndex].y, contrib.y);
                atomicAdd(&dev_img[path.pixelIndex].z, contrib.z);
#endif
            }
        }
    }
#endif // ENABLE_MIS && !ENABLE_OPTIX

    // === BRDF Sample (indirect) ===
    const glm::vec2 xi = glm::vec2(curand_uniform(&local_rand_state), curand_uniform(&local_rand_state));
    glm::vec3 wiLocal = squareToHemisphereCosine(xi);
    glm::mat3 tbn = TangentSpaceToWorld(nor);
    glm::vec3 wiWorld = tbn * wiLocal;
    float cos_theta = glm::dot(wiWorld, nor);

#if ENABLE_SPECTRAL_RENDERING
    for (int i = 0; i < SPECTRAL_N; i++) {
        path.throughputs[i] *= spectral_reflectance_from_rgb(material.color, path.wavelengths[i]);
    }
#else
    path.color *= material.color;
#endif

    path.lastBrdfPdf = fmaxf(cos_theta / PI, 1e-8f);
    path.remainingBounces--;
    rand_states[path.pixelIndex] = local_rand_state;
    path.ray.origin = hitPt + nor * EPSILON;
    path.ray.direction = wiWorld;
}

__global__ void kernShadeSpecular(int num_hit, SpecularHitWorkItem* queue, PathSegment* paths, Material* materials,
    glm::vec3* dev_albedo, glm::vec3* dev_normal, int depth)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_hit) return;

    SpecularHitWorkItem item = queue[idx];
    PathSegment& path = paths[item.path_idx];
    Material material = materials[item.material_id];

    glm::vec3 reflected_dir = glm::reflect(item.incident_ray_dir, item.surface_normal);

    // AOV: first bounce specular -> material color albedo, surface normal
    if (depth == 0 && dev_albedo && dev_normal) {
        dev_albedo[path.pixelIndex] = material.color;
        dev_normal[path.pixelIndex] = glm::normalize(item.surface_normal);
    }

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
    path.lastBrdfPdf = -1.0f;  // delta distribution, no MIS

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

__global__ void kernShadeGlass(int num_hit, GlassHitWorkItem* queue, PathSegment* paths, Material* materials, curandState* rand_states,
    glm::vec3* dev_albedo, glm::vec3* dev_normal, int depth)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_hit) return;

    GlassHitWorkItem item = queue[idx];
    PathSegment& path = paths[item.path_idx];
    Material material = materials[item.material_id];
    curandState local_rand_state = rand_states[path.pixelIndex];

    glm::vec3 normal = glm::normalize(item.surface_normal);
    path.ray.origin = item.intersect_point;

    // AOV: first bounce glass -> material color albedo, surface normal
    if (depth == 0 && dev_albedo && dev_normal) {
        dev_albedo[path.pixelIndex] = material.color;
        dev_normal[path.pixelIndex] = normal;
    }

    glm::vec3 incident_dir = glm::normalize(path.ray.direction);  // normalize for stability
    float cos_theta = glm::dot(incident_dir, normal);

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
        new_direction = glm::reflect(incident_dir, oriented_normal);
    }
    else {
        path.ray.origin += 0.0002f * incident_dir;
        new_direction = glm::refract(incident_dir, oriented_normal, eta1 / eta2_val);
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
    path.lastBrdfPdf = -1.0f;  // delta distribution, no MIS
    path.remainingBounces--;

    rand_states[path.pixelIndex] = local_rand_state;
}

// ============================================================================
// OptiX Shadow Rays: Prepare + Apply (replaces brute-force shadowRayOccluded)
// ============================================================================

#if ENABLE_OPTIX && ENABLE_MIS

#include "optix_params.h"

__global__ void kernPrepareShadowRays(
    int num_hit,
    LambertianHitWorkItem* queue,
    PathSegment* paths,
    Material* materials,
    curandState* rand_states,
    Geom* geoms, int geoms_size, glm::vec3* positions,
    int* light_indices, int num_lights,
    Geom* nonTriGeoms, int numNonTriGeoms,
    ShadowRayRequest* shadowRays,
    cudaTextureObject_t* textureObjects, int numTextures)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_hit || num_lights == 0) {
        if (idx < num_hit) {
            shadowRays[idx].neeContrib = glm::vec3(0.0f);
            shadowRays[idx].occludedByPrimitive = 1;
        }
        return;
    }

    LambertianHitWorkItem item = queue[idx];
    PathSegment& path = paths[item.path_idx];
    Material material = materials[item.material_id];

    // Sample texture
    if (material.textureId >= 0 && material.textureId < numTextures && textureObjects != nullptr) {
        float4 texColor = tex2D<float4>(textureObjects[material.textureId], item.uv.x, item.uv.y);
        material.color *= glm::vec3(texColor.x, texColor.y, texColor.z);
    }

    curandState local_rand_state = rand_states[path.pixelIndex];

    glm::vec3 nor = item.surface_normal;
    glm::vec3 hitPt = item.intersect_point;

    // Sample light
    int li = (int)(curand_uniform(&local_rand_state) * (float)num_lights);
    if (li >= num_lights) li = num_lights - 1;
    int lightGeomIdx = light_indices[li];
    Geom lightGeom = geoms[lightGeomIdx];
    Material lightMat = materials[lightGeom.materialid];

    LightSample ls = sampleGeomLight(lightGeom, positions, &local_rand_state);

    glm::vec3 toLight = ls.position - hitPt;
    float d2 = glm::dot(toLight, toLight);
    float d = sqrtf(d2);
    glm::vec3 wi = toLight / d;

    float cos_surface = glm::dot(wi, nor);
    float cos_light = glm::dot(-wi, ls.normal);

    rand_states[path.pixelIndex] = local_rand_state;

    // Check geometry
    if (cos_surface <= 0.0f || cos_light <= 0.0f) {
        shadowRays[idx].neeContrib = glm::vec3(0.0f);
        shadowRays[idx].occludedByPrimitive = 1;
        return;
    }

    float pdf_light = (d2 / (ls.area * cos_light)) / (float)num_lights;
    float pdf_brdf = cos_surface / PI;
    float mis_w = powerHeuristic(pdf_light, pdf_brdf);

    // Pre-compute contribution (assumes visible)
    glm::vec3 contrib;
#if ENABLE_SPECTRAL_RENDERING
    contrib = glm::vec3(0.0f);
    float range = (float)(MAX_SAMPLE_WAVELENGTH - MIN_SAMPLE_WAVELENGTH);
    for (int i = 0; i < SPECTRAL_N; i++) {
        float refl = spectral_reflectance_from_rgb(material.color, path.wavelengths[i]);
        float le = spectral_reflectance_from_rgb(lightMat.color, path.wavelengths[i]) * lightMat.emittance;
        float is_w = 1.0f / (range * path.pdfs[i]);
        float c = path.throughputs[i] * (refl / PI) * le * cos_surface * mis_w / pdf_light * is_w;
        contrib += wavelength_to_RGB(path.wavelengths[i]) * c;
    }
#else
    glm::vec3 f_brdf = material.color / PI;
    glm::vec3 Le = lightMat.color * lightMat.emittance;
    contrib = path.color * f_brdf * Le * cos_surface * mis_w / pdf_light;
#endif

    // Build shadow ray
    glm::vec3 shadowOrigin = hitPt + nor * EPSILON;
    glm::vec3 shadowDir = wi;
    float maxDist = d - EPSILON * 20.0f;

    // Quick test against non-triangle geoms (only 6 box/sphere)
    bool occByPrim = false;
    Ray shadowRay = { shadowOrigin + shadowDir * EPSILON * 10.0f, shadowDir };
    for (int i = 0; i < numNonTriGeoms; i++) {
        if (nonTriGeoms[i].materialid == lightGeom.materialid) continue; // skip light itself
        float t = -1.0f;
        glm::vec3 tmp_p, tmp_n;
        bool tmp_o;
        if (nonTriGeoms[i].type == CUBE)
            t = boxIntersectionTest(nonTriGeoms[i], shadowRay, tmp_p, tmp_n, tmp_o);
        else if (nonTriGeoms[i].type == SPHERE)
            t = sphereIntersectionTest(nonTriGeoms[i], shadowRay, tmp_p, tmp_n, tmp_o);
        if (t > 0.0f && t < maxDist) { occByPrim = true; break; }
    }

    shadowRays[idx].origin = shadowOrigin;
    shadowRays[idx].direction = shadowDir;
    shadowRays[idx].maxDist = maxDist;
    shadowRays[idx].neeContrib = contrib;
    shadowRays[idx].pixelIndex = path.pixelIndex;
    shadowRays[idx].occludedByPrimitive = occByPrim ? 1 : 0;
}

__global__ void kernApplyShadowResults(
    int num_rays,
    ShadowRayRequest* shadowRays,
    int* shadowOccluded,
    glm::vec3* dev_img)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rays) return;

    // If visible (not occluded by triangles AND not occluded by primitives)
    if (!shadowOccluded[idx] && !shadowRays[idx].occludedByPrimitive) {
        glm::vec3 c = shadowRays[idx].neeContrib;
        int px = shadowRays[idx].pixelIndex;
        atomicAdd(&dev_img[px].x, c.x);
        atomicAdd(&dev_img[px].y, c.y);
        atomicAdd(&dev_img[px].z, c.z);
    }
}

#endif // ENABLE_OPTIX && ENABLE_MIS