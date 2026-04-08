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
// Environment Map Importance Sampling Helpers (used by both kernShadeMiss and NEE)
// ============================================================================

__device__ int binarySearchCDF(const float* cdf, int size, float u) {
    int lo = 0, hi = size - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (cdf[mid] <= u) lo = mid + 1;
        else hi = mid;
    }
    return max(0, lo - 1);
}

__device__ float envMapPdf(
    const float* marginalCDF, const float* conditionalCDF,
    int envW, int envH, glm::vec3 dir)
{
    float u_coord = 0.5f + atan2f(dir.z, dir.x) / (2.0f * PI);
    float v_coord = 0.5f - asinf(glm::clamp(dir.y, -1.0f, 1.0f)) / PI;
    int x = glm::clamp((int)(u_coord * envW), 0, envW - 1);
    int y = glm::clamp((int)(v_coord * envH), 0, envH - 1);
    float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - dir.y * dir.y));
    if (sinTheta < 1e-6f) return 0.0f;
    float py = marginalCDF[y + 1] - marginalCDF[y];
    float px_given_y = conditionalCDF[y * (envW + 1) + x + 1] - conditionalCDF[y * (envW + 1) + x];
    return py * px_given_y * envW * envH / (2.0f * PI * PI * sinTheta);
}

// ============================================================================
// Shading Kernels
// ============================================================================

__global__ void kernShadeMiss(int num_hit, MissWorkItem* queue, PathSegment* paths, glm::vec3* dev_img,
    glm::vec3* dev_albedo, glm::vec3* dev_normal, int depth,
    cudaTextureObject_t envMap, bool hasEnvMap,
    const float* envCDF_marginal, const float* envCDF_conditional,
    int envW, int envH) {
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

        if (depth == 0) {
            // Direct view of sky: apply ACES tone mapping so sun has detail
            auto aces = [](float x) {
                return fminf(fmaxf((x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f), 0.0f), 1.0f);
            };
            Le = glm::vec3(aces(Le.x), aces(Le.y), aces(Le.z));
        } else {
            // Indirect bounces: keep linear but clamp extreme outliers
            Le = glm::min(Le, glm::vec3(50.0f));
        }

        // MIS weight for BSDF-sampled direction
        float mis_weight = 1.0f;
        if (envCDF_marginal && path.lastBrdfPdf > 0.0f) {
            float pdf_env = envMapPdf(envCDF_marginal, envCDF_conditional, envW, envH, dir);
            if (pdf_env > 0.0f) {
                mis_weight = powerHeuristic(path.lastBrdfPdf, pdf_env);
            }
        }

#if ENABLE_SPECTRAL_RENDERING
        glm::vec3 c(0.0f);
        float range = (float)(MAX_SAMPLE_WAVELENGTH - MIN_SAMPLE_WAVELENGTH);

        float max_Le = fmaxf(Le.x, fmaxf(Le.y, Le.z));
        glm::vec3 norm_Le = max_Le > 0.0f ? Le / max_Le : glm::vec3(0.0f);

        for (int i = 0; i < SPECTRAL_N; i++) {
            float le_wl = max_Le * spectral_reflectance_from_rgb(norm_Le, path.wavelengths[i]);
            float is_weight = 1.0f / (range * path.pdfs[i]);
            float radiance = path.throughputs[i] * le_wl * is_weight * mis_weight;
            c += spectral_to_sRGB(path.wavelengths[i], radiance);
        }
        atomicAdd(&dev_img[path.pixelIndex].x, c.x);
        atomicAdd(&dev_img[path.pixelIndex].y, c.y);
        atomicAdd(&dev_img[path.pixelIndex].z, c.z);
#else
        glm::vec3 contribution = path.color * Le * mis_weight;
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
        glm::vec3 c = spectral_to_sRGB(path.wavelengths[i], radiance);
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
    glm::vec3* dev_albedo, glm::vec3* dev_normal, int depth,
#if ENABLE_NRC
    NRCQueryWorkItem* nrc_queries,
    int* nrc_query_counter,
    float nrc_train_fraction,
    int* nrc_train_sample_counter,
    NRCTrainingSample* nrc_train_samples,
#endif
    int traceDepth)
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

    glm::vec3 V = -path.ray.direction;

#if ENABLE_NRC
    if (depth == NRC_QUERY_DEPTH && nrc_queries && nrc_query_counter) {
        float theta = atan2f(V.z, V.x);
        if (theta < 0.0f) theta += 2.0f * PI;
        float phi = acosf(glm::clamp(V.y, -1.0f, 1.0f));

        float rnd = curand_uniform(&local_rand_state);
        if (rnd < nrc_train_fraction && nrc_train_sample_counter && nrc_train_samples) {
            int slot = atomicAdd(nrc_train_sample_counter, 1);
            if (slot < NRC_TRAIN_RAYS) {
                path.nrcTrainIdx = slot;
                path.nrcRgbThroughput = glm::vec3(1.0f);
                path.nrcTargetRadiance = glm::vec3(0.0f);
                
                NRCTrainingSample s;
                s.position = hitPt;
                s.normal = nor;
                s.theta = theta;
                s.phi = phi;
                s.albedo = material.color;
                s.target_radiance = glm::vec3(0.0f);
                nrc_train_samples[slot] = s;
                
                // Do not terminate, let it continue tracing for ground truth
            } else {
                path.remainingBounces = 0; // Terminate if full
            }
        } else {
            NRCQueryWorkItem nrc_q;
            nrc_q.path_idx = item.path_idx;
            nrc_q.position = hitPt;
            nrc_q.normal = nor;
            nrc_q.viewDir = glm::vec2(theta, phi);
            nrc_q.albedo = material.color;
            
            int slot = atomicAdd(nrc_query_counter, 1);
            nrc_queries[slot] = nrc_q;
            path.remainingBounces = 0;
            return;
        }
    }
#endif

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
                glm::vec3 c_sum(0.0f);
                glm::vec3 L_val_sum(0.0f);
                for (int i = 0; i < SPECTRAL_N; i++) {
                    float refl = spectral_reflectance_from_rgb(material.color, path.wavelengths[i]);
                    float le = spectral_reflectance_from_rgb(lightMat.color, path.wavelengths[i])
                             * lightMat.emittance;
                    float is_w = 1.0f / (range * path.pdfs[i]);
                    float L_val_wl = (refl / PI) * le * cos_surface * mis_w / pdf_light * is_w;
                    float contrib = path.throughputs[i] * L_val_wl;
                    c_sum += spectral_to_sRGB(path.wavelengths[i], contrib);
                    L_val_sum += spectral_to_sRGB(path.wavelengths[i], L_val_wl);
                }
                atomicAdd(&dev_img[path.pixelIndex].x, c_sum.x);
                atomicAdd(&dev_img[path.pixelIndex].y, c_sum.y);
                atomicAdd(&dev_img[path.pixelIndex].z, c_sum.z);
#if ENABLE_NRC
                if (path.nrcTrainIdx != -1) {
                    path.nrcTargetRadiance += L_val_sum * path.nrcRgbThroughput;
                }
#endif
#else
                glm::vec3 f_brdf = material.color / PI;
                glm::vec3 Le = lightMat.color * lightMat.emittance;
                glm::vec3 L_val = f_brdf * Le * cos_surface * mis_w / pdf_light;
                glm::vec3 contrib = path.color * L_val;
                atomicAdd(&dev_img[path.pixelIndex].x, contrib.x);
                atomicAdd(&dev_img[path.pixelIndex].y, contrib.y);
                atomicAdd(&dev_img[path.pixelIndex].z, contrib.z);
#if ENABLE_NRC
                if (path.nrcTrainIdx != -1) {
                    path.nrcTargetRadiance += L_val * path.nrcRgbThroughput;
                }
#endif
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

#if ENABLE_NRC
    if (path.nrcTrainIdx != -1) {
        path.nrcRgbThroughput *= material.color;
    }
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
    if (did_reflect) {
        // Reflection: all wavelengths share the same reflection direction,
        // so standard hero wavelength importance weighting is correct
        for (int i = 0; i < SPECTRAL_N; i++) {
            float ior_i = compute_dispersion_ior(dispersion_abbe, base_IOR, path.wavelengths[i]);
            float eta1_i, eta2_i;
            if (cos_theta < 0.0f) { eta1_i = 1.0f; eta2_i = ior_i; }
            else                  { eta1_i = ior_i; eta2_i = 1.0f; }
            float F_i = FresnelDielectricEval(abs_cos_theta, eta1_i / eta2_i);
            path.throughputs[i] *= (F_hero > 1e-6f) ? (F_i / F_hero) : 0.0f;
            path.throughputs[i] *= spectral_reflectance_from_rgb(material.color, path.wavelengths[i]);
        }
    } else {
        // Refraction with dispersion: each wavelength refracts at a DIFFERENT angle.
        // Since we can only trace ONE direction (the hero wavelength's), the non-hero
        // wavelengths end up at the WRONG screen position. To produce correct spatial
        // color separation (rainbow), we must only keep the hero's contribution and
        // compensate by multiplying by SPECTRAL_N.
        for (int i = 0; i < SPECTRAL_N; i++) {
            if (i == 0) {
                // Hero wavelength: keep and scale up to compensate for dropped samples,
                // but only if this is the FIRST dispersion event (others still alive).
                // On subsequent refractions, non-hero are already 0 so don't multiply again.
                bool others_alive = false;
                for (int j = 1; j < SPECTRAL_N; j++) {
                    if (path.throughputs[j] > 0.0f) { others_alive = true; break; }
                }
                path.throughputs[0] *= spectral_reflectance_from_rgb(material.color, path.wavelengths[0]);
                if (others_alive) {
                    path.throughputs[0] *= (float)SPECTRAL_N;
                }
            } else {
                // Non-hero: zero out since they'd be at the wrong position
                path.throughputs[i] = 0.0f;
            }
        }
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
// Disney GGX BRDF Device Functions
// ============================================================================

// GGX (Trowbridge-Reitz) normal distribution function
__device__ float D_GGX(float NdotH, float alpha) {
    float a2 = alpha * alpha;
    float denom = NdotH * NdotH * (a2 - 1.0f) + 1.0f;
    return a2 / (PI * denom * denom + 1e-7f);
}

// GTR1 distribution for clearcoat (Berry distribution)
__device__ float D_GTR1(float NdotH, float alpha) {
    if (alpha >= 1.0f) return 1.0f / PI;
    float a2 = alpha * alpha;
    float t = 1.0f + (a2 - 1.0f) * NdotH * NdotH;
    return (a2 - 1.0f) / (PI * logf(a2) * t + 1e-7f);
}

// Smith G1 for GGX
__device__ float G1_SmithGGX(float NdotX, float alpha) {
    float a2 = alpha * alpha;
    return 2.0f * NdotX / (NdotX + sqrtf(a2 + (1.0f - a2) * NdotX * NdotX) + 1e-7f);
}

// Smith separable geometry term
__device__ float G_SmithGGX(float NdotV, float NdotL, float alpha) {
    return G1_SmithGGX(NdotV, alpha) * G1_SmithGGX(NdotL, alpha);
}

// Schlick Fresnel with vec3 F0
__device__ glm::vec3 F_SchlickVec3(float cosTheta, glm::vec3 F0) {
    float x = 1.0f - cosTheta;
    float x2 = x * x;
    float x5 = x2 * x2 * x;
    return F0 + (glm::vec3(1.0f) - F0) * x5;
}

// Schlick Fresnel scalar
__device__ float F_SchlickScalar(float cosTheta, float F0) {
    float x = 1.0f - cosTheta;
    float x2 = x * x;
    float x5 = x2 * x2 * x;
    return F0 + (1.0f - F0) * x5;
}

// Disney diffuse BRDF with retro-reflection
__device__ float disneyDiffuseFactor(float NdotL, float NdotV, float LdotH, float roughness) {
    float FL = F_SchlickScalar(NdotL, 0.0f);  // 1 - (1-NdotL)^5
    float FV = F_SchlickScalar(NdotV, 0.0f);
    // Compute (1-NdotL)^5 and (1-NdotV)^5
    float oneMinusNdotL = 1.0f - NdotL;
    float pow5L = oneMinusNdotL * oneMinusNdotL * oneMinusNdotL * oneMinusNdotL * oneMinusNdotL;
    float oneMinusNdotV = 1.0f - NdotV;
    float pow5V = oneMinusNdotV * oneMinusNdotV * oneMinusNdotV * oneMinusNdotV * oneMinusNdotV;

    float Fd90 = 0.5f + 2.0f * LdotH * LdotH * roughness;
    float FdL = 1.0f + (Fd90 - 1.0f) * pow5L;
    float FdV = 1.0f + (Fd90 - 1.0f) * pow5V;
    return FdL * FdV / PI;
}

// Sample GGX VNDF (Heitz 2018)
// Returns a half-vector in tangent space (Z-up)
__device__ glm::vec3 sampleGGXVNDF(glm::vec2 xi, float alpha_x, float alpha_y, glm::vec3 V_local) {
    // Stretch V
    glm::vec3 Vh = glm::normalize(glm::vec3(alpha_x * V_local.x, alpha_y * V_local.y, V_local.z));

    // Orthonormal basis around Vh
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    glm::vec3 T1 = lensq > 0.0f ? glm::vec3(-Vh.y, Vh.x, 0.0f) / sqrtf(lensq) : glm::vec3(1.0f, 0.0f, 0.0f);
    glm::vec3 T2 = glm::cross(Vh, T1);

    // Parameterization of the projected area
    float r = sqrtf(xi.x);
    float phi = 2.0f * PI * xi.y;
    float t1 = r * cosf(phi);
    float t2 = r * sinf(phi);
    float s = 0.5f * (1.0f + Vh.z);
    t2 = (1.0f - s) * sqrtf(fmaxf(0.0f, 1.0f - t1 * t1)) + s * t2;

    // Construct half-vector in stretched space
    glm::vec3 Nh = t1 * T1 + t2 * T2 + sqrtf(fmaxf(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;

    // Un-stretch
    glm::vec3 H = glm::normalize(glm::vec3(alpha_x * Nh.x, alpha_y * Nh.y, fmaxf(0.0f, Nh.z)));
    return H;
}

// PDF for VNDF sampling
__device__ float pdfGGXVNDF(float NdotH, float VdotH, float NdotV, float alpha) {
    float D = D_GGX(NdotH, alpha);
    float G1 = G1_SmithGGX(NdotV, alpha);
    return D * G1 * fmaxf(VdotH, 0.0f) / (NdotV + 1e-7f);
}

// Evaluate full Disney BRDF for a given wi direction (used by NEE)
__device__ glm::vec3 evaluateDisneyBRDF(
    glm::vec3 V, glm::vec3 L, glm::vec3 N, const Material& mat,
    float& pdf_out)
{
    float NdotL = glm::dot(N, L);
    float NdotV = glm::dot(N, V);
    if (NdotL <= 0.0f || NdotV <= 0.0f) {
        pdf_out = 0.0f;
        return glm::vec3(0.0f);
    }

    glm::vec3 H = glm::normalize(V + L);
    float NdotH = fmaxf(glm::dot(N, H), 0.0f);
    float VdotH = fmaxf(glm::dot(V, H), 0.0f);
    float LdotH = fmaxf(glm::dot(L, H), 0.0f);

    float alpha = fmaxf(mat.roughness * mat.roughness, 0.001f);

    // F0 for specular
    float luminance = 0.2126f * mat.color.x + 0.7152f * mat.color.y + 0.0722f * mat.color.z;
    glm::vec3 Ctint = luminance > 0.0f ? mat.color / luminance : glm::vec3(1.0f);
    glm::vec3 F0_dielectric = glm::mix(glm::vec3(0.04f), glm::vec3(0.04f) * Ctint, mat.specularTint);
    glm::vec3 F0 = glm::mix(F0_dielectric, mat.color, mat.metallic);

    // Diffuse lobe
    float diffWeight = (1.0f - mat.metallic);
    float diffFactor = disneyDiffuseFactor(NdotL, NdotV, LdotH, mat.roughness);
    glm::vec3 f_diff = mat.color * diffFactor * diffWeight;

    // Specular lobe (GGX microfacet)
    float D = D_GGX(NdotH, alpha);
    float G = G_SmithGGX(NdotV, NdotL, alpha);
    glm::vec3 F = F_SchlickVec3(VdotH, F0);
    glm::vec3 f_spec = D * G * F / (4.0f * NdotV * NdotL + 1e-7f);

    // Clearcoat lobe
    glm::vec3 f_clearcoat(0.0f);
    float ccPdf = 0.0f;
    if (mat.clearcoat > 0.0f) {
        float alpha_cc = glm::mix(0.1f, 0.001f, mat.clearcoatGloss);
        float D_cc = D_GTR1(NdotH, alpha_cc);
        float G_cc = G_SmithGGX(NdotV, NdotL, 0.25f);   // fixed roughness 0.25
        float F_cc = F_SchlickScalar(LdotH, 0.04f);       // polyurethane IOR ~1.5
        f_clearcoat = glm::vec3(mat.clearcoat * 0.25f * D_cc * G_cc * F_cc / (4.0f * NdotV * NdotL + 1e-7f));
        ccPdf = D_cc * NdotH / (4.0f * VdotH + 1e-7f);
    }

    // Total BRDF
    glm::vec3 f_total = f_diff + f_spec + f_clearcoat;

    // PDF: weighted mixture of diffuse + specular + clearcoat PDFs
    float pdf_diff = NdotL / PI;
    float pdf_spec = pdfGGXVNDF(NdotH, VdotH, NdotV, alpha) / (4.0f * VdotH + 1e-7f);

    float wDiff = diffWeight * 0.5f;
    float wSpec = 0.5f;
    float wCC = mat.clearcoat > 0.0f ? 0.15f : 0.0f;
    float total_w = wDiff + wSpec + wCC;
    wDiff /= total_w; wSpec /= total_w; wCC /= total_w;

    pdf_out = wDiff * pdf_diff + wSpec * pdf_spec + wCC * ccPdf;

    return f_total;
}

#if ENABLE_SPECTRAL_RENDERING
// Evaluate full Disney BRDF for a given wi direction, spectrally
__device__ float evaluateDisneyBRDF_Spectral(
    glm::vec3 V, glm::vec3 L, glm::vec3 N, const Material& mat,
    float wavelength, float& pdf_out)
{
    float NdotL = glm::dot(N, L);
    float NdotV = glm::dot(N, V);
    if (NdotL <= 0.0f || NdotV <= 0.0f) {
        pdf_out = 0.0f;
        return 0.0f;
    }

    glm::vec3 H = glm::normalize(V + L);
    float NdotH = fmaxf(glm::dot(N, H), 0.0f);
    float VdotH = fmaxf(glm::dot(V, H), 0.0f);
    float LdotH = fmaxf(glm::dot(L, H), 0.0f);

    float alpha = fmaxf(mat.roughness * mat.roughness, 0.001f);

    float luminance = 0.2126f * mat.color.x + 0.7152f * mat.color.y + 0.0722f * mat.color.z;
    float mat_wl = spectral_reflectance_from_rgb(mat.color, wavelength);
    
    // F0 for specular (spectral)
    float Ctint_wl = luminance > 0.0f ? mat_wl / luminance : 1.0f;
    // clamping F0 tint to reasonably small bounds if needed, but it's fine
    float F0_dielectric_wl = glm::mix(0.04f, 0.04f * Ctint_wl, mat.specularTint);
    float F0_wl = glm::mix(F0_dielectric_wl, mat_wl, mat.metallic);

    // Diffuse lobe (spectral)
    float diffWeight = (1.0f - mat.metallic);
    float diffFactor = disneyDiffuseFactor(NdotL, NdotV, LdotH, mat.roughness);
    float f_diff_wl = mat_wl * diffFactor * diffWeight;

    // Specular lobe (spectral)
    float D = D_GGX(NdotH, alpha);
    float G = G_SmithGGX(NdotV, NdotL, alpha);
    float F_wl = F_SchlickScalar(VdotH, F0_wl);
    float f_spec_wl = D * G * F_wl / (4.0f * NdotV * NdotL + 1e-7f);

    // Clearcoat lobe (achromatic, but we evaluate its scalar value)
    float f_clearcoat = 0.0f;
    float ccPdf = 0.0f;
    if (mat.clearcoat > 0.0f) {
        float alpha_cc = glm::mix(0.1f, 0.001f, mat.clearcoatGloss);
        float D_cc = D_GTR1(NdotH, alpha_cc);
        float G_cc = G_SmithGGX(NdotV, NdotL, 0.25f);   // fixed roughness 0.25
        float F_cc = F_SchlickScalar(LdotH, 0.04f);       // polyurethane IOR ~1.5
        f_clearcoat = mat.clearcoat * 0.25f * D_cc * G_cc * F_cc / (4.0f * NdotV * NdotL + 1e-7f);
        ccPdf = D_cc * NdotH / (4.0f * VdotH + 1e-7f);
    }

    // Total BRDF
    float f_total_wl = f_diff_wl + f_spec_wl + f_clearcoat;

    // PDF remains exactly the same as RGB
    float pdf_diff = NdotL / PI;
    float pdf_spec = pdfGGXVNDF(NdotH, VdotH, NdotV, alpha) / (4.0f * VdotH + 1e-7f);

    float wDiff = diffWeight * 0.5f;
    float wSpec = 0.5f;
    float wCC = mat.clearcoat > 0.0f ? 0.15f : 0.0f;
    float total_w = wDiff + wSpec + wCC;
    wDiff /= total_w; wSpec /= total_w; wCC /= total_w;

    pdf_out = wDiff * pdf_diff + wSpec * pdf_spec + wCC * ccPdf;

    return f_total_wl;
}
#endif

// ============================================================================
// Disney GGX Shading Kernel
// ============================================================================

__global__ void kernShadeDisneyGGX(
    int num_hit, DisneyGGXHitWorkItem* queue, PathSegment* paths,
    Material* materials, curandState* rand_states, glm::vec3* dev_img,
    Geom* geoms, int geoms_size, glm::vec3* positions,
    int* light_indices, int num_lights,
    cudaTextureObject_t* textureObjects, int numTextures,
    glm::vec3* dev_albedo, glm::vec3* dev_normal, int depth,
#if ENABLE_NRC
    NRCQueryWorkItem* nrc_queries,
    int* nrc_query_counter,
    float nrc_train_fraction,
    int* nrc_train_sample_counter,
    NRCTrainingSample* nrc_train_samples,
#endif
    int traceDepth)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_hit) return;

    DisneyGGXHitWorkItem item = queue[idx];
    PathSegment& path = paths[item.path_idx];
    Material material = materials[item.material_id];

    // Sample base color texture if available
    if (material.textureId >= 0 && material.textureId < numTextures && textureObjects != nullptr) {
        float4 texColor = tex2D<float4>(textureObjects[material.textureId], item.uv.x, item.uv.y);
        material.color *= glm::vec3(texColor.x, texColor.y, texColor.z);
    }

    curandState local_rand_state = rand_states[path.pixelIndex];

    glm::vec3 nor = glm::normalize(item.surface_normal);
    glm::vec3 hitPt = item.intersect_point;
    glm::vec3 V = -glm::normalize(item.incident_ray_dir);

    // AOV: first bounce
    if (depth == 0 && dev_albedo && dev_normal) {
        dev_albedo[path.pixelIndex] = material.color;
        dev_normal[path.pixelIndex] = nor;
    }

#if ENABLE_NRC
    if (depth == NRC_QUERY_DEPTH && nrc_queries && nrc_query_counter) {
        float theta = atan2f(V.z, V.x);
        if (theta < 0.0f) theta += 2.0f * PI;
        float phi = acosf(glm::clamp(V.y, -1.0f, 1.0f));

        float rnd = curand_uniform(&local_rand_state);
        if (rnd < nrc_train_fraction && nrc_train_sample_counter && nrc_train_samples) {
            int slot = atomicAdd(nrc_train_sample_counter, 1);
            if (slot < NRC_TRAIN_RAYS) {
                path.nrcTrainIdx = slot;
                path.nrcRgbThroughput = glm::vec3(1.0f);
                path.nrcTargetRadiance = glm::vec3(0.0f);
                
                NRCTrainingSample s;
                s.position = hitPt;
                s.normal = nor;
                s.theta = theta;
                s.phi = phi;
                s.albedo = material.color;
                s.target_radiance = glm::vec3(0.0f);
                nrc_train_samples[slot] = s;
            } else {
                path.remainingBounces = 0;
            }
        } else {
            NRCQueryWorkItem nrc_q;
            nrc_q.path_idx = item.path_idx;
            nrc_q.position = hitPt;
            nrc_q.normal = nor;
            nrc_q.viewDir = glm::vec2(theta, phi);
            nrc_q.albedo = material.color;
            
            int slot = atomicAdd(nrc_query_counter, 1);
            nrc_queries[slot] = nrc_q;
            path.remainingBounces = 0;
            return;
        }
    }
#endif

    float NdotV = fmaxf(glm::dot(nor, V), 1e-5f);
    float alpha = fmaxf(material.roughness * material.roughness, 0.001f);

    // F0 computation
    float luminance = 0.2126f * material.color.x + 0.7152f * material.color.y + 0.0722f * material.color.z;
    glm::vec3 Ctint = luminance > 0.0f ? material.color / luminance : glm::vec3(1.0f);
    glm::vec3 F0_dielectric = glm::mix(glm::vec3(0.04f), glm::vec3(0.04f) * Ctint, material.specularTint);
    glm::vec3 F0 = glm::mix(F0_dielectric, material.color, material.metallic);

    // Lobe selection weights
    float diffWeight = (1.0f - material.metallic) * 0.5f;
    float specWeight = 0.5f;
    float ccWeight = material.clearcoat > 0.0f ? 0.15f : 0.0f;
    float totalWeight = diffWeight + specWeight + ccWeight;
    diffWeight /= totalWeight;
    specWeight /= totalWeight;
    ccWeight /= totalWeight;

#if ENABLE_MIS && !ENABLE_OPTIX
    // === NEE: Direct Light Sampling (brute-force -- only used without OptiX) ===
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

                float brdf_pdf;
                glm::vec3 f_brdf = evaluateDisneyBRDF(V, wi, nor, material, brdf_pdf);

                float mis_w = powerHeuristic(pdf_light, brdf_pdf);

#if ENABLE_SPECTRAL_RENDERING
                glm::vec3 contrib_sum(0.0f);
                glm::vec3 L_val_sum(0.0f);
                float range = (float)(MAX_SAMPLE_WAVELENGTH - MIN_SAMPLE_WAVELENGTH);
                for (int i = 0; i < SPECTRAL_N; i++) {
                    float wl_pdf;
                    float f_brdf_wl = evaluateDisneyBRDF_Spectral(V, wi, nor, material, path.wavelengths[i], wl_pdf);
                    float mis_w_wl = powerHeuristic(pdf_light, wl_pdf);
                    float le = spectral_reflectance_from_rgb(lightMat.color, path.wavelengths[i]) * lightMat.emittance;
                    float is_w = 1.0f / (range * path.pdfs[i]);
                    float L_val_wl = f_brdf_wl * le * cos_surface * mis_w_wl / pdf_light * is_w;
                    float contrib = path.throughputs[i] * L_val_wl;
                    contrib_sum += spectral_to_sRGB(path.wavelengths[i], contrib);
                    L_val_sum += spectral_to_sRGB(path.wavelengths[i], L_val_wl);
                }
                atomicAdd(&dev_img[path.pixelIndex].x, contrib_sum.x);
                atomicAdd(&dev_img[path.pixelIndex].y, contrib_sum.y);
                atomicAdd(&dev_img[path.pixelIndex].z, contrib_sum.z);
#if ENABLE_NRC
                if (path.nrcTrainIdx != -1) {
                    path.nrcTargetRadiance += L_val_sum * path.nrcRgbThroughput;
                }
#endif
#else
                glm::vec3 Le = lightMat.color * lightMat.emittance;
                glm::vec3 L_val = f_brdf * Le * cos_surface * mis_w / pdf_light;
                glm::vec3 contrib = path.color * L_val;
                atomicAdd(&dev_img[path.pixelIndex].x, contrib.x);
                atomicAdd(&dev_img[path.pixelIndex].y, contrib.y);
                atomicAdd(&dev_img[path.pixelIndex].z, contrib.z);
#if ENABLE_NRC
                if (path.nrcTrainIdx != -1) {
                    path.nrcTargetRadiance += L_val * path.nrcRgbThroughput;
                }
#endif
#endif
            }
        }
    }
#endif // ENABLE_MIS && !ENABLE_OPTIX

    // === BRDF Sample (indirect) ===
    float r_lobe = curand_uniform(&local_rand_state);
    glm::vec2 xi(curand_uniform(&local_rand_state), curand_uniform(&local_rand_state));

    glm::mat3 tbn = TangentSpaceToWorld(nor);
    glm::mat3 tbnInv = WorldToTangentSpace(nor);
    glm::vec3 V_local = tbnInv * V;

    glm::vec3 wiWorld;

    if (r_lobe < diffWeight) {
        // Diffuse lobe: cosine-weighted hemisphere
        glm::vec3 wiLocal = squareToHemisphereCosine(xi);
        wiWorld = tbn * wiLocal;
        float NdotL = fmaxf(glm::dot(wiWorld, nor), 0.0f);
    } else if (r_lobe < diffWeight + specWeight) {
        // Specular lobe: GGX VNDF importance sampling
        glm::vec3 H_local = sampleGGXVNDF(xi, alpha, alpha, V_local);
        glm::vec3 H_world = glm::normalize(tbn * H_local);
        wiWorld = glm::reflect(-V, H_world);
        float VdotH = fmaxf(glm::dot(V, H_world), 0.0f);
    } else {
        // Clearcoat lobe: GTR1 sampling
        float alpha_cc = glm::mix(0.1f, 0.001f, material.clearcoatGloss);
        float a2 = alpha_cc * alpha_cc;
        float cosTheta = sqrtf(fmaxf(0.0f, (1.0f - powf(a2, 1.0f - xi.x)) / (1.0f - a2)));
        float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
        float phi = 2.0f * PI * xi.y;
        glm::vec3 H_local(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta);
        glm::vec3 H_world = glm::normalize(tbn * H_local);
        wiWorld = glm::reflect(-V, H_world);
        float NdotH = fmaxf(glm::dot(nor, H_world), 0.0f);
        float VdotH = fmaxf(glm::dot(V, H_world), 0.0f);
        float D_cc = D_GTR1(NdotH, alpha_cc);
    }

    float NdotL = glm::dot(wiWorld, nor);
    if (NdotL <= 0.0f) {
        // Below hemisphere -- kill ray
        path.remainingBounces = 0;
        rand_states[path.pixelIndex] = local_rand_state;
        return;
    }

    // Evaluate full BRDF for the sampled direction
#if ENABLE_SPECTRAL_RENDERING
    // We evaluate purely to get the mixed PDF first (using RGB)
    // Wait, the PDF does not depend on color. It only depends on roughness and weights. 
    // Wait, diffWeight, specWeight, ccWeight don't depend on color. So PDF is achromatic.
    float NdotH_pdf = fmaxf(glm::dot(nor, glm::normalize(V + wiWorld)), 0.0f);
    float VdotH_pdf = fmaxf(glm::dot(V, glm::normalize(V + wiWorld)), 0.0f);
    float pdf_diff = NdotL / PI;
    float pdf_spec = pdfGGXVNDF(NdotH_pdf, VdotH_pdf, NdotV, alpha) / (4.0f * VdotH_pdf + 1e-7f);
    float pdf_cc = 0.0f;
    if (material.clearcoat > 0.0f) {
        float alpha_cc = glm::mix(0.1f, 0.001f, material.clearcoatGloss);
        pdf_cc = D_GTR1(NdotH_pdf, alpha_cc) * NdotH_pdf / (4.0f * VdotH_pdf + 1e-7f);
    }
    float mixed_pdf = diffWeight * pdf_diff + specWeight * pdf_spec + ccWeight * pdf_cc;
    mixed_pdf = fmaxf(mixed_pdf, 1e-8f);

    for (int i = 0; i < SPECTRAL_N; i++) {
        float full_pdf_wl;
        float f_brdf_wl = evaluateDisneyBRDF_Spectral(V, wiWorld, nor, material, path.wavelengths[i], full_pdf_wl);
        float throughput_factor = f_brdf_wl * NdotL / mixed_pdf;
        throughput_factor = fminf(throughput_factor, 10.0f);
        path.throughputs[i] *= throughput_factor;
    }
    path.lastBrdfPdf = mixed_pdf;
#else
    float full_pdf;
    glm::vec3 f_brdf = evaluateDisneyBRDF(V, wiWorld, nor, material, full_pdf);

    // Mixed PDF from all lobes
    glm::vec3 H_eval = glm::normalize(V + wiWorld);
    float NdotH = fmaxf(glm::dot(nor, H_eval), 0.0f);
    float VdotH = fmaxf(glm::dot(V, H_eval), 0.0f);
    float LdotH = fmaxf(glm::dot(wiWorld, H_eval), 0.0f);

    float pdf_diff = NdotL / PI;
    float pdf_spec = pdfGGXVNDF(NdotH, VdotH, NdotV, alpha) / (4.0f * VdotH + 1e-7f);
    float pdf_cc = 0.0f;
    if (material.clearcoat > 0.0f) {
        float alpha_cc = glm::mix(0.1f, 0.001f, material.clearcoatGloss);
        pdf_cc = D_GTR1(NdotH, alpha_cc) * NdotH / (4.0f * VdotH + 1e-7f);
    }
    float mixed_pdf = diffWeight * pdf_diff + specWeight * pdf_spec + ccWeight * pdf_cc;
    mixed_pdf = fmaxf(mixed_pdf, 1e-8f);

    // Throughput update: f * cos(theta) / pdf
    glm::vec3 throughput_factor = f_brdf * NdotL / mixed_pdf;
    // Clamp to prevent fireflies
    throughput_factor = glm::min(throughput_factor, glm::vec3(10.0f));

    path.color *= throughput_factor;
    path.lastBrdfPdf = mixed_pdf;
#endif

    path.lastBrdfPdf = mixed_pdf;
    path.remainingBounces--;
    rand_states[path.pixelIndex] = local_rand_state;
    path.ray.origin = hitPt + nor * EPSILON;
    path.ray.direction = glm::normalize(wiWorld);
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
        contrib += spectral_to_sRGB(path.wavelengths[i], c);
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

// ============================================================================
// Environment Map NEE Shadow Rays
// ============================================================================

__global__ void kernPrepareEnvMapShadowRays(
    int num_hit,
    LambertianHitWorkItem* queue,
    PathSegment* paths,
    Material* materials,
    curandState* rand_states,
    Geom* nonTriGeoms, int numNonTriGeoms,
    ShadowRayRequest* shadowRays,
    cudaTextureObject_t* textureObjects, int numTextures,
    // Env map data
    cudaTextureObject_t envMap,
    const float* marginalCDF, const float* conditionalCDF,
    int envW, int envH, float envTotalPower)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_hit) return;

    LambertianHitWorkItem item = queue[idx];
    PathSegment& path = paths[item.path_idx];
    Material material = materials[item.material_id];

    // Sample texture
    if (material.textureId >= 0 && material.textureId < numTextures && textureObjects != nullptr) {
        float4 texColor = tex2D<float4>(textureObjects[material.textureId], item.uv.x, item.uv.y);
        material.color *= glm::vec3(texColor.x, texColor.y, texColor.z);
    }

    curandState local_rand_state = rand_states[path.pixelIndex];
    float u1 = curand_uniform(&local_rand_state);
    float u2 = curand_uniform(&local_rand_state);
    rand_states[path.pixelIndex] = local_rand_state;

    glm::vec3 nor = item.surface_normal;
    glm::vec3 hitPt = item.intersect_point;

    // Sample direction from env map CDF
    // 1. Sample row (v) from marginal CDF
    int y = binarySearchCDF(marginalCDF, envH + 1, u1);
    float py = marginalCDF[y + 1] - marginalCDF[y];
    if (py <= 0.0f) {
        shadowRays[idx].neeContrib = glm::vec3(0.0f);
        shadowRays[idx].occludedByPrimitive = 1;
        return;
    }
    float dv = (u1 - marginalCDF[y]) / py;
    float v = (y + dv + 0.5f) / envH;

    // 2. Sample column (u) from conditional CDF for row y
    const float* rowCDF = conditionalCDF + y * (envW + 1);
    int x = binarySearchCDF(rowCDF, envW + 1, u2);
    float px = rowCDF[x + 1] - rowCDF[x];
    if (px <= 0.0f) {
        shadowRays[idx].neeContrib = glm::vec3(0.0f);
        shadowRays[idx].occludedByPrimitive = 1;
        return;
    }
    float du = (u2 - rowCDF[x]) / px;
    float u = (x + du + 0.5f) / envW;

    // Convert UV to direction (inverse of equirectangular mapping)
    // Original: u = 0.5 + atan2(z, x) / (2*PI)
    //           v = 0.5 - asin(y) / PI
    float phi = (u - 0.5f) * 2.0f * PI;
    float y_val = sinf((0.5f - v) * PI);
    float xz = sqrtf(fmaxf(0.0f, 1.0f - y_val * y_val));
    glm::vec3 wi(xz * cosf(phi), y_val, xz * sinf(phi));
    wi = glm::normalize(wi);

    float cos_theta = glm::dot(wi, nor);

    // Check if direction is above surface
    if (cos_theta <= 0.0f) {
        shadowRays[idx].neeContrib = glm::vec3(0.0f);
        shadowRays[idx].occludedByPrimitive = 1;
        return;
    }

    // Compute PDF in solid angle
    float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - y_val * y_val));
    float pdf_env = py * px * envW * envH / (2.0f * PI * PI * fmaxf(sinTheta, 1e-6f));
    if (pdf_env <= 0.0f) {
        shadowRays[idx].neeContrib = glm::vec3(0.0f);
        shadowRays[idx].occludedByPrimitive = 1;
        return;
    }

    // Read env map value at sampled direction
    float4 envColor = tex2D<float4>(envMap, u, v);
    glm::vec3 Le(envColor.x, envColor.y, envColor.z);

    // MIS weight: env sampling vs BRDF sampling
    float pdf_brdf = cos_theta / PI;
    float mis_w = powerHeuristic(pdf_env, pdf_brdf);

    // Compute contribution: BRDF * Le * cos_theta * mis_weight / pdf_env
    glm::vec3 contrib;
#if ENABLE_SPECTRAL_RENDERING
    contrib = glm::vec3(0.0f);
    float range = (float)(MAX_SAMPLE_WAVELENGTH - MIN_SAMPLE_WAVELENGTH);
    // HDR scaling logic for spectral mapping
    float max_Le = fmaxf(Le.x, fmaxf(Le.y, Le.z));
    glm::vec3 norm_Le = max_Le > 0.0f ? Le / max_Le : glm::vec3(0.0f);

    for (int i = 0; i < SPECTRAL_N; i++) {
        float refl = spectral_reflectance_from_rgb(material.color, path.wavelengths[i]);
        float le_wl = max_Le * spectral_reflectance_from_rgb(norm_Le, path.wavelengths[i]);
        float is_w = 1.0f / (range * path.pdfs[i]);
        float c = path.throughputs[i] * (refl / PI) * le_wl * cos_theta * mis_w / pdf_env * is_w;
        contrib += spectral_to_sRGB(path.wavelengths[i], c);
    }
#else
    glm::vec3 f_brdf = material.color / PI;
    contrib = path.color * f_brdf * Le * cos_theta * mis_w / pdf_env;
#endif

    // Test against non-triangle geoms
    bool occByPrim = false;
    glm::vec3 shadowOrigin = hitPt + nor * EPSILON;
    Ray shadowRayR = { shadowOrigin + wi * EPSILON * 10.0f, wi };
    for (int i = 0; i < numNonTriGeoms; i++) {
        float t = -1.0f;
        glm::vec3 tmp_p, tmp_n;
        bool tmp_o;
        if (nonTriGeoms[i].type == CUBE)
            t = boxIntersectionTest(nonTriGeoms[i], shadowRayR, tmp_p, tmp_n, tmp_o);
        else if (nonTriGeoms[i].type == SPHERE)
            t = sphereIntersectionTest(nonTriGeoms[i], shadowRayR, tmp_p, tmp_n, tmp_o);
        if (t > 0.0f) { occByPrim = true; break; }
    }

    shadowRays[idx].origin = shadowOrigin;
    shadowRays[idx].direction = wi;
    shadowRays[idx].maxDist = 1e20f;  // infinity for env map
    shadowRays[idx].neeContrib = contrib;
    shadowRays[idx].pixelIndex = path.pixelIndex;
    shadowRays[idx].occludedByPrimitive = occByPrim ? 1 : 0;
}

// ============================================================================
// Disney GGX Shadow Rays (Geometry Lights)
// ============================================================================

__global__ void kernPrepareShadowRaysDisneyGGX(
    int num_hit,
    DisneyGGXHitWorkItem* queue,
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

    DisneyGGXHitWorkItem item = queue[idx];
    PathSegment& path = paths[item.path_idx];
    Material material = materials[item.material_id];

    // Sample texture
    if (material.textureId >= 0 && material.textureId < numTextures && textureObjects != nullptr) {
        float4 texColor = tex2D<float4>(textureObjects[material.textureId], item.uv.x, item.uv.y);
        material.color *= glm::vec3(texColor.x, texColor.y, texColor.z);
    }

    curandState local_rand_state = rand_states[path.pixelIndex];

    glm::vec3 nor = glm::normalize(item.surface_normal);
    glm::vec3 hitPt = item.intersect_point;
    glm::vec3 V = -glm::normalize(item.incident_ray_dir);

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

    if (cos_surface <= 0.0f || cos_light <= 0.0f) {
        shadowRays[idx].neeContrib = glm::vec3(0.0f);
        shadowRays[idx].occludedByPrimitive = 1;
        return;
    }

    float pdf_light = (d2 / (ls.area * cos_light)) / (float)num_lights;

    // Evaluate Disney BRDF for this light direction
#if ENABLE_SPECTRAL_RENDERING
    glm::vec3 contrib(0.0f);
    float range = (float)(MAX_SAMPLE_WAVELENGTH - MIN_SAMPLE_WAVELENGTH);
    for (int i = 0; i < SPECTRAL_N; i++) {
        float brdf_pdf;
        float f_brdf_wl = evaluateDisneyBRDF_Spectral(V, wi, nor, material, path.wavelengths[i], brdf_pdf);
        float mis_w = powerHeuristic(pdf_light, brdf_pdf);
        float le = spectral_reflectance_from_rgb(lightMat.color, path.wavelengths[i]) * lightMat.emittance;
        float is_w = 1.0f / (range * path.pdfs[i]);
        float c = path.throughputs[i] * f_brdf_wl * le * cos_surface * mis_w / pdf_light * is_w;
        contrib += spectral_to_sRGB(path.wavelengths[i], c);
    }
#else
    float brdf_pdf;
    glm::vec3 f_brdf = evaluateDisneyBRDF(V, wi, nor, material, brdf_pdf);
    float mis_w = powerHeuristic(pdf_light, brdf_pdf);
    glm::vec3 Le = lightMat.color * lightMat.emittance;
    glm::vec3 contrib = path.color * f_brdf * Le * cos_surface * mis_w / pdf_light;
#endif

    // Build shadow ray + prim occlusion test
    glm::vec3 shadowOrigin = hitPt + nor * EPSILON;
    glm::vec3 shadowDir = wi;
    float maxDist = d - EPSILON * 20.0f;

    bool occByPrim = false;
    Ray shadowRay = { shadowOrigin + shadowDir * EPSILON * 10.0f, shadowDir };
    for (int i = 0; i < numNonTriGeoms; i++) {
        if (nonTriGeoms[i].materialid == lightGeom.materialid) continue;
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

// ============================================================================
// Disney GGX Shadow Rays (Environment Map)
// ============================================================================

__global__ void kernPrepareEnvMapShadowRaysDisneyGGX(
    int num_hit,
    DisneyGGXHitWorkItem* queue,
    PathSegment* paths,
    Material* materials,
    curandState* rand_states,
    Geom* nonTriGeoms, int numNonTriGeoms,
    ShadowRayRequest* shadowRays,
    cudaTextureObject_t* textureObjects, int numTextures,
    cudaTextureObject_t envMap,
    const float* marginalCDF, const float* conditionalCDF,
    int envW, int envH, float envTotalPower)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_hit) return;

    DisneyGGXHitWorkItem item = queue[idx];
    PathSegment& path = paths[item.path_idx];
    Material material = materials[item.material_id];

    if (material.textureId >= 0 && material.textureId < numTextures && textureObjects != nullptr) {
        float4 texColor = tex2D<float4>(textureObjects[material.textureId], item.uv.x, item.uv.y);
        material.color *= glm::vec3(texColor.x, texColor.y, texColor.z);
    }

    curandState local_rand_state = rand_states[path.pixelIndex];
    float u1 = curand_uniform(&local_rand_state);
    float u2 = curand_uniform(&local_rand_state);
    rand_states[path.pixelIndex] = local_rand_state;

    glm::vec3 nor = glm::normalize(item.surface_normal);
    glm::vec3 hitPt = item.intersect_point;
    glm::vec3 V = -glm::normalize(item.incident_ray_dir);

    // Sample direction from env map CDF
    int y = binarySearchCDF(marginalCDF, envH + 1, u1);
    float py = marginalCDF[y + 1] - marginalCDF[y];
    if (py <= 0.0f) {
        shadowRays[idx].neeContrib = glm::vec3(0.0f);
        shadowRays[idx].occludedByPrimitive = 1;
        return;
    }
    float dv = (u1 - marginalCDF[y]) / py;
    float v = (y + dv + 0.5f) / envH;

    const float* rowCDF = conditionalCDF + y * (envW + 1);
    int x = binarySearchCDF(rowCDF, envW + 1, u2);
    float px = rowCDF[x + 1] - rowCDF[x];
    if (px <= 0.0f) {
        shadowRays[idx].neeContrib = glm::vec3(0.0f);
        shadowRays[idx].occludedByPrimitive = 1;
        return;
    }
    float du = (u2 - rowCDF[x]) / px;
    float u = (x + du + 0.5f) / envW;

    float phi = (u - 0.5f) * 2.0f * PI;
    float y_val = sinf((0.5f - v) * PI);
    float xz = sqrtf(fmaxf(0.0f, 1.0f - y_val * y_val));
    glm::vec3 wi(xz * cosf(phi), y_val, xz * sinf(phi));
    wi = glm::normalize(wi);

    float cos_theta = glm::dot(wi, nor);
    if (cos_theta <= 0.0f) {
        shadowRays[idx].neeContrib = glm::vec3(0.0f);
        shadowRays[idx].occludedByPrimitive = 1;
        return;
    }

    float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - y_val * y_val));
    float pdf_env = py * px * envW * envH / (2.0f * PI * PI * fmaxf(sinTheta, 1e-6f));
    if (pdf_env <= 0.0f) {
        shadowRays[idx].neeContrib = glm::vec3(0.0f);
        shadowRays[idx].occludedByPrimitive = 1;
        return;
    }

    float4 envColor = tex2D<float4>(envMap, u, v);
    glm::vec3 Le(envColor.x, envColor.y, envColor.z);

    // Evaluate Disney BRDF
#if ENABLE_SPECTRAL_RENDERING
    glm::vec3 contrib(0.0f);
    float range = (float)(MAX_SAMPLE_WAVELENGTH - MIN_SAMPLE_WAVELENGTH);
    // HDR scaling logic for spectral mapping
    float max_Le = fmaxf(Le.x, fmaxf(Le.y, Le.z));
    glm::vec3 norm_Le = max_Le > 0.0f ? Le / max_Le : glm::vec3(0.0f);

    for (int i = 0; i < SPECTRAL_N; i++) {
        float brdf_pdf;
        float f_brdf_wl = evaluateDisneyBRDF_Spectral(V, wi, nor, material, path.wavelengths[i], brdf_pdf);
        float mis_w = powerHeuristic(pdf_env, brdf_pdf);
        float le_wl = max_Le * spectral_reflectance_from_rgb(norm_Le, path.wavelengths[i]);
        float is_w = 1.0f / (range * path.pdfs[i]);
        float c = path.throughputs[i] * f_brdf_wl * le_wl * cos_theta * mis_w / pdf_env * is_w;
        contrib += spectral_to_sRGB(path.wavelengths[i], c);
    }
#else
    float brdf_pdf;
    glm::vec3 f_brdf = evaluateDisneyBRDF(V, wi, nor, material, brdf_pdf);
    float mis_w = powerHeuristic(pdf_env, brdf_pdf);
    glm::vec3 contrib = path.color * f_brdf * Le * cos_theta * mis_w / pdf_env;
#endif

    // Non-triangle occlusion test
    bool occByPrim = false;
    glm::vec3 shadowOrigin = hitPt + nor * EPSILON;
    Ray shadowRayR = { shadowOrigin + wi * EPSILON * 10.0f, wi };
    for (int i = 0; i < numNonTriGeoms; i++) {
        float t = -1.0f;
        glm::vec3 tmp_p, tmp_n;
        bool tmp_o;
        if (nonTriGeoms[i].type == CUBE)
            t = boxIntersectionTest(nonTriGeoms[i], shadowRayR, tmp_p, tmp_n, tmp_o);
        else if (nonTriGeoms[i].type == SPHERE)
            t = sphereIntersectionTest(nonTriGeoms[i], shadowRayR, tmp_p, tmp_n, tmp_o);
        if (t > 0.0f) { occByPrim = true; break; }
    }

    shadowRays[idx].origin = shadowOrigin;
    shadowRays[idx].direction = wi;
    shadowRays[idx].maxDist = 1e20f;
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