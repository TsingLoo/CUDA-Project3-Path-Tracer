// OptiX device programs: dual-mode raygen, closest-hit, miss
// Mode 0: Primary ray trace (triangle intersection -> HitResult buffer)
// Mode 1: Shadow ray trace (occlusion test -> shadowOccluded buffer)

#include <optix_device.h>
#include <cuda_runtime.h>

#include "glm/glm.hpp"
#include "optix_params.h"

extern "C" __constant__ OptixLaunchParams params;

// ============================================================================
// Raygen
// ============================================================================
extern "C" __global__ void __raygen__primary()
{
    const int idx = optixGetLaunchIndex().x;

    // === SHADOW RAY MODE ===
    if (params.mode == 1) {
        if (idx >= params.numShadowRays) return;

        ShadowRayRequest& req = params.shadowRays[idx];

        // Already occluded by box/sphere -- skip OptiX trace
        if (req.occludedByPrimitive) {
            params.shadowOccluded[idx] = 1;
            return;
        }

        float3 origin = make_float3(req.origin.x, req.origin.y, req.origin.z);
        float3 direction = make_float3(req.direction.x, req.direction.y, req.direction.z);

        unsigned int occluded = 0;
        unsigned int p1 = 0;

        optixTrace(
            params.traversable,
            origin,
            direction,
            0.001f,             // tmin
            req.maxDist,        // tmax -- only test up to light distance
            0.0f,               // rayTime
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            0, 1, 0,
            occluded, p1
        );

        params.shadowOccluded[idx] = occluded;
        return;
    }

    // === PRIMARY RAY MODE ===
    if (idx >= params.numPaths) return;

    PathSegment& path = params.paths[idx];
    if (path.remainingBounces <= 0) {
        params.hitResults[idx].t = -1.0f;
        return;
    }

    float3 origin = make_float3(path.ray.origin.x, path.ray.origin.y, path.ray.origin.z);
    float3 direction = make_float3(path.ray.direction.x, path.ray.direction.y, path.ray.direction.z);

    unsigned int p0 = idx;
    unsigned int p1 = 0;

    optixTrace(
        params.traversable,
        origin,
        direction,
        0.001f,           // tmin
        1e16f,            // tmax
        0.0f,             // rayTime
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0, 1, 0,
        p0, p1
    );
}

// ============================================================================
// Closest-Hit
// ============================================================================
extern "C" __global__ void __closesthit__primary()
{
    // Shadow mode: any hit means occluded
    if (params.mode == 1) {
        optixSetPayload_0(1);
        return;
    }

    // Primary mode: write hit data to buffer
    const int path_index = optixGetPayload_0();
    const int primIdx = optixGetPrimitiveIndex();

    const float2 bary = optixGetTriangleBarycentrics();
    const float baryU = bary.x;
    const float baryV = bary.y;
    const float baryW = 1.0f - baryU - baryV;

    const int i0 = params.vertexIndices[primIdx * 3 + 0];
    const int i1 = params.vertexIndices[primIdx * 3 + 1];
    const int i2 = params.vertexIndices[primIdx * 3 + 2];

    const glm::vec3 n0 = params.normals[i0];
    const glm::vec3 n1 = params.normals[i1];
    const glm::vec3 n2 = params.normals[i2];
    glm::vec3 normal = glm::normalize(baryW * n0 + baryU * n1 + baryV * n2);

    const glm::vec2 uv0 = params.triUV0[primIdx];
    const glm::vec2 uv1 = params.triUV1[primIdx];
    const glm::vec2 uv2 = params.triUV2[primIdx];
    glm::vec2 uv = baryW * uv0 + baryU * uv1 + baryV * uv2;

    const float3 ray_d = optixGetWorldRayDirection();
    glm::vec3 ray_dir(ray_d.x, ray_d.y, ray_d.z);
    if (glm::dot(normal, ray_dir) > 0.0f) {
        normal = -normal;
    }

    OptiXHitResult& result = params.hitResults[path_index];
    result.t = optixGetRayTmax();
    result.normal = normal;
    result.uv = uv;
    result.materialId = params.materialIds[primIdx];
}

// ============================================================================
// Miss
// ============================================================================
extern "C" __global__ void __miss__primary()
{
    if (params.mode == 1) {
        optixSetPayload_0(0);  // not occluded
        return;
    }

    // Primary mode: no triangle hit
    const int path_index = optixGetPayload_0();
    params.hitResults[path_index].t = -1.0f;
}
