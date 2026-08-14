#pragma once

// Shared data structures between OptiX device programs and host code.

#include <glm/glm.hpp>
#include "sceneStructs.h"

// Forward-declare OptixTraversableHandle so this header works without optix.h
#ifndef __optix_optix_types_h__
typedef unsigned long long OptixTraversableHandle;
#endif

// Per-ray hit result from OptiX (triangle-only)
struct OptiXHitResult {
    float t;            // hit distance (-1 = miss)
    glm::vec3 normal;   // interpolated surface normal
    glm::vec2 uv;       // interpolated UV
    int materialId;     // material index
};

// Shadow ray request for MIS direct lighting
struct ShadowRayRequest {
    glm::vec3 origin;
    glm::vec3 direction;
    float maxDist;
    glm::vec3 neeContrib;   // pre-computed direct light RGB contribution (assumes visible)
    int pixelIndex;         // pixel to accumulate to
    int occludedByPrimitive; // 1 if already occluded by box/sphere
};

// Launch parameters passed to all OptiX programs
struct OptixLaunchParams {
    int mode;  // 0 = primary ray, 1 = shadow ray

    // === Primary ray data (mode==0) ===
    PathSegment* paths;
    int* activeIndices;
    int numPaths;
    OptiXHitResult* hitResults;

    // Per-triangle data for closest-hit
    int* materialIds;
    glm::vec3* normals;
    int* vertexIndices;
    glm::vec2* triUV0;
    glm::vec2* triUV1;
    glm::vec2* triUV2;

    // === Shadow ray data (mode==1) ===
    ShadowRayRequest* shadowRays;
    int numShadowRays;
    int* shadowOccluded;  // output: 1=occluded by triangle, 0=visible

    // === Shared ===
    OptixTraversableHandle traversable;
};

// SBT record data
struct HitGroupSBTData {
    // Empty -- all data accessed via launch params
};
