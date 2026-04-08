#pragma once

#include "sceneStructs.h"
#include "nrc.h"

#include <glm/glm.hpp>
#include <thrust/random.h>
#include <curand_kernel.h>

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal, 
    thrust::default_random_engine& rng);

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 *
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 *
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */
__host__ __device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng);


__global__ void kernShadeMiss(
    int num_hit,
    MissWorkItem* queue,
    PathSegment* paths,
    glm::vec3* dev_img,
    glm::vec3* dev_albedo, glm::vec3* dev_normal, int depth,
    cudaTextureObject_t envMap, bool hasEnvMap,
    const float* envCDF_marginal, const float* envCDF_conditional,
    int envW, int envH);

__global__ void kernShadeHitLight(
    int num_hit,
    HitLightWorkItem* queue,
    PathSegment* paths,
    Material* materials,
    glm::vec3* dev_img,
    Geom* geoms,
    glm::vec3* positions,
    int* light_indices,
    int num_lights,
    glm::vec3* dev_albedo, glm::vec3* dev_normal, int depth);

__global__ void kernShadeLambertian(
    int num_hit, 
    LambertianHitWorkItem* queue, 
    PathSegment* paths, 
    Material* materials,
    curandState* rand_states,
    glm::vec3* dev_img,
    Geom* geoms,
    int geoms_size,
    glm::vec3* positions,
    int* light_indices,
    int num_lights,
    cudaTextureObject_t* textureObjects,
    int numTextures,
    glm::vec3* dev_albedo, glm::vec3* dev_normal, int depth,
#if ENABLE_NRC
    NRCQueryWorkItem* nrc_queries = nullptr,
    int* nrc_query_counter = nullptr,
    float nrc_train_fraction = 0.0f,
    int* nrc_train_sample_counter = nullptr,
    NRCTrainingSample* nrc_train_samples = nullptr,
#endif
    int traceDepth = 8);

__global__ void kernShadeSpecular(
    int num_hit,
    SpecularHitWorkItem* queue,
    PathSegment* paths,
    Material* materials,
    glm::vec3* dev_albedo, glm::vec3* dev_normal, int depth);

__global__ void kernShadeGlass(
    int num_hit,
    GlassHitWorkItem* queue,
    PathSegment* paths,
    Material* materials,
    curandState* rand_states,
    glm::vec3* dev_albedo, glm::vec3* dev_normal, int depth);

__global__ void kernShadeDisneyGGX(
    int num_hit,
    DisneyGGXHitWorkItem* queue,
    PathSegment* paths,
    Material* materials,
    curandState* rand_states,
    glm::vec3* dev_img,
    Geom* geoms,
    int geoms_size,
    glm::vec3* positions,
    int* light_indices,
    int num_lights,
    cudaTextureObject_t* textureObjects,
    int numTextures,
    glm::vec3* dev_albedo, glm::vec3* dev_normal, int depth,
#if ENABLE_NRC
    NRCQueryWorkItem* nrc_queries = nullptr,
    int* nrc_query_counter = nullptr,
    float nrc_train_fraction = 0.0f,
    int* nrc_train_sample_counter = nullptr,
    NRCTrainingSample* nrc_train_samples = nullptr,
#endif
    int traceDepth = 8);