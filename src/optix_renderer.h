#pragma once

#include <optix.h>
#include <optix_stubs.h>
#include <vector>
#include <glm/glm.hpp>

#include "sceneStructs.h"
#include "optix_params.h"

class Scene;

class OptixRenderer {
public:
    void init();
    void buildAccel(Scene* scene);

    // Mode 0: Trace primary rays against triangles. Results in dev_hitResults.
    void trace(PathSegment* dev_paths, int numPaths);

    // Mode 1: Trace shadow rays against triangles. Results in dev_shadowOccluded.
    void traceShadowRays(ShadowRayRequest* dev_shadowRays, int numRays, int* dev_shadowOccluded);

    void cleanup();

    void setNormals(glm::vec3* normals) { d_normals = normals; }

    // Public device buffers
    OptiXHitResult* dev_hitResults = nullptr;

#if ENABLE_DENOISER
    // Denoiser: init, run, cleanup
    void initDenoiser(int width, int height);
    void denoise(glm::vec3* dev_colorAccum, glm::vec3* dev_albedo, glm::vec3* dev_normal,
                 glm::vec3* dev_denoisedOut, int width, int height, int iter);
    void cleanupDenoiser();
#endif

private:
    void createContext();
    void createModule();
    void createProgramGroups();
    void createPipeline();
    void createSBT();

    OptixDeviceContext context = nullptr;
    OptixModule module = nullptr;
    OptixPipeline pipeline = nullptr;

    OptixProgramGroup raygenPG = nullptr;
    OptixProgramGroup missPG = nullptr;
    OptixProgramGroup hitgroupPG = nullptr;

    OptixShaderBindingTable sbt = {};
    CUdeviceptr d_raygenRecord = 0;
    CUdeviceptr d_missRecord = 0;
    CUdeviceptr d_hitgroupRecord = 0;

    OptixTraversableHandle traversable = 0;
    CUdeviceptr d_gasOutputBuffer = 0;

    CUdeviceptr d_materialIds = 0;
    CUdeviceptr d_vertexIndices = 0;
    CUdeviceptr d_triUV0 = 0;
    CUdeviceptr d_triUV1 = 0;
    CUdeviceptr d_triUV2 = 0;

    glm::vec3* d_normals = nullptr;

    OptixLaunchParams launchParams = {};
    CUdeviceptr d_launchParams = 0;

    int maxPaths = 0;

    bool initialized = false;

#if ENABLE_DENOISER
    OptixDenoiser denoiser = nullptr;
    CUdeviceptr d_denoiserState = 0;
    CUdeviceptr d_denoiserScratch = 0;
    CUdeviceptr d_denoiserIntensity = 0;
    CUdeviceptr d_hdrInput = 0;        // normalized HDR input (color / iter)
    size_t denoiserStateSize = 0;
    size_t denoiserScratchSize = 0;
    bool denoiserInitialized = false;
#endif
};

