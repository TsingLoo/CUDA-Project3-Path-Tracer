// OptiX host-side renderer: context, module, pipeline, SBT, GAS, trace dispatch.
// OptiX handles TRIANGLE intersection only. Box/sphere are handled by CUDA.

#include "optix_renderer.h"
#include "scene.h"

#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <cstring>

#define OPTIX_CHECK(call)                                                      \
    do {                                                                       \
        OptixResult res = call;                                                \
        if (res != OPTIX_SUCCESS) {                                            \
            fprintf(stderr, "OptiX error '%s' at %s:%d\n",                     \
                    optixGetErrorString(res), __FILE__, __LINE__);             \
            exit(1);                                                           \
        }                                                                      \
    } while(0)

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error '%s' at %s:%d\n",                      \
                    cudaGetErrorString(err), __FILE__, __LINE__);              \
            exit(1);                                                           \
        }                                                                      \
    } while(0)

static void optixLogCallback(unsigned int level, const char* tag, const char* message, void*) {
    if (level <= 2) {
        fprintf(stderr, "[OptiX %s] %s\n", tag, message);
    }
}

template <typename T>
struct SbtRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<int>             RaygenRecord;
typedef SbtRecord<int>             MissRecord;
typedef SbtRecord<HitGroupSBTData> HitGroupRecord;

// ============================================================================
// Initialization
// ============================================================================

void OptixRenderer::init() {
    if (initialized) return;
    createContext();
    createModule();
    createProgramGroups();
    createPipeline();

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_launchParams), sizeof(OptixLaunchParams)));

    initialized = true;
    printf("OptiX initialized successfully (RT Core acceleration enabled)\n");
}

void OptixRenderer::createContext() {
    CUDA_CHECK(cudaFree(0));
    OPTIX_CHECK(optixInit());

    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &optixLogCallback;
    options.logCallbackLevel = 4;

    CUcontext cuCtx = 0;
    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));
}

void OptixRenderer::createModule() {
    std::vector<std::string> searchPaths = {
        "optix_programs.ptx",
        "../optix_programs.ptx",
        "bin/optix_programs.ptx",
        "../bin/optix_programs.ptx",
        "build/bin/optix_programs.ptx",
        "../build/bin/optix_programs.ptx",
    };

    std::ifstream ptxFile;
    for (const auto& path : searchPaths) {
        ptxFile.open(path, std::ios::binary);
        if (ptxFile.is_open()) {
            printf("Loaded OptiX PTX from: %s\n", path.c_str());
            break;
        }
    }

    if (!ptxFile.is_open()) {
        fprintf(stderr, "ERROR: Could not find optix_programs.ptx\n");
        exit(1);
    }

    std::stringstream ss;
    ss << ptxFile.rdbuf();
    std::string ptxStr = ss.str();

    OptixModuleCompileOptions moduleCompileOptions = {};
    moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    OptixPipelineCompileOptions pco = {};
    pco.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pco.usesMotionBlur = false;
    pco.numPayloadValues = 2;
    pco.numAttributeValues = 2;
    pco.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pco.pipelineLaunchParamsVariableName = "params";
    pco.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

    char log[2048];
    size_t logSize = sizeof(log);
    OPTIX_CHECK(optixModuleCreate(context, &moduleCompileOptions, &pco,
        ptxStr.c_str(), ptxStr.size(), log, &logSize, &module));

    if (logSize > 1) printf("OptiX module log: %s\n", log);
}

void OptixRenderer::createProgramGroups() {
    OptixProgramGroupOptions pgOptions = {};
    char log[2048];
    size_t logSize;

    // Raygen
    OptixProgramGroupDesc raygenDesc = {};
    raygenDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygenDesc.raygen.module = module;
    raygenDesc.raygen.entryFunctionName = "__raygen__primary";
    logSize = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(context, &raygenDesc, 1, &pgOptions, log, &logSize, &raygenPG));

    // Miss
    OptixProgramGroupDesc missDesc = {};
    missDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missDesc.miss.module = module;
    missDesc.miss.entryFunctionName = "__miss__primary";
    logSize = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(context, &missDesc, 1, &pgOptions, log, &logSize, &missPG));

    // Hit group
    OptixProgramGroupDesc hitDesc = {};
    hitDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitDesc.hitgroup.moduleCH = module;
    hitDesc.hitgroup.entryFunctionNameCH = "__closesthit__primary";
    hitDesc.hitgroup.moduleAH = nullptr;
    hitDesc.hitgroup.entryFunctionNameAH = nullptr;
    logSize = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(context, &hitDesc, 1, &pgOptions, log, &logSize, &hitgroupPG));
}

void OptixRenderer::createPipeline() {
    OptixProgramGroup programGroups[] = { raygenPG, missPG, hitgroupPG };

    OptixPipelineLinkOptions plo = {};
    plo.maxTraceDepth = 1;

    OptixPipelineCompileOptions pco = {};
    pco.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pco.usesMotionBlur = false;
    pco.numPayloadValues = 2;
    pco.numAttributeValues = 2;
    pco.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pco.pipelineLaunchParamsVariableName = "params";
    pco.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

    char log[2048];
    size_t logSize = sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(context, &pco, &plo, programGroups, 3,
        log, &logSize, &pipeline));

    OPTIX_CHECK(optixPipelineSetStackSize(pipeline, 2*1024, 2*1024, 2*1024, 1));
}

void OptixRenderer::createSBT() {
    RaygenRecord raygenRecord;
    OPTIX_CHECK(optixSbtRecordPackHeader(raygenPG, &raygenRecord));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygenRecord), sizeof(RaygenRecord)));
    CUDA_CHECK(cudaMemcpy((void*)d_raygenRecord, &raygenRecord, sizeof(RaygenRecord), cudaMemcpyHostToDevice));

    MissRecord missRecord;
    OPTIX_CHECK(optixSbtRecordPackHeader(missPG, &missRecord));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_missRecord), sizeof(MissRecord)));
    CUDA_CHECK(cudaMemcpy((void*)d_missRecord, &missRecord, sizeof(MissRecord), cudaMemcpyHostToDevice));

    HitGroupRecord hitRecord;
    OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPG, &hitRecord));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hitgroupRecord), sizeof(HitGroupRecord)));
    CUDA_CHECK(cudaMemcpy((void*)d_hitgroupRecord, &hitRecord, sizeof(HitGroupRecord), cudaMemcpyHostToDevice));

    sbt.raygenRecord = d_raygenRecord;
    sbt.missRecordBase = d_missRecord;
    sbt.missRecordStrideInBytes = sizeof(MissRecord);
    sbt.missRecordCount = 1;
    sbt.hitgroupRecordBase = d_hitgroupRecord;
    sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
    sbt.hitgroupRecordCount = 1;
}

// ============================================================================
// Acceleration Structure
// ============================================================================

void OptixRenderer::buildAccel(Scene* scene) {
    int numTriangles = 0;
    for (const auto& geom : scene->geoms) {
        if (geom.type == TRIANGLE) numTriangles++;
    }

    if (numTriangles == 0) {
        printf("OptiX: No triangles to accelerate.\n");
        // Still need dev_hitResults for the merge kernel — fill with "miss" (t = -1)
        const Camera& cam = scene->state.camera;
        maxPaths = cam.resolution.x * cam.resolution.y;
        CUDA_CHECK(cudaMalloc(&dev_hitResults, maxPaths * sizeof(OptiXHitResult)));
        // Initialize all hits as misses
        std::vector<OptiXHitResult> initHits(maxPaths);
        for (auto& h : initHits) { h.t = -1.0f; h.materialId = -1; }
        CUDA_CHECK(cudaMemcpy(dev_hitResults, initHits.data(), maxPaths * sizeof(OptiXHitResult), cudaMemcpyHostToDevice));
        return;
    }

    printf("OptiX: Building GAS for %d triangles...\n", numTriangles);

    // Allocate hit results buffer (sized for max pixel count)
    const Camera& cam = scene->state.camera;
    maxPaths = cam.resolution.x * cam.resolution.y;
    CUDA_CHECK(cudaMalloc(&dev_hitResults, maxPaths * sizeof(OptiXHitResult)));

    // Build vertex and index arrays
    std::vector<float3> vertices(scene->positions.size());
    for (size_t i = 0; i < scene->positions.size(); i++) {
        vertices[i] = make_float3(scene->positions[i].x, scene->positions[i].y, scene->positions[i].z);
    }

    std::vector<uint3> indices(numTriangles);
    std::vector<int> materialIdsHost(numTriangles);
    std::vector<int> vertexIndicesHost(numTriangles * 3);
    std::vector<glm::vec2> triUV0Host(numTriangles);
    std::vector<glm::vec2> triUV1Host(numTriangles);
    std::vector<glm::vec2> triUV2Host(numTriangles);

    int triIdx = 0;
    for (const auto& geom : scene->geoms) {
        if (geom.type != TRIANGLE) continue;
        indices[triIdx] = make_uint3(geom.v0, geom.v1, geom.v2);
        materialIdsHost[triIdx] = geom.materialid;
        vertexIndicesHost[triIdx * 3 + 0] = geom.v0;
        vertexIndicesHost[triIdx * 3 + 1] = geom.v1;
        vertexIndicesHost[triIdx * 3 + 2] = geom.v2;
        triUV0Host[triIdx] = geom.uv0;
        triUV1Host[triIdx] = geom.uv1;
        triUV2Host[triIdx] = geom.uv2;
        triIdx++;
    }

    CUdeviceptr d_vertices, d_indices;
    CUDA_CHECK(cudaMalloc((void**)&d_vertices, vertices.size() * sizeof(float3)));
    CUDA_CHECK(cudaMemcpy((void*)d_vertices, vertices.data(), vertices.size() * sizeof(float3), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void**)&d_indices, indices.size() * sizeof(uint3)));
    CUDA_CHECK(cudaMemcpy((void*)d_indices, indices.data(), indices.size() * sizeof(uint3), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void**)&d_materialIds, materialIdsHost.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy((void*)d_materialIds, materialIdsHost.data(), materialIdsHost.size() * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void**)&d_vertexIndices, vertexIndicesHost.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy((void*)d_vertexIndices, vertexIndicesHost.data(), vertexIndicesHost.size() * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void**)&d_triUV0, triUV0Host.size() * sizeof(glm::vec2)));
    CUDA_CHECK(cudaMemcpy((void*)d_triUV0, triUV0Host.data(), triUV0Host.size() * sizeof(glm::vec2), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void**)&d_triUV1, triUV1Host.size() * sizeof(glm::vec2)));
    CUDA_CHECK(cudaMemcpy((void*)d_triUV1, triUV1Host.data(), triUV1Host.size() * sizeof(glm::vec2), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void**)&d_triUV2, triUV2Host.size() * sizeof(glm::vec2)));
    CUDA_CHECK(cudaMemcpy((void*)d_triUV2, triUV2Host.data(), triUV2Host.size() * sizeof(glm::vec2), cudaMemcpyHostToDevice));

    // Build GAS
    OptixBuildInput buildInput = {};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    buildInput.triangleArray.vertexStrideInBytes = sizeof(float3);
    buildInput.triangleArray.numVertices = (unsigned int)vertices.size();
    buildInput.triangleArray.vertexBuffers = &d_vertices;
    buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    buildInput.triangleArray.indexStrideInBytes = sizeof(uint3);
    buildInput.triangleArray.numIndexTriplets = (unsigned int)numTriangles;
    buildInput.triangleArray.indexBuffer = d_indices;

    unsigned int flags = OPTIX_GEOMETRY_FLAG_NONE;
    buildInput.triangleArray.flags = &flags;
    buildInput.triangleArray.numSbtRecords = 1;

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes bufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accelOptions, &buildInput, 1, &bufferSizes));

    CUdeviceptr d_tempBuffer;
    CUDA_CHECK(cudaMalloc((void**)&d_tempBuffer, bufferSizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_gasOutputBuffer, bufferSizes.outputSizeInBytes));

    OPTIX_CHECK(optixAccelBuild(context, 0, &accelOptions, &buildInput, 1,
        d_tempBuffer, bufferSizes.tempSizeInBytes,
        d_gasOutputBuffer, bufferSizes.outputSizeInBytes,
        &traversable, nullptr, 0));

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree((void*)d_tempBuffer));

    createSBT();

    printf("OptiX: GAS built successfully. Traversable handle: %llu\n", (unsigned long long)traversable);
}

// ============================================================================
// Trace -- writes results to dev_hitResults buffer
// ============================================================================

void OptixRenderer::trace(PathSegment* dev_paths, int numPaths) {
    if (traversable == 0) return;

    launchParams.mode = 0;  // primary ray mode
    launchParams.paths = dev_paths;
    launchParams.numPaths = numPaths;
    launchParams.hitResults = dev_hitResults;
    launchParams.traversable = traversable;

    launchParams.materialIds = (int*)d_materialIds;
    launchParams.vertexIndices = (int*)d_vertexIndices;
    launchParams.normals = d_normals;
    launchParams.triUV0 = (glm::vec2*)d_triUV0;
    launchParams.triUV1 = (glm::vec2*)d_triUV1;
    launchParams.triUV2 = (glm::vec2*)d_triUV2;

    CUDA_CHECK(cudaMemcpy((void*)d_launchParams, &launchParams, sizeof(OptixLaunchParams), cudaMemcpyHostToDevice));

    OPTIX_CHECK(optixLaunch(pipeline, 0, d_launchParams, sizeof(OptixLaunchParams),
        &sbt, numPaths, 1, 1));
}

// ============================================================================
// Shadow Ray Trace -- batch trace shadow rays for MIS
// ============================================================================

void OptixRenderer::traceShadowRays(ShadowRayRequest* dev_shadowRays, int numRays, int* dev_shadowOccluded) {
    if (traversable == 0 || numRays == 0) return;

    launchParams.mode = 1;  // shadow ray mode
    launchParams.shadowRays = dev_shadowRays;
    launchParams.numShadowRays = numRays;
    launchParams.shadowOccluded = dev_shadowOccluded;
    launchParams.traversable = traversable;

    CUDA_CHECK(cudaMemcpy((void*)d_launchParams, &launchParams, sizeof(OptixLaunchParams), cudaMemcpyHostToDevice));

    OPTIX_CHECK(optixLaunch(pipeline, 0, d_launchParams, sizeof(OptixLaunchParams),
        &sbt, numRays, 1, 1));
}

// ============================================================================
// Cleanup
// ============================================================================

void OptixRenderer::cleanup() {
    if (!initialized) return;

#if ENABLE_DENOISER
    cleanupDenoiser();
#endif

    if (dev_hitResults) cudaFree(dev_hitResults);
    if (d_gasOutputBuffer) cudaFree((void*)d_gasOutputBuffer);
    if (d_materialIds) cudaFree((void*)d_materialIds);
    if (d_vertexIndices) cudaFree((void*)d_vertexIndices);
    if (d_triUV0) cudaFree((void*)d_triUV0);
    if (d_triUV1) cudaFree((void*)d_triUV1);
    if (d_triUV2) cudaFree((void*)d_triUV2);

    if (d_raygenRecord) cudaFree((void*)d_raygenRecord);
    if (d_missRecord) cudaFree((void*)d_missRecord);
    if (d_hitgroupRecord) cudaFree((void*)d_hitgroupRecord);
    if (d_launchParams) cudaFree((void*)d_launchParams);

    if (pipeline) optixPipelineDestroy(pipeline);
    if (hitgroupPG) optixProgramGroupDestroy(hitgroupPG);
    if (missPG) optixProgramGroupDestroy(missPG);
    if (raygenPG) optixProgramGroupDestroy(raygenPG);
    if (module) optixModuleDestroy(module);
    if (context) optixDeviceContextDestroy(context);

    initialized = false;
    printf("OptiX renderer cleaned up.\n");
}

// ============================================================================
// OptiX AI Denoiser (Mode 3: Beauty + Albedo + Normal)
// ============================================================================

#if ENABLE_DENOISER

__global__ void kernNormalizeImage(int pixelcount, glm::vec3* input, float* output, int iter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pixelcount) return;
    float divisor = (float)iter;
#if ENABLE_SPECTRAL_RENDERING
    divisor *= SPECTRAL_N;
#endif
    glm::vec3 c = input[idx] / divisor;
    output[idx * 3 + 0] = c.x;
    output[idx * 3 + 1] = c.y;
    output[idx * 3 + 2] = c.z;
}

__global__ void kernVec3ToFloat3(int pixelcount, glm::vec3* input, float* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pixelcount) return;
    output[idx * 3 + 0] = input[idx].x;
    output[idx * 3 + 1] = input[idx].y;
    output[idx * 3 + 2] = input[idx].z;
}

void OptixRenderer::initDenoiser(int width, int height) {
    if (denoiserInitialized) return;
    if (!context) {
        fprintf(stderr, "ERROR: OptiX context not initialized before denoiser init\n");
        return;
    }

    OptixDenoiserOptions options = {};
    options.guideAlbedo = 1;
    options.guideNormal = 1;

    OPTIX_CHECK(optixDenoiserCreate(context, OPTIX_DENOISER_MODEL_KIND_HDR, &options, &denoiser));

    // Compute memory requirements
    OptixDenoiserSizes denoiserSizes;
    OPTIX_CHECK(optixDenoiserComputeMemoryResources(denoiser, width, height, &denoiserSizes));

    denoiserStateSize = denoiserSizes.stateSizeInBytes;
    denoiserScratchSize = denoiserSizes.withOverlapScratchSizeInBytes;

    CUDA_CHECK(cudaMalloc((void**)&d_denoiserState, denoiserStateSize));
    CUDA_CHECK(cudaMalloc((void**)&d_denoiserScratch, denoiserScratchSize));
    CUDA_CHECK(cudaMalloc((void**)&d_denoiserIntensity, sizeof(float)));

    int pixelcount = width * height;
    CUDA_CHECK(cudaMalloc((void**)&d_hdrInput, pixelcount * 3 * sizeof(float)));

    OPTIX_CHECK(optixDenoiserSetup(denoiser, 0,
        width, height,
        d_denoiserState, denoiserStateSize,
        d_denoiserScratch, denoiserScratchSize));

    denoiserInitialized = true;
    printf("OptiX Denoiser initialized (HDR + Albedo + Normal guides, %dx%d)\n", width, height);
}

void OptixRenderer::denoise(glm::vec3* dev_colorAccum, glm::vec3* dev_albedo, glm::vec3* dev_normal,
                             glm::vec3* dev_denoisedOut, int width, int height, int iter) {
    if (!denoiserInitialized) return;

    int pixelcount = width * height;
    dim3 blocks = (pixelcount + 255) / 256;

    // Normalize accumulated image -> d_hdrInput
    kernNormalizeImage<<<blocks, 256>>>(pixelcount, dev_colorAccum, (float*)d_hdrInput, iter);

    // Convert albedo and normal to float3 scratch buffers
    // We reuse scratch space: albedo and normal are already in the right format (glm::vec3 = 3 floats)
    // But OptixImage2D expects contiguous float3 data. glm::vec3 is already 12 bytes, same as float3.
    // So we can use dev_albedo and dev_normal directly, cast to float*.

    // Setup OptixImage2D for input color
    OptixImage2D inputColor = {};
    inputColor.data = d_hdrInput;
    inputColor.width = width;
    inputColor.height = height;
    inputColor.rowStrideInBytes = width * 3 * sizeof(float);
    inputColor.pixelStrideInBytes = 3 * sizeof(float);
    inputColor.format = OPTIX_PIXEL_FORMAT_FLOAT3;

    // Setup guide albedo
    OptixImage2D guideAlbedo = {};
    guideAlbedo.data = (CUdeviceptr)dev_albedo;
    guideAlbedo.width = width;
    guideAlbedo.height = height;
    guideAlbedo.rowStrideInBytes = width * 3 * sizeof(float);
    guideAlbedo.pixelStrideInBytes = 3 * sizeof(float);
    guideAlbedo.format = OPTIX_PIXEL_FORMAT_FLOAT3;

    // Setup guide normal
    OptixImage2D guideNormal = {};
    guideNormal.data = (CUdeviceptr)dev_normal;
    guideNormal.width = width;
    guideNormal.height = height;
    guideNormal.rowStrideInBytes = width * 3 * sizeof(float);
    guideNormal.pixelStrideInBytes = 3 * sizeof(float);
    guideNormal.format = OPTIX_PIXEL_FORMAT_FLOAT3;

    // Setup output
    OptixImage2D outputImage = {};
    outputImage.data = (CUdeviceptr)dev_denoisedOut;
    outputImage.width = width;
    outputImage.height = height;
    outputImage.rowStrideInBytes = width * 3 * sizeof(float);
    outputImage.pixelStrideInBytes = 3 * sizeof(float);
    outputImage.format = OPTIX_PIXEL_FORMAT_FLOAT3;

    // Compute intensity for HDR
    {
        OptixResult res = optixDenoiserComputeIntensity(denoiser, 0,
            &inputColor, d_denoiserIntensity,
            d_denoiserScratch, denoiserScratchSize);
        if (res != OPTIX_SUCCESS) {
            fprintf(stderr, "Denoiser computeIntensity failed: %s\n", optixGetErrorString(res));
            return;
        }
    }

    // Guide layer
    OptixDenoiserGuideLayer guideLayer = {};
    guideLayer.albedo = guideAlbedo;
    guideLayer.normal = guideNormal;

    // Denoiser layer (single layer)
    OptixDenoiserLayer layer = {};
    layer.input = inputColor;
    layer.output = outputImage;

    // Denoiser parameters
    OptixDenoiserParams params = {};
    params.hdrIntensity = d_denoiserIntensity;
    params.blendFactor = 0.0f;  // 0 = fully denoised, 1 = fully noisy

    {
        OptixResult res = optixDenoiserInvoke(denoiser, 0,
            &params,
            d_denoiserState, denoiserStateSize,
            &guideLayer, &layer, 1,
            0, 0,
            d_denoiserScratch, denoiserScratchSize);
        if (res != OPTIX_SUCCESS) {
            fprintf(stderr, "Denoiser invoke failed: %s\n", optixGetErrorString(res));
            return;
        }
    }
}

void OptixRenderer::cleanupDenoiser() {
    if (!denoiserInitialized) return;

    if (d_denoiserState) { cudaFree((void*)d_denoiserState); d_denoiserState = 0; }
    if (d_denoiserScratch) { cudaFree((void*)d_denoiserScratch); d_denoiserScratch = 0; }
    if (d_denoiserIntensity) { cudaFree((void*)d_denoiserIntensity); d_denoiserIntensity = 0; }
    if (d_hdrInput) { cudaFree((void*)d_hdrInput); d_hdrInput = 0; }

    if (denoiser) { optixDenoiserDestroy(denoiser); denoiser = nullptr; }

    denoiserInitialized = false;
    printf("OptiX Denoiser cleaned up.\n");
}

#endif // ENABLE_DENOISER

