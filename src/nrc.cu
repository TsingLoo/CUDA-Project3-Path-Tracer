// ============================================================================
// IMPORTANT: tiny-cuda-nn must be included BEFORE any project header that
// defines a  #define PI ...  macro.  tcnn's common_device.h uses PI() as an
// unqualified function call inside template bodies. Those bodies are
// instantiated at call-site, so if the PI macro is in scope at that point
// the call expands to e.g.  3.14f ()  which is a syntax error.
//
// Strategy:
//   1. Include tcnn headers first (PI macro not yet defined).
//   2. Include project headers (utilities.h defines  #define PI 3.14...f ).
//   3. Immediately #undef PI so future tcnn template instantiations are safe.
//   4. Use the local constant NRC_PI throughout this file instead of PI.
// ============================================================================

// Explicit CUDA headers to appease Visual Studio IntelliSense
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/config.h>
#include <iostream>

// Project headers (utilities.h is pulled in transitively by sceneStructs.h)
#include "nrc.h"
#include "sceneStructs.h"

// Kill the PI macro so tcnn template instantiations in this .cu don't break
#ifdef PI
#  undef PI
#endif

// Local replacement used in our kernels
static constexpr float NRC_PI = 3.14159265358979323846f;

#ifndef checkCUDAError
#define checkCUDAError(msg) checkCUDAErrorFn(msg, __FILE__, __LINE__)
static inline void checkCUDAErrorFn(const char* msg, const char* file, int line) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) return;
    std::cerr << "CUDA error: " << msg << " " << file << " " << line << " " << cudaGetErrorString(err) << std::endl;
    exit(1);
}
#endif

using namespace tcnn;

// Global state for NRC
static TrainableModel g_model;
static GPUMemory<NRCTrainingSample> g_train_samples;
static int g_current_train_sample_count = 0;

// Reusable buffers for tiny-cuda-nn
static GPUMemory<float> g_train_input_matrix;
static GPUMemory<float> g_train_target_matrix;
static GPUMemory<float> g_inference_input_matrix;
static GPUMemory<float> g_inference_output_matrix;

void nrcInit() {
    g_train_samples.resize(NRC_TRAIN_RAYS);
    
    // Memory allocation for the matrices
    g_train_input_matrix.resize(11 * NRC_TRAIN_RAYS);
    g_train_target_matrix.resize(3 * NRC_TRAIN_RAYS);

    // Initial capacity for inference, but we will resize dynamically if needed
    g_inference_input_matrix.resize(11 * 1024 * 1024);
    g_inference_output_matrix.resize(3 * 1024 * 1024);

    json config = {
        {"loss", {
            {"otype", "RelativeL2Luminance"}
        }},
        {"optimizer", {
            {"otype", "Adam"},
            {"learning_rate", 1e-2},
            {"beta1", 0.9},
            {"beta2", 0.99},
            {"epsilon", 1e-8},
            {"l2_reg", 1e-6}
        }},
        {"encoding", {
            {"otype", "Composite"},
            {"nested", {
                {{"otype", "TriangleWave"}, {"n_dims_to_encode", 3}, {"n_frequencies", 12}},
                {{"otype", "OneBlob"}, {"n_dims_to_encode", 3}, {"n_bins", 4}},
                {{"otype", "OneBlob"}, {"n_dims_to_encode", 2}, {"n_bins", 4}},
                {{"otype", "Identity"}, {"n_dims_to_encode", 3}}
            }}
        }},
        {"network", {
            {"otype", "FullyFusedMLP"},
            {"activation", "ReLU"},
            {"output_activation", "None"},
            {"n_neurons", 64},
            {"n_hidden_layers", 4}
        }}
    };

    try {
        g_model = create_from_config(11, 3, config);
        std::cout << "Successfully initialized tiny-cuda-nn NRC model." << std::endl;
        g_model.network->set_jit_fusion(true);
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize NRC: " << e.what() << std::endl;
        exit(1);
    }
}

void nrcFree() {
    g_model.loss.reset();
    g_model.optimizer.reset();
    g_model.network.reset();
    g_model.trainer.reset();

    g_train_samples.resize(0);
    g_train_input_matrix.resize(0);
    g_train_target_matrix.resize(0);
    g_inference_input_matrix.resize(0);
    g_inference_output_matrix.resize(0);
}

void nrcBeginFrame() {
    g_current_train_sample_count = 0;
}

NRCTrainingSample* nrcGetTrainingBuffer() {
    return g_train_samples.data();
}

void nrcSetTrainingSampleCount(int count) {
    g_current_train_sample_count = std::min(count, NRC_TRAIN_RAYS);
}

// -------------------------------------------------------------------------
// Helper Kernels
// -------------------------------------------------------------------------

__device__ inline float normalizePos(float x) {
    // Map bounds [-100, 100] approximately to [0, 1]
    return glm::clamp((x + 100.0f) / 200.0f, 0.0f, 1.0f);
}

__device__ inline float normalizeDir(float x) {
    // Map bounds [-1, 1] to [0, 1]
    return glm::clamp((x + 1.0f) / 2.0f, 0.0f, 1.0f);
}

__global__ void kernPackInferenceInput(
    int num_elements,
    const NRCQueryWorkItem* dev_queries,
    float* packed_matrix) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements) return;

    NRCQueryWorkItem q = dev_queries[i];

    int dims = 11;
    // position
    packed_matrix[i * dims + 0] = normalizePos(q.position.x);
    packed_matrix[i * dims + 1] = normalizePos(q.position.y);
    packed_matrix[i * dims + 2] = normalizePos(q.position.z);
    // normal
    packed_matrix[i * dims + 3] = normalizeDir(q.normal.x);
    packed_matrix[i * dims + 4] = normalizeDir(q.normal.y);
    packed_matrix[i * dims + 5] = normalizeDir(q.normal.z);
    // viewdir: theta [0, 2pi] -> [0, 1],  phi [0, pi] -> [0, 1]
    packed_matrix[i * dims + 6] = q.viewDir.x / (NRC_PI * 2.0f);
    packed_matrix[i * dims + 7] = q.viewDir.y / NRC_PI;
    // albedo
    packed_matrix[i * dims + 8] = glm::clamp(q.albedo.x, 0.0f, 1.0f);
    packed_matrix[i * dims + 9] = glm::clamp(q.albedo.y, 0.0f, 1.0f);
    packed_matrix[i * dims + 10] = glm::clamp(q.albedo.z, 0.0f, 1.0f);
}

__global__ void kernUnpackTrainingSamples(
    int num_elements,
    const NRCTrainingSample* samples,
    float* packed_input,
    float* packed_target)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements) return;

    NRCTrainingSample sample = samples[i];
    int dims = 11;

    packed_input[i * dims + 0] = normalizePos(sample.position.x);
    packed_input[i * dims + 1] = normalizePos(sample.position.y);
    packed_input[i * dims + 2] = normalizePos(sample.position.z);

    packed_input[i * dims + 3] = normalizeDir(sample.normal.x);
    packed_input[i * dims + 4] = normalizeDir(sample.normal.y);
    packed_input[i * dims + 5] = normalizeDir(sample.normal.z);

    packed_input[i * dims + 6] = sample.theta / (NRC_PI * 2.0f);
    packed_input[i * dims + 7] = sample.phi / NRC_PI;

    packed_input[i * dims + 8] = glm::clamp(sample.albedo.x, 0.0f, 1.0f);
    packed_input[i * dims + 9] = glm::clamp(sample.albedo.y, 0.0f, 1.0f);
    packed_input[i * dims + 10] = glm::clamp(sample.albedo.z, 0.0f, 1.0f);

    packed_target[i * 3 + 0] = sample.target_radiance.x;
    packed_target[i * 3 + 1] = sample.target_radiance.y;
    packed_target[i * 3 + 2] = sample.target_radiance.z;
}


// -------------------------------------------------------------------------
// Implementations
// -------------------------------------------------------------------------

void nrcInference(
    const NRCQueryWorkItem* dev_queries,
    float* dev_out_radiance,
    int batch_size,
    cudaStream_t stream)
{
    if (batch_size == 0) return;

    // tcnn::BATCH_SIZE_GRANULARITY == 256 (see common.h line 246); batch MUST be a multiple of this.
    int padded_batch_size = ((batch_size + 255) / 256) * 256;

    if (g_inference_input_matrix.size() < padded_batch_size * 11) {
        g_inference_input_matrix.resize(padded_batch_size * 11);
        g_inference_output_matrix.resize(padded_batch_size * 3);
    }

    int blockSize = 256;
    int blocks = (batch_size + blockSize - 1) / blockSize;

    kernPackInferenceInput<<<blocks, blockSize, 0, stream>>>(
        batch_size, dev_queries, g_inference_input_matrix.data()
    );
    checkCUDAError("kernPackInferenceInput");

    // tcnn convention: matrix is [rows=features, cols=batch] in column-major (AoS) storage.
    // Must pad batch size to 128 to prevent tcnn internal exceptions!
    GPUMatrix<float> input(g_inference_input_matrix.data(), 11, padded_batch_size);
    GPUMatrix<float> output(g_inference_output_matrix.data(), 3, padded_batch_size);

    try {
        g_model.network->inference(stream, input, output);
    } catch (const std::exception& e) {
        std::cerr << "NRC inference exception: " << e.what() << std::endl;
        exit(1);
    }

    // Copy to output buffer
    cudaMemcpyAsync(dev_out_radiance, g_inference_output_matrix.data(), batch_size * 3 * sizeof(float), cudaMemcpyDeviceToDevice, stream);
}

void nrcTrain(cudaStream_t stream) {
    if (g_current_train_sample_count == 0) return;

    int batch_size = g_current_train_sample_count;

    // tcnn::BATCH_SIZE_GRANULARITY == 256 (see common.h line 246); batch MUST be a multiple of this.
    int padded_batch_size = ((batch_size + 255) / 256) * 256;

    int blockSize = 256;
    int blocks = (batch_size + blockSize - 1) / blockSize;

    kernUnpackTrainingSamples<<<blocks, blockSize, 0, stream>>>(
        batch_size, g_train_samples.data(), g_train_input_matrix.data(), g_train_target_matrix.data()
    );
    checkCUDAError("kernUnpackTrainingSamples");

    // Explicitly zero out the padding elements so the model doesn't train on uninitialized memory
    int pad_diff = padded_batch_size - batch_size;
    if (pad_diff > 0) {
        cudaMemsetAsync(g_train_input_matrix.data() + batch_size * 11, 0, pad_diff * 11 * sizeof(float), stream);
        cudaMemsetAsync(g_train_target_matrix.data() + batch_size * 3, 0, pad_diff * 3 * sizeof(float), stream);
    }

    // GPUMatrix<float> = column-major (AoS) statically typed – required by trainer->training_step()
    GPUMatrix<float> input(g_train_input_matrix.data(), 11, padded_batch_size);
    GPUMatrix<float> target(g_train_target_matrix.data(), 3, padded_batch_size);

    try {
        for (int step = 0; step < NRC_TRAIN_STEPS; step++) {
            g_model.trainer->training_step(stream, input, target);
        }
    } catch (const std::exception& e) {
        std::cerr << "NRC training_step exception: " << e.what() << std::endl;
        exit(1);
    }
}


