#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>

// Configuration for Neural Radiance Cache
#ifndef ENABLE_NRC
#define ENABLE_NRC 1
#endif

// Parameters definition
#define NRC_TRAIN_RAYS (1 << 16) // Amount of training rays generated per frame
#define NRC_TRAIN_STEPS 4        // Amount of adam optimization steps per frame
#define NRC_QUERY_DEPTH 2        // 0-indexed, meaning it will happen at the 3rd bounce

struct NRCTrainingSample {
    glm::vec3 position;
    glm::vec3 normal;
    float theta;
    float phi;
    glm::vec3 albedo;
    glm::vec3 target_radiance;
};

// Initializes the neural radiance cache components
void nrcInit();

// Cleans up the NRC components
void nrcFree();

// Prepares the NRC buffers for a new rendering frame
void nrcBeginFrame();

// Query the neural radiance cache for a batch of path vertices
// Returns the estimated RGB radiance
void nrcInference(
    const struct NRCQueryWorkItem* dev_queries,
    float* dev_out_radiance,
    int batch_size,
    cudaStream_t stream = 0
);

// Get the device pointer to the training ring buffer
// The capacity is NRC_TRAIN_RAYS
NRCTrainingSample* nrcGetTrainingBuffer();

// Set the number of accumulated training samples for the current frame
void nrcSetTrainingSampleCount(int count);

// Execute the training step using collected samples
void nrcTrain(cudaStream_t stream = 0);

