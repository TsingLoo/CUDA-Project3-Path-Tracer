#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <curand_kernel.h>

#ifndef ENABLE_RESTIR_DI
#define ENABLE_RESTIR_DI 0
#endif

#define RESTIR_M_INITIAL 8
#define RESTIR_SPATIAL_TAPS 3
#define RESTIR_SPATIAL_RADIUS 20

// Basic candidate for DI: pointing to a specific light geometry.
struct RestirLightCandidate {
    int lightGeomIdx;
    glm::vec3 position;
    glm::vec3 normal;
    float area;
};

struct RestirReservoir {
    RestirLightCandidate y;  // The selected candidate
    float w_sum;             // Sum of weights
    float W;                 // Unbiased contribution weight
    int M;                   // Number of candidates seen so far
    
    __device__ void init() {
        w_sum = 0.0f;
        W = 0.0f;
        M = 0;
        y.lightGeomIdx = -1;
    }

    // Returns true if the candidate was selected
    __device__ bool update(const RestirLightCandidate& candidate, float weight, curandState* rng) {
        w_sum += weight;
        M++;
        if (curand_uniform(rng) < (weight / w_sum)) {
            y = candidate;
            return true;
        }
        return false;
    }

    // Merge a reservoir blindly
    __device__ void merge(const RestirReservoir& r, float target_pdf_y, curandState* rng) {
        if (r.M > 0) {
            float weight = target_pdf_y * r.W * r.M;
            if (update(r.y, weight, rng)) {
                // candidate replaced implicitly
            }
            // Add to M conceptually (or specific bounded logic to avoid blowing up)
            M += r.M - 1; // since update() already added 1
        }
    }
    
    // Finalize the W value given the current chosen sample and its Target PDF (p_hat)
    __device__ void finalize(float target_pdf) {
        if (target_pdf > 0.0f && M > 0) {
            W = w_sum / (M * target_pdf);
        } else {
            W = 0.0f;
        }
    }
};
