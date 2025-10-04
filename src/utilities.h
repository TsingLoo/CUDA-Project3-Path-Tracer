#pragma once

#include "glm/glm.hpp"

#include "cuda_runtime.h"

#include <algorithm>
#include <istream>
#include <iterator>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#define PI                3.1415926535897932384626422832795028841971f
#define TWO_PI            6.2831853071795864769252867665590057683943f
#define SQRT_OF_ONE_THIRD 0.5773502691896257645091487805019574556476f
#define EPSILON           0.00001f

inline __host__ __device__ bool solveQuadratic(float A, float B, float C, float& t0, float& t1)
{
    // A more robust and stable quadratic solver
    float invA = 1.0f / A;
    float b = B * invA;
    float c = C * invA;

    float neg_half_b = -b * 0.5f;
    float discriminant_sq = neg_half_b * neg_half_b - c;

    if (discriminant_sq < 0.0f) {
        return false; // No real roots, the ray misses
    }

    float sqrt_discriminant = sqrtf(discriminant_sq);
    t0 = neg_half_b - sqrt_discriminant;
    t1 = neg_half_b + sqrt_discriminant;

    return true;
}


class GuiDataContainer
{
public:
    GuiDataContainer() : TracedDepth(0) {}
    int TracedDepth;
};

namespace utilityCore
{
    extern float clamp(float f, float min, float max);
    extern bool replaceString(std::string& str, const std::string& from, const std::string& to);
    extern glm::vec3 clampRGB(glm::vec3 color);
    extern bool epsilonCheck(float a, float b);
    extern std::vector<std::string> tokenizeString(std::string str);
    extern glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);
    extern std::string convertIntToString(int number);
    extern std::istream& safeGetline(std::istream& is, std::string& t); //Thanks to http://stackoverflow.com/a/6089413
}