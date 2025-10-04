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

inline __device__ glm::vec2 squareToDiskConcentric(glm::vec2 xi)
{
    float a = 2.0 * xi.x - 1.0;
    float b = 2.0 * xi.y - 1.0;

    if (a == 0.0 && b == 0.0)
    {
        return glm::vec2(0.0);
    }

    float r, phi;

    if (abs(a) > abs(b))
    {
        r = a;
        phi = (PI / 4.0) * (b / a);
    }
    else
    {
        r = b;
        phi = (PI / 2.0) - (a / b) * (PI / 4.0);
    }

    float x = r * cos(phi);
    float y = r * sin(phi);
    return glm::vec2(x, y);
}

inline __device__ glm::vec3 squareToHemisphereCosine(glm::vec2 xi)
{
    glm::vec2 disk = squareToDiskConcentric(xi);
    float r2 = glm::dot(disk, disk);
    float z = glm::sqrt(glm::max(0.0, 1.0 - r2));
    return glm::vec3(disk.x, disk.y, z);
}

/// <summary>
/// build a tangent space given a normal vector
/// </summary>
/// <param name="nor">the z-axis of the space</param>
/// <param name="v2">the tangent of the space, x-axis</param>
/// <param name="v3"></param>
/// <returns></returns>
inline __device__ void buildTangentSpace(const glm::vec3 nor, glm::vec3& v2, glm::vec3& v3)
{
    if (abs(nor.x) > abs(nor.y))
        //cross(v1, vec3(0, 1, 0)) = vec3(-v1.z, 0, v1.x)
        v2 = glm::vec3(-nor.z, 0, nor.x) / sqrt(nor.x * nor.x + nor.z * nor.z);
    else
        v2 = glm::vec3(0, nor.z, -nor.y) / sqrt(nor.y * nor.y + nor.z * nor.z);
    v3 = glm::cross(nor, v2);
}

inline __device__ glm::mat3 TangentSpaceToWorld(const glm::vec3 nor) {
    glm::vec3 tan, bit;
    buildTangentSpace(nor, tan, bit);
    return glm::mat3(tan, bit, nor);
}

inline __device__ glm::mat3 WorldToTangentSpace(const glm::vec3 nor) {
    return glm::transpose(TangentSpaceToWorld(nor));
}

inline __device__ glm::vec3 f_diffuse(glm::vec3 albedo) {
    return albedo / PI;
}

inline __device__ glm::vec3 Sample_f_diffuse(glm::vec3 albedo, glm::vec2 xi, glm::vec3 nor,
    glm::vec3& wiW, float& pdf) {
    glm::vec3 wiLocal = squareToHemisphereCosine(xi);
    pdf = wiLocal.z / PI;
    glm::mat3 mat = TangentSpaceToWorld(nor);
    wiW = mat * wiLocal;
    return f_diffuse(albedo);
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