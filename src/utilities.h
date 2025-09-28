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

#define VEC3_TANGENT_NORMAL glm::vec3(0.0f, 0.0f, 1.0f)

#define DEBUG_WHITE_COLOR glm::vec3(1.0f)
#define DEBUG_YELLOW_COLOR glm::vec3(1.0f, 1.0f, 0.0f)
#define DEBUG_PINK_COLOR glm::vec3(1.0f, 0.0f, 1.0f)
#define DEBUG_BLUE_COLOR glm::vec3(0.0f, 1.0f, 1.0f)
#define DEBUG_EMPTY_COLOR glm::vec3(0.0f, 0.0f, 0.0f)
#define DEBUG_intersection_COLOR glm::vec3(intersection.t,intersection.t,intersection.t)


/// <summary>
/// w has been transformed to a tangent coordinate system
/// where the surface normal is the Z-axis(0,0,1)
/// </summary>
/// <param name="w"></param>
/// <returns></returns>
inline __device__ float CosTheta(const glm::vec3 w) { return w.z; }
inline __device__ float Cos2Theta(const glm::vec3 w) { return w.z * w.z; }
inline __device__ float AbsCosTheta(const glm::vec3 w) { return glm::abs(w.z); }

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

/// <summary>
/// if v and normal is not at the same hemisphere, revert the normal, v is the outgoing direction 
/// the surface normal that lies in the same hemisphere as an outgoing ray is frequently needed
/// </summary>
/// <param name="normal"></param>
/// <param name="v">is the outgoing direction</param>
/// <returns></returns>
inline __device__ glm::vec3 FaceForward(const glm::vec3 normal, const glm::vec3 v) {
    return glm::dot(normal, v) < 0.f ? -1.f * normal : normal;
}

/// <summary>
/// 
/// </summary>
/// <param name="wi">ray facing the surface</param>
/// <param name="n">surface normal at the same side as the incoming lie</param>
/// <param name="eta">eta ratio</param>
/// <param name="wt">ray direction after the refraction</param>
/// <returns></returns>
inline __device__ bool Refract(glm::vec3 wi, glm::vec3 n, float eta, glm::vec3& wt) {
    // Compute cos theta using Snell's law
    float cosThetaI = dot(n, wi);
    float sin2ThetaI = glm::max(float(0), float(1 - cosThetaI * cosThetaI));
    float sin2ThetaT = eta * eta * sin2ThetaI;

    // Handle total internal reflection for transmission
    if (sin2ThetaT >= 1) return false;
    float cosThetaT = sqrt(1 - sin2ThetaT);
    wt = eta * -wi + (eta * cosThetaI - cosThetaT) * n;
    return true;
}

inline __device__ float AbsDot(const glm::vec3 a, const glm::vec3 b) 
{
    return glm::abs(glm::dot(a, b));
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
