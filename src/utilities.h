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

#define ENABLE_FRESNEL_IN_TRANSMISSION 0

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

//__host__ __device__ glm::vec3 step(const glm::vec3& edge, const glm::vec3& x) {
//    return glm::vec3(
//        x.x < edge.x ? 0.0f : 1.0f,
//        x.y < edge.y ? 0.0f : 1.0f,
//        x.z < edge.z ? 0.0f : 1.0f
//    );
//}
//
//__host__ __device__ glm::vec3 sign(const glm::vec3& v) {
//    return glm::vec3(
//        (v.x > 0.0f) ? 1.0f : ((v.x < 0.0f) ? -1.0f : 0.0f),
//        (v.y > 0.0f) ? 1.0f : ((v.y < 0.0f) ? -1.0f : 0.0f),
//        (v.z > 0.0f) ? 1.0f : ((v.z < 0.0f) ? -1.0f : 0.0f)
//    );
//}


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

inline __device__ float SafeSqrt(float x) { return glm::sqrt(glm::max(0.f, x)); }


/// <summary>
/// 
/// </summary>
/// <param name="wi">incident direction wi</param>
/// <param name="n">surface normal at the same side as li</param>
/// <param name="eta">eta ratio</param>
/// <param name="wt">ray direction after the refraction</param>
/// <returns></returns>
inline __device__ bool Refract(glm::vec3 wi, glm::vec3 n, float eta, glm::vec3& wt) {
   
    float cosThetaI = dot(n, wi);
    //Potentially flip interface orientation for Snells law
    if (cosThetaI < 0) {
        eta = 1 / eta;
        cosThetaI = -cosThetaI;
        n = -n;
    }

    //Compute  cosThetaT using Snell law
    float sin2ThetaI = glm::max(0.0f, 1.0f - cosThetaI * cosThetaI);

    float sin2ThetaT = eta * eta * sin2ThetaI;

    // Handle total internal reflection for transmission
    if (sin2ThetaT >= 1) return false;
    float cosThetaT = SafeSqrt(1 - sin2ThetaT);
    wt = eta * -wi + (eta * cosThetaI - cosThetaT) * n;

    //Provide relative IOR along ray to caller
    //if (etap)
    //    *etap = eta;

    return true;
}

inline __device__ float AbsDot(const glm::vec3 a, const glm::vec3 b) 
{
    return glm::abs(glm::dot(a, b));
}


//reference: https://www.pbr-book.org/4ed/Reflection_Models/Specular_Reflection_and_Transmission
inline __device__ float FrDielectric(float cosTheta_i, float eta) {
    cosTheta_i = glm::clamp(cosTheta_i, -1.0f, 1.0f);

    //Potentially swap indices of refraction
    if (cosTheta_i < 0) {
        eta = 1 / eta;
        cosTheta_i = -cosTheta_i;
    }


    float sin2Theta_i = 1.0f - cosTheta_i * cosTheta_i;
    float sin2Theta_t = sin2Theta_i / (eta * eta);
    if (sin2Theta_t >= 1.0f)
        return 1.0f;
    float cosTheta_t = SafeSqrt(1.0f - sin2Theta_t);

    float r_parl = (eta * cosTheta_i - cosTheta_t) /
        (eta * cosTheta_i + cosTheta_t);
    float r_perp = (cosTheta_i - eta * cosTheta_t) /
        (cosTheta_i + eta * cosTheta_t);
    return (r_parl * r_parl + r_perp * r_perp) / 2;
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
