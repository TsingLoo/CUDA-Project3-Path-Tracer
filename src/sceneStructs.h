#pragma once

#include "utilities.h"

#include <string>
#include <vector>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE,
    CUBE,
    TRIANGLE
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Geom
{
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;

    int v0, v1, v2;
    glm::vec2 uv0, uv1, uv2;  // per-vertex UVs for TRIANGLE type
};

enum MaterialType {
    LAMBERTIAN,
    SPECULAR,
    GLASS,
};

struct Material
{
    MaterialType type;

    glm::vec3 color;
    struct
    {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float abbe;
    float emittance;
    int textureId;       // index into device texture array, -1 = no texture
    float roughness;     // stored from glTF PBR, for future use
};

struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;


    //in mm
    float focalLength;
    float fAperture;
    float focusDistance;
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment
{
    Ray ray;
    int pixelIndex;
    int remainingBounces;

#if ENABLE_SPECTRAL_RENDERING
#define SPECTRAL_N 4  // number of hero wavelengths per ray
    float wavelengths[SPECTRAL_N];
    float throughputs[SPECTRAL_N];
    float pdfs[SPECTRAL_N];  // CIE importance sampling PDF per wavelength
#else
    glm::vec3 color;
#endif 

    float lastBrdfPdf;  // BRDF pdf from last bounce (-1 = specular/camera, for MIS)
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
  glm::vec2 uv;  // interpolated texture coordinate
};

struct MissWorkItem {
    int path_idx;
};

struct HitLightWorkItem {
    int path_idx;
    int material_id = -1;
    int geom_idx = -1;
    glm::vec3 hit_point;
    glm::vec3 hit_normal;
};

struct LambertianHitWorkItem {
    int path_idx;
    int material_id = -1;
    glm::vec3 intersect_point;
    glm::vec3 surface_normal;
    glm::vec2 uv;  // interpolated texture coordinate
};

struct SpecularHitWorkItem {
    int path_idx;
    int material_id = -1;
    glm::vec3 intersect_point;
    glm::vec3 surface_normal;
    glm::vec3 incident_ray_dir;
    glm::vec2 uv;
};

struct GlassHitWorkItem {
    int path_idx;
    int material_id = -1;
    float IOR;
    glm::vec3 intersect_point;
    glm::vec3 surface_normal;
    glm::vec3 incident_ray_dir;
    glm::vec2 uv;
};