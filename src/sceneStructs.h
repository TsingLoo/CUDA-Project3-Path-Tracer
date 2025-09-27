#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <string>
#include <vector>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE,
    CUBE,
    TRIANGLE,
    SQUARE_PLANE
};

struct Ray
{
    /// <summary>
    /// In world space
    /// </summary>
    glm::vec3 origin;

    /// <summary>
    /// In world space
    /// </summary>
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
};

struct Triangle : public Geom
{
    glm::vec3 v0;
    glm::vec3 v1;
    glm::vec3 v2;
    glm::vec3 normal;
};

enum MaterialType
{
    DIFFUSE_REFL,
    SPEC_REFL,
    SPEC_TRANS,
    SPEC_GLASS,
    MICROFACET_REFL,
    PLASTIC,
    DIFFUSE_TRANS,
    EMITTIVE
};

struct Material
{
    glm::vec3 albedo;
    float roughness;
    float eta;
    float emittance;

    MaterialType type;
};



struct Camera
{
    glm::ivec2 resolution;
    /// <summary>
    /// World Space
    /// </summary>
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
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
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
	glm::vec3 throughput;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;
  /// <summary>
  /// The surfaceNormal is in world space
  /// </summary>
  glm::vec3 surfaceNormal;
  int materialId;
  bool outside;
};
