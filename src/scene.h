#pragma once

#include "tiny_gltf.h"
#include "sceneStructs.h"
#include <vector>
#include <string>

// Host-side texture data loaded from glTF
struct TextureData {
    int width = 0;
    int height = 0;
    int channels = 0;  // typically 4 (RGBA)
    std::vector<unsigned char> pixels;
};

// Host-side HDR environment map
struct EnvironmentMap {
    int width = 0;
    int height = 0;
    std::vector<float> pixels;  // RGB float data (width * height * 3)
    bool loaded = false;
};

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    bool loadGLTF(const std::string& filename, const glm::mat4& instance_transform,
                  int material_type_override = -1, float ior_override = 0.0f, float abbe_override = 0.0f);

    void setupDefaultCamera();
    void setupCameraFromGLTF(const tinygltf::Model& model);

    // Helper: process materials from a glTF model
    int processGLTFMaterials(const tinygltf::Model& model,
                             int material_type_override = -1, float ior_override = 0.0f, float abbe_override = 0.0f);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;

    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> texcoords;

    std::vector<TextureData> textures;  // host-side texture data for CUDA texture objects

    EnvironmentMap envMap;  // HDRI environment map

    // Load/reload environment map from .hdr or .exr file
    bool loadEnvironmentMap(const std::string& path);
};

