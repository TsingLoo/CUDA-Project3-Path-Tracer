#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>

#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <algorithm>

using namespace std;
using json = nlohmann::json;

// ============================================================================
// Constructor
// ============================================================================

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else if (ext == ".gltf" || ext == ".glb")
    {
        loadGLTF(filename, glm::mat4(1.0f));
        // Try to extract a camera from the glTF, otherwise use defaults
        // Camera setup is done inside loadGLTF for standalone mode
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

// ============================================================================
// Default Camera Setup
// ============================================================================

void Scene::setupDefaultCamera() {
    Camera& camera = this->state.camera;

    this->state.iterations = 1024;
    this->state.traceDepth = 8;
    this->state.imageName = "default_render.png";

    camera.resolution = glm::ivec2(800, 800);
    float fovy = 45.0f;

    camera.position = glm::vec3(0, 0, 10);
    camera.lookAt = glm::vec3(0, 0, 0);
    camera.up = glm::vec3(0, 1, 0);

    camera.focalLength = 50.0f;
    camera.fAperture = 22.0f;
    camera.focusDistance = glm::length(camera.lookAt - camera.position);

    camera.view = glm::normalize(camera.lookAt - camera.position);
    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.up = glm::normalize(glm::cross(camera.right, camera.view));

    float yscaled = tan(fovy * (PI / 180.0f));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180.0f) / PI;
    camera.fov = glm::vec2(fovx, fovy);
    camera.pixelLength = glm::vec2(2.0f * xscaled / (float)camera.resolution.x,
        2.0f * yscaled / (float)camera.resolution.y);

    int pixel_count = camera.resolution.x * camera.resolution.y;
    this->state.image.resize(pixel_count);
    std::fill(this->state.image.begin(), this->state.image.end(), glm::vec3(0.0f));
}

// ============================================================================
// glTF Camera Extraction
// ============================================================================

void Scene::setupCameraFromGLTF(const tinygltf::Model& model) {
    // Look for a camera node in the scene
    int cameraNodeIdx = -1;
    int cameraIdx = -1;

    std::function<void(int, const glm::mat4&)> findCamera =
        [&](int node_idx, const glm::mat4& parent_transform) {
        const tinygltf::Node& node = model.nodes[node_idx];
        if (node.camera >= 0 && cameraIdx < 0) {
            cameraIdx = node.camera;
            cameraNodeIdx = node_idx;
        }
        for (int child_idx : node.children) {
            findCamera(child_idx, parent_transform);
        }
    };

    const tinygltf::Scene& gltf_scene = model.scenes[model.defaultScene > -1 ? model.defaultScene : 0];
    for (int node_idx : gltf_scene.nodes) {
        findCamera(node_idx, glm::mat4(1.0f));
    }

    if (cameraIdx < 0 || cameraIdx >= (int)model.cameras.size()) {
        // No camera found, use defaults
        setupDefaultCamera();
        return;
    }

    // Compute the node's world transform
    // For simplicity, compute from the camera node directly
    const tinygltf::Node& camNode = model.nodes[cameraNodeIdx];
    glm::mat4 camTransform(1.0f);
    if (camNode.matrix.size() == 16) {
        float m[16]; for (int i = 0; i < 16; i++) m[i] = (float)camNode.matrix[i];
        camTransform = glm::make_mat4(m);
    } else {
        if (camNode.translation.size() == 3) {
            camTransform = glm::translate(camTransform, glm::vec3(camNode.translation[0], camNode.translation[1], camNode.translation[2]));
        }
        if (camNode.rotation.size() == 4) {
            camTransform *= glm::mat4_cast(glm::quat((float)camNode.rotation[3], (float)camNode.rotation[0], (float)camNode.rotation[1], (float)camNode.rotation[2]));
        }
        if (camNode.scale.size() == 3) {
            camTransform = glm::scale(camTransform, glm::vec3(camNode.scale[0], camNode.scale[1], camNode.scale[2]));
        }
    }

    Camera& camera = this->state.camera;
    const tinygltf::Camera& gltfCam = model.cameras[cameraIdx];

    this->state.iterations = 1024;
    this->state.traceDepth = 8;
    this->state.imageName = "gltf_render.png";

    camera.resolution = glm::ivec2(800, 800);

    // Extract camera position and orientation from transform
    camera.position = glm::vec3(camTransform * glm::vec4(0, 0, 0, 1));
    glm::vec3 forward = glm::normalize(glm::vec3(camTransform * glm::vec4(0, 0, -1, 0)));
    camera.up = glm::normalize(glm::vec3(camTransform * glm::vec4(0, 1, 0, 0)));
    camera.lookAt = camera.position + forward;

    camera.focalLength = 50.0f;
    camera.fAperture = 22.0f;
    camera.focusDistance = 10.0f;

    float fovy = 45.0f;
    if (gltfCam.type == "perspective") {
        fovy = glm::degrees((float)gltfCam.perspective.yfov);
    }

    camera.view = glm::normalize(camera.lookAt - camera.position);
    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.up = glm::normalize(glm::cross(camera.right, camera.view));

    float yscaled = tan(fovy * (PI / 180.0f));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180.0f) / PI;
    camera.fov = glm::vec2(fovx, fovy);
    camera.pixelLength = glm::vec2(2.0f * xscaled / (float)camera.resolution.x,
        2.0f * yscaled / (float)camera.resolution.y);

    int pixel_count = camera.resolution.x * camera.resolution.y;
    this->state.image.resize(pixel_count);
    std::fill(this->state.image.begin(), this->state.image.end(), glm::vec3(0.0f));

    std::cout << "Using glTF camera at position " << glm::to_string(camera.position) << std::endl;
}

// ============================================================================
// glTF Material Processing
// ============================================================================

int Scene::processGLTFMaterials(const tinygltf::Model& model,
                                 int material_type_override, float ior_override, float abbe_override) {
    const int material_offset = this->materials.size();

    for (size_t matIdx = 0; matIdx < model.materials.size(); ++matIdx) {
        const auto& mat = model.materials[matIdx];
        Material newMaterial = {};
        newMaterial.textureId = -1;
        newMaterial.roughness = 1.0f;
        newMaterial.abbe = 20.0f;
        newMaterial.indexOfRefraction = 1.5f;

        const auto& pbr = mat.pbrMetallicRoughness;

        // --- Base color ---
        if (pbr.baseColorFactor.size() == 4) {
            newMaterial.color = glm::vec3(
                (float)pbr.baseColorFactor[0],
                (float)pbr.baseColorFactor[1],
                (float)pbr.baseColorFactor[2]);
        } else {
            newMaterial.color = glm::vec3(0.8f);
        }

        // --- Base color texture ---
        if (pbr.baseColorTexture.index >= 0) {
            int texIdx = pbr.baseColorTexture.index;
            if (texIdx < (int)model.textures.size()) {
                int imgIdx = model.textures[texIdx].source;
                if (imgIdx >= 0 && imgIdx < (int)model.images.size()) {
                    const auto& img = model.images[imgIdx];
                    // Check if we already loaded this image
                    // Simple: just store it (may duplicate if same image used by multiple materials)
                    TextureData tex;
                    tex.width = img.width;
                    tex.height = img.height;
                    tex.channels = img.component;
                    tex.pixels = img.image; // copy
                    newMaterial.textureId = (int)this->textures.size();
                    this->textures.push_back(std::move(tex));
                    std::cout << "  Loaded texture " << newMaterial.textureId
                              << " (" << tex.width << "x" << tex.height << ")" << std::endl;
                }
            }
        }

        // --- Material type override from JSON ---
        if (material_type_override >= 0) {
            newMaterial.type = (MaterialType)material_type_override;
            if (ior_override > 0.0f) newMaterial.indexOfRefraction = ior_override;
            if (abbe_override > 0.0f) newMaterial.abbe = abbe_override;
        }
        // --- PBR-based material type selection ---
        else if (mat.alphaMode == "BLEND") {
            // Transparent material -> Glass
            newMaterial.type = GLASS;
        }
        else if (pbr.metallicFactor >= 0.5) {
            newMaterial.type = SPECULAR;
        }
        else {
            newMaterial.type = LAMBERTIAN;
        }

        newMaterial.roughness = (float)pbr.roughnessFactor;

        // --- Emissive ---
        if (mat.emissiveFactor.size() == 3) {
            float emissiveLuminance = (float)(
                0.2126 * mat.emissiveFactor[0] +
                0.7152 * mat.emissiveFactor[1] +
                0.0722 * mat.emissiveFactor[2]);
            if (emissiveLuminance > 0.0f) {
                newMaterial.emittance = emissiveLuminance * 10.0f; // Scale up for visible emission
                newMaterial.color = glm::vec3(
                    (float)mat.emissiveFactor[0],
                    (float)mat.emissiveFactor[1],
                    (float)mat.emissiveFactor[2]);
            }
        }

        std::cout << "  Material[" << matIdx << "] \"" << mat.name << "\""
                  << " type=" << newMaterial.type
                  << " color=" << glm::to_string(newMaterial.color)
                  << " roughness=" << newMaterial.roughness
                  << " texId=" << newMaterial.textureId
                  << std::endl;

        this->materials.push_back(newMaterial);
    }

    return material_offset;
}

// ============================================================================
// Helper: Read accessor data with proper byte stride
// ============================================================================

static const unsigned char* getAccessorData(const tinygltf::Model& model, const tinygltf::Accessor& accessor, int& stride) {
    const tinygltf::BufferView& bv = model.bufferViews[accessor.bufferView];
    const tinygltf::Buffer& buf = model.buffers[bv.buffer];
    const unsigned char* data = &buf.data[bv.byteOffset + accessor.byteOffset];
    stride = accessor.ByteStride(bv);
    return data;
}

// ============================================================================
// Consolidated glTF Loader
// ============================================================================

bool Scene::loadGLTF(const std::string& filename, const glm::mat4& instance_transform,
                     int material_type_override, float ior_override, float abbe_override) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    std::cout << "Loading glTF from: " << filename << std::endl;

    // Try ASCII first, then binary
    auto ext = filename.substr(filename.find_last_of('.'));
    bool ret;
    if (ext == ".glb") {
        ret = loader.LoadBinaryFromFile(&model, &err, &warn, filename);
    } else {
        ret = loader.LoadASCIIFromFile(&model, &err, &warn, filename);
        if (!ret) {
            ret = loader.LoadBinaryFromFile(&model, &err, &warn, filename);
        }
    }

    if (!warn.empty()) {
        std::cout << "glTF Warning: " << warn << std::endl;
    }
    if (!err.empty()) {
        std::cerr << "glTF Error: " << err << std::endl;
    }
    if (!ret) {
        std::cerr << "Failed to load glTF: " << filename << std::endl;
        return false;
    }

    // Determine if this is standalone mode (no JSON parent calling us)
    // When called from the constructor with identity transform and no material override,
    // it's standalone mode.
    bool isStandalone = (material_type_override < 0 &&
                         instance_transform == glm::mat4(1.0f));

    // --- Process Materials ---
    int material_offset = processGLTFMaterials(model, material_type_override, ior_override, abbe_override);

    // Add a default material if the model has no materials
    if (model.materials.empty()) {
        Material defaultMat = {};
        defaultMat.type = LAMBERTIAN;
        defaultMat.color = glm::vec3(0.7f);
        defaultMat.textureId = -1;
        defaultMat.roughness = 1.0f;
        defaultMat.abbe = 20.0f;
        defaultMat.indexOfRefraction = 1.5f;
        this->materials.push_back(defaultMat);
        std::cout << "  Added default grey material" << std::endl;
    }

    // --- Process Nodes and Meshes ---
    std::function<void(int, const glm::mat4&)> processNode =
        [&](int node_idx, const glm::mat4& parent_transform) {

        const tinygltf::Node& node = model.nodes[node_idx];
        glm::mat4 node_transform = parent_transform;

        // Apply node's local transform
        if (node.matrix.size() == 16) {
            float m[16]; for (int i = 0; i < 16; i++) m[i] = (float)node.matrix[i];
            node_transform *= glm::make_mat4(m);
        }
        else {
            if (node.translation.size() == 3) {
                node_transform = glm::translate(node_transform, glm::vec3(node.translation[0], node.translation[1], node.translation[2]));
            }
            if (node.rotation.size() == 4) {
                node_transform *= glm::mat4_cast(glm::quat((float)node.rotation[3], (float)node.rotation[0], (float)node.rotation[1], (float)node.rotation[2]));
            }
            if (node.scale.size() == 3) {
                node_transform = glm::scale(node_transform, glm::vec3(node.scale[0], node.scale[1], node.scale[2]));
            }
        }

        if (node.mesh > -1) {
            const tinygltf::Mesh& mesh = model.meshes[node.mesh];
            for (const auto& primitive : mesh.primitives) {
                // Must have indices and positions
                if (primitive.indices < 0 || primitive.attributes.find("POSITION") == primitive.attributes.end()) {
                    continue;
                }

                // --- Get Indices ---
                const tinygltf::Accessor& indicesAccessor = model.accessors[primitive.indices];
                int idx_stride;
                const unsigned char* idx_data = getAccessorData(model, indicesAccessor, idx_stride);

                // --- Get Positions ---
                const tinygltf::Accessor& posAccessor = model.accessors.at(primitive.attributes.find("POSITION")->second);
                int pos_stride;
                const unsigned char* pos_data = getAccessorData(model, posAccessor, pos_stride);

                // --- Get Normals (optional) ---
                const unsigned char* norm_data = nullptr;
                int norm_stride = 0;
                bool hasNormals = primitive.attributes.find("NORMAL") != primitive.attributes.end();
                if (hasNormals) {
                    const tinygltf::Accessor& normAccessor = model.accessors.at(primitive.attributes.find("NORMAL")->second);
                    norm_data = getAccessorData(model, normAccessor, norm_stride);
                }

                // --- Get TexCoords (optional) ---
                const unsigned char* uv_data = nullptr;
                int uv_stride = 0;
                bool hasTexCoords = primitive.attributes.find("TEXCOORD_0") != primitive.attributes.end();
                if (hasTexCoords) {
                    const tinygltf::Accessor& uvAccessor = model.accessors.at(primitive.attributes.find("TEXCOORD_0")->second);
                    uv_data = getAccessorData(model, uvAccessor, uv_stride);
                }

                // Normal transform for non-uniform scaling
                glm::mat3 normalMatrix = glm::mat3(glm::transpose(glm::inverse(node_transform)));

                // Vertex offset for this primitive
                const int current_vertex_offset = this->positions.size();

                // --- Add Vertices ---
                for (size_t i = 0; i < posAccessor.count; ++i) {
                    const float* p = reinterpret_cast<const float*>(pos_data + i * pos_stride);
                    glm::vec3 pos(p[0], p[1], p[2]);
                    this->positions.push_back(glm::vec3(node_transform * glm::vec4(pos, 1.0f)));

                    if (hasNormals && norm_data) {
                        const float* n = reinterpret_cast<const float*>(norm_data + i * norm_stride);
                        glm::vec3 norm(n[0], n[1], n[2]);
                        this->normals.push_back(glm::normalize(normalMatrix * norm));
                    } else {
                        // Placeholder — will be overwritten per-face below
                        this->normals.push_back(glm::vec3(0, 1, 0));
                    }

                    if (hasTexCoords && uv_data) {
                        const float* uv = reinterpret_cast<const float*>(uv_data + i * uv_stride);
                        this->texcoords.push_back(glm::vec2(uv[0], uv[1]));
                    } else {
                        this->texcoords.push_back(glm::vec2(0.0f));
                    }
                }

                // --- Determine material ID ---
                int mat_id;
                if (primitive.material < 0) {
                    // Use default material (last material added, or 0)
                    mat_id = (model.materials.empty()) ? material_offset : material_offset;
                } else {
                    mat_id = material_offset + primitive.material;
                }

                // --- Add triangles ---
                auto addTriangles = [&](auto getIndex) {
                    for (size_t i = 0; i + 2 < indicesAccessor.count; i += 3) {
                        int i0 = current_vertex_offset + getIndex(i + 0);
                        int i1 = current_vertex_offset + getIndex(i + 1);
                        int i2 = current_vertex_offset + getIndex(i + 2);

                        Geom tri_geom = {};
                        tri_geom.type = TRIANGLE;
                        tri_geom.materialid = mat_id;
                        tri_geom.v0 = i0;
                        tri_geom.v1 = i1;
                        tri_geom.v2 = i2;

                        // Identity transforms for mesh triangles (vertices are pre-transformed)
                        tri_geom.transform = glm::mat4(1.0f);
                        tri_geom.inverseTransform = glm::mat4(1.0f);
                        tri_geom.invTranspose = glm::mat4(1.0f);

                        // Per-vertex UVs
                        if (hasTexCoords && uv_data) {
                            const float* uv0 = reinterpret_cast<const float*>(uv_data + (size_t)(i0 - current_vertex_offset) * uv_stride);
                            const float* uv1 = reinterpret_cast<const float*>(uv_data + (size_t)(i1 - current_vertex_offset) * uv_stride);
                            const float* uv2 = reinterpret_cast<const float*>(uv_data + (size_t)(i2 - current_vertex_offset) * uv_stride);
                            tri_geom.uv0 = glm::vec2(uv0[0], uv0[1]);
                            tri_geom.uv1 = glm::vec2(uv1[0], uv1[1]);
                            tri_geom.uv2 = glm::vec2(uv2[0], uv2[1]);
                        } else {
                            tri_geom.uv0 = tri_geom.uv1 = tri_geom.uv2 = glm::vec2(0.0f);
                        }

                        // Compute face normal if no normals provided
                        if (!hasNormals) {
                            glm::vec3 v0 = this->positions[i0];
                            glm::vec3 v1 = this->positions[i1];
                            glm::vec3 v2 = this->positions[i2];
                            glm::vec3 faceNormal = glm::normalize(glm::cross(v1 - v0, v2 - v0));
                            this->normals[i0] = faceNormal;
                            this->normals[i1] = faceNormal;
                            this->normals[i2] = faceNormal;
                        }

                        this->geoms.push_back(tri_geom);
                    }
                };

                // Handle different index types (UNSIGNED_BYTE, UNSIGNED_SHORT, UNSIGNED_INT)
                switch (indicesAccessor.componentType) {
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
                    addTriangles([idx_data, idx_stride](size_t i) -> int {
                        return (int)*(idx_data + i * idx_stride);
                    });
                    break;
                }
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
                    addTriangles([idx_data, idx_stride](size_t i) -> int {
                        return (int)*reinterpret_cast<const unsigned short*>(idx_data + i * idx_stride);
                    });
                    break;
                }
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
                    addTriangles([idx_data, idx_stride](size_t i) -> int {
                        return (int)*reinterpret_cast<const unsigned int*>(idx_data + i * idx_stride);
                    });
                    break;
                }
                default:
                    std::cerr << "Unsupported index component type: " << indicesAccessor.componentType << std::endl;
                    break;
                }
            }
        }

        for (int child_idx : node.children) {
            processNode(child_idx, node_transform);
        }
    };

    const tinygltf::Scene& gltf_scene = model.scenes[model.defaultScene > -1 ? model.defaultScene : 0];
    for (int node_idx : gltf_scene.nodes) {
        processNode(node_idx, instance_transform);
    }

    std::cout << "glTF loaded: " << this->geoms.size() << " triangles, "
              << this->materials.size() << " materials, "
              << this->textures.size() << " textures" << std::endl;

    // --- Camera setup for standalone mode ---
    if (isStandalone) {
        setupCameraFromGLTF(model);
    }

    return true;
}

// ============================================================================
// JSON Scene Loader
// ============================================================================

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);

    // Extract base directory from the JSON path for resolving relative paths
    std::string baseDir = "";
    size_t lastSlash = jsonName.find_last_of("/\\");
    if (lastSlash != std::string::npos) {
        baseDir = jsonName.substr(0, lastSlash + 1);
    }

    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        newMaterial.textureId = -1;
        newMaterial.roughness = 1.0f;

        if (p["TYPE"] == "Diffuse")
        {
            newMaterial.type = LAMBERTIAN;
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting")
        {
            newMaterial.type = LAMBERTIAN;
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            newMaterial.type = SPECULAR;
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Glass")
        {
            newMaterial.type = GLASS;
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.indexOfRefraction = p.value("IOR", 1.5f);
            newMaterial.abbe = p.value("ABBE", 20.0f);
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        if (type == "gltf")
        {
            std::string path = p["PATH"];
            // Resolve relative path based on JSON file location
            if (path.size() > 0 && path[0] != '/' && path.find(':') == std::string::npos) {
                path = baseDir + path;
            }

            std::cout << "Loading glTF model from: " << path << std::endl;

            const auto& trans = p["TRANS"];
            const auto& rotat = p["ROTAT"];
            const auto& scale = p["SCALE"];
            glm::vec3 translation = glm::vec3(trans[0], trans[1], trans[2]);
            glm::vec3 rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
            glm::vec3 scaling = glm::vec3(scale[0], scale[1], scale[2]);

            glm::mat4 instance_transform = utilityCore::buildTransformationMatrix(translation, rotation, scaling);

            // Allow optional material override from JSON
            int mat_type = -1;
            float ior = 0.0f;
            float abbe = 0.0f;
            if (p.contains("MATERIAL_TYPE")) {
                std::string mt = p["MATERIAL_TYPE"];
                if (mt == "Glass") mat_type = GLASS;
                else if (mt == "Specular") mat_type = SPECULAR;
                else if (mt == "Diffuse") mat_type = LAMBERTIAN;
                ior = p.value("IOR", 0.0f);
                abbe = p.value("ABBE", 0.0f);
            }

            this->loadGLTF(path, instance_transform, mat_type, ior, abbe);
            continue;
        }

        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else
        {
            newGeom.type = SPHERE;
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);
        newGeom.uv0 = newGeom.uv1 = newGeom.uv2 = glm::vec2(0.0f);

        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];

    const float defaultFocalLength = 50.0f;
    const float defaultFAperture = 22.0f;
    const float defaultFocusDist = 1.0f;

    camera.focalLength = cameraData.value("FOCALLENGTH", defaultFocalLength);
    camera.fAperture = cameraData.value("FAPERTURE", defaultFAperture);
    camera.focusDistance = cameraData.value("FOCUSDISTANCE", defaultFocusDist);

    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}
