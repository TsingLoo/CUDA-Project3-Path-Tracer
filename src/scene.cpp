#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/matrix_transform.hpp> // For glm::translate, glm::rotate, glm::scale
#include <glm/gtc/type_ptr.hpp>         // For glm::make_mat4
#include <glm/gtc/quaternion.hpp>       // For glm::quat and glm::mat4_cast

#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

using namespace std;
using json = nlohmann::json;

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
    else if (ext == ".gltf")
    {
        loadGLTFScene(filename);
        setupDefaultCamera();
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::setupDefaultCamera() {
    // --- Direct Access to State ---
    Camera& camera = this->state.camera;

    // --- Set Default Render State ---
    this->state.iterations = 1024; // Default number of samples per pixel
    this->state.traceDepth = 8;     // Default max bounces
    this->state.imageName = "default_render.png";

    // --- Set Default Camera Intrinsics ---
    camera.resolution = glm::ivec2(600, 600);
    float fovy = 45.0f; // 45-degree vertical field of view is standard

    // --- Set Default Camera Extrinsics (Position & Orientation) ---
    camera.position = glm::vec3(0, 0, 10); // Positioned 10 units back on the Z-axis
    camera.lookAt = glm::vec3(0, 0, 0);   // Looking at the world origin
    camera.up = glm::vec3(0, 1, 0);     // Y-axis is up

    // --- Set Default Physical Lens Properties ---
    camera.focalLength = 50.0f; // Standard 50mm lens
    camera.fAperture = 22.0f;   // Small aperture for deep depth of field
    camera.focusDistance = glm::length(camera.lookAt - camera.position); // Focus on the lookAt point

    // --- Calculate Derived Camera Vectors ---
    // The order of these calculations is important!
    camera.view = glm::normalize(camera.lookAt - camera.position);
    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.up = glm::normalize(glm::cross(camera.right, camera.view)); // Re-orthogonalize

    // --- Calculate FOV and Pixel Dimensions ---
    float yscaled = tan(fovy * (PI / 180.0f));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180.0f) / PI;
    camera.fov = glm::vec2(fovx, fovy);
    camera.pixelLength = glm::vec2(2.0f * xscaled / (float)camera.resolution.x,
        2.0f * yscaled / (float)camera.resolution.y);

    // --- Initialize Image Buffer ---
    int pixel_count = camera.resolution.x * camera.resolution.y;
    this->state.image.resize(pixel_count);
    std::fill(this->state.image.begin(), this->state.image.end(), glm::vec3(0.0f));
}

bool Scene::loadGLTF(const std::string& filename, const glm::mat4& instance_transform) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, filename);
    if (!ret) {
        // Also check for binary .glb files as a fallback
        ret = loader.LoadBinaryFromFile(&model, &err, &warn, filename);
    }

    if (!warn.empty()) {
        std::cout << "glTF Loader Warning: " << warn << std::endl;
    }
    if (!err.empty()) {
        std::cerr << "glTF Loader Error: " << err << std::endl;
    }
    if (!ret) {
        return false;
    }

    // --- Handle Offsets ---
    // This allows us to append multiple glTF models to the same scene
    const int material_offset = this->materials.size();

    // 1. Process Materials from the glTF file
    for (const auto& mat : model.materials) {
        Material newMaterial = {};
        const auto& pbr = mat.pbrMetallicRoughness;

        if (mat.alphaMode == "BLEND") {
            newMaterial.type = GLASS;
            newMaterial.indexOfRefraction = 1.5f;
            newMaterial.abbe = 10.0f;
        }
        else {
            newMaterial.type = GLASS;
            newMaterial.indexOfRefraction = 1.5f;
            newMaterial.abbe = 10.0f;
        }

        newMaterial.type = GLASS;
        newMaterial.indexOfRefraction = 1.3f;
        newMaterial.abbe = 30.0f;

        if (pbr.baseColorFactor.size() == 4) {
            newMaterial.color = glm::vec3(pbr.baseColorFactor[0], pbr.baseColorFactor[1], pbr.baseColorFactor[2]);
            newMaterial.color = glm::vec3(0.5, 1.0, 0.5f);
            newMaterial.color = glm::vec3(1.0, 1.0, 1.0f);
        }

        newMaterial.color = glm::vec3(1.0, 1.0, 1.0f);

        this->materials.push_back(newMaterial);
    }

    // 2. Process Nodes and Meshes
    std::function<void(int, const glm::mat4&)> processNode =
        [&](int node_idx, const glm::mat4& parent_transform) {

        const tinygltf::Node& node = model.nodes[node_idx];
        glm::mat4 node_transform = parent_transform;

        // Apply this node's local transform
        if (node.matrix.size() == 16) {
            node_transform *= glm::mat4(glm::make_mat4(node.matrix.data()));
        }
        else {
            // Apply TRS properties (Translate, Rotate, Scale) if no matrix is given
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
                // Check that we have the necessary attributes
                if (primitive.indices < 0 || primitive.attributes.find("POSITION") == primitive.attributes.end()) {
                    continue;
                }

                // --- Get pointers to the raw buffer data ---
                const tinygltf::Accessor& indicesAccessor = model.accessors[primitive.indices];
                const tinygltf::Accessor& posAccessor = model.accessors.at(primitive.attributes.find("POSITION")->second);

                const tinygltf::BufferView& indicesBufferView = model.bufferViews[indicesAccessor.bufferView];
                const tinygltf::BufferView& posBufferView = model.bufferViews[posAccessor.bufferView];

                const tinygltf::Buffer& indicesBuffer = model.buffers[indicesBufferView.buffer];
                const tinygltf::Buffer& posBuffer = model.buffers[posBufferView.buffer];

                const float* positions = reinterpret_cast<const float*>(&posBuffer.data[posBufferView.byteOffset + posAccessor.byteOffset]);

                // Get normals if they exist
                const float* normals = nullptr;
                if (primitive.attributes.find("NORMAL") != primitive.attributes.end()) {
                    const tinygltf::Accessor& normAccessor = model.accessors.at(primitive.attributes.find("NORMAL")->second);
                    const tinygltf::BufferView& normBufferView = model.bufferViews[normAccessor.bufferView];
                    const tinygltf::Buffer& normBuffer = model.buffers[normBufferView.buffer];
                    normals = reinterpret_cast<const float*>(&normBuffer.data[normBufferView.byteOffset + normAccessor.byteOffset]);
                }

                // This is the offset for vertex indices for this specific mesh primitive
                const int current_vertex_offset = this->positions.size();

                // Add vertices to our global scene lists
                for (size_t i = 0; i < posAccessor.count; ++i) {
                    glm::vec3 pos = glm::vec3(positions[i * 3 + 0], positions[i * 3 + 1], positions[i * 3 + 2]);

                    // Apply the final combined transform (instance * node) to the vertex
                    this->positions.push_back(glm::vec3(node_transform * glm::vec4(pos, 1.0f)));

                    if (normals) {
                        glm::vec3 norm = glm::vec3(normals[i * 3 + 0], normals[i * 3 + 1], normals[i * 3 + 2]);
                        // Apply transform to normals as well (use inverse transpose for non-uniform scaling)
                        this->normals.push_back(glm::normalize(glm::mat3(glm::transpose(glm::inverse(node_transform))) * norm));
                    }
                    else {
                        // If no normals are provided, we can add a placeholder
                        this->normals.push_back(glm::vec3(0, 1, 0));
                    }
                }

                // Add triangles to our scene's geometry list
                const void* indices_data = &indicesBuffer.data[indicesBufferView.byteOffset + indicesAccessor.byteOffset];

                // Handle different index types (unsigned short, unsigned int, etc.)
                switch (indicesAccessor.componentType) {
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
                    const unsigned short* indices = static_cast<const unsigned short*>(indices_data);
                    for (size_t i = 0; i < indicesAccessor.count; i += 3) {
                        Geom tri_geom = {};
                        tri_geom.type = TRIANGLE;
                        tri_geom.materialid = (primitive.material < 0) ? 0 : material_offset + primitive.material;
                        tri_geom.v0 = current_vertex_offset + indices[i + 0];
                        tri_geom.v1 = current_vertex_offset + indices[i + 1];
                        tri_geom.v2 = current_vertex_offset + indices[i + 2];
                        this->geoms.push_back(tri_geom);
                    }
                    break;
                }
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
                    const unsigned int* indices = static_cast<const unsigned int*>(indices_data);
                    for (size_t i = 0; i < indicesAccessor.count; i += 3) {
                        Geom tri_geom = {};
                        tri_geom.type = TRIANGLE;
                        tri_geom.materialid = (primitive.material < 0) ? 0 : material_offset + primitive.material;
                        tri_geom.v0 = current_vertex_offset + indices[i + 0];
                        tri_geom.v1 = current_vertex_offset + indices[i + 1];
                        tri_geom.v2 = current_vertex_offset + indices[i + 2];
                        this->geoms.push_back(tri_geom);
                    }
                    break;
                }
                                                         // Add other cases like UNSIGNED_BYTE if needed
                }
            }
        }

        for (int child_idx : node.children) {
            processNode(child_idx, node_transform);
        }
        };

    const tinygltf::Scene& gltf_scene = model.scenes[model.defaultScene > -1 ? model.defaultScene : 0];
    for (int node_idx : gltf_scene.nodes) {
        // Start recursion with the instance transform from the JSON file
        processNode(node_idx, instance_transform);
    }

    return true;
}

bool Scene::loadGLTFScene(const std::string& filename) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, filename);
    // For binary .glb files, use: bool ret = loader.LoadBinaryFromFile(...)

	std::cout << "Loading glTF scene from " << filename << " ..." << std::endl;

    if (!warn.empty()) {
        printf("Warn: %s\n", warn.c_str());
    }
    if (!err.empty()) {
        printf("Err: %s\n", err.c_str());
        return false;
    }
    if (!ret) {
        printf("Failed to parse glTF\n");
        return false;
    }

    // --- Process Materials (example) ---
    // This part will need to be adapted to your Material struct
    for (const auto& mat : model.materials) {
        Material newMaterial = {};
        const auto& pbr = mat.pbrMetallicRoughness;
        newMaterial.type = LAMBERTIAN; // Default to Lambertian
        newMaterial.color = glm::vec3(
            pbr.baseColorFactor[0],
            pbr.baseColorFactor[1],
            pbr.baseColorFactor[2]
        );
        this->materials.push_back(newMaterial);
    }

    // --- Process Nodes and Meshes ---
    // This lambda function will recursively process the scene graph
    std::function<void(int, const glm::mat4&)> processNode =
        [&](int node_idx, const glm::mat4& parent_transform) {

        const tinygltf::Node& node = model.nodes[node_idx];
        glm::mat4 transform = parent_transform;

        // Apply this node's transform
        if (node.matrix.size() == 16) {
            transform *= glm::mat4(glm::make_mat4(node.matrix.data()));
        }
        else {
            if (node.translation.size() == 3) {
                transform = glm::translate(transform, glm::vec3(node.translation[0], node.translation[1], node.translation[2]));
            }
            if (node.rotation.size() == 4) {
                transform *= glm::mat4_cast(glm::quat((float)node.rotation[3], (float)node.rotation[0], (float)node.rotation[1], (float)node.rotation[2]));
            }
            if (node.scale.size() == 3) {
                transform = glm::scale(transform, glm::vec3(node.scale[0], node.scale[1], node.scale[2]));
            }
        }

        if (node.mesh > -1) {
            const tinygltf::Mesh& mesh = model.meshes[node.mesh];
            for (const auto& primitive : mesh.primitives) {
                const auto& indicesAccessor = model.accessors[primitive.indices];
                const auto& positionsAccessor = model.accessors[primitive.attributes.find("POSITION")->second];
                const auto& normalsAccessor = model.accessors[primitive.attributes.find("NORMAL")->second];

                const tinygltf::BufferView& indicesBufferView = model.bufferViews[indicesAccessor.bufferView];
                const tinygltf::BufferView& positionsBufferView = model.bufferViews[positionsAccessor.bufferView];
                const tinygltf::BufferView& normalsBufferView = model.bufferViews[normalsAccessor.bufferView];

                const tinygltf::Buffer& indicesBuffer = model.buffers[indicesBufferView.buffer];
                const tinygltf::Buffer& positionsBuffer = model.buffers[positionsBufferView.buffer];
                const tinygltf::Buffer& normalsBuffer = model.buffers[normalsBufferView.buffer];

                // Get pointers to the data
                const unsigned short* indices = reinterpret_cast<const unsigned short*>(&indicesBuffer.data[indicesBufferView.byteOffset + indicesAccessor.byteOffset]);
                const float* positions = reinterpret_cast<const float*>(&positionsBuffer.data[positionsBufferView.byteOffset + positionsAccessor.byteOffset]);
                const float* normals = reinterpret_cast<const float*>(&normalsBuffer.data[normalsBufferView.byteOffset + normalsAccessor.byteOffset]);

                // This is the offset for vertex indices for this specific mesh
                int vertex_offset = this->positions.size();

                // Add vertices to our global lists
                for (size_t i = 0; i < positionsAccessor.count; ++i) {
                    glm::vec3 pos = glm::vec3(positions[i * 3 + 0], positions[i * 3 + 1], positions[i * 3 + 2]);
                    glm::vec3 norm = glm::vec3(normals[i * 3 + 0], normals[i * 3 + 1], normals[i * 3 + 2]);

                    // Apply the node's transform to the vertex position and normal
                    this->positions.push_back(glm::vec3(transform * glm::vec4(pos, 1.0f)));
                    this->normals.push_back(glm::normalize(glm::mat3(transform) * norm));
                }

                // Add triangles (as Geoms) to our scene
                for (size_t i = 0; i < indicesAccessor.count; i += 3) {
                    Geom tri_geom = {};
                    tri_geom.type = TRIANGLE;
                    tri_geom.materialid = primitive.material;
                    tri_geom.v0 = vertex_offset + indices[i + 0];
                    tri_geom.v1 = vertex_offset + indices[i + 1];
                    tri_geom.v2 = vertex_offset + indices[i + 2];
                    this->geoms.push_back(tri_geom);
                }
            }
        }

        for (int child_idx : node.children) {
            processNode(child_idx, transform);
        }
        };

    const tinygltf::Scene& gltf_scene = model.scenes[model.defaultScene > -1 ? model.defaultScene : 0];
    for (int node_idx : gltf_scene.nodes) {
        processNode(node_idx, glm::mat4(1.0f));
    }

    return true;
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
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

			std::cout << "Trying to load glTF model from: " << path << std::endl;

            // Build the instance transform matrix for this glTF model from the JSON
            const auto& trans = p["TRANS"];
            const auto& rotat = p["ROTAT"];
            const auto& scale = p["SCALE"];
            glm::vec3 translation = glm::vec3(trans[0], trans[1], trans[2]);
            glm::vec3 rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
            glm::vec3 scaling = glm::vec3(scale[0], scale[1], scale[2]);

            glm::mat4 instance_transform = utilityCore::buildTransformationMatrix(translation, rotation, scaling);

            // This function will add many triangle Geoms to the `geoms` vector itself.
            this->loadGLTF(path, instance_transform);
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
