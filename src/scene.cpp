#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
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
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

glm::mat4 buildTransformationMatrix(glm::vec3 t, glm::vec3 r, glm::vec3 s) {
    glm::mat4 trans = glm::translate(t);
    glm::mat4 rot = glm::rotate(glm::radians(r.z), glm::vec3(0, 0, 1)) *
        glm::rotate(glm::radians(r.y), glm::vec3(0, 1, 0)) *
        glm::rotate(glm::radians(r.x), glm::vec3(1, 0, 0));
    glm::mat4 scale = glm::scale(s);
    return trans * rot * scale;
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
            const auto& col = p["RGB"];
            newMaterial.albedo = glm::vec3(col[0], col[1], col[2]);
            newMaterial.type = MaterialType::DIFFUSE_REFL;
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.albedo = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
            newMaterial.type = MaterialType::EMITTIVE;
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.albedo = glm::vec3(col[0], col[1], col[2]);
            newMaterial.type = MaterialType::SPEC_REFL;
            newMaterial.roughness = p["ROUGHNESS"];
        }
        else if (p["TYPE"] == "Specular_Trans")
        {
            const auto& col = p["RGB"];
            newMaterial.albedo = glm::vec3(col[0], col[1], col[2]);
            newMaterial.type = MaterialType::SPEC_TRANS;
            newMaterial.roughness = p["ROUGHNESS"];
            newMaterial.eta = p["eta"];
        }
        else if (p["TYPE"] == "Specular_Glass")
        {
            const auto& col = p["RGB"];
            newMaterial.albedo = glm::vec3(col[0], col[1], col[2]);
            newMaterial.type = MaterialType::SPEC_GLASS;
            newMaterial.roughness = p["ROUGHNESS"];
            newMaterial.eta = p["eta"];
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
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


void Scene::loadFromJSON_561(const std::string& jsonName) {
    std::ifstream f(jsonName);
    json data = json::parse(f);

    // 1. Navigate to the scene object (we'll just use the first frame)
    const auto& scene = data["frames"][0]["scene"];

    // 2. Parse Materials and create a name-to-ID map
    // This map is crucial for linking primitives to their materials by name.
    std::unordered_map<std::string, uint32_t> materialNameToId;
    if (scene.contains("materials")) {
        const auto& materialsData = scene["materials"];
        for (const auto& mat_json : materialsData) {
            std::string name = mat_json["name"];
            Material newMaterial{};

            std::string type = mat_json["type"];
            if (type == "MatteMaterial") {
                const auto& kd = mat_json["Kd"];
                newMaterial.albedo = glm::vec3(kd[0], kd[1], kd[2]);
                newMaterial.emittance = 0.0f; // Not emissive
                newMaterial.type = MaterialType::DIFFUSE_REFL;
            }
            // Add else-if for MirrorMaterial, GlassMaterial etc. here

            // Store the material and map its name to its index in our vector
            materialNameToId[name] = materials.size();
            materials.emplace_back(newMaterial);
        }
    }

    // 3. Parse Primitives
    if (scene.contains("primitives")) {
        const auto& primitivesData = scene["primitives"];
        for (const auto& prim_json : primitivesData) {
            Geom newGeom;

            // Map shape string to enum
            std::string shape = prim_json["shape"];
            if (shape == "Cube") newGeom.type = CUBE;
            else if (shape == "SquarePlane") newGeom.type = SQUARE_PLANE;
            // Add else-if for "Sphere", "Mesh", etc.

            // Use the map to find the material ID from its name
            std::string materialName = prim_json["material"];
            newGeom.materialid = materialNameToId.at(materialName);

            // Parse the nested transform object
            const auto& transform_json = prim_json["transform"];
            glm::vec3 t(0), r(0), s(1); // Default values
            if (transform_json.contains("translate")) t = { transform_json["translate"][0], transform_json["translate"][1], transform_json["translate"][2] };
            if (transform_json.contains("rotate"))    r = { transform_json["rotate"][0],    transform_json["rotate"][1],    transform_json["rotate"][2] };
            if (transform_json.contains("scale"))     s = { transform_json["scale"][0],     transform_json["scale"][1],     transform_json["scale"][2] };

            newGeom.transform = buildTransformationMatrix(t, r, s);
            newGeom.inverseTransform = glm::inverse(newGeom.transform);
            newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

            geoms.push_back(newGeom);
        }
    }

    // 4. Parse Lights (treat them as primitives with special materials)
    if (scene.contains("lights")) {
        const auto& lightsData = scene["lights"];
        for (const auto& light_json : lightsData) {
            // A light is just another piece of geometry...
            Geom newGeom;
            std::string shape = light_json["shape"];
            if (shape == "SquarePlane") newGeom.type = SQUARE_PLANE;
            // Add other light shapes if needed

            // ...but it has a unique, new emissive material that we create on the fly.
            Material lightMaterial{};
            const auto& color = light_json["lightColor"];
            float intensity = light_json.value("intensity", 1.0f);
            lightMaterial.albedo = glm::vec3(color[0], color[1], color[2]);
            lightMaterial.emittance = intensity;
            lightMaterial.type = EMITTIVE;

            // Add the new light material to our list and get its ID
            newGeom.materialid = materials.size();
            materials.emplace_back(lightMaterial);

            // Parse the transform just like a regular primitive
            const auto& transform_json = light_json["transform"];
            glm::vec3 t(0), r(0), s(1);
            if (transform_json.contains("translate")) t = { transform_json["translate"][0], transform_json["translate"][1], transform_json["translate"][2] };
            if (transform_json.contains("rotate"))    r = { transform_json["rotate"][0],    transform_json["rotate"][1],    transform_json["rotate"][2] };
            if (transform_json.contains("scale"))     s = { transform_json["scale"][0],     transform_json["scale"][1],     transform_json["scale"][2] };

            newGeom.transform = buildTransformationMatrix(t, r, s);
            newGeom.inverseTransform = glm::inverse(newGeom.transform);
            newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

            geoms.push_back(newGeom);
        }
    }

    // 5. Parse Camera
    if (scene.contains("camera")) {
        const auto& cameraData = scene["camera"];
        Camera& camera = state.camera;

        camera.resolution.x = cameraData["width"];
        camera.resolution.y = cameraData["height"];

        const auto& eye = cameraData["eye"];
        const auto& target = cameraData["target"];
        const auto& up = cameraData["worldUp"];
        camera.position = glm::vec3(eye[0], eye[1], eye[2]);
        camera.lookAt = glm::vec3(target[0], target[1], target[2]);
        camera.up = glm::vec3(up[0], up[1], up[2]);

        camera.view = glm::normalize(camera.lookAt - camera.position);
        camera.right = glm::normalize(glm::cross(camera.view, camera.up));
        // Re-calculate the "true" up vector to ensure orthogonality
        camera.up = glm::normalize(glm::cross(camera.right, camera.view));

        float fovy_degrees = cameraData["fov"];
        camera.fov.y = fovy_degrees;

        // Calculate horizontal FOV from vertical FOV and aspect ratio
        float aspectRatio = (float)camera.resolution.x / (float)camera.resolution.y;
        float yscaled = tan(glm::radians(fovy_degrees * 0.5f));
        float xscaled = yscaled * aspectRatio;
        camera.fov.x = glm::degrees(atan(xscaled) * 2.0f);

        // Update pixel length for ray generation
        camera.pixelLength = glm::vec2(2.0f * xscaled / (float)camera.resolution.x,
            2.0f * yscaled / (float)camera.resolution.y);
    }

    // Other settings (optional, provide defaults if not present)
    state.iterations = scene.value("iterations", 1024);
    state.traceDepth = scene.value("depth", 5);
    state.imageName = scene.value("outputFile", "render.png");

    int arraylen = state.camera.resolution.x * state.camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}