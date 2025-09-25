#pragma once

#include <glm/gtx/transform.hpp>
#include "sceneStructs.h"
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void loadFromJSON_561(const std::string& jsonName);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
};
