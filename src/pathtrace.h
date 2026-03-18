#pragma once

#include "scene.h"
#include "utilities.h"
#include "dispersion.h"

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration, bool denoiserEnabled);

// Runtime reload
void pathtraceReloadEnvMap(const EnvironmentMap& envMap);
void pathtraceReloadScene(Scene* newScene);
