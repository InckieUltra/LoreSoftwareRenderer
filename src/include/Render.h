#pragma once
#include "Scene.h"

uint32_t toARGB(uint8_t r, uint8_t g, uint8_t b, uint8_t a);
Texture RenderScene(Scene& scene, int width, int height);
Texture RenderSceneRayTracing(Scene& scene, int width, int height);