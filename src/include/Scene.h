#pragma once
#include "Primitives.h"
#include "Camera.h"

class Scene{
public:

    void init();
    void render();
    void update(float deltaTime);
    void cleanup();

public:
    Camera camera;
    std::vector<Object> objects;
    std::vector<DirectionalLight> lights;
};