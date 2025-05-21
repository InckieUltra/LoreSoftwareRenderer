#include <SDL.h>
#include <SDL_image.h>
#include <iostream>
#include <algorithm>
#include "Render.h"
#include "ResourceManager.h"

const int WIDTH =  800;
const int HEIGHT = 600;

bool middleMouseDown = false;
bool shiftHeld = false;
int lastMouseX, lastMouseY;

void handleEvent(const SDL_Event& e, Camera& camera) {
    if (e.type == SDL_MOUSEWHEEL) {
        camera.zoom((float)e.wheel.y);  // y > 0: forward, y < 0: backward
    }

    if (e.type == SDL_MOUSEBUTTONDOWN) {
        if (e.button.button == SDL_BUTTON_MIDDLE) {
            middleMouseDown = true;
            lastMouseX = e.button.x;
            lastMouseY = e.button.y;
        }
    }

    if (e.type == SDL_MOUSEBUTTONUP) {
        if (e.button.button == SDL_BUTTON_MIDDLE) {
            middleMouseDown = false;
        }
    }

    if (e.type == SDL_MOUSEMOTION) {
        if (middleMouseDown) {
            int dx = e.motion.x - lastMouseX;
            int dy = e.motion.y - lastMouseY;
            lastMouseX = e.motion.x;
            lastMouseY = e.motion.y;

            if (shiftHeld) {
                camera.moveView((float)dx, (float)dy);  // Shift + 中键
            } else {
                camera.pan((float)dx, (float)dy);       // 中键平移
            }
        }
    }

    if (e.type == SDL_KEYDOWN) {
        if (e.key.keysym.sym == SDLK_LSHIFT || e.key.keysym.sym == SDLK_RSHIFT) {
            shiftHeld = true;
        }
    }

    if (e.type == SDL_KEYUP) {
        if (e.key.keysym.sym == SDLK_LSHIFT || e.key.keysym.sym == SDLK_RSHIFT) {
            shiftHeld = false;
        }
    }
}

void InitializeScene(Scene& scene) {
	std::vector<Mesh> meshes;
	Transform transform;

	transform.position = { 0, -0.05f, 0 };
	transform.rotation = Eigen::Quaternionf(Eigen::AngleAxisf(0.0f, Eigen::Vector3f(0, 1, 0)));
	transform.scale = { 1, 1, 1 };

	std::vector<Mesh> meshes2;
	ResourceManager::loadGeometryFromObj(RESOURCE_DIR "/CornellBox.obj", meshes2);

	Texture* CornellTexture = new Texture();
	bool success2 = ResourceManager::loadTextureFromPNG(RESOURCE_DIR "/CornellBox.png", 
		*CornellTexture);
	for (auto& mesh : meshes2){
		mesh.texture = CornellTexture;
	}
	scene.objects.push_back(Object(meshes2, transform));

	scene.camera.target = transform.position;

	scene.lights.push_back(DirectionalLight{ {1, 1, 1}, {1, 1, 1}, 1.0f });
}

void AddSomething(Scene& scene){
	std::vector<Mesh> meshes;
	Transform transform;

	ResourceManager::loadGeometryFromObj(RESOURCE_DIR "/Pandora Bunny.obj", meshes);
	transform.position = { 0.3f, 0, -0.3f };
	transform.rotation = Eigen::Quaternionf(Eigen::AngleAxisf(0.0f, Eigen::Vector3f(1, 0, 0)));
	transform.scale = { 1, 1, 1 };

	Texture* BunnyTexture = new Texture();
	bool success = ResourceManager::loadTextureFromPNG(RESOURCE_DIR "/Pandora Bunny.png", 
		*BunnyTexture);

	for (auto& mesh : meshes){
		mesh.texture = BunnyTexture;
	}
	scene.objects.push_back(Object(meshes, transform));
}

int main(int argc, char* argv[]) {
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::cerr << "SDL_Init failed: " << SDL_GetError() << "\n";
        return 1;
    }

	if (!(IMG_Init(IMG_INIT_PNG) & IMG_INIT_PNG)) {
	    std::cerr << "IMG_Init failed: " << IMG_GetError() << std::endl;
	    return -1;
	}

    SDL_Window* window = SDL_CreateWindow("Software Triangle", 100, 100, WIDTH, HEIGHT, SDL_WINDOW_SHOWN);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, 0);
    SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, WIDTH, HEIGHT);

    bool running = true;
    SDL_Event e;

	Scene scene;

	InitializeScene(scene);
	//AddSomething(scene);

    while (running) {
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT)
                running = false;
			handleEvent(e, scene.camera);
        }

        uint32_t* pixels = nullptr;
        int pitch;
        SDL_LockTexture(texture, nullptr, (void**)&pixels, &pitch);

        // Clear background
        std::fill(pixels, pixels + WIDTH * HEIGHT, 0xFF202020); // dark gray

		if (scene.objects.size() > 1) {
			scene.objects[1].transform.rotation = Eigen::AngleAxisf(0.005f, Eigen::Vector3f(0, 1, 0)) * scene.objects[1].transform.rotation;
		}
		
		Texture renderedTexture = RenderSceneRayTracing(scene, WIDTH, HEIGHT);
		for (int y = 0; y < HEIGHT; ++y) {
			for (int x = 0; x < WIDTH; ++x) {
				int index = y * WIDTH + x;
				uint8_t r = renderedTexture.data[index * 4 + 0];
				uint8_t g = renderedTexture.data[index * 4 + 1];
				uint8_t b = renderedTexture.data[index * 4 + 2];
				uint8_t a = renderedTexture.data[index * 4 + 3];
				pixels[index] = toARGB(r, g, b, a);
			}
		}

        SDL_UnlockTexture(texture);
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, nullptr, nullptr);
        SDL_RenderPresent(renderer);
    }

    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
