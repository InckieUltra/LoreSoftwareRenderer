#pragma once
#include <vector>
#include <filesystem>
#include "Primitives.h"

class ResourceManager {
public:
	static bool loadGeometryFromObj(
		const std::filesystem::path& path,
		std::vector<Mesh>& meshes
	);
    static bool loadTextureFromPNG(
		const std::filesystem::path& path,
		Texture& texture
	);
};