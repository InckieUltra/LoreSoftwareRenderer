// In ResourceManager.cpp
#include "ResourceManager.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#define TINYOBJLOADER_IMPLEMENTATION // add this to exactly 1 of your C++ files
#include "tiny_obj_loader.h"

#include <SDL_image.h>

bool ResourceManager::loadGeometryFromObj(
	const std::filesystem::path& path,
	std::vector<Mesh>& meshes
) {
	tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;

    std::string warn;
    std::string err;

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, path.string().c_str());

    if (!warn.empty()) {
        std::cout << warn << std::endl;
    }

    if (!err.empty()) {
        std::cerr << err << std::endl;
    }

    if (!ret) {
        return false;
    }

    // Filling in mesh.vertices:
	for (const auto& shape : shapes) {
	    Mesh mesh;
	    size_t index_offset = 0;

	    // Iterate over all faces (triangles, since we triangulated the OBJ)
	    for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
	        // Assume triangulated mesh (3 vertices per face)
	        assert(shape.mesh.num_face_vertices[f] == 3);

	        // For each vertex of the face
	        for (size_t v = 0; v < 3; v++) {
	            tinyobj::index_t idx = shape.mesh.indices[index_offset + v];

	            Vertex vertex;
			
	            // Position (with Y-up to Z-up conversion if needed)
	            vertex.position = {
	                attrib.vertices[3 * idx.vertex_index + 0],
	                attrib.vertices[3 * idx.vertex_index + 1],
	                attrib.vertices[3 * idx.vertex_index + 2]
	            };

	            if (idx.normal_index >= 0 && !attrib.normals.empty()) {
	                vertex.normal = {
	                    attrib.normals[3 * idx.normal_index + 0],
	                    attrib.normals[3 * idx.normal_index + 1],
	                    attrib.normals[3 * idx.normal_index + 2]
	                };
	            } else {
	                vertex.normal = {0.0f, 1.0f, 0.0f};
	            }

	            if (idx.texcoord_index >= 0 && !attrib.texcoords.empty()) {
	                vertex.uv = {
	                    attrib.texcoords[2 * idx.texcoord_index + 0],
	                    1.0f - attrib.texcoords[2 * idx.texcoord_index + 1] // Flip Y
	                };
	            } else {
	                vertex.uv = {0.0f, 0.0f};
	            }

	            if (!attrib.colors.empty()) {
	                vertex.color = {
	                    attrib.colors[3 * idx.vertex_index + 0],
	                    attrib.colors[3 * idx.vertex_index + 1],
	                    attrib.colors[3 * idx.vertex_index + 2],
	                    1.0f
	                };
	            } else {
	                vertex.color = {1.0f, 1.0f, 1.0f, 1.0f};
	            }

	            mesh.vertices.push_back(vertex);
	            mesh.indices.push_back(static_cast<uint32_t>(mesh.vertices.size() - 1));
	        }

	        index_offset += 3;
	    }

	    meshes.push_back(mesh);
	}

    return true;
}

bool ResourceManager::loadTextureFromPNG(
	const std::filesystem::path& path,
	Texture& texture
) {
    SDL_Surface* surface = IMG_Load(path.string().c_str());
    if (!surface) {
        std::cerr << "Failed to load image: " << path << "\n";
        std::cerr << "SDL_image error: " << IMG_GetError() << "\n";
        return false;
    }

    // Ensure surface is in RGBA32 format
    SDL_Surface* formatted = SDL_ConvertSurfaceFormat(surface, SDL_PIXELFORMAT_RGBA32, 0);
    SDL_FreeSurface(surface); // free original

    if (!formatted) {
        std::cerr << "Failed to convert image to RGBA32: " << path << "\n";
        return false;
    }

    texture.width = formatted->w;
    texture.height = formatted->h;
    size_t dataSize = texture.width * texture.height * 4; // 4 bytes per pixel (RGBA)

    texture.data.resize(dataSize);
    std::memcpy(texture.data.data(), formatted->pixels, dataSize);

    SDL_FreeSurface(formatted);
    return true;
}