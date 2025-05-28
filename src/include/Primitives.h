#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>

struct Texture {
    int width, height;
    std::vector<uint8_t> data; // Assuming RGBA format
};

struct Vertex {
    Eigen::Vector3f position;
    Eigen::Vector3f normal;
    Eigen::Vector4f color;
    Eigen::Vector2f uv;
};

struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    Texture* texture = nullptr;
};

struct Transform {
    Eigen::Vector3f position;
    Eigen::Quaternionf rotation;
    Eigen::Vector3f scale;
};

struct Object {
    Object() {}
    Object(std::vector<Mesh> meshes, Transform transform)
        : meshes(meshes), transform(transform) {
            update_matrix();
        }

    std::vector<Mesh> meshes;
    Transform transform;
    Eigen::Matrix4f modelMatrix;
    Eigen::Matrix4f invModelMatrix;

    void update_matrix(){
        modelMatrix = Eigen::Affine3f(
            Eigen::Translation3f(transform.position) *
            transform.rotation.toRotationMatrix() *
            Eigen::Scaling(transform.scale)
        ).matrix();
        invModelMatrix = modelMatrix.inverse();
    }

    void update_transform(const Transform& new_transform) {
        transform = new_transform;
        update_matrix();
    }
};

struct DirectionalLight {
    Eigen::Vector3f direction;
    Eigen::Vector3f color;
    float intensity;
};

struct PointLight {
    Eigen::Vector3f position;
    Eigen::Vector3f color;
    float intensity;
};

struct DepthBuffer {
    int width, height;
    std::vector<float> data; // Assuming float depth values
};