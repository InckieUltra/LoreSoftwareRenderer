#pragma once
#include <Eigen/Dense>

class Camera {
public:
    Eigen::Vector3f position;
    Eigen::Vector3f target;
    Eigen::Vector3f up;

    float fov; // Field of view
    float nearPlane;
    float farPlane;

    float zoomSpeed = 1.0f;
    float moveSpeed = 0.01f;
    float panSpeed = 0.01f;

    Camera()
        : position(0.0f, 0.0f, 5.0f),
          target(0.0f, 0.0f, 0.0f),
          up(0.0f, 1.0f, 0.0f),
          fov(45.0f),
          nearPlane(0.1f),
          farPlane(100.0f) {}

    Eigen::Matrix4f getViewMatrix() const {
        Eigen::Vector3f f = (target - position).normalized();
        Eigen::Vector3f zaxis = -f;
        Eigen::Vector3f xaxis = up.cross(zaxis).normalized();
        Eigen::Vector3f yaxis = zaxis.cross(xaxis);

        Eigen::Matrix4f view = Eigen::Matrix4f::Identity();
        view.block<1, 3>(0, 0) = xaxis.transpose();
        view.block<1, 3>(1, 0) = yaxis.transpose();
        view.block<1, 3>(2, 0) = zaxis.transpose();

        view(0, 3) = -xaxis.dot(position);
        view(1, 3) = -yaxis.dot(position);
        view(2, 3) = -zaxis.dot(position);

        return view;
    }

    void moveView(float dx, float dy) {
        Eigen::Vector3f viewDir = (target - position).normalized();
        Eigen::Vector3f right = viewDir.cross(up).normalized();
        Eigen::Vector3f upMove = right.cross(viewDir).normalized();

        position += -right * dx * panSpeed + upMove * dy * panSpeed;
        target += -right * dx * panSpeed + upMove * dy * panSpeed;
    }

    void pan(float dx, float dy) {
        Eigen::Vector3f viewDir = (target - position).normalized();
        Eigen::Vector3f right = -viewDir.cross(up).normalized();
        Eigen::Vector3f upMove = right.cross(viewDir).normalized();

        position += right * dx * moveSpeed + upMove * -dy * moveSpeed;
    }

    void zoom(float delta) {
        Eigen::Vector3f viewDir = (target - position).normalized();
        float distance = (target - position).norm();
        float zoomAmount = delta * zoomSpeed;
        if (distance - zoomAmount < 1.0f) {
            zoomAmount = 1.0f - distance;
        }

        position += viewDir * zoomAmount;
    }
};
