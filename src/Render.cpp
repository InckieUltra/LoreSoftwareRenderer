#include "Render.h"
#include <iostream>
#include <corecrt_math_defines.h>
#include <execution>
#include "BVH.h"
#include <stack>

uint32_t toARGB(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    return (a << 24) | (b << 16) | (g << 8) | r;
}

void drawFilledTriangle(Texture& renderTarget, DepthBuffer& depthBuffer,
    const Eigen::Vector4f& p0, const Eigen::Vector4f& p1, const Eigen::Vector4f& p2,
    const Eigen::Vector4f& c0, const Eigen::Vector4f& c1, const Eigen::Vector4f& c2,
    const Eigen::Vector4f& n0, const Eigen::Vector4f& n1, const Eigen::Vector4f& n2,
    const Eigen::Vector3f& vp0, const Eigen::Vector3f& vp1, const Eigen::Vector3f& vp2,
    const Eigen::Vector2f& uv0, const Eigen::Vector2f& uv1, const Eigen::Vector2f& uv2,
    const Texture* texture,
    const DirectionalLight& light, const Eigen::Vector3f& cameraPos) {
    // Barycentric rasterization
    int minX = std::max(0, (int)std::floor(std::min({p0.x(), p1.x(), p2.x()})));
    int maxX = std::min(renderTarget.width - 1, (int)std::ceil(std::max({p0.x(), p1.x(), p2.x()})));
    int minY = std::max(0, (int)std::floor(std::min({p0.y(), p1.y(), p2.y()})));
    int maxY = std::min(renderTarget.height - 1, (int)std::ceil(std::max({p0.y(), p1.y(), p2.y()})));

    float denom = (p1.y() - p2.y()) * (p0.x() - p2.x()) + (p2.x() - p1.x()) * (p0.y() - p2.y());
    if (denom == 0) return;

    for (int y = minY; y <= maxY; ++y) {
        for (int x = minX; x <= maxX; ++x) {
            float w0 = ((p1.y() - p2.y()) * (x - p2.x()) + (p2.x() - p1.x()) * (y - p2.y())) / denom;
            float w1 = ((p2.y() - p0.y()) * (x - p2.x()) + (p0.x() - p2.x()) * (y - p2.y())) / denom;
            float w2 = 1.0f - w0 - w1;

            if (w0 >= 0 && w1 >= 0 && w2 >= 0) {
                // Compute depth value
                float depth = w0 * p0.z() + w1 * p1.z() + w2 * p2.z();
                int index = y * renderTarget.width + x;

                // Check depth buffer
                if (depth < depthBuffer.data[index]) {
                    depthBuffer.data[index] = depth;
                    // Interpolate color
                    Eigen::Vector4f color = w0 * c0 + w1 * c1 + w2 * c2;
                    if (texture != nullptr){
                        float u = w0 * uv0.x() + w1 * uv1.x() + w2 * uv2.x();
                        float v = w0 * uv0.y() + w1 * uv1.y() + w2 * uv2.y();
                        int texX = static_cast<int>(u * texture->width);
                        int texY = static_cast<int>(v * texture->height);
                        texX = std::clamp(texX, 0, texture->width - 1);
                        texY = std::clamp(texY, 0, texture->height - 1);
                        int offset = (texY * texture->width + texX) * 4;
                        color = Eigen::Vector4f(
                            texture->data[offset + 2] / 255.0f,  // R
                            texture->data[offset + 1] / 255.0f,  // G
                            texture->data[offset + 0] / 255.0f,  // B
                            texture->data[offset + 3] / 255.0f   // A
                        );
                    }
                    Eigen::Vector3f normal = w0 * n0.head<3>() + w1 * n1.head<3>() + w2 * n2.head<3>();
                    normal.normalize();
                    Eigen::Vector3f lightDir = light.direction.normalized();
                    Eigen::Vector3f viewDir = (cameraPos - 
                        (w0*vp0.head<3>()+w1*vp1.head<3>()+w2*vp2.head<3>())).normalized();
                    Eigen::Vector3f reflectDir = (2.0f * normal.dot(lightDir) * normal - lightDir).normalized();
                    float specAngle = std::max(0.0f, reflectDir.dot(viewDir));
                    float shininess = 32.0f; // 高光指数，值越大高光越集中
                    float specular = std::pow(specAngle, shininess);
                    float intensity = light.intensity * std::max(0.0f, normal.dot(lightDir)) + specular + 0.1f;
                    intensity = std::min(intensity, 1.0f); // Clamp to [0, 1]
                    color.head<3>() = color.head<3>().array() * light.color.array() * intensity;
                    renderTarget.data[index * 4 + 0] = static_cast<uint8_t>(color.x() * 255);
                    renderTarget.data[index * 4 + 1] = static_cast<uint8_t>(color.y() * 255);
                    renderTarget.data[index * 4 + 2] = static_cast<uint8_t>(color.z() * 255);
                    renderTarget.data[index * 4 + 3] = static_cast<uint8_t>(color.w() * 255);
                }
            }
        }
    }
}

Texture RenderScene(Scene& scene, int width, int height) {
    // Create a texture to hold the rendered image
    Texture texture;
    texture.width = width;
    texture.height = height;
    texture.data.resize(width * height * 4); // Assuming RGBA format

    // Clear the texture with a solid color (e.g., black)
    std::fill(texture.data.begin(), texture.data.end(), 0);

    DepthBuffer depthBuffer;
    depthBuffer.width = width;
    depthBuffer.height = height;
    depthBuffer.data.resize(width * height, std::numeric_limits<float>::max());

    float fovY = 45.0f * (float)M_PI / 180.0f;  // 垂直视角，单位：弧度
    float aspect = width / (float)height;
    float near = scene.camera.nearPlane;
    float far =  scene.camera.farPlane;
    float tanHalfFovy = tan(fovY / 2.0f);

    Eigen::Matrix4f projectionMatrix = (Eigen::Matrix4f() << 
        1.0f / (aspect * tanHalfFovy), 0, 0, 0,
        0, 1.0f / tanHalfFovy, 0, 0,
        0, 0, -(far + near) / (far - near), -2.0f * far * near / (far - near),
        0, 0, -1.0f, 0
    ).finished();

    //float left = -4.0f * aspect;
    //float right = 4.0f * aspect;
    //float bottom = -4.0f;
    //float top = 4.0f;
////
    //Eigen::Matrix4f projectionMatrix = (Eigen::Matrix4f() << 
    //    2.0f / (right - left), 0, 0, -(right + left) / (right - left),
    //    0, 2.0f / (top - bottom), 0, -(top + bottom) / (top - bottom),
    //    0, 0, -2.0f / (far - near), -(far + near) / (far - near),
    //    0, 0, 0, 1
    //).finished();

    Eigen::Matrix4f viewMatrix = scene.camera.getViewMatrix();

    Eigen::Matrix4f vpMatrix = projectionMatrix * viewMatrix;

    // Iterate over each object in the scene
    for (const auto& object : scene.objects) {
        for (const auto& mesh : object.meshes) {
            // Iterate over every triangle
            for (size_t i = 0; i < mesh.indices.size(); i += 3) {
                // Get the vertices of the triangle
                const Vertex& v0 = mesh.vertices[mesh.indices[i]];
                const Vertex& v1 = mesh.vertices[mesh.indices[i + 1]];
                const Vertex& v2 = mesh.vertices[mesh.indices[i + 2]];

                // Get model matrix for the object
                Eigen::Matrix4f modelMatrix = Eigen::Affine3f(
                    Eigen::Translation3f(object.transform.position) *
                    object.transform.rotation.toRotationMatrix() *
                    Eigen::Scaling(object.transform.scale)
                ).matrix();

                // Combine model, view, and projection matrices
                Eigen::Matrix4f mvpMatrix = vpMatrix * modelMatrix;

                // Transform the vertices to screen space
                Eigen::Vector4f p0 = mvpMatrix * Eigen::Vector4f(v0.position.x(), v0.position.y(), v0.position.z(), 1.0f);
                Eigen::Vector4f p1 = mvpMatrix * Eigen::Vector4f(v1.position.x(), v1.position.y(), v1.position.z(), 1.0f);
                Eigen::Vector4f p2 = mvpMatrix * Eigen::Vector4f(v2.position.x(), v2.position.y(), v2.position.z(), 1.0f);

                Eigen::Vector4f n0 = mvpMatrix * Eigen::Vector4f(v0.normal.x(), v0.normal.y(), v0.normal.z(), 0.0f);
                Eigen::Vector4f n1 = mvpMatrix * Eigen::Vector4f(v1.normal.x(), v1.normal.y(), v1.normal.z(), 0.0f);
                Eigen::Vector4f n2 = mvpMatrix * Eigen::Vector4f(v2.normal.x(), v2.normal.y(), v2.normal.z(), 0.0f);

                // Convert to screen coordinates
                p0 /= p0.w();
                p1 /= p1.w();
                p2 /= p2.w();

                Eigen::Vector4f* points[] = { &p0, &p1, &p2 };
                for (auto* p : points) {
                    p->x() = (p->x() + 1.0f) * 0.5f * width;
                    p->y() = (1.0f - p->y()) * 0.5f * height;
                }

                // Draw the triangle (you can use your own rasterization method here)
                drawFilledTriangle(texture, depthBuffer,
                    p0, p1, p2,
                    v0.color, v1.color, v2.color,
                    n0, n1, n2,
                    v0.position, v1.position, v2.position,
                    v0.uv, v1.uv, v2.uv,
                    mesh.texture,
                    scene.lights[0], // Assuming a single directional light,
                    scene.camera.position // Camera position for lighting calculations
                );
            }
        }
    }

    return texture;
}

struct MeshAccel {
    const Mesh* mesh;                     // 指向原 mesh
    std::vector<BVHNode> bvhNodes;        // 当前 mesh 的 BVH
    std::vector<TriangleRef> triangleRefs;
    int rootNode = -1;                // BVH 的根节点索引
};

bool intersectRayTriangle(
    const Eigen::Vector3f& orig, const Eigen::Vector3f& dir,
    const Eigen::Vector3f& v0, const Eigen::Vector3f& v1, const Eigen::Vector3f& v2,
    float& t, float& u, float& v) 
{
    const float EPSILON = 1e-6f;
    Eigen::Vector3f edge1 = v1 - v0;
    Eigen::Vector3f edge2 = v2 - v0;
    Eigen::Vector3f h = dir.cross(edge2);
    float a = edge1.dot(h);
    if (fabs(a) < EPSILON)
        return false;
    float f = 1.0f / a;
    Eigen::Vector3f s = orig - v0;
    u = f * s.dot(h);
    if (u < 0.0f || u > 1.0f)
        return false;
    Eigen::Vector3f q = s.cross(edge1);
    v = f * dir.dot(q);
    if (v < 0.0f || u + v > 1.0f)
        return false;
    t = f * edge2.dot(q);
    return t > EPSILON;
}

bool intersectBVH(
    const std::vector<BVHNode>& nodes,
    const int rootNode,
    const std::vector<Vertex>& vertices,
    const std::vector<uint32_t>& indices,
    const std::vector<TriangleRef>& triangles,
    const Ray& ray,
    float& outT, float& outU, float& outV, int& hitTriangleIndex
) {
    struct StackEntry { int nodeIdx; };
    std::stack<StackEntry> stack;
    stack.push({rootNode});

    bool hit = false;
    outT = std::numeric_limits<float>::max();
    hitTriangleIndex = -1;

    while (!stack.empty()) {
        int nodeIdx = stack.top().nodeIdx;
        stack.pop();
        const BVHNode& node = nodes[nodeIdx];

        float tMin = 0.0f, tMax = outT;
        if (!node.bounds.intersect(ray, tMin, tMax))
            continue;

        if (node.isLeaf()) {
            for (int i = node.start; i < node.start + node.count; ++i) {
                const TriangleRef& tri = triangles[i];
                float t, u, v;
                if (intersectRayTriangle(ray.origin, ray.dir, 
                    vertices[indices[tri.triangleIndex * 3 + 0]].position,
                    vertices[indices[tri.triangleIndex * 3 + 1]].position,
                    vertices[indices[tri.triangleIndex * 3 + 2]].position,
                    t, u, v)
                ) {
                    if (t < outT && t > 1e-4f) {
                        hit = true;
                        outT = t;
                        outU = u;
                        outV = v;
                        hitTriangleIndex = tri.triangleIndex;
                    }
                }
            }
        } else {
            stack.push({node.left});
            stack.push({node.right});
        }
    }

    return hit;
}

//Eigen::Vector3f traceRay(
//    const Ray& ray,
//    const Scene& scene,
//    const std::vector<BVHNode>& nodes,
//    const std::vector<TriangleRef>& refs,
//    int depth
//) {
//    if (depth > 3) return Eigen::Vector3f(0, 0, 0);  // 最多反射 3 次
//
//    float t, u, v;
//    TriangleRef hitTri;
//    const Mesh* hitMesh = nullptr;
//
//    bool hit = intersectBVH(ray, scene, nodes, refs, t, u, v, hitTri);
//
//    if (!hit) return Eigen::Vector3f(0.0f, 0.0f, 0.0f);  // 没命中，返回背景色
//
//    // === 命中后的插值计算 ===
//    int triId = hitTri.triangleIndex;
//    const auto& indices = hitMesh->indices;
//    const auto& vertices = hitMesh->vertices;
//
//    const Vertex& v0 = vertices[indices[3 * triId + 0]];
//    const Vertex& v1 = vertices[indices[3 * triId + 1]];
//    const Vertex& v2 = vertices[indices[3 * triId + 2]];
//    float w = 1 - u - v;
//
//    Eigen::Vector3f pos = ray.origin + t * ray.dir;
//    Eigen::Vector3f normal = (w * v0.normal + u * v1.normal + v * v2.normal).normalized();
//    Eigen::Vector3f color = (w * v0.color.head<3>() + u * v1.color.head<3>() + v * v2.color.head<3>());
//
//    // === 光照 ===
//    Eigen::Vector3f lightDir = -scene.lights[0].direction.normalized();
//    float diff = std::max(0.0f, normal.dot(lightDir));
//    Eigen::Vector3f localColor = color * diff;
//
//    // === 反射光线 ===
//    Eigen::Vector3f reflectDir = reflect(ray.dir.normalized(), normal).normalized();
//    Ray reflectedRay(pos + 1e-4f * reflectDir, reflectDir);  // 避免自交
//
//    Eigen::Vector3f reflectedColor = traceRay(reflectedRay, scene, nodes, refs, depth + 1);
//
//    float reflectivity = 0.4f;  // 固定反射率，可以根据材质设置
//
//    return (1 - reflectivity) * localColor + reflectivity * reflectedColor;
//}

Texture RenderSceneRayTracing(Scene& scene, int width, int height) {
    Texture texture;
    texture.width = width;
    texture.height = height;
    texture.data.resize(width * height * 4, 0); // Clear to black

    Eigen::Vector3f cameraPos = scene.camera.position;
    Eigen::Matrix4f viewMatrix = scene.camera.getViewMatrix();
    Eigen::Matrix4f invView = viewMatrix.inverse();

    float fovY = 45.0f * (float)M_PI / 180.0f;
    float aspect = width / (float)height;
    float tanHalfFovY = tan(fovY / 2.0f);

    std::vector<MeshAccel> meshAccels;

    for (auto& object: scene.objects) {
        Eigen::Matrix4f modelMatrix = Eigen::Affine3f(
            Eigen::Translation3f(object.transform.position) *
            object.transform.rotation.toRotationMatrix() *
            Eigen::Scaling(object.transform.scale)
        ).matrix();
        for (const auto& mesh : object.meshes) {
            MeshAccel accel;
            accel.mesh = &mesh;
            std::vector<Eigen::Vector3f> positions;
            for (size_t i = 0; i < mesh.indices.size(); i += 3) {
                TriangleRef tri;
                tri.triangleIndex = i/3;
                Eigen::Vector3f v0 = mesh.vertices[mesh.indices[i]].position;
                Eigen::Vector3f v1 = mesh.vertices[mesh.indices[i + 1]].position;
                Eigen::Vector3f v2 = mesh.vertices[mesh.indices[i + 2]].position;
                tri.centroid = (v0 + v1 + v2) / 3.0f;
                accel.triangleRefs.push_back(tri);
            }
            accel.rootNode = buildBVH(
                mesh.vertices,
                mesh.indices,
                accel.bvhNodes,
                accel.triangleRefs,
                0,
                mesh.indices.size() / 3
            );
            meshAccels.push_back(accel);
        }
    }

    // 遍历每个像素
    std::vector<int> indices(width * height);
    std::iota(indices.begin(), indices.end(), 0);

    std::for_each(std::execution::par, indices.begin(), indices.end(), [&](int loopIndex) {
            int x = loopIndex % width;
            int y = loopIndex / width;
            // 将屏幕像素坐标映射到 [-1, 1]
            float px = (2.0f * (x + 0.5f) / width - 1.0f) * aspect * tanHalfFovY;
            float py = (1.0f - 2.0f * (y + 0.5f) / height) * tanHalfFovY;

            // 构造光线（摄像机坐标系 -> 世界坐标系）
            Eigen::Vector3f rayDir = (invView.block<3,3>(0,0) * Eigen::Vector3f(px, py, -1.0f)).normalized();

            // 与场景求交
            Eigen::Vector3f hitColor(0, 0, 0);
            float minT = std::numeric_limits<float>::max();
            for (const auto& object : scene.objects) {
                Eigen::Matrix4f modelMatrix = Eigen::Affine3f(
                    Eigen::Translation3f(object.transform.position) *
                    object.transform.rotation.toRotationMatrix() *
                    Eigen::Scaling(object.transform.scale)
                ).matrix();
                Eigen::Matrix4f invModel = modelMatrix.inverse();
//
                for (const auto& accel : meshAccels) {
                    // 把 ray 从世界空间变换到当前 mesh 局部空间
                    Eigen::Vector3f rayOriginLocal = (invModel * cameraPos.homogeneous()).hnormalized();
                    Eigen::Vector3f rayDirLocal = (invModel * rayDir.homogeneous()).hnormalized();
                
                    // 遍历 BVH
                    float t, u, v;
                    int triangleIndex = -1;
                    Ray ray(rayOriginLocal, rayDirLocal);
                    if (intersectBVH(accel.bvhNodes, accel.rootNode, accel.mesh->vertices, accel.mesh->indices,
                        accel.triangleRefs, ray, t, u, v, triangleIndex)) {
                        if (t < minT) {
                            minT = t;
                        
                            // 获取三角形信息
                            const TriangleRef& tri = accel.triangleRefs[triangleIndex];
                            const Vertex& v0 = accel.mesh->vertices[accel.mesh->indices[triangleIndex*3+0]];
                            const Vertex& v1 = accel.mesh->vertices[accel.mesh->indices[triangleIndex*3+1]];
                            const Vertex& v2 = accel.mesh->vertices[accel.mesh->indices[triangleIndex*3+2]];
                        
                            // 插值颜色
                            Eigen::Vector3f diffuse(1.0f, 1.0f, 1.0f);
                            if (accel.mesh->texture != nullptr) {
                                float texU = v0.uv.x() * (1 - u - v) + v1.uv.x() * u + v2.uv.x() * v;
                                float texV = v0.uv.y() * (1 - u - v) + v1.uv.y() * u + v2.uv.y() * v;
                                int texX = static_cast<int>(texU * accel.mesh->texture->width);
                                int texY = static_cast<int>(texV * accel.mesh->texture->height);
                                texX = std::clamp(texX, 0, accel.mesh->texture->width - 1);
                                texY = std::clamp(texY, 0, accel.mesh->texture->height - 1);
                                int offset = (texY * accel.mesh->texture->width + texX) * 4;
                                diffuse = Eigen::Vector3f(
                                    accel.mesh->texture->data[offset + 2] / 255.0f,
                                    accel.mesh->texture->data[offset + 1] / 255.0f,
                                    accel.mesh->texture->data[offset + 0] / 255.0f
                                );
                            }
                        
                            // 插值法线，转回世界空间
                            Eigen::Vector3f n = ((1 - u - v) * v0.normal + u * v1.normal + v * v2.normal).normalized();
                            n = (modelMatrix * n.homogeneous()).hnormalized();  // 回到世界空间
                        
                            // 光照
                            Eigen::Vector3f lightDir = -scene.lights[0].direction.normalized();
                            float intensity = std::max(0.0f, n.dot(lightDir));
                            Eigen::Vector3f shading = intensity * Eigen::Vector3f(1, 1, 1);
                            hitColor = diffuse.cwiseProduct(shading);
                        }
                    }
                    //for (int i=0;i<object.meshes[0].indices.size();i+=3){
                    //    const Vertex& v0 = object.meshes[0].vertices[object.meshes[0].indices[i]];
                    //    const Vertex& v1 = object.meshes[0].vertices[object.meshes[0].indices[i+1]];
                    //    const Vertex& v2 = object.meshes[0].vertices[object.meshes[0].indices[i+2]];
                    //    if (intersectRayTriangle(
                    //        rayOriginLocal, rayDirLocal,
                    //        v0.position, v1.position, v2.position,
                    //        t, u, v
                    //    )) {
                    //        if (t < minT) {
                    //            minT = t;
                    //            hitColor = Eigen::Vector3f(1, 0, 0); // Red color for hit
                    //        }
                    //    }
                    //}
                }
//
            }

            // 写入颜色
            int idx = (y * width + x) * 4;
            texture.data[idx + 0] = std::min(255.0f, hitColor.x() * 255);
            texture.data[idx + 1] = std::min(255.0f, hitColor.y() * 255);
            texture.data[idx + 2] = std::min(255.0f, hitColor.z() * 255);
            texture.data[idx + 3] = 255;
    });

    return texture;
}
