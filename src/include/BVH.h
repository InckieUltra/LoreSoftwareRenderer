#pragma once
#include <Eigen/Core>
#include "Primitives.h"

struct Ray {
    Ray(const Eigen::Vector3f& origin, const Eigen::Vector3f& dir)
        : origin(origin), dir(dir.normalized()) {}

    Eigen::Vector3f origin;  // 光线起点
    Eigen::Vector3f dir;     // 光线方向（应当单位化）
    float tMin = 0.001f;     // 可选：起始 t，用于避免自交
    float tMax = std::numeric_limits<float>::max(); // 可选：最大距离
};


struct AABB {
    Eigen::Vector3f min = Eigen::Vector3f::Constant( std::numeric_limits<float>::max() );
    Eigen::Vector3f max = Eigen::Vector3f::Constant(-std::numeric_limits<float>::max());

    void expand(const Eigen::Vector3f& p) {
        min = min.cwiseMin(p);
        max = max.cwiseMax(p);
    }

    void expand(const AABB& box) {
        expand(box.min);
        expand(box.max);
    }

    bool intersect(const Ray& ray, float& tMin, float& tMax) const {
        // 使用 slab method 实现
        for (int i = 0; i < 3; ++i) {
            float invD = 1.0f / ray.dir[i];
            float t0 = (min[i] - ray.origin[i]) * invD;
            float t1 = (max[i] - ray.origin[i]) * invD;
            if (invD < 0.0f) std::swap(t0, t1);
            tMin = std::max(tMin, t0);
            tMax = std::min(tMax, t1);
            if (tMax < tMin) return false;
        }
        return true;
    }

    int longestAxis() const {
        Eigen::Vector3f diag = max - min;
        if (diag[0] >= diag[1] && diag[0] >= diag[2]) return 0;
        if (diag[1] >= diag[0] && diag[1] >= diag[2]) return 1;
        return 2;
    }
};

struct BVHNode {
    AABB bounds;
    int left = -1, right = -1;
    int start = 0, count = 0; // 三角形索引范围（单位是 triangle 数）
    bool isLeaf() const { return left == -1; }
};

struct TriangleRef {
    int triangleIndex; // 即 i/3 in mesh.indices
    Eigen::Vector3f centroid;
};

int buildBVH(const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices,
             std::vector<BVHNode>& nodes, std::vector<TriangleRef>& refs,
             int start, int end
) {
    BVHNode node;
    AABB bounds;
    AABB centroidBounds;

    for (int i = start; i < end; ++i) {
        int idx = refs[i].triangleIndex;
        Eigen::Vector3f v0 = vertices[indices[3 * idx + 0]].position;
        Eigen::Vector3f v1 = vertices[indices[3 * idx + 1]].position;
        Eigen::Vector3f v2 = vertices[indices[3 * idx + 2]].position;

        bounds.expand(v0); bounds.expand(v1); bounds.expand(v2);
        centroidBounds.expand(refs[i].centroid);
    }

    node.bounds = bounds;

    int triCount = end - start;
    if (triCount <= 4) {
        node.start = start;
        node.count = triCount;
        nodes.push_back(node);
        return nodes.size() - 1;
    }

    int axis = centroidBounds.longestAxis();
    int mid = (start + end) / 2;

    std::nth_element(refs.begin() + start, refs.begin() + mid, refs.begin() + end,
                     [axis](const TriangleRef& a, const TriangleRef& b) {
                         return a.centroid[axis] < b.centroid[axis];
                     });

    int left =  buildBVH(vertices, indices, nodes, refs, start, mid);
    int right = buildBVH(vertices, indices, nodes, refs, mid, end);

    node.left = left;
    node.right = right;
    nodes.push_back(node);
    return nodes.size() - 1;
}