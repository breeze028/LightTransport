#pragma once

#include "lt/material.h"
#include "lt/math.h"

#include <memory>
#include <string>
#include <vector>

namespace lt {

struct LightComponent {
    bool enabled = false;
    Vec3 color = {1.0f, 1.0f, 1.0f};
    float intensity = 1.0f;
};

struct Mesh {
    std::string name = "Mesh";
    std::vector<Vec3> vertices;
    std::vector<Vec2> texcoords;
    std::vector<uint32_t> indices;
    int material = 0;
    Vec3 translation = {};
    Vec3 rotation = {};
    Vec3 scale = {1.0f, 1.0f, 1.0f};
    LightComponent light;
};

struct Camera {
    Vec3 position = {0.0f, 1.0f, 4.0f};
    Vec3 target = {0.0f, 1.0f, 0.0f};
    Vec3 up = {0.0f, 1.0f, 0.0f};
    float fov_degrees = 45.0f;
};

struct Scene {
    Scene() = default;
    Scene(const Scene& other);
    Scene& operator=(const Scene& other);
    Scene(Scene&&) noexcept = default;
    Scene& operator=(Scene&&) noexcept = default;

    Camera camera;
    std::vector<std::shared_ptr<Material>> materials;
    std::vector<std::shared_ptr<Texture>> textures;
    std::vector<Mesh> meshes;
};

struct Triangle {
    Vec3 v0;
    Vec3 v1;
    Vec3 v2;
    Vec3 normal;
    Vec3 centroid;
    Vec2 uv0;
    Vec2 uv1;
    Vec2 uv2;
    int material = 0;
    int mesh = -1;
};

struct Aabb {
    Vec3 min = {kInfinity, kInfinity, kInfinity};
    Vec3 max = {-kInfinity, -kInfinity, -kInfinity};
};

struct BvhNode {
    Aabb bounds;
    int left = -1;
    int right = -1;
    int first = 0;
    int count = 0;
};

struct RenderScene {
    std::vector<Triangle> triangles;
    std::vector<int> triangle_indices;
    std::vector<int> light_triangle_indices;
    std::vector<BvhNode> bvh_nodes;
};

struct SceneLoadResult {
    Scene scene;
    std::string error;
};

Scene make_default_scene();
SceneLoadResult load_scene(const std::string& path);
bool save_scene(const Scene& scene, const std::string& path, std::string& error);
int find_material(const Scene& scene, const std::string& name);
RenderScene build_render_scene(const Scene& scene);
Mesh make_cube_mesh(const std::string& name, int material, Vec3 translation, float scale);
Mesh make_uv_sphere_mesh(const std::string& name, int material, Vec3 translation, float radius, int segments = 24, int rings = 12);
Mesh make_quad_mesh(const std::string& name, int material, Vec3 a, Vec3 b, Vec3 c, Vec3 d);

} // namespace lt
