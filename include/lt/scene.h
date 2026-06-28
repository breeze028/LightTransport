#pragma once

#include "lt/material.h"
#include "lt/math.h"

#include <memory>
#include <string>
#include <vector>

namespace lt {

struct LightComponent {
    bool enabled = false;
    bool double_sided = false;
    Vec3 color = {1.0f, 1.0f, 1.0f};
    float intensity = 1.0f;
};

struct Mesh {
    std::string name = "Mesh";
    std::vector<Vec3> vertices;
    std::vector<Vec3> normals;
    std::vector<Vec2> texcoords;
    std::vector<uint32_t> indices;
    int material = 0;
    Vec3 translation = {};
    Vec3 rotation = {};
    Vec3 scale = {1.0f, 1.0f, 1.0f};
    LightComponent light;
};

struct Sphere {
    std::string name = "Sphere";
    int material = 0;
    Vec3 center = {};
    float radius = 0.5f;
};

struct Camera {
    Vec3 position = {0.0f, 1.0f, 4.0f};
    Vec3 target = {0.0f, 1.0f, 0.0f};
    Vec3 up = {0.0f, 1.0f, 0.0f};
    float fov_degrees = 45.0f;
    float right_sign = 1.0f;
};

struct Environment {
    enum class Mapping { Equirectangular = 0, EqualArea = 1 };

    Vec3 color = {1.0f, 1.0f, 1.0f};
    float strength = 1.0f;
    std::shared_ptr<Texture> texture;
    bool constant = false;
    Mapping mapping = Mapping::Equirectangular;
    Vec3 light_from_world_x = {1.0f, 0.0f, 0.0f};
    Vec3 light_from_world_y = {0.0f, 1.0f, 0.0f};
    Vec3 light_from_world_z = {0.0f, 0.0f, 1.0f};
};

struct SceneRenderSettings {
    int stylized_samples = 8;
    int stylized_max_depth = 1;
    bool use_irradiance_volume = false;
    int irradiance_volume_grid_resolution = 7;
    int irradiance_volume_subgrid_resolution = 3;
    int irradiance_volume_direction_resolution = 9;
    int irradiance_volume_bake_samples = 1;
    int irradiance_volume_bake_bounces = 4;
    float irradiance_volume_bounds_inset = 0.01f;
    bool irradiance_volume_principled_gi = false;
    bool irradiance_volume_debug_probes = false;
    float irradiance_volume_debug_probe_radius_scale = 0.10f;
    bool irradiance_volume_cache_enabled = true;
    bool irradiance_volume_auto_update = true;
    bool irradiance_volume_manual_bounds = false;
    Vec3 irradiance_volume_bounds_min = {-1.0f, -1.0f, -1.0f};
    Vec3 irradiance_volume_bounds_max = {1.0f, 1.0f, 1.0f};
};

struct Scene {
    Scene() = default;
    Scene(const Scene& other);
    Scene& operator=(const Scene& other);
    Scene(Scene&&) noexcept = default;
    Scene& operator=(Scene&&) noexcept = default;

    Camera camera;
    Environment environment;
    SceneRenderSettings render_settings;
    bool has_render_settings = false;
    bool uses_builtin_default_meshes = false;
    std::vector<std::shared_ptr<Material>> materials;
    std::vector<std::shared_ptr<Texture>> textures;
    std::vector<Mesh> meshes;
    std::vector<Sphere> spheres;
};

struct Triangle {
    Vec3 v0;
    Vec3 v1;
    Vec3 v2;
    Vec3 normal;
    Vec3 n0;
    Vec3 n1;
    Vec3 n2;
    Vec3 tangent;
    Vec3 bitangent;
    Vec3 centroid;
    Vec2 uv0;
    Vec2 uv1;
    Vec2 uv2;
    int material = 0;
    int mesh = -1;
};

struct RenderSphere {
    Vec3 center;
    float radius = 0.5f;
    int material = 0;
    int sphere = -1;
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
    struct MeshInstance {
        Aabb bounds;
        int bvh_root = -1;
        int mesh = -1;
    };

    std::vector<Triangle> triangles;
    std::vector<RenderSphere> spheres;
    std::vector<int> triangle_indices;
    std::vector<int> flat_triangle_indices;
    std::vector<int> light_triangle_indices;
    std::vector<BvhNode> bvh_nodes;
    std::vector<BvhNode> flat_bvh_nodes;
    std::vector<MeshInstance> mesh_instances;
    std::vector<int> mesh_instance_indices;
    std::vector<BvhNode> tlas_nodes;
};

struct SceneLoadResult {
    Scene scene;
    std::string error;
};

Scene make_default_scene();
SceneLoadResult load_scene(const std::string& path);
SceneLoadResult load_gltf_scene(const std::string& path);
SceneLoadResult load_pbrt_scene(const std::string& path);
bool save_scene(const Scene& scene, const std::string& path, std::string& error);
int find_material(const Scene& scene, const std::string& name);
RenderScene build_render_scene(const Scene& scene);
Mesh make_cube_mesh(const std::string& name, int material, Vec3 translation, float scale);
Mesh make_uv_sphere_mesh(const std::string& name, int material, Vec3 translation, float radius, int segments = 24, int rings = 12);
Mesh make_quad_mesh(const std::string& name, int material, Vec3 a, Vec3 b, Vec3 c, Vec3 d);

} // namespace lt
