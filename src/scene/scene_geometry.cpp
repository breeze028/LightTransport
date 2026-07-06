#include "scene_internal.h"

#include <algorithm>
#include <cmath>

namespace lt::scene_detail {
namespace {

void expand(Aabb& bounds, Vec3 point) {
    bounds.min = min(bounds.min, point);
    bounds.max = max(bounds.max, point);
}

} // namespace

void add_triangle(Mesh& mesh, uint32_t a, uint32_t b, uint32_t c) {
    mesh.indices.push_back(a);
    mesh.indices.push_back(b);
    mesh.indices.push_back(c);
}

Vec2 default_uv(Vec3 vertex) {
    return {vertex.x + 0.5f, vertex.z + 0.5f};
}

void ensure_default_uvs(Mesh& mesh) {
    if (mesh.texcoords.size() == mesh.vertices.size()) {
        return;
    }
    mesh.texcoords.resize(mesh.vertices.size());
    for (size_t i = 0; i < mesh.vertices.size(); ++i) {
        mesh.texcoords[i] = default_uv(mesh.vertices[i]);
    }
}

Vec3 rotate_xyz(Vec3 point, Vec3 rotation) {
    const float cx = std::cos(rotation.x);
    const float sx = std::sin(rotation.x);
    const float cy = std::cos(rotation.y);
    const float sy = std::sin(rotation.y);
    const float cz = std::cos(rotation.z);
    const float sz = std::sin(rotation.z);
    point = {point.x, point.y * cx - point.z * sx, point.y * sx + point.z * cx};
    point = {point.x * cy + point.z * sy, point.y, -point.x * sy + point.z * cy};
    point = {point.x * cz - point.y * sz, point.x * sz + point.y * cz, point.z};
    return point;
}

void center_mesh_origin(Mesh& mesh) {
    if (mesh.vertices.empty()) {
        return;
    }

    Aabb bounds;
    for (Vec3 vertex : mesh.vertices) {
        expand(bounds, vertex);
    }
    const Vec3 local_center = (bounds.min + bounds.max) * 0.5f;
    if (length(local_center) <= 1.0e-6f) {
        return;
    }

    for (Vec3& vertex : mesh.vertices) {
        vertex = vertex - local_center;
    }
    mesh.translation = mesh.translation + rotate_xyz(local_center * mesh.scale, mesh.rotation);
}

bool has_light_emission(Vec3 emission) {
    return emission.x > 0.0f || emission.y > 0.0f || emission.z > 0.0f;
}

LightComponent make_light_from_emission(Vec3 emission) {
    LightComponent light;
    light.intensity = std::max({emission.x, emission.y, emission.z});
    light.enabled = light.intensity > 0.0f;
    light.double_sided = false;
    light.color = light.enabled ? emission / light.intensity : Vec3{1.0f, 1.0f, 1.0f};
    return light;
}

} // namespace lt::scene_detail

namespace lt {

Mesh make_quad_mesh(const std::string& name, int material, Vec3 a, Vec3 b, Vec3 c, Vec3 d) {
    Mesh mesh;
    mesh.name = name;
    mesh.material = material;
    mesh.vertices = {a, b, c, d};
    mesh.texcoords = {{0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f}};
    mesh.indices = {0, 1, 2, 0, 2, 3};
    scene_detail::center_mesh_origin(mesh);
    return mesh;
}

Mesh make_cube_mesh(const std::string& name, int material, Vec3 translation, float scale) {
    Mesh mesh;
    mesh.name = name;
    mesh.material = material;
    mesh.translation = translation;
    mesh.scale = Vec3{scale};
    mesh.vertices = {
        {-0.5f, -0.5f, -0.5f}, {0.5f, -0.5f, -0.5f}, {0.5f, 0.5f, -0.5f}, {-0.5f, 0.5f, -0.5f},
        {-0.5f, -0.5f, 0.5f}, {0.5f, -0.5f, 0.5f}, {0.5f, 0.5f, 0.5f}, {-0.5f, 0.5f, 0.5f},
    };
    scene_detail::ensure_default_uvs(mesh);
    const uint32_t faces[] = {
        0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4,
        3, 6, 2, 3, 7, 6, 1, 2, 6, 1, 6, 5, 0, 4, 7, 0, 7, 3,
    };
    mesh.indices.assign(std::begin(faces), std::end(faces));
    return mesh;
}

Mesh make_uv_sphere_mesh(const std::string& name, int material, Vec3 translation, float radius, int segments, int rings) {
    Mesh mesh;
    mesh.name = name;
    mesh.material = material;
    mesh.translation = translation;
    mesh.scale = Vec3{radius};
    for (int y = 0; y <= rings; ++y) {
        const float v = static_cast<float>(y) / static_cast<float>(rings);
        const float theta = v * kPi;
        for (int x = 0; x <= segments; ++x) {
            const float u = static_cast<float>(x) / static_cast<float>(segments);
            const float phi = u * 2.0f * kPi;
            mesh.vertices.push_back({std::sin(theta) * std::cos(phi), std::cos(theta), std::sin(theta) * std::sin(phi)});
            mesh.texcoords.push_back({u, 1.0f - v});
        }
    }
    for (int y = 0; y < rings; ++y) {
        for (int x = 0; x < segments; ++x) {
            const uint32_t a = static_cast<uint32_t>(y * (segments + 1) + x);
            const uint32_t b = a + 1;
            const uint32_t c = static_cast<uint32_t>((y + 1) * (segments + 1) + x + 1);
            const uint32_t d = static_cast<uint32_t>((y + 1) * (segments + 1) + x);
            scene_detail::add_triangle(mesh, a, b, d);
            scene_detail::add_triangle(mesh, b, c, d);
        }
    }
    return mesh;
}

Scene make_default_scene() {
    Scene scene;
    scene.uses_builtin_default_meshes = true;
    scene.camera = {{0.0f, 1.05f, -1.25f}, {0.0f, 0.85f, 0.0f}, {0.0f, 1.0f, 0.0f}, 45.0f};
    scene.materials = {
        make_material("white", {0.78f, 0.78f, 0.72f}, BrdfModel::Lambertian, 0.65f, 0.0f),
        make_material("red", {0.85f, 0.18f, 0.14f}, BrdfModel::Lambertian, 0.65f, 0.0f),
        make_material("green", {0.18f, 0.70f, 0.24f}, BrdfModel::Lambertian, 0.65f, 0.0f),
        make_material("blue", {0.22f, 0.32f, 0.90f}, BrdfModel::Principled, 0.35f, 0.0f),
        make_material("light", {1.0f, 0.9f, 0.72f}, BrdfModel::Lambertian, 0.5f, 0.0f),
    };
    scene.meshes.push_back(make_quad_mesh("Floor", 0, {-1.4f, 0.0f, -1.5f}, {1.4f, 0.0f, -1.5f}, {1.4f, 0.0f, 1.5f}, {-1.4f, 0.0f, 1.5f}));
    scene.meshes.push_back(make_quad_mesh("Ceiling", 0, {-1.4f, 2.2f, 1.5f}, {1.4f, 2.2f, 1.5f}, {1.4f, 2.2f, -1.5f}, {-1.4f, 2.2f, -1.5f}));
    scene.meshes.push_back(make_quad_mesh("Left Wall", 1, {-1.4f, 0.0f, 1.5f}, {-1.4f, 2.2f, 1.5f}, {-1.4f, 2.2f, -1.5f}, {-1.4f, 0.0f, -1.5f}));
    scene.meshes.push_back(make_quad_mesh("Right Wall", 2, {1.4f, 0.0f, -1.5f}, {1.4f, 2.2f, -1.5f}, {1.4f, 2.2f, 1.5f}, {1.4f, 0.0f, 1.5f}));
    scene.meshes.push_back(make_quad_mesh("Back Wall", 0, {-1.4f, 0.0f, 1.5f}, {1.4f, 0.0f, 1.5f}, {1.4f, 2.2f, 1.5f}, {-1.4f, 2.2f, 1.5f}));
    scene.meshes.push_back(make_quad_mesh("Area_Light", 4, {-0.28f, 2.18f, 0.28f}, {-0.28f, 2.18f, -0.28f}, {0.28f, 2.18f, -0.28f}, {0.28f, 2.18f, 0.28f}));
    scene.meshes.back().light = {true, false, {1.0f, 0.888889f, 0.666667f}, 9.0f};
    scene.spheres.push_back({"Blue_Sphere", 3, {-0.55f, 0.42f, 0.0f}, 0.42f});
    scene.meshes.push_back(make_cube_mesh("White Block", 0, {0.48f, 0.32f, -0.55f}, 0.62f));
    return scene;
}

int find_material(const Scene& scene, const std::string& name) {
    for (int i = 0; i < static_cast<int>(scene.materials.size()); ++i) {
        if (scene.materials[static_cast<size_t>(i)] && scene.materials[static_cast<size_t>(i)]->name == name) {
            return i;
        }
    }
    return -1;
}

} // namespace lt
