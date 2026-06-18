#include "lt/scene.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <functional>
#include <numeric>
#include <sstream>
#include <memory>
#include <unordered_map>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#include <urlmon.h>
#endif

namespace lt {
namespace {

void add_triangle(Mesh& mesh, uint32_t a, uint32_t b, uint32_t c) {
    mesh.indices.push_back(a);
    mesh.indices.push_back(b);
    mesh.indices.push_back(c);
}

Vec2 default_uv(Vec3 v) {
    return {v.x + 0.5f, v.z + 0.5f};
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

Aabb triangle_bounds(const Triangle& tri) {
    constexpr float kBoundsEpsilon = 1.0e-4f;
    Aabb bounds;
    bounds.min = min(tri.v0, min(tri.v1, tri.v2));
    bounds.max = max(tri.v0, max(tri.v1, tri.v2));
    bounds.min = bounds.min - Vec3{kBoundsEpsilon};
    bounds.max = bounds.max + Vec3{kBoundsEpsilon};
    return bounds;
}

void expand(Aabb& bounds, Vec3 p) {
    bounds.min = min(bounds.min, p);
    bounds.max = max(bounds.max, p);
}

void expand(Aabb& bounds, const Aabb& other) {
    expand(bounds, other.min);
    expand(bounds, other.max);
}

float axis_value(Vec3 v, int axis) {
    return axis == 0 ? v.x : axis == 1 ? v.y : v.z;
}

int build_bvh_recursive(RenderScene& render_scene, std::vector<int>& indices, std::vector<BvhNode>& nodes, int first, int count) {
    BvhNode node;
    for (int i = first; i < first + count; ++i) {
        expand(node.bounds, triangle_bounds(render_scene.triangles[static_cast<size_t>(indices[static_cast<size_t>(i)])]));
    }

    const int node_index = static_cast<int>(nodes.size());
    nodes.push_back(node);

    if (count <= 4) {
        nodes[static_cast<size_t>(node_index)].first = first;
        nodes[static_cast<size_t>(node_index)].count = count;
        return node_index;
    }

    Aabb centroid_bounds;
    for (int i = first; i < first + count; ++i) {
        const Triangle& tri = render_scene.triangles[static_cast<size_t>(indices[static_cast<size_t>(i)])];
        expand(centroid_bounds, tri.centroid);
    }
    const Vec3 extent = centroid_bounds.max - centroid_bounds.min;
    int axis = 0;
    if (extent.y > extent.x && extent.y >= extent.z) {
        axis = 1;
    } else if (extent.z > extent.x && extent.z >= extent.y) {
        axis = 2;
    }

    const int mid = first + count / 2;
    std::nth_element(
        indices.begin() + first,
        indices.begin() + mid,
        indices.begin() + first + count,
        [&](int a, int b) {
            return axis_value(render_scene.triangles[static_cast<size_t>(a)].centroid, axis) <
                axis_value(render_scene.triangles[static_cast<size_t>(b)].centroid, axis);
        });

    const int left = build_bvh_recursive(render_scene, indices, nodes, first, mid - first);
    const int right = build_bvh_recursive(render_scene, indices, nodes, mid, first + count - mid);
    nodes[static_cast<size_t>(node_index)].left = left;
    nodes[static_cast<size_t>(node_index)].right = right;
    return node_index;
}

int build_tlas_recursive(RenderScene& render_scene, int first, int count) {
    BvhNode node;
    for (int i = first; i < first + count; ++i) {
        const int instance_index = render_scene.mesh_instance_indices[static_cast<size_t>(i)];
        if (instance_index >= 0 && instance_index < static_cast<int>(render_scene.mesh_instances.size())) {
            expand(node.bounds, render_scene.mesh_instances[static_cast<size_t>(instance_index)].bounds);
        }
    }

    const int node_index = static_cast<int>(render_scene.tlas_nodes.size());
    render_scene.tlas_nodes.push_back(node);

    if (count <= 2) {
        render_scene.tlas_nodes[static_cast<size_t>(node_index)].first = first;
        render_scene.tlas_nodes[static_cast<size_t>(node_index)].count = count;
        return node_index;
    }

    Aabb centroid_bounds;
    for (int i = first; i < first + count; ++i) {
        const int instance_index = render_scene.mesh_instance_indices[static_cast<size_t>(i)];
        const Aabb& bounds = render_scene.mesh_instances[static_cast<size_t>(instance_index)].bounds;
        expand(centroid_bounds, (bounds.min + bounds.max) * 0.5f);
    }
    const Vec3 extent = centroid_bounds.max - centroid_bounds.min;
    int axis = 0;
    if (extent.y > extent.x && extent.y >= extent.z) {
        axis = 1;
    } else if (extent.z > extent.x && extent.z >= extent.y) {
        axis = 2;
    }

    const int mid = first + count / 2;
    std::nth_element(
        render_scene.mesh_instance_indices.begin() + first,
        render_scene.mesh_instance_indices.begin() + mid,
        render_scene.mesh_instance_indices.begin() + first + count,
        [&](int a, int b) {
            const Aabb& bounds_a = render_scene.mesh_instances[static_cast<size_t>(a)].bounds;
            const Aabb& bounds_b = render_scene.mesh_instances[static_cast<size_t>(b)].bounds;
            return axis_value((bounds_a.min + bounds_a.max) * 0.5f, axis) <
                axis_value((bounds_b.min + bounds_b.max) * 0.5f, axis);
        });

    const int left = build_tlas_recursive(render_scene, first, mid - first);
    const int right = build_tlas_recursive(render_scene, mid, first + count - mid);
    render_scene.tlas_nodes[static_cast<size_t>(node_index)].left = left;
    render_scene.tlas_nodes[static_cast<size_t>(node_index)].right = right;
    return node_index;
}

int material_id(const std::unordered_map<std::string, int>& material_ids, const std::string& name) {
    const auto it = material_ids.find(name);
    return it == material_ids.end() ? -1 : it->second;
}

bool has_light_emission(Vec3 emission) {
    return emission.x > 0.0f || emission.y > 0.0f || emission.z > 0.0f;
}

LightComponent make_light_from_emission(Vec3 emission) {
    LightComponent light;
    light.intensity = std::max({emission.x, emission.y, emission.z});
    light.enabled = light.intensity > 0.0f;
    light.double_sided = true;
    light.color = light.enabled ? emission / light.intensity : Vec3{1.0f, 1.0f, 1.0f};
    return light;
}

bool parse_float_token(const std::string& token, float& value) {
    std::istringstream in(token);
    in >> value;
    return in && in.eof();
}

bool is_absolute_path(const std::string& path) {
    return (path.size() >= 2 && path[1] == ':') || (!path.empty() && (path[0] == '/' || path[0] == '\\'));
}

std::string parent_path(const std::string& path) {
    const size_t slash = path.find_last_of("\\/");
    return slash == std::string::npos ? std::string{} : path.substr(0, slash);
}

std::string join_path(const std::string& parent, const std::string& child) {
    if (parent.empty() || is_absolute_path(child)) {
        return child;
    }
    const char last = parent.back();
    return parent + ((last == '\\' || last == '/') ? "" : "\\") + child;
}

std::string lowercase_extension(const std::string& path) {
    const size_t dot = path.find_last_of('.');
    const size_t query = path.find_first_of("?#", dot == std::string::npos ? 0 : dot);
    std::string ext = dot == std::string::npos ? std::string{} : path.substr(dot, query == std::string::npos ? std::string::npos : query - dot);
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext;
}

bool is_url(const std::string& path) {
    return path.rfind("http://", 0) == 0 || path.rfind("https://", 0) == 0;
}

SceneLoadResult load_url_scene(const std::string& url) {
#if defined(_WIN32)
    char temp_dir[MAX_PATH] = {};
    if (GetTempPathA(MAX_PATH, temp_dir) == 0) {
        return {make_default_scene(), "Could not find temporary directory for URL scene cache"};
    }
    std::string ext = lowercase_extension(url);
    if (ext.empty()) {
        ext = ".glb";
    }
    const std::string cached = std::string(temp_dir) + "LightTransport_" + std::to_string(std::hash<std::string>{}(url)) + ext;
    const HRESULT hr = URLDownloadToFileA(nullptr, url.c_str(), cached.c_str(), 0, nullptr);
    if (FAILED(hr)) {
        return {make_default_scene(), "Could not download scene URL: " + url};
    }
    return load_scene(cached);
#else
    return {make_default_scene(), "URL scene loading is only implemented on Windows builds"};
#endif
}

Vec3 rotate_xyz(Vec3 p, Vec3 r) {
    const float cx = std::cos(r.x);
    const float sx = std::sin(r.x);
    const float cy = std::cos(r.y);
    const float sy = std::sin(r.y);
    const float cz = std::cos(r.z);
    const float sz = std::sin(r.z);
    p = {p.x, p.y * cx - p.z * sx, p.y * sx + p.z * cx};
    p = {p.x * cy + p.z * sy, p.y, -p.x * sy + p.z * cy};
    p = {p.x * cz - p.y * sz, p.x * sz + p.y * cz, p.z};
    return p;
}

void center_mesh_origin(Mesh& mesh) {
    if (mesh.vertices.empty()) {
        return;
    }

    Aabb bounds;
    for (Vec3 v : mesh.vertices) {
        expand(bounds, v);
    }
    const Vec3 local_center = (bounds.min + bounds.max) * 0.5f;
    if (length(local_center) <= 1.0e-6f) {
        return;
    }

    for (Vec3& v : mesh.vertices) {
        v = v - local_center;
    }
    mesh.translation = mesh.translation + rotate_xyz(local_center * mesh.scale, mesh.rotation);
}

} // namespace

Scene::Scene(const Scene& other) : camera(other.camera), environment(other.environment), uses_builtin_default_meshes(other.uses_builtin_default_meshes), meshes(other.meshes) {
    textures = other.textures;
    materials.reserve(other.materials.size());
    for (const std::shared_ptr<Material>& material : other.materials) {
        materials.push_back(material ? material->clone() : nullptr);
    }
}

Scene& Scene::operator=(const Scene& other) {
    if (this == &other) {
        return *this;
    }
    camera = other.camera;
    environment = other.environment;
    uses_builtin_default_meshes = other.uses_builtin_default_meshes;
    meshes = other.meshes;
    textures = other.textures;
    materials.clear();
    materials.reserve(other.materials.size());
    for (const std::shared_ptr<Material>& material : other.materials) {
        materials.push_back(material ? material->clone() : nullptr);
    }
    return *this;
}

Mesh make_quad_mesh(const std::string& name, int material, Vec3 a, Vec3 b, Vec3 c, Vec3 d) {
    Mesh mesh;
    mesh.name = name;
    mesh.material = material;
    mesh.vertices = {a, b, c, d};
    mesh.texcoords = {{0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f}};
    mesh.indices = {0, 1, 2, 0, 2, 3};
    center_mesh_origin(mesh);
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
    ensure_default_uvs(mesh);
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
            add_triangle(mesh, a, d, b);
            add_triangle(mesh, b, d, c);
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
    scene.meshes.back().light = {true, true, {1.0f, 0.888889f, 0.666667f}, 9.0f};
    scene.meshes.push_back(make_uv_sphere_mesh("Blue Mesh", 3, {-0.55f, 0.42f, 0.0f}, 0.42f));
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

std::shared_ptr<Texture> find_texture(const Scene& scene, const std::string& name) {
    for (const std::shared_ptr<Texture>& texture : scene.textures) {
        if (texture && texture->name == name) {
            return texture;
        }
    }
    return nullptr;
}

SceneLoadResult load_scene(const std::string& path) {
    if (is_url(path)) {
        return load_url_scene(path);
    }

    const std::string ext = lowercase_extension(path);
    if (ext == ".glb" || ext == ".gltf") {
        return load_gltf_scene(path);
    }
    if (ext == ".pbrt") {
        return load_pbrt_scene(path);
    }

    std::ifstream file(path);
    if (!file) {
        return {make_default_scene(), "Could not open scene: " + path};
    }

    Scene scene;
    const std::string scene_dir = parent_path(path);
    std::unordered_map<std::string, int> material_ids;
    std::vector<Vec3> legacy_material_emissions;
    std::vector<std::string> material_texture_names;
    std::vector<std::pair<std::string, LightComponent>> pending_lights;
    std::string environment_texture_name;
    std::string line;
    int line_no = 0;
    bool saw_camera = false;

    while (std::getline(file, line)) {
        ++line_no;
        const auto hash = line.find('#');
        if (hash != std::string::npos) {
            line.resize(hash);
        }

        std::istringstream in(line);
        std::string tag;
        if (!(in >> tag)) {
            continue;
        }

        if (tag == "camera") {
            in >> scene.camera.position.x >> scene.camera.position.y >> scene.camera.position.z;
            in >> scene.camera.target.x >> scene.camera.target.y >> scene.camera.target.z;
            in >> scene.camera.fov_degrees;
            saw_camera = true;
        } else if (tag == "texture") {
            std::string texture_name;
            std::string texture_path;
            in >> texture_name >> texture_path;
            if (!in) {
                return {scene, "Invalid texture at line " + std::to_string(line_no)};
            }
            Texture texture;
            std::string error;
            const std::string resolved_texture_path = join_path(scene_dir, texture_path);
            if (!load_texture_file(texture_name, resolved_texture_path, texture, error)) {
                return {scene, error};
            }
            texture.path = texture_path;
            scene.textures.push_back(std::make_shared<Texture>(std::move(texture)));
        } else if (tag == "environment") {
            std::string texture_name;
            in >> scene.environment.color.x >> scene.environment.color.y >> scene.environment.color.z >> scene.environment.strength;
            if (!in) {
                return {scene, "Invalid environment at line " + std::to_string(line_no)};
            }
            if (in >> texture_name) {
                scene.environment.texture = find_texture(scene, texture_name);
                scene.environment.constant = false;
                environment_texture_name = texture_name;
            } else {
                scene.environment.texture = nullptr;
                scene.environment.constant = true;
                environment_texture_name.clear();
            }
        } else if (tag == "material") {
            std::string material_name;
            Vec3 albedo;
            Vec3 legacy_emission;
            in >> material_name >> albedo.x >> albedo.y >> albedo.z;
            if (!in) {
                return {scene, "Invalid material at line " + std::to_string(line_no)};
            }
            std::vector<std::string> rest;
            for (std::string token; in >> token;) {
                rest.push_back(token);
            }
            if (rest.size() >= 3 &&
                parse_float_token(rest[0], legacy_emission.x) &&
                parse_float_token(rest[1], legacy_emission.y) &&
                parse_float_token(rest[2], legacy_emission.z)) {
                rest.erase(rest.begin(), rest.begin() + 3);
            }
            BrdfModel brdf = BrdfModel::Lambertian;
            float roughness = 0.5f;
            float metallic = 0.0f;
            if (rest.size() >= 3) {
                brdf = parse_brdf_model(rest[0]);
                parse_float_token(rest[1], roughness);
                parse_float_token(rest[2], metallic);
            }
            roughness = brdf == BrdfModel::Dielectric ? std::clamp(roughness, 1.0f, 3.0f) : std::clamp(roughness, 0.02f, 1.0f);
            metallic = std::clamp(metallic, 0.0f, 1.0f);
            material_ids[material_name] = static_cast<int>(scene.materials.size());
            std::shared_ptr<Material> material = make_material(material_name, albedo, brdf, roughness, metallic);
            const std::string texture_name = rest.size() >= 4 ? rest[3] : std::string{};
            material->albedo_texture = texture_name.empty() ? nullptr : find_texture(scene, texture_name);
            scene.materials.push_back(material);
            material_texture_names.push_back(texture_name);
            legacy_material_emissions.push_back(legacy_emission);
        } else if (tag == "light") {
            std::string mesh_name;
            LightComponent light;
            in >> mesh_name >> light.color.x >> light.color.y >> light.color.z >> light.intensity;
            if (!in || light.intensity < 0.0f) {
                return {scene, "Invalid light at line " + std::to_string(line_no)};
            }
            int double_sided = light.double_sided ? 1 : 0;
            if (in >> double_sided) {
                light.double_sided = double_sided != 0;
            }
            light.enabled = light.intensity > 0.0f;
            pending_lights.push_back({mesh_name, light});
        } else if (tag == "mesh") {
            Mesh mesh;
            std::string material_name;
            int vertex_count = 0;
            int triangle_count = 0;
            in >> mesh.name >> material_name >> mesh.translation.x >> mesh.translation.y >> mesh.translation.z >>
                mesh.rotation.x >> mesh.rotation.y >> mesh.rotation.z >> mesh.scale.x;
            if (!(in >> mesh.scale.y >> mesh.scale.z >> vertex_count >> triangle_count)) {
                mesh.scale.y = mesh.scale.x;
                mesh.scale.z = mesh.scale.x;
                in.clear();
                std::istringstream retry(line);
                retry >> tag >> mesh.name >> material_name >> mesh.translation.x >> mesh.translation.y >> mesh.translation.z >>
                    mesh.rotation.x >> mesh.rotation.y >> mesh.rotation.z >> mesh.scale.x >> vertex_count >> triangle_count;
            }
            mesh.material = material_id(material_ids, material_name);
            if (!in || mesh.material < 0 || vertex_count < 0 || triangle_count < 0) {
                return {scene, "Invalid mesh header at line " + std::to_string(line_no)};
            }
            for (int i = 0; i < vertex_count; ++i) {
                Vec3 v;
                if (!(file >> v.x >> v.y >> v.z)) {
                    return {scene, "Invalid mesh vertex data after line " + std::to_string(line_no)};
                }
                mesh.vertices.push_back(v);
                mesh.texcoords.push_back(default_uv(v));
            }
            for (int i = 0; i < triangle_count; ++i) {
                uint32_t a = 0;
                uint32_t b = 0;
                uint32_t c = 0;
                if (!(file >> a >> b >> c)) {
                    return {scene, "Invalid mesh index data after line " + std::to_string(line_no)};
                }
                if (a >= mesh.vertices.size() || b >= mesh.vertices.size() || c >= mesh.vertices.size()) {
                    return {scene, "Mesh index out of range after line " + std::to_string(line_no)};
                }
                mesh.indices.push_back(a);
                mesh.indices.push_back(b);
                mesh.indices.push_back(c);
            }
            center_mesh_origin(mesh);
            ensure_default_uvs(mesh);
            if (mesh.material >= 0 && mesh.material < static_cast<int>(legacy_material_emissions.size()) &&
                has_light_emission(legacy_material_emissions[static_cast<size_t>(mesh.material)])) {
                mesh.light = make_light_from_emission(legacy_material_emissions[static_cast<size_t>(mesh.material)]);
            }
            scene.meshes.push_back(mesh);
            scene.uses_builtin_default_meshes = false;
        } else {
            return {scene, "Unknown scene token '" + tag + "' at line " + std::to_string(line_no)};
        }
    }

    if (scene.meshes.empty()) {
        Scene defaults = make_default_scene();
        if (!saw_camera) {
            scene.camera = defaults.camera;
        }
        if (scene.materials.empty()) {
            scene.materials = std::move(defaults.materials);
        }
        scene.meshes = std::move(defaults.meshes);
        scene.uses_builtin_default_meshes = true;
    }
    for (const auto& pending : pending_lights) {
        for (Mesh& mesh : scene.meshes) {
            if (mesh.name == pending.first) {
                mesh.light = pending.second;
                break;
            }
        }
    }
    for (size_t i = 0; i < scene.materials.size() && i < material_texture_names.size(); ++i) {
        if (scene.materials[i] && !material_texture_names[i].empty() && !scene.materials[i]->albedo_texture) {
            scene.materials[i]->albedo_texture = find_texture(scene, material_texture_names[i]);
        }
    }
    if (!environment_texture_name.empty()) {
        scene.environment.texture = find_texture(scene, environment_texture_name);
    }
    return {scene, {}};
}

bool save_scene(const Scene& scene, const std::string& path, std::string& error) {
    std::ofstream out(path);
    if (!out) {
        error = "Could not write scene: " + path;
        return false;
    }

    if (scene.uses_builtin_default_meshes) {
        out << "# Mesh-based Cornell-ish test scene.\n";
        out << "# If no mesh blocks are listed, the built-in mesh Cornell scene is used.\n";
    }

    out << "camera "
        << scene.camera.position.x << ' ' << scene.camera.position.y << ' ' << scene.camera.position.z << "   "
        << scene.camera.target.x << ' ' << scene.camera.target.y << ' ' << scene.camera.target.z << "   "
        << scene.camera.fov_degrees << "\n\n";

    for (const std::shared_ptr<Texture>& texture : scene.textures) {
        if (texture) {
            out << "texture " << texture->name << ' ' << texture->path << '\n';
        }
    }
    if (!scene.textures.empty()) {
        out << '\n';
    }

    if (scene.environment.constant || scene.environment.texture) {
        out << "environment "
            << scene.environment.color.x << ' ' << scene.environment.color.y << ' ' << scene.environment.color.z << ' '
            << scene.environment.strength;
        if (scene.environment.texture) {
            out << ' ' << scene.environment.texture->name;
        }
        out << "\n\n";
    }

    for (const std::shared_ptr<Material>& material : scene.materials) {
        if (!material) {
            continue;
        }
        float roughness = 0.5f;
        float metallic = 0.0f;
        if (const auto* principled = dynamic_cast<const PrincipledMaterial*>(material.get())) {
            roughness = principled->roughness;
            metallic = principled->metallic;
        } else if (const auto* dielectric = dynamic_cast<const DielectricMaterial*>(material.get())) {
            roughness = dielectric->ior;
        }
        out << "material " << material->name << ' '
            << material->albedo.x << ' ' << material->albedo.y << ' ' << material->albedo.z << ' '
            << material->model_name() << ' '
            << roughness << ' ' << metallic;
        if (material->albedo_texture) {
            out << ' ' << material->albedo_texture->name;
        }
        out << '\n';
    }
    out << '\n';

    if (scene.uses_builtin_default_meshes) return true;

    for (const Mesh& mesh : scene.meshes) {
        if (!mesh.light.enabled || mesh.light.intensity <= 0.0f) {
            continue;
        }
        out << "light " << mesh.name << ' '
            << mesh.light.color.x << ' ' << mesh.light.color.y << ' ' << mesh.light.color.z << ' '
            << mesh.light.intensity << ' ' << (mesh.light.double_sided ? 1 : 0) << '\n';
    }
    out << '\n';

    for (const Mesh& mesh : scene.meshes) {
        out << "mesh " << mesh.name << ' ' << scene.materials[static_cast<size_t>(mesh.material)]->name << ' '
            << mesh.translation.x << ' ' << mesh.translation.y << ' ' << mesh.translation.z << ' '
            << mesh.rotation.x << ' ' << mesh.rotation.y << ' ' << mesh.rotation.z << ' '
            << mesh.scale.x << ' ' << mesh.scale.y << ' ' << mesh.scale.z << ' ' << mesh.vertices.size() << ' ' << mesh.indices.size() / 3 << '\n';
        for (Vec3 v : mesh.vertices) {
            out << v.x << ' ' << v.y << ' ' << v.z << '\n';
        }
        for (size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
            out << mesh.indices[i] << ' ' << mesh.indices[i + 1] << ' ' << mesh.indices[i + 2] << '\n';
        }
    }
    return true;
}

RenderScene build_render_scene(const Scene& scene) {
    RenderScene render_scene;
    for (int mesh_index = 0; mesh_index < static_cast<int>(scene.meshes.size()); ++mesh_index) {
        const Mesh& mesh = scene.meshes[static_cast<size_t>(mesh_index)];
        const int triangle_index_begin = static_cast<int>(render_scene.triangles.size());
        const int index_begin = static_cast<int>(render_scene.triangle_indices.size());
        for (size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
            const size_t i0 = static_cast<size_t>(mesh.indices[i]);
            const size_t i1 = static_cast<size_t>(mesh.indices[i + 1]);
            const size_t i2 = static_cast<size_t>(mesh.indices[i + 2]);
            if (i0 >= mesh.vertices.size() || i1 >= mesh.vertices.size() || i2 >= mesh.vertices.size()) {
                continue;
            }
            const Vec3 v0 = mesh.translation + rotate_xyz(mesh.vertices[i0] * mesh.scale, mesh.rotation);
            const Vec3 v1 = mesh.translation + rotate_xyz(mesh.vertices[i1] * mesh.scale, mesh.rotation);
            const Vec3 v2 = mesh.translation + rotate_xyz(mesh.vertices[i2] * mesh.scale, mesh.rotation);
            const Vec3 n = normalize(cross(v1 - v0, v2 - v0));
            if (dot(n, n) <= 0.0f) {
                continue;
            }
            Vec3 n0 = n;
            Vec3 n1 = n;
            Vec3 n2 = n;
            if (i0 < mesh.normals.size() && i1 < mesh.normals.size() && i2 < mesh.normals.size()) {
                n0 = normalize(rotate_xyz(mesh.normals[i0], mesh.rotation));
                n1 = normalize(rotate_xyz(mesh.normals[i1], mesh.rotation));
                n2 = normalize(rotate_xyz(mesh.normals[i2], mesh.rotation));
                if (dot(n0, n0) <= 0.0f || dot(n1, n1) <= 0.0f || dot(n2, n2) <= 0.0f) {
                    n0 = n1 = n2 = n;
                }
            }
            const int triangle_index = static_cast<int>(render_scene.triangles.size());
            const Vec2 uv0 = i0 < mesh.texcoords.size() ? mesh.texcoords[i0] : default_uv(mesh.vertices[i0]);
            const Vec2 uv1 = i1 < mesh.texcoords.size() ? mesh.texcoords[i1] : default_uv(mesh.vertices[i1]);
            const Vec2 uv2 = i2 < mesh.texcoords.size() ? mesh.texcoords[i2] : default_uv(mesh.vertices[i2]);
            Vec3 tangent;
            Vec3 bitangent;
            const Vec3 edge1 = v1 - v0;
            const Vec3 edge2 = v2 - v0;
            const Vec2 duv1 = uv1 - uv0;
            const Vec2 duv2 = uv2 - uv0;
            const float det = duv1.x * duv2.y - duv1.y * duv2.x;
            if (std::fabs(det) > 1.0e-8f) {
                const float inv_det = 1.0f / det;
                tangent = normalize((edge1 * duv2.y - edge2 * duv1.y) * inv_det);
                bitangent = normalize((edge2 * duv1.x - edge1 * duv2.x) * inv_det);
            }
            if (dot(tangent, tangent) <= 0.0f || dot(bitangent, bitangent) <= 0.0f) {
                const Vec3 up = std::fabs(n.z) < 0.999f ? Vec3{0.0f, 0.0f, 1.0f} : Vec3{1.0f, 0.0f, 0.0f};
                tangent = normalize(cross(up, n));
                bitangent = cross(n, tangent);
            }
            render_scene.triangles.push_back({v0, v1, v2, n, n0, n1, n2, tangent, bitangent, (v0 + v1 + v2) / 3.0f, uv0, uv1, uv2, mesh.material, mesh_index});
            render_scene.triangle_indices.push_back(triangle_index);
            Vec3 material_emission;
            if (mesh.material >= 0 && mesh.material < static_cast<int>(scene.materials.size()) && scene.materials[static_cast<size_t>(mesh.material)]) {
                material_emission = scene.materials[static_cast<size_t>(mesh.material)]->emission;
            }
            if ((mesh.light.enabled && mesh.light.intensity > 0.0f) || has_light_emission(material_emission)) {
                render_scene.light_triangle_indices.push_back(triangle_index);
            }
        }
        const int triangle_count = static_cast<int>(render_scene.triangles.size()) - triangle_index_begin;
        if (triangle_count > 0) {
            const int root = build_bvh_recursive(render_scene, render_scene.triangle_indices, render_scene.bvh_nodes, index_begin, triangle_count);
            RenderScene::MeshInstance instance;
            instance.bounds = render_scene.bvh_nodes[static_cast<size_t>(root)].bounds;
            instance.bvh_root = root;
            instance.mesh = mesh_index;
            render_scene.mesh_instance_indices.push_back(static_cast<int>(render_scene.mesh_instances.size()));
            render_scene.mesh_instances.push_back(instance);
        }
    }
    if (!render_scene.mesh_instance_indices.empty()) {
        build_tlas_recursive(render_scene, 0, static_cast<int>(render_scene.mesh_instance_indices.size()));
    }
    render_scene.flat_triangle_indices.resize(render_scene.triangles.size());
    std::iota(render_scene.flat_triangle_indices.begin(), render_scene.flat_triangle_indices.end(), 0);
    if (!render_scene.flat_triangle_indices.empty()) {
        build_bvh_recursive(render_scene, render_scene.flat_triangle_indices, render_scene.flat_bvh_nodes, 0, static_cast<int>(render_scene.flat_triangle_indices.size()));
    }
    return render_scene;
}

} // namespace lt
