#include "lt/scene.h"
#include "lt/log.h"

#include <ufbx.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace lt {
namespace {

struct FbxMaterialCache {
    std::unordered_map<const ufbx_material*, int> materials;
    std::unordered_map<const ufbx_texture*, std::shared_ptr<Texture>> color_textures_by_object;
    std::unordered_map<const ufbx_texture*, std::shared_ptr<Texture>> data_textures_by_object;
    std::unordered_map<std::string, std::shared_ptr<Texture>> textures_by_path;
    std::unordered_set<std::string> failed_texture_paths;
    std::unordered_set<std::string> skipped_texture_paths;
};

struct BoundsAccumulator {
    Vec3 min = {kInfinity, kInfinity, kInfinity};
    Vec3 max = {-kInfinity, -kInfinity, -kInfinity};
    bool valid = false;
};

struct MeshBuilder {
    Mesh mesh;
    std::unordered_map<uint32_t, uint32_t> local_indices;
    bool initialized = false;
};

struct PysceneMaterialTweaks {
    std::optional<float> roughness;
    std::optional<float> metallic;
    std::optional<float> ior;
    std::optional<float> transmission;
    std::optional<bool> double_sided;
};

std::string ufbx_string_to_string(ufbx_string value) {
    return value.data && value.length > 0 ? std::string(value.data, value.length) : std::string{};
}

std::string lower_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::string path_key(const std::filesystem::path& path) {
    return lower_copy(path.lexically_normal().string());
}

std::string trim_copy(std::string value) {
    const auto begin = std::find_if_not(value.begin(), value.end(), [](unsigned char c) {
        return std::isspace(c);
    });
    const auto end = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char c) {
        return std::isspace(c);
    }).base();
    return begin < end ? std::string(begin, end) : std::string{};
}

std::string material_base_name(std::string name) {
    const size_t pipe = name.find_last_of("|:");
    if (pipe != std::string::npos) {
        name = name.substr(pipe + 1);
    }
    for (char& c : name) {
        if (c == ' ') {
            c = '_';
        }
    }
    return name;
}

Vec3 to_vec3(ufbx_vec3 value) {
    return {
        static_cast<float>(value.x),
        static_cast<float>(value.y),
        static_cast<float>(value.z),
    };
}

Vec2 to_vec2(ufbx_vec2 value) {
    return {
        static_cast<float>(value.x),
        static_cast<float>(value.y),
    };
}

TextureTransform texture_transform_from_fbx(const ufbx_texture* texture) {
    TextureTransform transform;
    if (!texture || !texture->has_uv_transform) {
        return transform;
    }
    const float scale_x = static_cast<float>(texture->uv_transform.scale.x);
    const float scale_y = static_cast<float>(texture->uv_transform.scale.y);
    const float offset_x = static_cast<float>(texture->uv_transform.translation.x);
    const float offset_y = static_cast<float>(texture->uv_transform.translation.y);
    transform.scale = {
        scale_x == 0.0f ? 1.0f : scale_x,
        scale_y == 0.0f ? 1.0f : scale_y,
    };
    transform.offset = {
        offset_x,
        1.0f - transform.scale.y - offset_y,
    };
    const ufbx_vec3 rotation = ufbx_quat_to_euler(texture->uv_transform.rotation, UFBX_ROTATION_ORDER_XYZ);
    transform.rotation = static_cast<float>(rotation.z * kPi / 180.0);
    return transform;
}

void set_material_input(
    MaterialInput& input,
    std::shared_ptr<Texture> texture,
    const ufbx_texture* fbx_texture,
    MaterialInputChannel channel,
    TextureRole role,
    TextureColorSpace color_space) {
    input.texture = std::move(texture);
    input.channel = channel;
    input.role = role;
    input.color_space = color_space;
    input.transform = texture_transform_from_fbx(fbx_texture);
    if (input.texture) {
        apply_texture_role(*input.texture, role, color_space);
    }
}

ufbx_vec3 to_ufbx_vec3(Vec3 value) {
    return {
        static_cast<ufbx_real>(value.x),
        static_cast<ufbx_real>(value.y),
        static_cast<ufbx_real>(value.z),
    };
}

Vec3 matrix_position(const ufbx_matrix& matrix) {
    return to_vec3(matrix.cols[3]);
}

Vec3 transform_direction(const ufbx_matrix& matrix, Vec3 direction) {
    return to_vec3(ufbx_transform_direction(&matrix, to_ufbx_vec3(direction)));
}

Vec3 coordinate_axis_vector(ufbx_coordinate_axis axis, Vec3 fallback) {
    switch (axis) {
    case UFBX_COORDINATE_AXIS_POSITIVE_X: return {1.0f, 0.0f, 0.0f};
    case UFBX_COORDINATE_AXIS_NEGATIVE_X: return {-1.0f, 0.0f, 0.0f};
    case UFBX_COORDINATE_AXIS_POSITIVE_Y: return {0.0f, 1.0f, 0.0f};
    case UFBX_COORDINATE_AXIS_NEGATIVE_Y: return {0.0f, -1.0f, 0.0f};
    case UFBX_COORDINATE_AXIS_POSITIVE_Z: return {0.0f, 0.0f, 1.0f};
    case UFBX_COORDINATE_AXIS_NEGATIVE_Z: return {0.0f, 0.0f, -1.0f};
    case UFBX_COORDINATE_AXIS_UNKNOWN:
    default:
        return fallback;
    }
}

Vec3 material_vec3(const ufbx_material_map& map, Vec3 fallback) {
    if (!map.has_value) {
        return fallback;
    }
    if (map.value_components >= 3) {
        return to_vec3(map.value_vec3);
    }
    if (map.value_components == 1) {
        const float value = static_cast<float>(map.value_real);
        return {value, value, value};
    }
    return fallback;
}

float material_real(const ufbx_material_map& map, float fallback) {
    if (!map.has_value) {
        return fallback;
    }
    return static_cast<float>(map.value_real);
}

std::optional<Vec3> prop_vec3(const ufbx_props& props, const char* name) {
    const ufbx_prop* prop = ufbx_find_prop(&props, name);
    if (!prop || (prop->flags & UFBX_PROP_FLAG_NOT_FOUND) || !(prop->flags & UFBX_PROP_FLAG_VALUE_VEC3)) {
        return std::nullopt;
    }
    return to_vec3(prop->value_vec3);
}

std::filesystem::path resolve_texture_path(const std::filesystem::path& scene_dir, const std::string& filename) {
    if (filename.empty()) {
        return {};
    }
    std::filesystem::path path(filename);
    return path.is_absolute() ? path : scene_dir / path;
}

std::optional<std::filesystem::path> first_existing_texture(
    const std::filesystem::path& scene_dir,
    const std::string& base_name,
    const std::string& suffix) {
    static constexpr std::array<const char*, 5> kExtensions = {".dds", ".png", ".tga", ".jpg", ".jpeg"};
    for (const char* extension : kExtensions) {
        const std::filesystem::path path = scene_dir / "Textures" / (base_name + suffix + extension);
        std::error_code ec;
        if (std::filesystem::exists(path, ec)) {
            return path;
        }
    }
    return std::nullopt;
}

uint32_t read_le32(const unsigned char* data) {
    return static_cast<uint32_t>(data[0]) |
        (static_cast<uint32_t>(data[1]) << 8) |
        (static_cast<uint32_t>(data[2]) << 16) |
        (static_cast<uint32_t>(data[3]) << 24);
}

bool is_bc5_dds(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return false;
    }
    std::array<unsigned char, 148> header = {};
    input.read(reinterpret_cast<char*>(header.data()), static_cast<std::streamsize>(header.size()));
    const std::streamsize bytes_read = input.gcount();
    if (bytes_read < 128 ||
        header[0] != 'D' ||
        header[1] != 'D' ||
        header[2] != 'S' ||
        header[3] != ' ' ||
        read_le32(header.data() + 4) != 124u) {
        return false;
    }
    const char* fourcc = reinterpret_cast<const char*>(header.data() + 84);
    if (std::memcmp(fourcc, "ATI2", 4) == 0 || std::memcmp(fourcc, "BC5U", 4) == 0) {
        return true;
    }
    if (std::memcmp(fourcc, "DX10", 4) == 0 && bytes_read >= 148) {
        const uint32_t dxgi_format = read_le32(header.data() + 128);
        return dxgi_format == 83u || dxgi_format == 84u;
    }
    return false;
}

std::shared_ptr<Texture> load_texture_path(
    Scene& scene,
    FbxMaterialCache& cache,
    const std::filesystem::path& scene_dir,
    const std::filesystem::path& path,
    const std::string& name_hint,
    bool color_texture,
    bool optional_normal = false) {
    if (path.empty()) {
        return nullptr;
    }

    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        return nullptr;
    }

    const std::string key = path_key(path) + (color_texture ? "|color" : "|data");
    if (const auto it = cache.textures_by_path.find(key); it != cache.textures_by_path.end()) {
        return it->second;
    }
    if (cache.failed_texture_paths.find(key) != cache.failed_texture_paths.end()) {
        return nullptr;
    }

    if (optional_normal && lower_copy(path.extension().string()) == ".dds" && is_bc5_dds(path)) {
        if (cache.skipped_texture_paths.insert(key).second) {
            LT_LOG_DEBUG("FBX normal texture '{}' was skipped: BC5/ATI2 DDS normals are not expanded by default", path.string());
        }
        cache.failed_texture_paths.insert(key);
        return nullptr;
    }

    Texture texture;
    std::string error;
    std::string name = name_hint.empty() ? path.stem().string() : name_hint;
    if (!load_texture_file(name, path.string(), texture, error)) {
        cache.failed_texture_paths.insert(key);
        LT_LOG_DEBUG("FBX texture '{}' was skipped: {}", path.string(), error);
        return nullptr;
    }
    apply_texture_role(texture, color_texture ? TextureRole::Color : TextureRole::Data, color_texture ? TextureColorSpace::SRGB : TextureColorSpace::Raw);

    std::error_code rel_ec;
    const std::filesystem::path relative = std::filesystem::relative(path, scene_dir, rel_ec);
    texture.path = rel_ec ? path.string() : relative.string();
    auto shared = std::make_shared<Texture>(std::move(texture));
    cache.textures_by_path[key] = shared;
    scene.textures.push_back(shared);
    return shared;
}

std::shared_ptr<Texture> load_texture_object(
    Scene& scene,
    FbxMaterialCache& cache,
    const std::filesystem::path& scene_dir,
    const ufbx_texture* texture,
    const std::string& name_hint,
    bool color_texture,
    bool optional_normal = false) {
    if (!texture) {
        return nullptr;
    }
    auto& textures_by_object = color_texture ? cache.color_textures_by_object : cache.data_textures_by_object;
    if (const auto it = textures_by_object.find(texture); it != textures_by_object.end()) {
        return it->second;
    }

    std::shared_ptr<Texture> result;
    if (texture->content.data && texture->content.size > 0) {
        Texture decoded;
        std::string error;
        const std::string name = name_hint.empty() ? ufbx_string_to_string(texture->name) : name_hint;
        if (load_texture_memory(name, static_cast<const unsigned char*>(texture->content.data), texture->content.size, decoded, error)) {
            apply_texture_role(decoded, color_texture ? TextureRole::Color : TextureRole::Data, color_texture ? TextureColorSpace::SRGB : TextureColorSpace::Raw);
            decoded.path = name;
            result = std::make_shared<Texture>(std::move(decoded));
            scene.textures.push_back(result);
        } else {
            LT_LOG_WARN("FBX embedded texture '{}' was skipped: {}", name, error);
        }
    }

    if (!result) {
        const std::array<std::string, 3> filenames = {
            ufbx_string_to_string(texture->filename),
            ufbx_string_to_string(texture->relative_filename),
            ufbx_string_to_string(texture->absolute_filename),
        };
        for (const std::string& filename : filenames) {
            result = load_texture_path(scene, cache, scene_dir, resolve_texture_path(scene_dir, filename), name_hint, color_texture, optional_normal);
            if (result) {
                break;
            }
        }
    }

    textures_by_object[texture] = result;
    return result;
}

std::shared_ptr<Texture> load_texture_map(
    Scene& scene,
    FbxMaterialCache& cache,
    const std::filesystem::path& scene_dir,
    const ufbx_material_map& map,
    const std::string& name_hint,
    bool color_texture,
    bool optional_normal = false) {
    if (!map.texture_enabled || !map.texture) {
        return nullptr;
    }
    return load_texture_object(scene, cache, scene_dir, map.texture, name_hint, color_texture, optional_normal);
}

std::shared_ptr<Texture> load_convention_texture(
    Scene& scene,
    FbxMaterialCache& cache,
    const std::filesystem::path& scene_dir,
    const std::string& material_name,
    const std::string& suffix,
    bool color_texture,
    bool optional_normal = false) {
    const std::string base_name = material_base_name(material_name);
    const std::optional<std::filesystem::path> path = first_existing_texture(scene_dir, base_name, suffix);
    return path ? load_texture_path(scene, cache, scene_dir, *path, base_name + suffix, color_texture, optional_normal) : nullptr;
}

bool has_alpha_below_one(const Texture& texture) {
    return std::any_of(texture.alpha.begin(), texture.alpha.end(), [](float alpha) {
        return alpha < 0.999f;
    });
}

void copy_common_material_fields(const Material& source, Material& target) {
    target.alpha = source.alpha;
    target.alpha_cutoff = source.alpha_cutoff;
    target.alpha_mode = source.alpha_mode;
    target.double_sided = source.double_sided;
    target.albedo_texture = source.albedo_texture;
    target.normal_texture = source.normal_texture;
    target.normal_scale = source.normal_scale;
    target.emission = source.emission;
    target.emission_texture = source.emission_texture;
    target.npr = source.npr;
}

int default_material(Scene& scene) {
    for (int i = 0; i < static_cast<int>(scene.materials.size()); ++i) {
        if (scene.materials[static_cast<size_t>(i)] && scene.materials[static_cast<size_t>(i)]->name == "default") {
            return i;
        }
    }
    scene.materials.push_back(make_material("default", {0.8f, 0.8f, 0.8f}, BrdfModel::Principled, 0.6f, 0.0f));
    return static_cast<int>(scene.materials.size() - 1);
}

int import_material(Scene& scene, FbxMaterialCache& cache, const std::filesystem::path& scene_dir, const ufbx_material* material) {
    if (!material) {
        return default_material(scene);
    }
    if (const auto it = cache.materials.find(material); it != cache.materials.end()) {
        return it->second;
    }

    const std::string name = ufbx_string_to_string(material->name).empty()
        ? "material_" + std::to_string(scene.materials.size())
        : ufbx_string_to_string(material->name);

    Vec3 albedo = material_vec3(material->pbr.base_color, material_vec3(material->fbx.diffuse_color, {0.8f, 0.8f, 0.8f}));
    const float base_factor = material_real(material->pbr.base_factor, 1.0f);
    albedo = albedo * std::max(0.0f, base_factor);

    float roughness = material_real(material->pbr.roughness, 0.55f);
    float metallic = material_real(material->pbr.metalness, 0.0f);
    auto imported = std::make_shared<StandardSurfaceMaterial>(name, albedo, std::clamp(roughness, 0.02f, 1.0f), std::clamp(metallic, 0.0f, 1.0f));
    imported->double_sided = true;

    imported->albedo_texture =
        load_texture_map(scene, cache, scene_dir, material->pbr.base_color, name + "_BaseColor", true);
    if (!imported->albedo_texture) {
        imported->albedo_texture =
            load_texture_map(scene, cache, scene_dir, material->fbx.diffuse_color, name + "_BaseColor", true);
    }
    if (!imported->albedo_texture) {
        imported->albedo_texture = load_convention_texture(scene, cache, scene_dir, name, "_BaseColor", true);
    }
    if (imported->albedo_texture) {
        imported->albedo = {1.0f, 1.0f, 1.0f};
        const ufbx_texture* fbx_texture = material->pbr.base_color.texture_enabled && material->pbr.base_color.texture
            ? material->pbr.base_color.texture
            : material->fbx.diffuse_color.texture_enabled ? material->fbx.diffuse_color.texture : nullptr;
        set_material_input(imported->base_color_input, imported->albedo_texture, fbx_texture, MaterialInputChannel::RGB, TextureRole::Color, TextureColorSpace::SceneLinear);
    }
    if (imported->albedo_texture && has_alpha_below_one(*imported->albedo_texture)) {
        imported->alpha_mode = AlphaMode::Blend;
    }

    imported->normal_texture =
        load_texture_map(scene, cache, scene_dir, material->pbr.normal_map, name + "_Normal", false, true);
    if (!imported->normal_texture) {
        imported->normal_texture =
            load_texture_map(scene, cache, scene_dir, material->fbx.normal_map, name + "_Normal", false, true);
    }
    if (!imported->normal_texture) {
        imported->normal_texture = load_convention_texture(scene, cache, scene_dir, name, "_Normal", false, true);
    }

    std::shared_ptr<Texture> roughness_texture =
        load_texture_map(scene, cache, scene_dir, material->pbr.roughness, name + "_Roughness", false);
    if (roughness_texture) {
        imported->roughness = 1.0f;
        set_material_input(imported->roughness_input, roughness_texture, material->pbr.roughness.texture, MaterialInputChannel::R, TextureRole::Data, TextureColorSpace::Raw);
    }

    std::shared_ptr<Texture> metalness_texture =
        load_texture_map(scene, cache, scene_dir, material->pbr.metalness, name + "_Metalness", false);
    if (metalness_texture) {
        imported->metalness = 1.0f;
        set_material_input(imported->metalness_input, metalness_texture, material->pbr.metalness.texture, MaterialInputChannel::R, TextureRole::Data, TextureColorSpace::Raw);
    }

    std::shared_ptr<Texture> specular_texture =
        load_texture_map(scene, cache, scene_dir, material->pbr.specular_factor, name + "_Specular", false);
    if (!specular_texture) {
        specular_texture =
            load_texture_map(scene, cache, scene_dir, material->fbx.specular_factor, name + "_Specular", false);
    }
    if (!specular_texture) {
        specular_texture =
            load_texture_map(scene, cache, scene_dir, material->pbr.specular_color, name + "_Specular", false);
    }
    if (!specular_texture) {
        specular_texture =
            load_texture_map(scene, cache, scene_dir, material->fbx.specular_color, name + "_Specular", false);
    }
    if (!specular_texture) {
        specular_texture = load_convention_texture(scene, cache, scene_dir, name, "_Specular", false);
    }
    if (specular_texture) {
        imported->specular_weight = 1.0f;
        const ufbx_texture* fbx_texture = material->pbr.specular_factor.texture_enabled && material->pbr.specular_factor.texture
            ? material->pbr.specular_factor.texture
            : material->fbx.specular_factor.texture_enabled && material->fbx.specular_factor.texture
                ? material->fbx.specular_factor.texture
                : material->pbr.specular_color.texture_enabled && material->pbr.specular_color.texture
                    ? material->pbr.specular_color.texture
                    : material->fbx.specular_color.texture_enabled ? material->fbx.specular_color.texture : nullptr;
        set_material_input(imported->specular_weight_input, specular_texture, fbx_texture, MaterialInputChannel::R, TextureRole::Data, TextureColorSpace::Raw);
    }

    const Vec3 emission_color = material_vec3(material->pbr.emission_color, material_vec3(material->fbx.emission_color, {}));
    const float emission_factor = material_real(material->pbr.emission_factor, material_real(material->fbx.emission_factor, 1.0f));
    imported->emission = emission_color * std::max(0.0f, emission_factor);
    imported->emission_texture =
        load_texture_map(scene, cache, scene_dir, material->pbr.emission_color, name + "_Emissive", true);
    if (!imported->emission_texture) {
        imported->emission_texture =
            load_texture_map(scene, cache, scene_dir, material->fbx.emission_color, name + "_Emissive", true);
    }
    if (!imported->emission_texture) {
        imported->emission_texture = load_convention_texture(scene, cache, scene_dir, name, "_Emissive", true);
    }
    if (imported->emission_texture && imported->emission.x == 0.0f && imported->emission.y == 0.0f && imported->emission.z == 0.0f) {
        imported->emission = {1.0f, 1.0f, 1.0f};
    }
    if (imported->emission_texture) {
        imported->emission_texture->role = TextureRole::Emission;
        const ufbx_texture* fbx_texture = material->pbr.emission_color.texture_enabled && material->pbr.emission_color.texture
            ? material->pbr.emission_color.texture
            : material->fbx.emission_color.texture_enabled ? material->fbx.emission_color.texture : nullptr;
        set_material_input(imported->emission_input, imported->emission_texture, fbx_texture, MaterialInputChannel::RGB, TextureRole::Emission, TextureColorSpace::SceneLinear);
    }

    scene.materials.push_back(imported);
    const int index = static_cast<int>(scene.materials.size() - 1);
    cache.materials[material] = index;
    return index;
}

void expand_bounds(BoundsAccumulator& bounds, Vec3 point) {
    bounds.min = bounds.valid ? min(bounds.min, point) : point;
    bounds.max = bounds.valid ? max(bounds.max, point) : point;
    bounds.valid = true;
}

uint32_t add_vertex(
    MeshBuilder& builder,
    const ufbx_mesh* source,
    const ufbx_matrix& position_matrix,
    const ufbx_matrix& normal_matrix,
    uint32_t fbx_index,
    BoundsAccumulator& bounds) {
    if (const auto it = builder.local_indices.find(fbx_index); it != builder.local_indices.end()) {
        return it->second;
    }

    const ufbx_vec3 p = ufbx_get_vertex_vec3(&source->vertex_position, fbx_index);
    const Vec3 position = to_vec3(ufbx_transform_position(&position_matrix, p));
    const uint32_t local = static_cast<uint32_t>(builder.mesh.vertices.size());
    builder.local_indices[fbx_index] = local;
    builder.mesh.vertices.push_back(position);
    expand_bounds(bounds, position);

    if (source->vertex_normal.exists) {
        const ufbx_vec3 n = ufbx_get_vertex_vec3(&source->vertex_normal, fbx_index);
        builder.mesh.normals.push_back(normalize(to_vec3(ufbx_transform_direction(&normal_matrix, n))));
    }
    if (source->vertex_uv.exists) {
        builder.mesh.texcoords.push_back(to_vec2(ufbx_get_vertex_vec2(&source->vertex_uv, fbx_index)));
    }
    return local;
}

void import_mesh_node(Scene& scene, FbxMaterialCache& cache, const std::filesystem::path& scene_dir, const ufbx_node* node, BoundsAccumulator& bounds) {
    const ufbx_mesh* source = node->mesh;
    if (!source || !source->vertex_position.exists || source->faces.count == 0 || !node->visible) {
        return;
    }

    std::unordered_map<int, MeshBuilder> builders;
    std::vector<uint32_t> triangles(std::max<size_t>(3, source->max_face_triangles * 3u));
    const ufbx_matrix position_matrix = node->geometry_to_world;
    const ufbx_matrix normal_matrix = ufbx_get_compatible_matrix_for_normals(node);
    const std::string node_name = ufbx_string_to_string(node->name).empty()
        ? ufbx_string_to_string(source->name)
        : ufbx_string_to_string(node->name);

    for (size_t face_index = 0; face_index < source->faces.count; ++face_index) {
        const ufbx_face face = source->faces.data[face_index];
        if (face.num_indices < 3) {
            continue;
        }

        uint32_t material_slot = 0;
        if (face_index < source->face_material.count) {
            material_slot = source->face_material.data[face_index];
        }
        const ufbx_material* material = material_slot < node->materials.count
            ? node->materials.data[material_slot]
            : material_slot < source->materials.count ? source->materials.data[material_slot] : nullptr;
        const int material_index = import_material(scene, cache, scene_dir, material);

        MeshBuilder& builder = builders[material_index];
        if (!builder.initialized) {
            builder.mesh.name = node_name + "_" + std::to_string(material_index);
            builder.mesh.material = material_index;
            builder.initialized = true;
        }

        const size_t max_face_indices = std::max<size_t>(1, source->max_face_triangles) * 3;
        if (triangles.size() < max_face_indices) {
            triangles.resize(max_face_indices);
        }
        const uint32_t triangle_count = ufbx_triangulate_face(triangles.data(), triangles.size(), source, face);
        for (uint32_t tri = 0; tri < triangle_count; ++tri) {
            const uint32_t a = add_vertex(builder, source, position_matrix, normal_matrix, triangles[tri * 3u + 0u], bounds);
            const uint32_t b = add_vertex(builder, source, position_matrix, normal_matrix, triangles[tri * 3u + 1u], bounds);
            const uint32_t c = add_vertex(builder, source, position_matrix, normal_matrix, triangles[tri * 3u + 2u], bounds);
            builder.mesh.indices.push_back(a);
            builder.mesh.indices.push_back(b);
            builder.mesh.indices.push_back(c);
        }
    }

    for (auto& item : builders) {
        Mesh& mesh = item.second.mesh;
        if (mesh.indices.size() < 3 || mesh.vertices.empty()) {
            continue;
        }
        if (mesh.normals.size() != mesh.vertices.size()) {
            mesh.normals.clear();
        }
        if (mesh.texcoords.size() != mesh.vertices.size()) {
            mesh.texcoords.clear();
        }
        scene.meshes.push_back(std::move(mesh));
    }
}

bool is_probably_interior_scene(const std::string& path) {
    const std::string lower = lower_copy(path);
    return lower.find("interior") != std::string::npos;
}

void set_default_camera(Scene& scene, const BoundsAccumulator& bounds, const std::string& path) {
    if (!bounds.valid) {
        scene.camera = {{0.0f, 1.0f, 4.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, 45.0f};
        return;
    }

    const Vec3 center = (bounds.min + bounds.max) * 0.5f;
    const Vec3 extent = bounds.max - bounds.min;
    const float radius = std::max(1.0f, length(extent) * 0.5f);
    if (is_probably_interior_scene(path)) {
        scene.camera.position = center + Vec3{extent.x * 0.05f, extent.y * 0.04f, -extent.z * 0.16f};
        scene.camera.target = center + Vec3{extent.x * 0.18f, extent.y * 0.02f, extent.z * 0.28f};
        scene.camera.up = {0.0f, 1.0f, 0.0f};
        scene.camera.fov_degrees = 55.0f;
        return;
    }
    scene.camera.target = center + Vec3{0.0f, extent.y * 0.06f, 0.0f};
    scene.camera.position = center + Vec3{radius * 0.35f, radius * 0.28f, -radius * 1.55f};
    scene.camera.up = {0.0f, 1.0f, 0.0f};
    scene.camera.fov_degrees = 45.0f;
}

bool import_first_camera(Scene& scene, const ufbx_scene* fbx, const BoundsAccumulator& bounds) {
    if (!fbx) {
        return false;
    }
    for (size_t i = 0; i < fbx->cameras.count; ++i) {
        const ufbx_camera* camera = fbx->cameras.data[i];
        if (!camera || camera->projection_mode != UFBX_PROJECTION_MODE_PERSPECTIVE || camera->instances.count == 0) {
            continue;
        }
        for (size_t instance_index = 0; instance_index < camera->instances.count; ++instance_index) {
            const ufbx_node* node = camera->instances.data[instance_index];
            if (!node || !node->visible) {
                continue;
            }

            const bool projection_axes_valid = ufbx_coordinate_axes_valid(camera->projection_axes);
            const Vec3 local_front = projection_axes_valid
                ? coordinate_axis_vector(camera->projection_axes.front, {-1.0f, 0.0f, 0.0f})
                : Vec3{-1.0f, 0.0f, 0.0f};
            const Vec3 local_up = projection_axes_valid
                ? coordinate_axis_vector(camera->projection_axes.up, {0.0f, 1.0f, 0.0f})
                : Vec3{0.0f, 1.0f, 0.0f};
            Vec3 forward = normalize(transform_direction(node->node_to_world, -local_front));
            Vec3 up = normalize(transform_direction(node->node_to_world, local_up));
            if (const std::optional<Vec3> target = prop_vec3(camera->props, "InterestPosition")) {
                const std::optional<Vec3> prop_position = prop_vec3(camera->props, "Position");
                const Vec3 prop_forward = normalize(*target - prop_position.value_or(to_vec3(node->node_to_world.cols[3])));
                if (dot(prop_forward, prop_forward) > 0.0f) {
                    forward = prop_forward;
                }
            }
            if (const std::optional<Vec3> prop_up = prop_vec3(camera->props, "UpVector")) {
                const Vec3 candidate_up = normalize(*prop_up);
                if (dot(candidate_up, candidate_up) > 0.0f) {
                    up = candidate_up;
                }
            }
            if (dot(forward, forward) <= 0.0f) {
                forward = {0.0f, 0.0f, -1.0f};
            }
            if (dot(up, up) <= 0.0f || std::fabs(dot(forward, up)) > 0.98f) {
                up = {0.0f, 1.0f, 0.0f};
            }

            const Vec3 position = matrix_position(node->node_to_world);
            const Vec3 center = bounds.valid ? (bounds.min + bounds.max) * 0.5f : position + forward;
            const float focus_distance = std::max(1.0f, length(center - position));
            const float fov_y = static_cast<float>(camera->field_of_view_deg.y > 0.0 ? camera->field_of_view_deg.y : camera->field_of_view_deg.x);
            scene.camera.position = position;
            scene.camera.target = position + forward * focus_distance;
            scene.camera.up = up;
            scene.camera.fov_degrees = std::clamp(fov_y, 10.0f, 120.0f);
            LT_LOG_INFO("Imported FBX camera '{}'", ufbx_string_to_string(camera->name));
            return true;
        }
    }
    return false;
}

int import_fbx_lights(Scene& scene, const ufbx_scene* fbx, const BoundsAccumulator& bounds) {
    if (!fbx || !bounds.valid) {
        return 0;
    }
    const Vec3 extent = bounds.max - bounds.min;
    const float light_size = std::clamp(length(extent) * 0.008f, 0.05f, 0.45f);
    int imported = 0;
    for (size_t i = 0; i < fbx->lights.count; ++i) {
        const ufbx_light* light = fbx->lights.data[i];
        if (!light || !light->cast_light || light->intensity <= 0.0) {
            continue;
        }
        if (light->type == UFBX_LIGHT_DIRECTIONAL || light->type == UFBX_LIGHT_VOLUME) {
            continue;
        }
        for (size_t instance_index = 0; instance_index < light->instances.count; ++instance_index) {
            const ufbx_node* node = light->instances.data[instance_index];
            if (!node || !node->visible) {
                continue;
            }

            Vec3 direction = normalize(to_vec3(ufbx_transform_direction(&node->node_to_world, light->local_direction)));
            if (dot(direction, direction) <= 0.0f) {
                direction = {0.0f, -1.0f, 0.0f};
            }
            const Vec3 helper = std::fabs(direction.y) < 0.95f ? Vec3{0.0f, 1.0f, 0.0f} : Vec3{1.0f, 0.0f, 0.0f};
            const Vec3 tangent = normalize(cross(helper, direction));
            const Vec3 bitangent = normalize(cross(direction, tangent));
            const Vec3 center = matrix_position(node->node_to_world);
            const float half = light_size * 0.5f;

            Mesh mesh;
            mesh.name = ufbx_string_to_string(node->name);
            if (mesh.name.empty()) {
                mesh.name = "fbx_light_" + std::to_string(imported);
            }
            mesh.material = default_material(scene);
            mesh.vertices = {
                center - tangent * half - bitangent * half,
                center + tangent * half - bitangent * half,
                center + tangent * half + bitangent * half,
                center - tangent * half + bitangent * half,
            };
            mesh.normals.assign(4, direction);
            mesh.texcoords = {{0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f}};
            mesh.indices = {0, 1, 2, 0, 2, 3};
            mesh.light.enabled = true;
            mesh.light.double_sided = light->type == UFBX_LIGHT_POINT;
            mesh.light.color = to_vec3(light->color);
            mesh.light.intensity = static_cast<float>(light->intensity);
            scene.meshes.push_back(std::move(mesh));
            ++imported;
        }
    }
    if (imported > 0) {
        LT_LOG_INFO("Imported {} FBX light nodes as mesh lights", imported);
    }
    return imported;
}

std::optional<std::string> quoted_argument(const std::string& line, const std::string& marker) {
    const size_t marker_pos = line.find(marker);
    if (marker_pos == std::string::npos) {
        return std::nullopt;
    }
    const size_t begin = line.find('"', marker_pos + marker.size());
    if (begin == std::string::npos) {
        return std::nullopt;
    }
    const size_t end = line.find('"', begin + 1);
    if (end == std::string::npos || end <= begin + 1) {
        return std::nullopt;
    }
    return line.substr(begin + 1, end - begin - 1);
}

std::optional<float> parse_float_after(const std::string& line, const std::string& marker) {
    const size_t pos = line.find(marker);
    if (pos == std::string::npos) {
        return std::nullopt;
    }
    std::istringstream input(line.substr(pos + marker.size()));
    float value = 0.0f;
    return input >> value ? std::optional<float>{value} : std::nullopt;
}

std::optional<float> parse_float_value(const std::string& value_text) {
    std::istringstream input(value_text);
    float value = 0.0f;
    return input >> value ? std::optional<float>{value} : std::nullopt;
}

std::optional<bool> parse_bool_value(std::string value_text) {
    value_text = lower_copy(trim_copy(std::move(value_text)));
    if (value_text == "true") {
        return true;
    }
    if (value_text == "false") {
        return false;
    }
    if (const auto value = parse_float_value(value_text)) {
        return *value != 0.0f;
    }
    return std::nullopt;
}

void convert_material_to_standard_surface(Scene& scene, int material_index, float transmission, float ior) {
    if (material_index < 0 || material_index >= static_cast<int>(scene.materials.size())) {
        return;
    }
    std::shared_ptr<Material>& material = scene.materials[static_cast<size_t>(material_index)];
    if (!material) {
        return;
    }
    const float clamped_ior = std::clamp(ior, 1.0f, 3.0f);
    if (auto* standard = dynamic_cast<StandardSurfaceMaterial*>(material.get())) {
        standard->transmission_weight = std::clamp(transmission, 0.0f, 1.0f);
        standard->specular_ior = clamped_ior;
        standard->transmission_color = {1.0f, 1.0f, 1.0f};
        standard->transmission_input = {};
        return;
    }

    float roughness = 0.02f;
    float metalness = 0.0f;
    if (const auto* principled = dynamic_cast<const PrincipledMaterial*>(material.get())) {
        roughness = principled->roughness;
        metalness = principled->metallic;
    }
    auto standard = std::make_shared<StandardSurfaceMaterial>(material->name, material->albedo, roughness, metalness);
    standard->transmission_weight = std::clamp(transmission, 0.0f, 1.0f);
    standard->transmission_color = {1.0f, 1.0f, 1.0f};
    standard->specular_ior = clamped_ior;
    copy_common_material_fields(*material, *standard);
    if (standard->albedo_texture) {
        standard->albedo = {1.0f, 1.0f, 1.0f};
        standard->base_color_input.texture = standard->albedo_texture;
        standard->base_color_input.role = TextureRole::Color;
        standard->base_color_input.color_space = TextureColorSpace::SceneLinear;
    }
    material = std::move(standard);
}

void apply_material_tweaks(Scene& scene, const std::unordered_map<int, PysceneMaterialTweaks>& tweaks) {
    for (const auto& [material_index, tweak] : tweaks) {
        if (material_index < 0 || material_index >= static_cast<int>(scene.materials.size())) {
            continue;
        }
        std::shared_ptr<Material>& material = scene.materials[static_cast<size_t>(material_index)];
        if (!material) {
            continue;
        }

        if (tweak.roughness || tweak.metallic) {
            if (auto* principled = dynamic_cast<PrincipledMaterial*>(material.get())) {
                if (tweak.roughness) {
                    principled->roughness = std::clamp(*tweak.roughness, 0.02f, 1.0f);
                }
                if (tweak.metallic) {
                    principled->metallic = std::clamp(*tweak.metallic, 0.0f, 1.0f);
                }
            } else if (auto* standard = dynamic_cast<StandardSurfaceMaterial*>(material.get())) {
                if (tweak.roughness) {
                    standard->roughness = std::clamp(*tweak.roughness, 0.02f, 1.0f);
                    standard->roughness_input = {};
                }
                if (tweak.metallic) {
                    standard->metalness = std::clamp(*tweak.metallic, 0.0f, 1.0f);
                    standard->metalness_input = {};
                }
            }
        }

        const float transmission = tweak.transmission.value_or(0.0f);
        if (transmission > 0.0f) {
            convert_material_to_standard_surface(scene, material_index, transmission, tweak.ior.value_or(1.5f));
            material = scene.materials[static_cast<size_t>(material_index)];
        } else if (tweak.ior) {
            if (auto* dielectric = dynamic_cast<DielectricMaterial*>(material.get())) {
                dielectric->ior = std::clamp(*tweak.ior, 1.0f, 3.0f);
            } else if (auto* standard = dynamic_cast<StandardSurfaceMaterial*>(material.get())) {
                standard->specular_ior = std::clamp(*tweak.ior, 1.0f, 3.0f);
            }
        }

        if (tweak.double_sided && material) {
            material->double_sided = *tweak.double_sided;
        }
    }
}

void apply_pyscene_directives(Scene& scene, const std::filesystem::path& pyscene_path) {
    std::ifstream input(pyscene_path);
    if (!input) {
        return;
    }

    const std::filesystem::path scene_dir = pyscene_path.parent_path();
    std::optional<std::string> env_map;
    float env_intensity = 1.0f;
    float emissive_multiplier = 1.0f;
    std::unordered_map<std::string, int> material_variables;
    std::unordered_map<int, PysceneMaterialTweaks> material_tweaks;
    std::string line;
    while (std::getline(input, line)) {
        if (const auto env = quoted_argument(line, "EnvMap(")) {
            env_map = *env;
        }
        if (const auto intensity = parse_float_after(line, "envMap.intensity =")) {
            env_intensity = std::max(0.0f, *intensity);
        }
        if (const auto multiplier = parse_float_after(line, "emissiveFactor *=")) {
            emissive_multiplier *= std::max(0.0f, *multiplier);
        }
        if (const auto material_name = quoted_argument(line, "sceneBuilder.getMaterial(")) {
            const size_t equals = line.find('=');
            if (equals != std::string::npos) {
                const std::string variable = trim_copy(line.substr(0, equals));
                const int material_index = find_material(scene, *material_name);
                if (!variable.empty() && material_index >= 0) {
                    material_variables[variable] = material_index;
                }
            }
        }

        const size_t dot = line.find('.');
        const size_t equals = dot == std::string::npos ? std::string::npos : line.find('=', dot + 1u);
        if (dot != std::string::npos && equals != std::string::npos) {
            const std::string variable = trim_copy(line.substr(0, dot));
            const auto variable_it = material_variables.find(variable);
            if (variable_it != material_variables.end()) {
                const std::string property = trim_copy(line.substr(dot + 1u, equals - dot - 1u));
                const std::string value = trim_copy(line.substr(equals + 1u));
                PysceneMaterialTweaks& tweak = material_tweaks[variable_it->second];
                if (property == "roughness") {
                    tweak.roughness = parse_float_value(value);
                } else if (property == "metallic") {
                    tweak.metallic = parse_float_value(value);
                } else if (property == "indexOfRefraction") {
                    tweak.ior = parse_float_value(value);
                } else if (property == "specularTransmission") {
                    tweak.transmission = parse_float_value(value);
                } else if (property == "doubleSided") {
                    tweak.double_sided = parse_bool_value(value);
                }
            }
        }
    }

    if (env_map) {
        Texture texture;
        std::string error;
        const std::filesystem::path env_path = resolve_texture_path(scene_dir, *env_map);
        if (load_texture_file("environment", env_path.string(), texture, error)) {
            texture.path = *env_map;
            scene.textures.push_back(std::make_shared<Texture>(std::move(texture)));
            scene.environment.texture = scene.textures.back();
            scene.environment.color = {1.0f, 1.0f, 1.0f};
            scene.environment.strength = env_intensity;
            scene.environment.constant = false;
            scene.environment.mapping = Environment::Mapping::Equirectangular;
            LT_LOG_INFO("Loaded pyscene environment map '{}' (intensity={})", env_path.string(), env_intensity);
        } else {
            LT_LOG_WARN("pyscene environment '{}' was skipped: {}", env_path.string(), error);
        }
    }

    if (emissive_multiplier != 1.0f) {
        int emissive_materials = 0;
        for (size_t material_index = 0; material_index < scene.materials.size(); ++material_index) {
            const std::shared_ptr<Material>& material = scene.materials[material_index];
            if (!material) {
                continue;
            }
            const bool has_emission = material->emission.x != 0.0f || material->emission.y != 0.0f || material->emission.z != 0.0f;
            if (has_emission || material->emission_texture) {
                if (!has_emission) {
                    material->emission = {1.0f, 1.0f, 1.0f};
                }
                material->emission = material->emission * emissive_multiplier;
                int mesh_refs = 0;
                for (const Mesh& mesh : scene.meshes) {
                    if (mesh.material == static_cast<int>(material_index)) {
                        ++mesh_refs;
                    }
                }
                LT_LOG_DEBUG(
                    "pyscene emissive material '{}' index={} mesh_refs={} texture={}",
                    material->name,
                    material_index,
                    mesh_refs,
                    material->emission_texture ? material->emission_texture->name : "<none>");
                ++emissive_materials;
            }
        }
        LT_LOG_INFO("Applied pyscene emissive multiplier {} to {} material(s)", emissive_multiplier, emissive_materials);
    }
    apply_material_tweaks(scene, material_tweaks);
}

SceneLoadResult load_fbx_scene_internal(const std::string& path, bool apply_sidecar) {
    LT_LOG_INFO("Loading FBX scene '{}'", path);
    ufbx_load_opts opts = {};
    opts.generate_missing_normals = true;
    opts.evaluate_skinning = true;
    opts.use_blender_pbr_material = true;

    ufbx_error error = {};
    ufbx_scene* fbx = ufbx_load_file(path.c_str(), &opts, &error);
    if (!fbx) {
        std::string description = error.description.data
            ? std::string(error.description.data, error.description.length)
            : "unknown FBX error";
        LT_LOG_ERROR("Failed to load FBX scene '{}': {}", path, description);
        return {make_default_scene(), description};
    }

    Scene scene;
    scene.environment.constant = false;
    scene.environment.color = {1.0f, 1.0f, 1.0f};
    scene.environment.strength = 1.0f;

    FbxMaterialCache cache;
    BoundsAccumulator bounds;
    const std::filesystem::path scene_dir = std::filesystem::path(path).parent_path();
    for (size_t i = 0; i < fbx->nodes.count; ++i) {
        const ufbx_node* node = fbx->nodes.data[i];
        if (!node || node->is_root || !node->mesh) {
            continue;
        }
        import_mesh_node(scene, cache, scene_dir, node, bounds);
    }

    if (scene.meshes.empty()) {
        ufbx_free_scene(fbx);
        LT_LOG_ERROR("FBX scene '{}' contains no supported triangle meshes", path);
        return {make_default_scene(), "FBX scene contains no supported triangle meshes: " + path};
    }
    const bool imported_camera = import_first_camera(scene, fbx, bounds);
    import_fbx_lights(scene, fbx, bounds);
    ufbx_free_scene(fbx);
    if (!imported_camera) {
        set_default_camera(scene, bounds, path);
    }

    if (apply_sidecar) {
        const std::filesystem::path sidecar = std::filesystem::path(path).replace_extension(".pyscene");
        std::error_code ec;
        if (std::filesystem::exists(sidecar, ec)) {
            apply_pyscene_directives(scene, sidecar);
        }
    }

    LT_LOG_INFO("Loaded FBX scene '{}' (meshes={}, materials={}, textures={})",
        path, scene.meshes.size(), scene.materials.size(), scene.textures.size());
    return {std::move(scene), {}};
}

} // namespace

SceneLoadResult load_fbx_scene(const std::string& path) {
    return load_fbx_scene_internal(path, true);
}

SceneLoadResult load_pyscene_scene(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        return {make_default_scene(), "Could not open pyscene: " + path};
    }

    std::optional<std::string> fbx_path;
    std::string line;
    while (std::getline(input, line)) {
        if (const auto imported = quoted_argument(line, "importScene(")) {
            fbx_path = *imported;
            break;
        }
    }
    if (!fbx_path) {
        return {make_default_scene(), "pyscene does not contain sceneBuilder.importScene(): " + path};
    }

    const std::filesystem::path resolved = resolve_texture_path(std::filesystem::path(path).parent_path(), *fbx_path);
    SceneLoadResult result = load_fbx_scene_internal(resolved.string(), false);
    if (result.error.empty()) {
        apply_pyscene_directives(result.scene, path);
    }
    return result;
}

} // namespace lt
