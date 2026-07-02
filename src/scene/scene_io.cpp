#include "scene_internal.h"
#include "lt/log.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <functional>
#include <sstream>
#include <unordered_map>
#include <utility>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#include <urlmon.h>
#endif

namespace lt {
namespace {

int material_id(const std::unordered_map<std::string, int>& material_ids, const std::string& name) {
    const auto it = material_ids.find(name);
    return it == material_ids.end() ? -1 : it->second;
}

std::shared_ptr<Texture> find_texture(const Scene& scene, const std::string& name) {
    for (const std::shared_ptr<Texture>& texture : scene.textures) {
        if (texture && texture->name == name) {
            return texture;
        }
    }
    return nullptr;
}

bool parse_float_token(const std::string& token, float& value) {
    std::istringstream input(token);
    input >> value;
    return input && input.eof();
}

bool parse_int_token(const std::string& token, int& value) {
    std::istringstream input(token);
    input >> value;
    return input && input.eof();
}

bool parse_vec3_tokens(const std::vector<std::string>& tokens, size_t offset, Vec3& value) {
    return offset + 2 < tokens.size() &&
        parse_float_token(tokens[offset], value.x) &&
        parse_float_token(tokens[offset + 1], value.y) &&
        parse_float_token(tokens[offset + 2], value.z);
}

Environment::Mapping parse_environment_mapping(const std::string& name) {
    if (name == "equal_area" || name == "equalarea" || name == "octahedral") {
        return Environment::Mapping::EqualArea;
    }
    return Environment::Mapping::Equirectangular;
}

const char* environment_mapping_name(Environment::Mapping mapping) {
    switch (mapping) {
    case Environment::Mapping::EqualArea: return "equal_area";
    case Environment::Mapping::Equirectangular:
    default:
        return "equirectangular";
    }
}

bool is_default_environment_basis(const Environment& environment) {
    return environment.light_from_world_x.x == 1.0f && environment.light_from_world_x.y == 0.0f && environment.light_from_world_x.z == 0.0f &&
        environment.light_from_world_y.x == 0.0f && environment.light_from_world_y.y == 1.0f && environment.light_from_world_y.z == 0.0f &&
        environment.light_from_world_z.x == 0.0f && environment.light_from_world_z.y == 0.0f && environment.light_from_world_z.z == 1.0f;
}

AlphaMode parse_alpha_mode_name(const std::string& name) {
    if (name == "mask" || name == "MASK") {
        return AlphaMode::Mask;
    }
    if (name == "blend" || name == "BLEND") {
        return AlphaMode::Blend;
    }
    return AlphaMode::Opaque;
}

const char* alpha_mode_name(AlphaMode mode) {
    switch (mode) {
    case AlphaMode::Mask: return "mask";
    case AlphaMode::Blend: return "blend";
    case AlphaMode::Opaque:
    default:
        return "opaque";
    }
}

TextureRole parse_texture_role_name(const std::string& name) {
    if (name == "color") return TextureRole::Color;
    if (name == "data" || name == "raw") return TextureRole::Data;
    if (name == "normal") return TextureRole::Normal;
    if (name == "emission" || name == "emissive") return TextureRole::Emission;
    if (name == "environment") return TextureRole::Environment;
    return TextureRole::Unknown;
}

const char* texture_role_name(TextureRole role) {
    switch (role) {
    case TextureRole::Color: return "color";
    case TextureRole::Data: return "data";
    case TextureRole::Normal: return "normal";
    case TextureRole::Emission: return "emission";
    case TextureRole::Environment: return "environment";
    case TextureRole::Unknown:
    default:
        return "unknown";
    }
}

TextureColorSpace parse_texture_color_space_name(const std::string& name) {
    if (name == "scene_linear" || name == "linear") return TextureColorSpace::SceneLinear;
    if (name == "srgb" || name == "sRGB") return TextureColorSpace::SRGB;
    if (name == "raw" || name == "data") return TextureColorSpace::Raw;
    return TextureColorSpace::Auto;
}

const char* texture_color_space_name(TextureColorSpace color_space) {
    switch (color_space) {
    case TextureColorSpace::SceneLinear: return "scene_linear";
    case TextureColorSpace::SRGB: return "srgb";
    case TextureColorSpace::Raw: return "raw";
    case TextureColorSpace::Auto:
    default:
        return "auto";
    }
}

MaterialInputChannel parse_material_input_channel_name(const std::string& name) {
    if (name == "r" || name == "R") return MaterialInputChannel::R;
    if (name == "g" || name == "G") return MaterialInputChannel::G;
    if (name == "b" || name == "B") return MaterialInputChannel::B;
    if (name == "a" || name == "A" || name == "alpha") return MaterialInputChannel::A;
    return MaterialInputChannel::RGB;
}

const char* material_input_channel_name(MaterialInputChannel channel) {
    switch (channel) {
    case MaterialInputChannel::R: return "r";
    case MaterialInputChannel::G: return "g";
    case MaterialInputChannel::B: return "b";
    case MaterialInputChannel::A: return "a";
    case MaterialInputChannel::RGB:
    default:
        return "rgb";
    }
}

bool has_nonzero(Vec3 value) {
    return value.x != 0.0f || value.y != 0.0f || value.z != 0.0f;
}

bool has_nondefault_material_properties(const Material& material) {
    return material.alpha != 1.0f ||
        material.alpha_cutoff != 0.5f ||
        material.alpha_mode != AlphaMode::Opaque ||
        material.double_sided ||
        material.normal_scale != 1.0f ||
        has_nonzero(material.emission);
}

bool assign_material_texture(Material& material, const std::string& slot, const std::shared_ptr<Texture>& texture) {
    if (slot == "albedo" || slot == "base_color" || slot == "baseColor") {
        if (texture) {
            apply_texture_role(*texture, TextureRole::Color, TextureColorSpace::SRGB);
        }
        material.albedo_texture = texture;
        if (auto* standard = dynamic_cast<StandardSurfaceMaterial*>(&material)) {
            standard->base_color_input.texture = texture;
            standard->base_color_input.role = TextureRole::Color;
            standard->base_color_input.color_space = TextureColorSpace::SceneLinear;
        }
        return true;
    }
    if (slot == "normal") {
        if (texture) {
            apply_texture_role(*texture, TextureRole::Normal, TextureColorSpace::Raw);
        }
        material.normal_texture = texture;
        return true;
    }
    if (slot == "emission" || slot == "emissive") {
        if (texture) {
            apply_texture_role(*texture, TextureRole::Emission, TextureColorSpace::SRGB);
        }
        material.emission_texture = texture;
        if (auto* standard = dynamic_cast<StandardSurfaceMaterial*>(&material)) {
            standard->emission_input.texture = texture;
            standard->emission_input.role = TextureRole::Emission;
            standard->emission_input.color_space = TextureColorSpace::SceneLinear;
        }
        return true;
    }
    if (auto* standard = dynamic_cast<StandardSurfaceMaterial*>(&material)) {
        if (texture) {
            apply_texture_role(*texture, TextureRole::Data, TextureColorSpace::Raw);
        }
        if (slot == "metallic_roughness" || slot == "metallicRoughness") {
            standard->roughness_input.texture = texture;
            standard->roughness_input.channel = MaterialInputChannel::G;
            standard->metalness_input.texture = texture;
            standard->metalness_input.channel = MaterialInputChannel::B;
        } else if (slot == "roughness") {
            standard->roughness_input.texture = texture;
            standard->roughness_input.channel = MaterialInputChannel::R;
        } else if (slot == "metalness" || slot == "metallic") {
            standard->metalness_input.texture = texture;
            standard->metalness_input.channel = MaterialInputChannel::R;
        } else if (slot == "transmission") {
            standard->transmission_input.texture = texture;
            standard->transmission_input.channel = MaterialInputChannel::R;
        } else if (slot == "opacity") {
            standard->opacity_input.texture = texture;
            standard->opacity_input.channel = MaterialInputChannel::A;
        } else if (slot == "coat" || slot == "clearcoat") {
            standard->coat_input.texture = texture;
            standard->coat_input.channel = MaterialInputChannel::R;
        } else if (slot == "coat_roughness" || slot == "clearcoat_roughness") {
            standard->coat_roughness_input.texture = texture;
            standard->coat_roughness_input.channel = MaterialInputChannel::G;
        } else if (slot == "sheen_color") {
            if (texture) {
                apply_texture_role(*texture, TextureRole::Color, TextureColorSpace::SRGB);
            }
            standard->sheen_color_input.texture = texture;
            standard->sheen_color_input.role = TextureRole::Color;
            standard->sheen_color_input.color_space = TextureColorSpace::SceneLinear;
        } else if (slot == "sheen_roughness") {
            standard->sheen_roughness_input.texture = texture;
            standard->sheen_roughness_input.channel = MaterialInputChannel::A;
        } else {
            return false;
        }
        return true;
    }
    auto* principled = dynamic_cast<PrincipledMaterial*>(&material);
    if (!principled) {
        return false;
    }
    if (slot == "metallic_roughness" || slot == "metallicRoughness") {
        principled->metallic_roughness_texture = texture;
    } else if (slot == "sheen_color") {
        principled->sheen_color_texture = texture;
    } else if (slot == "sheen_roughness") {
        principled->sheen_roughness_texture = texture;
    } else if (slot == "clearcoat") {
        principled->clearcoat_texture = texture;
    } else if (slot == "clearcoat_roughness") {
        principled->clearcoat_roughness_texture = texture;
    } else {
        return false;
    }
    return true;
}

MaterialInput* standard_material_input_for_slot(StandardSurfaceMaterial& material, const std::string& slot) {
    if (slot == "base_color" || slot == "baseColor" || slot == "albedo") return &material.base_color_input;
    if (slot == "roughness") return &material.roughness_input;
    if (slot == "metalness" || slot == "metallic") return &material.metalness_input;
    if (slot == "specular" || slot == "specular_weight") return &material.specular_weight_input;
    if (slot == "transmission") return &material.transmission_input;
    if (slot == "opacity") return &material.opacity_input;
    if (slot == "emission" || slot == "emissive") return &material.emission_input;
    if (slot == "coat" || slot == "clearcoat") return &material.coat_input;
    if (slot == "coat_roughness" || slot == "clearcoat_roughness") return &material.coat_roughness_input;
    if (slot == "sheen_color") return &material.sheen_color_input;
    if (slot == "sheen_roughness") return &material.sheen_roughness_input;
    return nullptr;
}

const MaterialInput* standard_material_input_for_slot(const StandardSurfaceMaterial& material, const std::string& slot) {
    return standard_material_input_for_slot(const_cast<StandardSurfaceMaterial&>(material), slot);
}

bool assign_standard_material_input(StandardSurfaceMaterial& material, const std::string& slot, MaterialInput input) {
    MaterialInput* target = standard_material_input_for_slot(material, slot);
    if (!target) {
        return false;
    }
    if (input.texture) {
        apply_texture_role(*input.texture, input.role, input.color_space);
    }
    *target = std::move(input);
    if (slot == "base_color" || slot == "baseColor" || slot == "albedo") {
        material.albedo_texture = target->texture;
    } else if (slot == "normal") {
        material.normal_texture = target->texture;
    } else if (slot == "emission" || slot == "emissive") {
        material.emission_texture = target->texture;
    }
    return true;
}

void write_material_texture(std::ostream& output, const Material& material, const char* slot, const std::shared_ptr<Texture>& texture) {
    if (texture) {
        output << "material_texture " << material.name << ' ' << slot << ' ' << texture->name << '\n';
    }
}

void write_material_input(std::ostream& output, const StandardSurfaceMaterial& material, const char* slot, const MaterialInput& input) {
    if (!input.texture) {
        return;
    }
    output << "material_input " << material.name << ' ' << slot << ' '
        << input.texture->name << ' '
        << texture_role_name(input.role) << ' '
        << texture_color_space_name(input.color_space) << ' '
        << material_input_channel_name(input.channel) << ' '
        << input.scalar_factor << ' '
        << input.color_factor.x << ' ' << input.color_factor.y << ' ' << input.color_factor.z << ' '
        << input.transform.uv_set << ' '
        << input.transform.offset.x << ' ' << input.transform.offset.y << ' '
        << input.transform.scale.x << ' ' << input.transform.scale.y << ' '
        << input.transform.rotation << '\n';
}

void parse_npr_settings(const std::vector<std::string>& tokens, NprSettings& npr) {
    if (tokens.empty()) {
        return;
    }
    npr.style = parse_npr_style(tokens[0]);
    if (npr.style == NprStyle::ColorMap) {
        if (tokens.size() > 1) parse_float_token(tokens[1], npr.value_min);
        if (tokens.size() > 2) parse_float_token(tokens[2], npr.value_max);
        npr.value_max = std::max(npr.value_max, npr.value_min + 1.0e-4f);
        return;
    }
    if (npr.style == NprStyle::XToon) {
        if (tokens.size() > 1) npr.xtoon_detail_mode = parse_xtoon_detail_mode(tokens[1]);
        if (tokens.size() > 2) parse_int_token(tokens[2], npr.xtoon_steps);
        parse_vec3_tokens(tokens, 3, npr.xtoon_shadow);
        parse_vec3_tokens(tokens, 6, npr.xtoon_mid);
        parse_vec3_tokens(tokens, 9, npr.xtoon_lit);
        parse_vec3_tokens(tokens, 12, npr.xtoon_accent);
        if (tokens.size() > 15) parse_float_token(tokens[15], npr.xtoon_detail_strength);
        if (tokens.size() > 16) parse_float_token(tokens[16], npr.xtoon_detail_threshold);
        if (tokens.size() > 17) parse_float_token(tokens[17], npr.xtoon_detail_power);
        if (tokens.size() > 18) parse_float_token(tokens[18], npr.xtoon_depth_near);
        if (tokens.size() > 19) parse_float_token(tokens[19], npr.xtoon_depth_far);
        npr.xtoon_steps = std::clamp(npr.xtoon_steps, 1, 8);
        npr.xtoon_detail_strength = std::clamp(npr.xtoon_detail_strength, 0.0f, 1.0f);
        npr.xtoon_detail_threshold = std::clamp(npr.xtoon_detail_threshold, 0.0f, 1.0f);
        npr.xtoon_detail_power = std::clamp(npr.xtoon_detail_power, 0.001f, 256.0f);
        npr.xtoon_depth_near = std::max(1.0e-4f, npr.xtoon_depth_near);
        npr.xtoon_depth_far = std::max(npr.xtoon_depth_far, npr.xtoon_depth_near + 1.0e-4f);
        return;
    }
    if (npr.style == NprStyle::CrossHatching) {
        if (tokens.size() > 1) parse_int_token(tokens[1], npr.hatch_sets);
        if (tokens.size() > 2) parse_float_token(tokens[2], npr.hatch_spacing);
        if (tokens.size() > 3) parse_float_token(tokens[3], npr.hatch_width);
        if (tokens.size() > 4) parse_float_token(tokens[4], npr.hatch_angle);
        if (tokens.size() > 5) parse_float_token(tokens[5], npr.hatch_value_min);
        if (tokens.size() > 6) parse_float_token(tokens[6], npr.hatch_value_max);
        parse_vec3_tokens(tokens, 7, npr.hatch_ink);
        parse_vec3_tokens(tokens, 10, npr.hatch_paper);
        if (tokens.size() > 13) {
            int passthrough = npr.hatch_passthrough ? 1 : 0;
            if (parse_int_token(tokens[13], passthrough)) {
                npr.hatch_passthrough = passthrough != 0;
            }
        }
        if (tokens.size() > 14) {
            int shadow_only = npr.hatch_shadow_only ? 1 : 0;
            if (parse_int_token(tokens[14], shadow_only)) {
                npr.hatch_shadow_only = shadow_only != 0;
            }
        }
        npr.hatch_sets = std::clamp(npr.hatch_sets, 1, 8);
        npr.hatch_spacing = std::max(1.0e-4f, npr.hatch_spacing);
        npr.hatch_width = std::clamp(npr.hatch_width, 1.0e-5f, npr.hatch_spacing);
        npr.hatch_value_max = std::max(npr.hatch_value_max, npr.hatch_value_min + 1.0e-4f);
    }
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
    std::string extension = dot == std::string::npos
        ? std::string{}
        : path.substr(dot, query == std::string::npos ? std::string::npos : query - dot);
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char value) {
        return static_cast<char>(std::tolower(value));
    });
    return extension;
}

bool is_url(const std::string& path) {
    return path.rfind("http://", 0) == 0 || path.rfind("https://", 0) == 0;
}

SceneLoadResult load_url_scene(const std::string& url) {
#if defined(_WIN32)
    LT_LOG_INFO("Downloading scene URL '{}'", url);
    char temp_dir[MAX_PATH] = {};
    if (GetTempPathA(MAX_PATH, temp_dir) == 0) {
        LT_LOG_ERROR("Could not find temporary directory for URL scene cache");
        return {make_default_scene(), "Could not find temporary directory for URL scene cache"};
    }
    std::string extension = lowercase_extension(url);
    if (extension.empty()) {
        extension = ".glb";
    }
    const std::string cached = std::string(temp_dir) + "LightTransport_" + std::to_string(std::hash<std::string>{}(url)) + extension;
    const HRESULT result = URLDownloadToFileA(nullptr, url.c_str(), cached.c_str(), 0, nullptr);
    if (FAILED(result)) {
        LT_LOG_ERROR("Could not download scene URL '{}'", url);
        return {make_default_scene(), "Could not download scene URL: " + url};
    }
    LT_LOG_DEBUG("Downloaded scene URL '{}' to '{}'", url, cached);
    return load_scene(cached);
#else
    LT_LOG_ERROR("URL scene loading is only implemented on Windows builds: '{}'", url);
    return {make_default_scene(), "URL scene loading is only implemented on Windows builds"};
#endif
}

} // namespace

SceneLoadResult load_scene(const std::string& path) {
    LT_LOG_INFO("Loading scene '{}'", path);
    if (is_url(path)) {
        return load_url_scene(path);
    }

    const std::string extension = lowercase_extension(path);
    if (extension == ".fbx") {
        SceneLoadResult result = load_fbx_scene(path);
        if (result.error.empty()) {
            LT_LOG_INFO("Loaded FBX scene '{}' (meshes={}, spheres={}, materials={}, textures={})",
                path, result.scene.meshes.size(), result.scene.spheres.size(), result.scene.materials.size(), result.scene.textures.size());
        } else {
            LT_LOG_ERROR("Failed to load FBX scene '{}': {}", path, result.error);
        }
        return result;
    }
    if (extension == ".pyscene") {
        SceneLoadResult result = load_pyscene_scene(path);
        if (result.error.empty()) {
            LT_LOG_INFO("Loaded pyscene '{}' (meshes={}, spheres={}, materials={}, textures={})",
                path, result.scene.meshes.size(), result.scene.spheres.size(), result.scene.materials.size(), result.scene.textures.size());
        } else {
            LT_LOG_ERROR("Failed to load pyscene '{}': {}", path, result.error);
        }
        return result;
    }
    if (extension == ".glb" || extension == ".gltf") {
        SceneLoadResult result = load_gltf_scene(path);
        if (result.error.empty()) {
            LT_LOG_INFO("Loaded glTF scene '{}' (meshes={}, spheres={}, materials={}, textures={})",
                path, result.scene.meshes.size(), result.scene.spheres.size(), result.scene.materials.size(), result.scene.textures.size());
        } else {
            LT_LOG_ERROR("Failed to load glTF scene '{}': {}", path, result.error);
        }
        return result;
    }
    if (extension == ".pbrt") {
        SceneLoadResult result = load_pbrt_scene(path);
        if (result.error.empty()) {
            LT_LOG_INFO("Loaded PBRT scene '{}' (meshes={}, spheres={}, materials={}, textures={}, directional_lights={})",
                path, result.scene.meshes.size(), result.scene.spheres.size(), result.scene.materials.size(), result.scene.textures.size(), result.scene.directional_lights.size());
        } else {
            LT_LOG_ERROR("Failed to load PBRT scene '{}': {}", path, result.error);
        }
        return result;
    }

    std::ifstream file(path);
    if (!file) {
        LT_LOG_ERROR("Could not open scene '{}'; using default scene fallback", path);
        return {make_default_scene(), "Could not open scene: " + path};
    }

    Scene scene;
    const auto fail = [&](std::string error) {
        LT_LOG_ERROR("Failed to load .lt scene '{}': {}", path, error);
        return SceneLoadResult{scene, std::move(error)};
    };
    const std::string scene_dir = parent_path(path);
    std::unordered_map<std::string, int> material_ids;
    std::vector<Vec3> legacy_material_emissions;
    std::vector<std::string> material_texture_names;
    std::vector<std::pair<std::string, LightComponent>> pending_lights;
    std::string environment_texture_name;
    std::string line;
    int line_number = 0;
    bool saw_camera = false;

    while (std::getline(file, line)) {
        ++line_number;
        const size_t comment = line.find('#');
        if (comment != std::string::npos) {
            line.resize(comment);
        }

        std::istringstream input(line);
        std::string tag;
        if (!(input >> tag)) {
            continue;
        }

        if (tag == "camera") {
            input >> scene.camera.position.x >> scene.camera.position.y >> scene.camera.position.z;
            input >> scene.camera.target.x >> scene.camera.target.y >> scene.camera.target.z;
            input >> scene.camera.fov_degrees;
            saw_camera = true;
        } else if (tag == "texture") {
            std::string texture_name;
            std::string texture_path;
            input >> texture_name >> texture_path;
            if (!input) {
                return fail("Invalid texture at line " + std::to_string(line_number));
            }
            Texture texture;
            std::string error;
            const std::string resolved_texture_path = join_path(scene_dir, texture_path);
            if (!load_texture_file(texture_name, resolved_texture_path, texture, error)) {
                return fail(error);
            }
            texture.path = texture_path;
            scene.textures.push_back(std::make_shared<Texture>(std::move(texture)));
        } else if (tag == "environment") {
            std::string texture_name;
            input >> scene.environment.color.x >> scene.environment.color.y >> scene.environment.color.z >> scene.environment.strength;
            if (!input) {
                return fail("Invalid environment at line " + std::to_string(line_number));
            }
            if (input >> texture_name) {
                scene.environment.texture = find_texture(scene, texture_name);
                scene.environment.constant = false;
                environment_texture_name = texture_name;
            } else {
                scene.environment.texture = nullptr;
                scene.environment.constant = true;
                environment_texture_name.clear();
            }
        } else if (tag == "environment_mapping") {
            std::string mapping_name;
            input >> mapping_name;
            if (!input) {
                return fail("Invalid environment_mapping at line " + std::to_string(line_number));
            }
            scene.environment.mapping = parse_environment_mapping(mapping_name);
        } else if (tag == "environment_basis") {
            input >> scene.environment.light_from_world_x.x >> scene.environment.light_from_world_x.y >> scene.environment.light_from_world_x.z;
            input >> scene.environment.light_from_world_y.x >> scene.environment.light_from_world_y.y >> scene.environment.light_from_world_y.z;
            input >> scene.environment.light_from_world_z.x >> scene.environment.light_from_world_z.y >> scene.environment.light_from_world_z.z;
            if (!input) {
                return fail("Invalid environment_basis at line " + std::to_string(line_number));
            }
        } else if (tag == "directional_light") {
            DirectionalLight light;
            input >> light.direction.x >> light.direction.y >> light.direction.z
                >> light.color.x >> light.color.y >> light.color.z
                >> light.intensity;
            if (!input || light.intensity < 0.0f) {
                return fail("Invalid directional_light at line " + std::to_string(line_number));
            }
            light.direction = normalize(light.direction);
            if (dot(light.direction, light.direction) > 0.0f && light.intensity > 0.0f) {
                scene.directional_lights.push_back(light);
            }
        } else if (tag == "npr_sampling") {
            input >> scene.render_settings.stylized_samples >> scene.render_settings.stylized_max_depth;
            if (!input) {
                return fail("Invalid npr_sampling at line " + std::to_string(line_number));
            }
            scene.render_settings.stylized_samples = std::clamp(scene.render_settings.stylized_samples, 1, 128);
            scene.render_settings.stylized_max_depth = std::clamp(scene.render_settings.stylized_max_depth, 0, 32);
            scene.has_render_settings = true;
        } else if (tag == "sampling") {
            int mode = 1; // NEE default
            int heuristic = 1; // Power default
            input >> mode >> heuristic;
            if (!input) {
                return fail("Invalid sampling at line " + std::to_string(line_number));
            }
            scene.render_settings.sampling_mode = std::clamp(mode, 0, 2);
            scene.render_settings.mis_heuristic = std::clamp(heuristic, 0, 1);
            scene.has_render_settings = true;
        } else if (tag == "irradiance_volume") {
            int enabled = scene.render_settings.use_irradiance_volume ? 1 : 0;
            int principled_gi = scene.render_settings.irradiance_volume_principled_gi ? 1 : 0;
            int debug_probes = scene.render_settings.irradiance_volume_debug_probes ? 1 : 0;
            int cache_enabled = scene.render_settings.irradiance_volume_cache_enabled ? 1 : 0;
            int auto_update = scene.render_settings.irradiance_volume_auto_update ? 1 : 0;
            int manual_bounds = scene.render_settings.irradiance_volume_manual_bounds ? 1 : 0;
            input >> enabled
                >> scene.render_settings.irradiance_volume_grid_resolution
                >> scene.render_settings.irradiance_volume_subgrid_resolution
                >> scene.render_settings.irradiance_volume_direction_resolution
                >> scene.render_settings.irradiance_volume_bake_samples
                >> scene.render_settings.irradiance_volume_bake_bounces
                >> scene.render_settings.irradiance_volume_bounds_inset
                >> principled_gi
                >> debug_probes
                >> scene.render_settings.irradiance_volume_debug_probe_radius_scale
                >> cache_enabled
                >> auto_update
                >> manual_bounds;
            if (!input) {
                return fail("Invalid irradiance_volume at line " + std::to_string(line_number));
            }
            scene.render_settings.use_irradiance_volume = enabled != 0;
            scene.render_settings.irradiance_volume_grid_resolution = std::clamp(scene.render_settings.irradiance_volume_grid_resolution, 2, 64);
            scene.render_settings.irradiance_volume_subgrid_resolution = std::clamp(scene.render_settings.irradiance_volume_subgrid_resolution, 2, 32);
            scene.render_settings.irradiance_volume_direction_resolution = std::clamp(scene.render_settings.irradiance_volume_direction_resolution, 1, 32);
            scene.render_settings.irradiance_volume_bake_samples = std::clamp(scene.render_settings.irradiance_volume_bake_samples, 1, 256);
            scene.render_settings.irradiance_volume_bake_bounces = std::clamp(scene.render_settings.irradiance_volume_bake_bounces, 1, 32);
            scene.render_settings.irradiance_volume_bounds_inset = std::clamp(scene.render_settings.irradiance_volume_bounds_inset, 0.0f, 0.45f);
            scene.render_settings.irradiance_volume_principled_gi = principled_gi != 0;
            scene.render_settings.irradiance_volume_debug_probes = debug_probes != 0;
            scene.render_settings.irradiance_volume_debug_probe_radius_scale = std::clamp(scene.render_settings.irradiance_volume_debug_probe_radius_scale, 0.0f, 2.0f);
            scene.render_settings.irradiance_volume_cache_enabled = cache_enabled != 0;
            scene.render_settings.irradiance_volume_auto_update = auto_update != 0;
            scene.render_settings.irradiance_volume_manual_bounds = manual_bounds != 0;
            int bake_backend = 0;
            if (!(input >> bake_backend)) {
                input.clear();  // old file format; default to GPU
                bake_backend = 0;
            }
            scene.render_settings.irradiance_volume_bake_backend = std::clamp(bake_backend, 0, 1);
            scene.has_render_settings = true;
        } else if (tag == "irradiance_volume_bounds") {
            input >> scene.render_settings.irradiance_volume_bounds_min.x
                >> scene.render_settings.irradiance_volume_bounds_min.y
                >> scene.render_settings.irradiance_volume_bounds_min.z
                >> scene.render_settings.irradiance_volume_bounds_max.x
                >> scene.render_settings.irradiance_volume_bounds_max.y
                >> scene.render_settings.irradiance_volume_bounds_max.z;
            if (!input) {
                return fail("Invalid irradiance_volume_bounds at line " + std::to_string(line_number));
            }
            scene.has_render_settings = true;
        } else if (tag == "lightmap") {
            int enabled = scene.render_settings.use_lightmap ? 1 : 0;
            int principled_gi = scene.render_settings.lightmap_principled_gi ? 1 : 0;
            int cache_enabled = scene.render_settings.lightmap_cache_enabled ? 1 : 0;
            int auto_update = scene.render_settings.lightmap_auto_update ? 1 : 0;
            input >> enabled
                >> scene.render_settings.lightmap_resolution
                >> scene.render_settings.lightmap_padding
                >> scene.render_settings.lightmap_dilation
                >> scene.render_settings.lightmap_bake_samples
                >> scene.render_settings.lightmap_bake_bounces
                >> principled_gi
                >> cache_enabled
                >> auto_update;
            if (!input) {
                return fail("Invalid lightmap at line " + std::to_string(line_number));
            }
            scene.render_settings.use_lightmap = enabled != 0;
            scene.render_settings.lightmap_resolution = std::clamp(scene.render_settings.lightmap_resolution, 16, 16384);
            scene.render_settings.lightmap_padding = std::clamp(scene.render_settings.lightmap_padding, 0, 64);
            scene.render_settings.lightmap_dilation = std::clamp(scene.render_settings.lightmap_dilation, 0, 64);
            scene.render_settings.lightmap_bake_samples = std::clamp(scene.render_settings.lightmap_bake_samples, 1, 1024);
            scene.render_settings.lightmap_bake_bounces = std::clamp(scene.render_settings.lightmap_bake_bounces, 1, 32);
            scene.render_settings.lightmap_principled_gi = principled_gi != 0;
            scene.render_settings.lightmap_cache_enabled = cache_enabled != 0;
            scene.render_settings.lightmap_auto_update = auto_update != 0;
            int lm_bake_backend = 0;
            if (!(input >> lm_bake_backend)) {
                input.clear();  // old file format; default to GPU
                lm_bake_backend = 0;
            }
            scene.render_settings.lightmap_bake_backend = std::clamp(lm_bake_backend, 0, 1);
            scene.has_render_settings = true;
        } else if (tag == "material") {
            std::string material_name;
            Vec3 albedo;
            Vec3 legacy_emission;
            input >> material_name >> albedo.x >> albedo.y >> albedo.z;
            if (!input) {
                return fail("Invalid material at line " + std::to_string(line_number));
            }

            std::vector<std::string> remaining_tokens;
            for (std::string token; input >> token;) {
                remaining_tokens.push_back(token);
            }
            if (remaining_tokens.size() >= 3 &&
                parse_float_token(remaining_tokens[0], legacy_emission.x) &&
                parse_float_token(remaining_tokens[1], legacy_emission.y) &&
                parse_float_token(remaining_tokens[2], legacy_emission.z)) {
                remaining_tokens.erase(remaining_tokens.begin(), remaining_tokens.begin() + 3);
            }

            BrdfModel brdf = BrdfModel::Lambertian;
            float roughness = 0.5f;
            float metallic = 0.0f;
            if (remaining_tokens.size() >= 3) {
                brdf = parse_brdf_model(remaining_tokens[0]);
                parse_float_token(remaining_tokens[1], roughness);
                parse_float_token(remaining_tokens[2], metallic);
            }
            roughness = brdf == BrdfModel::Dielectric
                ? std::clamp(roughness, 1.0f, 3.0f)
                : std::clamp(roughness, 0.02f, 1.0f);
            metallic = std::clamp(metallic, 0.0f, 1.0f);

            material_ids[material_name] = static_cast<int>(scene.materials.size());
            std::shared_ptr<Material> material = make_material(material_name, albedo, brdf, roughness, metallic);
            const std::string texture_name = remaining_tokens.size() >= 4 ? remaining_tokens[3] : std::string{};
            if (!texture_name.empty()) {
                std::shared_ptr<Texture> texture = find_texture(scene, texture_name);
                if (texture) {
                    assign_material_texture(*material, "base_color", texture);
                }
            }
            scene.materials.push_back(material);
            material_texture_names.push_back(texture_name);
            legacy_material_emissions.push_back(legacy_emission);
        } else if (tag == "material_properties") {
            std::string material_name;
            std::string alpha_mode;
            int double_sided = 0;
            input >> material_name;
            const int index = material_id(material_ids, material_name);
            if (!input || index < 0 || index >= static_cast<int>(scene.materials.size()) || !scene.materials[static_cast<size_t>(index)]) {
                return fail("Invalid material_properties material at line " + std::to_string(line_number));
            }
            Material& material = *scene.materials[static_cast<size_t>(index)];
            input >> material.alpha
                >> material.alpha_cutoff
                >> alpha_mode
                >> double_sided
                >> material.normal_scale
                >> material.emission.x
                >> material.emission.y
                >> material.emission.z;
            if (!input) {
                return fail("Invalid material_properties at line " + std::to_string(line_number));
            }
            material.alpha = std::clamp(material.alpha, 0.0f, 1.0f);
            material.alpha_cutoff = std::clamp(material.alpha_cutoff, 0.0f, 1.0f);
            material.alpha_mode = parse_alpha_mode_name(alpha_mode);
            material.double_sided = double_sided != 0;
            material.normal_scale = std::max(0.0f, material.normal_scale);
        } else if (tag == "material_texture") {
            std::string material_name;
            std::string slot;
            std::string texture_name;
            input >> material_name >> slot >> texture_name;
            const int index = material_id(material_ids, material_name);
            if (!input || index < 0 || index >= static_cast<int>(scene.materials.size()) || !scene.materials[static_cast<size_t>(index)]) {
                return fail("Invalid material_texture at line " + std::to_string(line_number));
            }
            std::shared_ptr<Texture> texture;
            if (texture_name != "none") {
                texture = find_texture(scene, texture_name);
                if (!texture) {
                    return fail("Unknown material_texture texture '" + texture_name + "' at line " + std::to_string(line_number));
                }
            }
            if (!assign_material_texture(*scene.materials[static_cast<size_t>(index)], slot, texture)) {
                return fail("Unknown material_texture slot '" + slot + "' at line " + std::to_string(line_number));
            }
        } else if (tag == "material_input") {
            std::string material_name;
            std::string slot;
            std::string texture_name;
            std::string role_name;
            std::string color_space_name;
            std::string channel_name;
            MaterialInput material_input;
            input >> material_name >> slot >> texture_name >> role_name >> color_space_name >> channel_name;
            const int index = material_id(material_ids, material_name);
            if (!input || index < 0 || index >= static_cast<int>(scene.materials.size()) || !scene.materials[static_cast<size_t>(index)]) {
                return fail("Invalid material_input at line " + std::to_string(line_number));
            }
            auto* standard = dynamic_cast<StandardSurfaceMaterial*>(scene.materials[static_cast<size_t>(index)].get());
            if (!standard) {
                return fail("material_input used on non-standard-surface material at line " + std::to_string(line_number));
            }
            if (texture_name != "none") {
                material_input.texture = find_texture(scene, texture_name);
                if (!material_input.texture) {
                    return fail("Unknown material_input texture '" + texture_name + "' at line " + std::to_string(line_number));
                }
            }
            material_input.role = parse_texture_role_name(role_name);
            material_input.color_space = parse_texture_color_space_name(color_space_name);
            material_input.channel = parse_material_input_channel_name(channel_name);
            input >> material_input.scalar_factor
                >> material_input.color_factor.x
                >> material_input.color_factor.y
                >> material_input.color_factor.z
                >> material_input.transform.uv_set
                >> material_input.transform.offset.x
                >> material_input.transform.offset.y
                >> material_input.transform.scale.x
                >> material_input.transform.scale.y
                >> material_input.transform.rotation;
            if (!input) {
                return fail("Invalid material_input values at line " + std::to_string(line_number));
            }
            if (!assign_standard_material_input(*standard, slot, std::move(material_input))) {
                return fail("Unknown material_input slot '" + slot + "' at line " + std::to_string(line_number));
            }
        } else if (tag == "principled_properties") {
            std::string material_name;
            input >> material_name;
            const int index = material_id(material_ids, material_name);
            if (!input || index < 0 || index >= static_cast<int>(scene.materials.size()) || !scene.materials[static_cast<size_t>(index)]) {
                return fail("Invalid principled_properties material at line " + std::to_string(line_number));
            }
            auto* principled = dynamic_cast<PrincipledMaterial*>(scene.materials[static_cast<size_t>(index)].get());
            if (!principled) {
                return fail("principled_properties used on non-principled material at line " + std::to_string(line_number));
            }
            input >> principled->sheen_color.x
                >> principled->sheen_color.y
                >> principled->sheen_color.z
                >> principled->sheen_roughness
                >> principled->clearcoat
                >> principled->clearcoat_roughness;
            if (!input) {
                return fail("Invalid principled_properties at line " + std::to_string(line_number));
            }
            principled->sheen_roughness = std::clamp(principled->sheen_roughness, 0.0f, 1.0f);
            principled->clearcoat = std::clamp(principled->clearcoat, 0.0f, 1.0f);
            principled->clearcoat_roughness = std::clamp(principled->clearcoat_roughness, 0.0f, 1.0f);
        } else if (tag == "dielectric_properties") {
            std::string material_name;
            input >> material_name;
            const int index = material_id(material_ids, material_name);
            if (!input || index < 0 || index >= static_cast<int>(scene.materials.size()) || !scene.materials[static_cast<size_t>(index)]) {
                return fail("Invalid dielectric_properties material at line " + std::to_string(line_number));
            }
            auto* dielectric = dynamic_cast<DielectricMaterial*>(scene.materials[static_cast<size_t>(index)].get());
            if (!dielectric) {
                return fail("dielectric_properties used on non-dielectric material at line " + std::to_string(line_number));
            }
            input >> dielectric->transmission_tint.x >> dielectric->transmission_tint.y >> dielectric->transmission_tint.z;
            if (!input) {
                return fail("Invalid dielectric_properties at line " + std::to_string(line_number));
            }
        } else if (tag == "standard_surface") {
            std::string material_name;
            input >> material_name;
            const int index = material_id(material_ids, material_name);
            if (!input || index < 0 || index >= static_cast<int>(scene.materials.size()) || !scene.materials[static_cast<size_t>(index)]) {
                return fail("Invalid standard_surface material at line " + std::to_string(line_number));
            }
            auto* standard = dynamic_cast<StandardSurfaceMaterial*>(scene.materials[static_cast<size_t>(index)].get());
            if (!standard) {
                return fail("standard_surface used on non-standard-surface material at line " + std::to_string(line_number));
            }
            input >> standard->specular_weight
                >> standard->specular_ior
                >> standard->transmission_weight
                >> standard->transmission_color.x
                >> standard->transmission_color.y
                >> standard->transmission_color.z
                >> standard->coat_weight
                >> standard->coat_roughness
                >> standard->sheen_color.x
                >> standard->sheen_color.y
                >> standard->sheen_color.z
                >> standard->sheen_weight
                >> standard->sheen_roughness
                >> standard->subsurface_weight
                >> standard->volume_density;
            if (!input) {
                return fail("Invalid standard_surface values at line " + std::to_string(line_number));
            }
            standard->specular_weight = std::max(0.0f, standard->specular_weight);
            standard->specular_ior = std::clamp(standard->specular_ior, 1.0f, 3.0f);
            standard->transmission_weight = std::clamp(standard->transmission_weight, 0.0f, 1.0f);
            standard->coat_weight = std::clamp(standard->coat_weight, 0.0f, 1.0f);
            standard->coat_roughness = std::clamp(standard->coat_roughness, 0.0f, 1.0f);
            standard->sheen_weight = std::clamp(standard->sheen_weight, 0.0f, 1.0f);
            standard->sheen_roughness = std::clamp(standard->sheen_roughness, 0.0f, 1.0f);
            standard->unsupported_subsurface = standard->subsurface_weight > 0.0f;
            standard->unsupported_volume = standard->volume_density > 0.0f;
        } else if (tag == "npr") {
            std::string material_name;
            std::string style_name;
            input >> material_name >> style_name;
            const int index = material_id(material_ids, material_name);
            if (!input || index < 0 || index >= static_cast<int>(scene.materials.size()) || !scene.materials[static_cast<size_t>(index)]) {
                return fail("Invalid npr material at line " + std::to_string(line_number));
            }
            std::vector<std::string> npr_tokens{style_name};
            for (std::string token; input >> token;) {
                npr_tokens.push_back(token);
            }
            parse_npr_settings(npr_tokens, scene.materials[static_cast<size_t>(index)]->npr);
        } else if (tag == "light") {
            std::string mesh_name;
            LightComponent light;
            input >> mesh_name >> light.color.x >> light.color.y >> light.color.z >> light.intensity;
            if (!input || light.intensity < 0.0f) {
                return fail("Invalid light at line " + std::to_string(line_number));
            }
            int double_sided = light.double_sided ? 1 : 0;
            if (input >> double_sided) {
                light.double_sided = double_sided != 0;
            }
            light.enabled = light.intensity > 0.0f;
            pending_lights.push_back({mesh_name, light});
        } else if (tag == "sphere") {
            Sphere sphere;
            std::string material_name;
            input >> sphere.name >> material_name >> sphere.center.x >> sphere.center.y >> sphere.center.z >> sphere.radius;
            sphere.material = material_id(material_ids, material_name);
            if (!input || sphere.material < 0 || sphere.radius <= 0.0f) {
                return fail("Invalid sphere at line " + std::to_string(line_number));
            }
            int exclude_from_irradiance_volume_bake = 0;
            if (input >> exclude_from_irradiance_volume_bake) {
                sphere.exclude_from_irradiance_volume_bake = exclude_from_irradiance_volume_bake != 0;
            }
            scene.spheres.push_back(sphere);
            scene.uses_builtin_default_meshes = false;
        } else if (tag == "mesh") {
            Mesh mesh;
            std::string material_name;
            int vertex_count = 0;
            int triangle_count = 0;
            input >> mesh.name >> material_name >> mesh.translation.x >> mesh.translation.y >> mesh.translation.z >>
                mesh.rotation.x >> mesh.rotation.y >> mesh.rotation.z >> mesh.scale.x;
            if (!(input >> mesh.scale.y >> mesh.scale.z >> vertex_count >> triangle_count)) {
                mesh.scale.y = mesh.scale.x;
                mesh.scale.z = mesh.scale.x;
                input.clear();
                std::istringstream retry(line);
                retry >> tag >> mesh.name >> material_name >> mesh.translation.x >> mesh.translation.y >> mesh.translation.z >>
                    mesh.rotation.x >> mesh.rotation.y >> mesh.rotation.z >> mesh.scale.x >> vertex_count >> triangle_count;
                int exclude_from_irradiance_volume_bake = 0;
                if (retry >> exclude_from_irradiance_volume_bake) {
                    mesh.exclude_from_irradiance_volume_bake = exclude_from_irradiance_volume_bake != 0;
                }
            }
            mesh.material = material_id(material_ids, material_name);
            if (!input || mesh.material < 0 || vertex_count < 0 || triangle_count < 0) {
                return fail("Invalid mesh header at line " + std::to_string(line_number));
            }
            int exclude_from_irradiance_volume_bake = 0;
            if (input >> exclude_from_irradiance_volume_bake) {
                mesh.exclude_from_irradiance_volume_bake = exclude_from_irradiance_volume_bake != 0;
            }

            for (int i = 0; i < vertex_count; ++i) {
                Vec3 vertex;
                if (!(file >> vertex.x >> vertex.y >> vertex.z)) {
                    return fail("Invalid mesh vertex data after line " + std::to_string(line_number));
                }
                mesh.vertices.push_back(vertex);
                mesh.texcoords.push_back(scene_detail::default_uv(vertex));
            }

            std::string first_index_token;
            if (triangle_count > 0) {
                if (!(file >> first_index_token)) {
                    return fail("Invalid mesh index data after line " + std::to_string(line_number));
                }
                while (first_index_token == "normals" || first_index_token == "texcoords") {
                    if (first_index_token == "normals") {
                        int normal_count = 0;
                        if (!(file >> normal_count) || normal_count < 0) {
                            return fail("Invalid mesh normal header after line " + std::to_string(line_number));
                        }
                        mesh.normals.clear();
                        mesh.normals.reserve(static_cast<size_t>(normal_count));
                        for (int i = 0; i < normal_count; ++i) {
                            Vec3 normal;
                            if (!(file >> normal.x >> normal.y >> normal.z)) {
                                return fail("Invalid mesh normal data after line " + std::to_string(line_number));
                            }
                            mesh.normals.push_back(normalize(normal));
                        }
                        if (!mesh.normals.empty() && mesh.normals.size() != mesh.vertices.size()) {
                            return fail("Mesh normal count must match vertex count after line " + std::to_string(line_number));
                        }
                    } else {
                        int texcoord_count = 0;
                        if (!(file >> texcoord_count) || texcoord_count < 0) {
                            return fail("Invalid mesh texcoord header after line " + std::to_string(line_number));
                        }
                        mesh.texcoords.clear();
                        mesh.texcoords.reserve(static_cast<size_t>(texcoord_count));
                        for (int i = 0; i < texcoord_count; ++i) {
                            Vec2 uv;
                            if (!(file >> uv.x >> uv.y)) {
                                return fail("Invalid mesh texcoord data after line " + std::to_string(line_number));
                            }
                            mesh.texcoords.push_back(uv);
                        }
                        if (!mesh.texcoords.empty() && mesh.texcoords.size() != mesh.vertices.size()) {
                            return fail("Mesh texcoord count must match vertex count after line " + std::to_string(line_number));
                        }
                    }
                    if (!(file >> first_index_token)) {
                        return fail("Invalid mesh index data after line " + std::to_string(line_number));
                    }
                }
            }

            for (int i = 0; i < triangle_count; ++i) {
                uint32_t a = 0;
                uint32_t b = 0;
                uint32_t c = 0;
                if (i == 0) {
                    std::istringstream first_index(first_index_token);
                    if (!(first_index >> a) || !first_index.eof() || !(file >> b >> c)) {
                        return fail("Invalid mesh index data after line " + std::to_string(line_number));
                    }
                } else if (!(file >> a >> b >> c)) {
                    return fail("Invalid mesh index data after line " + std::to_string(line_number));
                }
                if (a >= mesh.vertices.size() || b >= mesh.vertices.size() || c >= mesh.vertices.size()) {
                    return fail("Mesh index out of range after line " + std::to_string(line_number));
                }
                scene_detail::add_triangle(mesh, a, b, c);
            }

            scene_detail::center_mesh_origin(mesh);
            scene_detail::ensure_default_uvs(mesh);
            if (mesh.material >= 0 && mesh.material < static_cast<int>(legacy_material_emissions.size()) &&
                scene_detail::has_light_emission(legacy_material_emissions[static_cast<size_t>(mesh.material)])) {
                mesh.light = scene_detail::make_light_from_emission(legacy_material_emissions[static_cast<size_t>(mesh.material)]);
            }
            scene.meshes.push_back(mesh);
            scene.uses_builtin_default_meshes = false;
        } else {
            return fail("Unknown scene token '" + tag + "' at line " + std::to_string(line_number));
        }
    }

    if (scene.meshes.empty() && scene.spheres.empty()) {
        LT_LOG_WARN("Scene '{}' contains no geometry; using built-in default meshes", path);
        Scene defaults = make_default_scene();
        if (!saw_camera) {
            scene.camera = defaults.camera;
        }
        if (scene.materials.empty()) {
            scene.materials = std::move(defaults.materials);
        }
        scene.meshes = std::move(defaults.meshes);
        scene.spheres = std::move(defaults.spheres);
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
    LT_LOG_INFO("Loaded .lt scene '{}' (meshes={}, spheres={}, materials={}, textures={}, directional_lights={})",
        path, scene.meshes.size(), scene.spheres.size(), scene.materials.size(), scene.textures.size(), scene.directional_lights.size());
    return {scene, {}};
}

bool save_scene(const Scene& scene, const std::string& path, std::string& error) {
    std::ofstream output(path);
    if (!output) {
        error = "Could not write scene: " + path;
        LT_LOG_ERROR("Could not save scene '{}': {}", path, error);
        return false;
    }

    if (scene.uses_builtin_default_meshes) {
        output << "# Mesh-based Cornell-ish test scene.\n";
        output << "# If no mesh blocks are listed, the built-in mesh Cornell scene is used.\n";
    }

    output << "camera "
        << scene.camera.position.x << ' ' << scene.camera.position.y << ' ' << scene.camera.position.z << "   "
        << scene.camera.target.x << ' ' << scene.camera.target.y << ' ' << scene.camera.target.z << "   "
        << scene.camera.fov_degrees << "\n\n";

    for (const std::shared_ptr<Texture>& texture : scene.textures) {
        if (texture) {
            output << "texture " << texture->name << ' ' << texture->path << '\n';
        }
    }
    if (!scene.textures.empty()) {
        output << '\n';
    }

    if (scene.environment.constant || scene.environment.texture) {
        output << "environment "
            << scene.environment.color.x << ' ' << scene.environment.color.y << ' ' << scene.environment.color.z << ' '
            << scene.environment.strength;
        if (scene.environment.texture) {
            output << ' ' << scene.environment.texture->name;
        }
        output << '\n';
        if (scene.environment.mapping != Environment::Mapping::Equirectangular) {
            output << "environment_mapping " << environment_mapping_name(scene.environment.mapping) << '\n';
        }
        if (!is_default_environment_basis(scene.environment)) {
            output << "environment_basis "
                << scene.environment.light_from_world_x.x << ' ' << scene.environment.light_from_world_x.y << ' ' << scene.environment.light_from_world_x.z << ' '
                << scene.environment.light_from_world_y.x << ' ' << scene.environment.light_from_world_y.y << ' ' << scene.environment.light_from_world_y.z << ' '
                << scene.environment.light_from_world_z.x << ' ' << scene.environment.light_from_world_z.y << ' ' << scene.environment.light_from_world_z.z << '\n';
        }
        output << '\n';
    }

    for (const DirectionalLight& light : scene.directional_lights) {
        if (dot(light.direction, light.direction) <= 0.0f || light.intensity <= 0.0f) {
            continue;
        }
        output << "directional_light "
            << light.direction.x << ' ' << light.direction.y << ' ' << light.direction.z << ' '
            << light.color.x << ' ' << light.color.y << ' ' << light.color.z << ' '
            << light.intensity << '\n';
    }
    if (!scene.directional_lights.empty()) {
        output << '\n';
    }

    if (scene.has_render_settings) {
        const SceneRenderSettings& settings = scene.render_settings;
        output << "npr_sampling "
            << settings.stylized_samples << ' '
            << settings.stylized_max_depth << '\n';
        output << "sampling "
            << settings.sampling_mode << ' '
            << settings.mis_heuristic << '\n';
        output << "irradiance_volume "
            << (settings.use_irradiance_volume ? 1 : 0) << ' '
            << settings.irradiance_volume_grid_resolution << ' '
            << settings.irradiance_volume_subgrid_resolution << ' '
            << settings.irradiance_volume_direction_resolution << ' '
            << settings.irradiance_volume_bake_samples << ' '
            << settings.irradiance_volume_bake_bounces << ' '
            << settings.irradiance_volume_bounds_inset << ' '
            << (settings.irradiance_volume_principled_gi ? 1 : 0) << ' '
            << (settings.irradiance_volume_debug_probes ? 1 : 0) << ' '
            << settings.irradiance_volume_debug_probe_radius_scale << ' '
            << (settings.irradiance_volume_cache_enabled ? 1 : 0) << ' '
            << (settings.irradiance_volume_auto_update ? 1 : 0) << ' '
            << (settings.irradiance_volume_manual_bounds ? 1 : 0) << ' '
            << settings.irradiance_volume_bake_backend << '\n';
        if (settings.irradiance_volume_manual_bounds) {
            output << "irradiance_volume_bounds "
                << settings.irradiance_volume_bounds_min.x << ' '
                << settings.irradiance_volume_bounds_min.y << ' '
                << settings.irradiance_volume_bounds_min.z << ' '
                << settings.irradiance_volume_bounds_max.x << ' '
                << settings.irradiance_volume_bounds_max.y << ' '
                << settings.irradiance_volume_bounds_max.z << '\n';
        }
        output << "lightmap "
            << (settings.use_lightmap ? 1 : 0) << ' '
            << settings.lightmap_resolution << ' '
            << settings.lightmap_padding << ' '
            << settings.lightmap_dilation << ' '
            << settings.lightmap_bake_samples << ' '
            << settings.lightmap_bake_bounces << ' '
            << (settings.lightmap_principled_gi ? 1 : 0) << ' '
            << (settings.lightmap_cache_enabled ? 1 : 0) << ' '
            << (settings.lightmap_auto_update ? 1 : 0) << ' '
            << settings.lightmap_bake_backend << '\n';
        output << '\n';
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
        } else if (const auto* standard = dynamic_cast<const StandardSurfaceMaterial*>(material.get())) {
            roughness = standard->roughness;
            metallic = standard->metalness;
        }
        output << "material " << material->name << ' '
            << material->albedo.x << ' ' << material->albedo.y << ' ' << material->albedo.z << ' '
            << material->model_name() << ' ' << roughness << ' ' << metallic;
        if (material->albedo_texture) {
            output << ' ' << material->albedo_texture->name;
        }
        output << '\n';
    }

    for (const std::shared_ptr<Material>& material : scene.materials) {
        if (!material) {
            continue;
        }
        if (has_nondefault_material_properties(*material)) {
            output << "material_properties " << material->name << ' '
                << material->alpha << ' '
                << material->alpha_cutoff << ' '
                << alpha_mode_name(material->alpha_mode) << ' '
                << (material->double_sided ? 1 : 0) << ' '
                << material->normal_scale << ' '
                << material->emission.x << ' '
                << material->emission.y << ' '
                << material->emission.z << '\n';
        }
        write_material_texture(output, *material, "normal", material->normal_texture);
        write_material_texture(output, *material, "emission", material->emission_texture);
        if (const auto* principled = dynamic_cast<const PrincipledMaterial*>(material.get())) {
            if (has_nonzero(principled->sheen_color) ||
                principled->sheen_roughness != 0.0f ||
                principled->clearcoat != 0.0f ||
                principled->clearcoat_roughness != 0.0f) {
                output << "principled_properties " << material->name << ' '
                    << principled->sheen_color.x << ' '
                    << principled->sheen_color.y << ' '
                    << principled->sheen_color.z << ' '
                    << principled->sheen_roughness << ' '
                    << principled->clearcoat << ' '
                    << principled->clearcoat_roughness << '\n';
            }
            write_material_texture(output, *material, "metallic_roughness", principled->metallic_roughness_texture);
            write_material_texture(output, *material, "sheen_color", principled->sheen_color_texture);
            write_material_texture(output, *material, "sheen_roughness", principled->sheen_roughness_texture);
            write_material_texture(output, *material, "clearcoat", principled->clearcoat_texture);
            write_material_texture(output, *material, "clearcoat_roughness", principled->clearcoat_roughness_texture);
        } else if (const auto* dielectric = dynamic_cast<const DielectricMaterial*>(material.get())) {
            if (dielectric->transmission_tint.x != 1.0f ||
                dielectric->transmission_tint.y != 1.0f ||
                dielectric->transmission_tint.z != 1.0f) {
                output << "dielectric_properties " << material->name << ' '
                    << dielectric->transmission_tint.x << ' '
                    << dielectric->transmission_tint.y << ' '
                    << dielectric->transmission_tint.z << '\n';
            }
        } else if (const auto* standard = dynamic_cast<const StandardSurfaceMaterial*>(material.get())) {
            if (standard->specular_weight != 1.0f ||
                standard->specular_ior != 1.5f ||
                standard->transmission_weight != 0.0f ||
                standard->transmission_color.x != 1.0f ||
                standard->transmission_color.y != 1.0f ||
                standard->transmission_color.z != 1.0f ||
                standard->coat_weight != 0.0f ||
                standard->coat_roughness != 0.0f ||
                has_nonzero(standard->sheen_color) ||
                standard->sheen_weight != 0.0f ||
                standard->sheen_roughness != 0.0f ||
                standard->subsurface_weight != 0.0f ||
                standard->volume_density != 0.0f) {
                output << "standard_surface " << material->name << ' '
                    << standard->specular_weight << ' '
                    << standard->specular_ior << ' '
                    << standard->transmission_weight << ' '
                    << standard->transmission_color.x << ' '
                    << standard->transmission_color.y << ' '
                    << standard->transmission_color.z << ' '
                    << standard->coat_weight << ' '
                    << standard->coat_roughness << ' '
                    << standard->sheen_color.x << ' '
                    << standard->sheen_color.y << ' '
                    << standard->sheen_color.z << ' '
                    << standard->sheen_weight << ' '
                    << standard->sheen_roughness << ' '
                    << standard->subsurface_weight << ' '
                    << standard->volume_density << '\n';
            }
            write_material_input(output, *standard, "base_color", standard->base_color_input);
            write_material_input(output, *standard, "roughness", standard->roughness_input);
            write_material_input(output, *standard, "metalness", standard->metalness_input);
            write_material_input(output, *standard, "specular", standard->specular_weight_input);
            write_material_input(output, *standard, "transmission", standard->transmission_input);
            write_material_input(output, *standard, "opacity", standard->opacity_input);
            write_material_input(output, *standard, "emission", standard->emission_input);
            write_material_input(output, *standard, "coat", standard->coat_input);
            write_material_input(output, *standard, "coat_roughness", standard->coat_roughness_input);
            write_material_input(output, *standard, "sheen_color", standard->sheen_color_input);
            write_material_input(output, *standard, "sheen_roughness", standard->sheen_roughness_input);
        }
    }
    output << '\n';

    for (const std::shared_ptr<Material>& material : scene.materials) {
        if (!material || material->npr.style == NprStyle::None) {
            continue;
        }
        const NprSettings& npr = material->npr;
        output << "npr " << material->name << ' ' << npr_style_name(npr.style);
        if (npr.style == NprStyle::ColorMap) {
            output << ' ' << npr.value_min << ' ' << npr.value_max;
        } else if (npr.style == NprStyle::XToon) {
            output << ' ' << xtoon_detail_mode_name(npr.xtoon_detail_mode) << ' '
                << npr.xtoon_steps << ' '
                << npr.xtoon_shadow.x << ' ' << npr.xtoon_shadow.y << ' ' << npr.xtoon_shadow.z << ' '
                << npr.xtoon_mid.x << ' ' << npr.xtoon_mid.y << ' ' << npr.xtoon_mid.z << ' '
                << npr.xtoon_lit.x << ' ' << npr.xtoon_lit.y << ' ' << npr.xtoon_lit.z << ' '
                << npr.xtoon_accent.x << ' ' << npr.xtoon_accent.y << ' ' << npr.xtoon_accent.z << ' '
                << npr.xtoon_detail_strength << ' '
                << npr.xtoon_detail_threshold << ' '
                << npr.xtoon_detail_power << ' '
                << npr.xtoon_depth_near << ' '
                << npr.xtoon_depth_far;
        } else if (npr.style == NprStyle::CrossHatching) {
            output << ' ' << npr.hatch_sets << ' '
                << npr.hatch_spacing << ' '
                << npr.hatch_width << ' '
                << npr.hatch_angle << ' '
                << npr.hatch_value_min << ' '
                << npr.hatch_value_max << ' '
                << npr.hatch_ink.x << ' ' << npr.hatch_ink.y << ' ' << npr.hatch_ink.z << ' '
                << npr.hatch_paper.x << ' ' << npr.hatch_paper.y << ' ' << npr.hatch_paper.z << ' '
                << (npr.hatch_passthrough ? 1 : 0) << ' '
                << (npr.hatch_shadow_only ? 1 : 0);
        }
        output << '\n';
    }
    output << '\n';

    if (scene.uses_builtin_default_meshes) {
        LT_LOG_INFO("Saved scene '{}' (built-in default meshes, materials={}, textures={}, directional_lights={})",
            path, scene.materials.size(), scene.textures.size(), scene.directional_lights.size());
        return true;
    }

    for (const Mesh& mesh : scene.meshes) {
        if (!mesh.light.enabled || mesh.light.intensity <= 0.0f) {
            continue;
        }
        output << "light " << mesh.name << ' '
            << mesh.light.color.x << ' ' << mesh.light.color.y << ' ' << mesh.light.color.z << ' '
            << mesh.light.intensity << ' ' << (mesh.light.double_sided ? 1 : 0) << '\n';
    }
    output << '\n';

    for (const Sphere& sphere : scene.spheres) {
        if (sphere.material < 0 || sphere.material >= static_cast<int>(scene.materials.size()) || !scene.materials[static_cast<size_t>(sphere.material)]) {
            continue;
        }
        output << "sphere " << sphere.name << ' ' << scene.materials[static_cast<size_t>(sphere.material)]->name << ' '
            << sphere.center.x << ' ' << sphere.center.y << ' ' << sphere.center.z << ' ' << sphere.radius << ' '
            << (sphere.exclude_from_irradiance_volume_bake ? 1 : 0) << '\n';
    }
    if (!scene.spheres.empty()) {
        output << '\n';
    }

    for (const Mesh& mesh : scene.meshes) {
        output << "mesh " << mesh.name << ' ' << scene.materials[static_cast<size_t>(mesh.material)]->name << ' '
            << mesh.translation.x << ' ' << mesh.translation.y << ' ' << mesh.translation.z << ' '
            << mesh.rotation.x << ' ' << mesh.rotation.y << ' ' << mesh.rotation.z << ' '
            << mesh.scale.x << ' ' << mesh.scale.y << ' ' << mesh.scale.z << ' '
            << mesh.vertices.size() << ' ' << mesh.indices.size() / 3 << ' '
            << (mesh.exclude_from_irradiance_volume_bake ? 1 : 0) << '\n';
        for (Vec3 vertex : mesh.vertices) {
            output << vertex.x << ' ' << vertex.y << ' ' << vertex.z << '\n';
        }
        if (mesh.normals.size() == mesh.vertices.size()) {
            output << "normals " << mesh.normals.size() << '\n';
            for (Vec3 normal : mesh.normals) {
                output << normal.x << ' ' << normal.y << ' ' << normal.z << '\n';
            }
        }
        if (mesh.texcoords.size() == mesh.vertices.size()) {
            output << "texcoords " << mesh.texcoords.size() << '\n';
            for (Vec2 uv : mesh.texcoords) {
                output << uv.x << ' ' << uv.y << '\n';
            }
        }
        for (size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
            output << mesh.indices[i] << ' ' << mesh.indices[i + 1] << ' ' << mesh.indices[i + 2] << '\n';
        }
    }
    LT_LOG_INFO("Saved scene '{}' (meshes={}, spheres={}, materials={}, textures={}, directional_lights={})",
        path, scene.meshes.size(), scene.spheres.size(), scene.materials.size(), scene.textures.size(), scene.directional_lights.size());
    return true;
}

} // namespace lt
