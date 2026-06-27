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
            LT_LOG_INFO("Loaded PBRT scene '{}' (meshes={}, spheres={}, materials={}, textures={})",
                path, result.scene.meshes.size(), result.scene.spheres.size(), result.scene.materials.size(), result.scene.textures.size());
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
            material->albedo_texture = texture_name.empty() ? nullptr : find_texture(scene, texture_name);
            scene.materials.push_back(material);
            material_texture_names.push_back(texture_name);
            legacy_material_emissions.push_back(legacy_emission);
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
            }
            mesh.material = material_id(material_ids, material_name);
            if (!input || mesh.material < 0 || vertex_count < 0 || triangle_count < 0) {
                return fail("Invalid mesh header at line " + std::to_string(line_number));
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
                if (first_index_token == "normals") {
                    int normal_count = 0;
                    if (!(file >> normal_count) || normal_count < 0) {
                        return fail("Invalid mesh normal header after line " + std::to_string(line_number));
                    }
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
    LT_LOG_INFO("Loaded .lt scene '{}' (meshes={}, spheres={}, materials={}, textures={})",
        path, scene.meshes.size(), scene.spheres.size(), scene.materials.size(), scene.textures.size());
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
        output << "\n\n";
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
        output << "material " << material->name << ' '
            << material->albedo.x << ' ' << material->albedo.y << ' ' << material->albedo.z << ' '
            << material->model_name() << ' ' << roughness << ' ' << metallic;
        if (material->albedo_texture) {
            output << ' ' << material->albedo_texture->name;
        }
        output << '\n';
    }

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
        LT_LOG_INFO("Saved scene '{}' (built-in default meshes, materials={}, textures={})",
            path, scene.materials.size(), scene.textures.size());
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
            << sphere.center.x << ' ' << sphere.center.y << ' ' << sphere.center.z << ' ' << sphere.radius << '\n';
    }
    if (!scene.spheres.empty()) {
        output << '\n';
    }

    for (const Mesh& mesh : scene.meshes) {
        output << "mesh " << mesh.name << ' ' << scene.materials[static_cast<size_t>(mesh.material)]->name << ' '
            << mesh.translation.x << ' ' << mesh.translation.y << ' ' << mesh.translation.z << ' '
            << mesh.rotation.x << ' ' << mesh.rotation.y << ' ' << mesh.rotation.z << ' '
            << mesh.scale.x << ' ' << mesh.scale.y << ' ' << mesh.scale.z << ' '
            << mesh.vertices.size() << ' ' << mesh.indices.size() / 3 << '\n';
        for (Vec3 vertex : mesh.vertices) {
            output << vertex.x << ' ' << vertex.y << ' ' << vertex.z << '\n';
        }
        if (mesh.normals.size() == mesh.vertices.size()) {
            output << "normals " << mesh.normals.size() << '\n';
            for (Vec3 normal : mesh.normals) {
                output << normal.x << ' ' << normal.y << ' ' << normal.z << '\n';
            }
        }
        for (size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
            output << mesh.indices[i] << ' ' << mesh.indices[i + 1] << ' ' << mesh.indices[i + 2] << '\n';
        }
    }
    LT_LOG_INFO("Saved scene '{}' (meshes={}, spheres={}, materials={}, textures={})",
        path, scene.meshes.size(), scene.spheres.size(), scene.materials.size(), scene.textures.size());
    return true;
}

} // namespace lt
