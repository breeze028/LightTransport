#include "lt/scene.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace lt {
namespace {

struct Mat4 {
    float m[4][4] = {
        {1.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 1.0f},
    };
};

struct Param {
    std::string type;
    std::string name;
    std::vector<std::string> values;
};

struct Token {
    std::string text;
    int line = 1;
};

struct ImportState {
    Mat4 transform;
    int material = 0;
    LightComponent area_light;
};

struct MeshLoadResult {
    Mesh mesh;
    std::string error;
};

struct PendingMesh {
    std::string object;
    std::future<MeshLoadResult> future;
};

Mat4 multiply(const Mat4& a, const Mat4& b) {
    Mat4 r{};
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            r.m[y][x] = 0.0f;
            for (int k = 0; k < 4; ++k) {
                r.m[y][x] += a.m[y][k] * b.m[k][x];
            }
        }
    }
    return r;
}

Mat4 transpose(const Mat4& m) {
    Mat4 r;
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            r.m[y][x] = m.m[x][y];
        }
    }
    return r;
}

Vec3 transform_point(const Mat4& m, Vec3 p) {
    const float x = m.m[0][0] * p.x + m.m[0][1] * p.y + m.m[0][2] * p.z + m.m[0][3];
    const float y = m.m[1][0] * p.x + m.m[1][1] * p.y + m.m[1][2] * p.z + m.m[1][3];
    const float z = m.m[2][0] * p.x + m.m[2][1] * p.y + m.m[2][2] * p.z + m.m[2][3];
    const float w = m.m[3][0] * p.x + m.m[3][1] * p.y + m.m[3][2] * p.z + m.m[3][3];
    return std::fabs(w) > 1.0e-8f ? Vec3{x / w, y / w, z / w} : Vec3{x, y, z};
}

Vec3 transform_vector(const Mat4& m, Vec3 v) {
    return {
        m.m[0][0] * v.x + m.m[0][1] * v.y + m.m[0][2] * v.z,
        m.m[1][0] * v.x + m.m[1][1] * v.y + m.m[1][2] * v.z,
        m.m[2][0] * v.x + m.m[2][1] * v.y + m.m[2][2] * v.z,
    };
}

Vec3 inverse_transform_component_basis_assuming_rotation(const Mat4& m, int component) {
    return normalize({
        m.m[0][component],
        m.m[1][component],
        m.m[2][component],
    });
}

Mat4 translate(Vec3 t) {
    Mat4 m;
    m.m[0][3] = t.x;
    m.m[1][3] = t.y;
    m.m[2][3] = t.z;
    return m;
}

Mat4 scale(Vec3 s) {
    Mat4 m;
    m.m[0][0] = s.x;
    m.m[1][1] = s.y;
    m.m[2][2] = s.z;
    return m;
}

Mat4 rotate(float angle_degrees, Vec3 axis) {
    axis = normalize(axis);
    const float a = angle_degrees * kPi / 180.0f;
    const float c = std::cos(a);
    const float s = std::sin(a);
    const float t = 1.0f - c;
    Mat4 m;
    m.m[0][0] = t * axis.x * axis.x + c;
    m.m[0][1] = t * axis.x * axis.y - s * axis.z;
    m.m[0][2] = t * axis.x * axis.z + s * axis.y;
    m.m[1][0] = t * axis.x * axis.y + s * axis.z;
    m.m[1][1] = t * axis.y * axis.y + c;
    m.m[1][2] = t * axis.y * axis.z - s * axis.x;
    m.m[2][0] = t * axis.x * axis.z - s * axis.y;
    m.m[2][1] = t * axis.y * axis.z + s * axis.x;
    m.m[2][2] = t * axis.z * axis.z + c;
    return m;
}

bool parse_float(const std::string& token, float& value) {
    std::istringstream in(token);
    in >> value;
    return in && in.eof();
}

bool parse_int(const std::string& token, int& value) {
    std::istringstream in(token);
    in >> value;
    return in && in.eof();
}

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::filesystem::path resolve_path(const std::filesystem::path& base, const std::string& path) {
    std::filesystem::path p(path);
    return p.is_absolute() ? p : base / p;
}

std::vector<Token> tokenize(const std::string& text) {
    std::vector<Token> tokens;
    int line = 1;
    for (size_t i = 0; i < text.size();) {
        const char c = text[i];
        if (c == '\n') {
            ++line;
            ++i;
        } else if (std::isspace(static_cast<unsigned char>(c))) {
            ++i;
        } else if (c == '#') {
            while (i < text.size() && text[i] != '\n') ++i;
        } else if (c == '[' || c == ']') {
            tokens.push_back({std::string(1, c), line});
            ++i;
        } else if (c == '"') {
            std::string value;
            ++i;
            while (i < text.size() && text[i] != '"') {
                if (text[i] == '\\' && i + 1 < text.size()) ++i;
                value.push_back(text[i]);
                if (text[i] == '\n') ++line;
                ++i;
            }
            if (i < text.size()) ++i;
            tokens.push_back({std::move(value), line});
        } else {
            std::string value;
            while (i < text.size() && !std::isspace(static_cast<unsigned char>(text[i])) && text[i] != '[' && text[i] != ']' && text[i] != '#') {
                value.push_back(text[i++]);
            }
            tokens.push_back({std::move(value), line});
        }
    }
    return tokens;
}

bool is_directive(const std::string& token) {
    static const std::unordered_set<std::string> directives = {
        "WorldBegin", "WorldEnd", "AttributeBegin", "AttributeEnd", "TransformBegin", "TransformEnd",
        "Identity", "Translate", "Scale", "Rotate", "Transform", "ConcatTransform", "LookAt", "Camera",
        "Film", "Sampler", "Integrator", "PixelFilter", "Accelerator", "Include", "MakeNamedMaterial",
        "NamedMaterial", "Material", "Texture", "Shape", "AreaLightSource", "LightSource", "ReverseOrientation",
        "MediumInterface", "MakeNamedMedium", "ObjectBegin", "ObjectEnd", "ObjectInstance", "ActiveTransform",
        "CoordinateSystem", "CoordSysTransform"
    };
    return directives.count(token) > 0;
}

std::pair<std::string, std::string> split_param_name(const std::string& text) {
    const size_t space = text.find(' ');
    if (space == std::string::npos) return {"", text};
    return {lower(text.substr(0, space)), text.substr(space + 1)};
}

std::vector<Param> parse_params(const std::vector<Token>& tokens, size_t& pos) {
    std::vector<Param> params;
    while (pos < tokens.size() && !is_directive(tokens[pos].text)) {
        const auto [type, name] = split_param_name(tokens[pos++].text);
        if (name.empty()) break;
        Param param{type, name, {}};
        if (pos < tokens.size() && tokens[pos].text == "[") {
            ++pos;
            while (pos < tokens.size() && tokens[pos].text != "]") {
                param.values.push_back(tokens[pos++].text);
            }
            if (pos < tokens.size() && tokens[pos].text == "]") ++pos;
        } else if (pos < tokens.size() && !is_directive(tokens[pos].text)) {
            param.values.push_back(tokens[pos++].text);
        }
        params.push_back(std::move(param));
    }
    return params;
}

const Param* find_param(const std::vector<Param>& params, const std::string& name) {
    for (const Param& param : params) {
        if (param.name == name) return &param;
    }
    return nullptr;
}

float float_param(const std::vector<Param>& params, const std::string& name, float fallback) {
    const Param* param = find_param(params, name);
    if (!param || param->values.empty()) return fallback;
    float value = fallback;
    parse_float(param->values[0], value);
    return value;
}

Vec3 vec3_param(const std::vector<Param>& params, const std::string& name, Vec3 fallback) {
    const Param* param = find_param(params, name);
    if (!param || param->values.empty()) return fallback;
    if (param->values.size() == 1) {
        float value = 0.0f;
        return parse_float(param->values[0], value) ? Vec3{value} : fallback;
    }
    Vec3 value = fallback;
    parse_float(param->values[0], value.x);
    parse_float(param->values[1], value.y);
    parse_float(param->values[2], value.z);
    return value;
}

std::string string_param(const std::vector<Param>& params, const std::string& name, const std::string& fallback = {}) {
    const Param* param = find_param(params, name);
    return param && !param->values.empty() ? param->values[0] : fallback;
}

std::vector<float> float_array_param(const std::vector<Param>& params, const std::string& name) {
    std::vector<float> values;
    if (const Param* param = find_param(params, name)) {
        values.reserve(param->values.size());
        for (const std::string& token : param->values) {
            float value = 0.0f;
            if (parse_float(token, value)) values.push_back(value);
        }
    }
    return values;
}

std::vector<int> int_array_param(const std::vector<Param>& params, const std::string& name) {
    std::vector<int> values;
    if (const Param* param = find_param(params, name)) {
        values.reserve(param->values.size());
        for (const std::string& token : param->values) {
            int value = 0;
            if (parse_int(token, value)) values.push_back(value);
        }
    }
    return values;
}

int find_texture_index(const Scene& scene, const std::string& name) {
    for (int i = 0; i < static_cast<int>(scene.textures.size()); ++i) {
        if (scene.textures[static_cast<size_t>(i)] && scene.textures[static_cast<size_t>(i)]->name == name) return i;
    }
    return -1;
}

void apply_transform(Mesh& mesh, const Mat4& transform) {
    for (Vec3& vertex : mesh.vertices) {
        vertex = transform_point(transform, vertex);
    }
    for (Vec3& normal : mesh.normals) {
        normal = normalize(transform_vector(transform, normal));
    }
}

void triangulate_face(Mesh& mesh, const std::vector<uint32_t>& face) {
    for (size_t i = 1; i + 1 < face.size(); ++i) {
        mesh.indices.push_back(face[0]);
        mesh.indices.push_back(face[i]);
        mesh.indices.push_back(face[i + 1]);
    }
}

int scalar_size(const std::string& type) {
    const std::string t = lower(type);
    if (t == "char" || t == "uchar" || t == "int8" || t == "uint8") return 1;
    if (t == "short" || t == "ushort" || t == "int16" || t == "uint16") return 2;
    if (t == "int" || t == "uint" || t == "float" || t == "int32" || t == "uint32" || t == "float32") return 4;
    if (t == "double" || t == "float64") return 8;
    return 0;
}

double read_binary_scalar(std::istream& in, const std::string& type) {
    const std::string t = lower(type);
    if (t == "char" || t == "int8") { int8_t v = 0; in.read(reinterpret_cast<char*>(&v), 1); return v; }
    if (t == "uchar" || t == "uint8") { uint8_t v = 0; in.read(reinterpret_cast<char*>(&v), 1); return v; }
    if (t == "short" || t == "int16") { int16_t v = 0; in.read(reinterpret_cast<char*>(&v), 2); return v; }
    if (t == "ushort" || t == "uint16") { uint16_t v = 0; in.read(reinterpret_cast<char*>(&v), 2); return v; }
    if (t == "int" || t == "int32") { int32_t v = 0; in.read(reinterpret_cast<char*>(&v), 4); return v; }
    if (t == "uint" || t == "uint32") { uint32_t v = 0; in.read(reinterpret_cast<char*>(&v), 4); return v; }
    if (t == "float" || t == "float32") { float v = 0.0f; in.read(reinterpret_cast<char*>(&v), 4); return v; }
    if (t == "double" || t == "float64") { double v = 0.0; in.read(reinterpret_cast<char*>(&v), 8); return v; }
    return 0.0;
}

struct PlyProperty {
    bool list = false;
    std::string count_type;
    std::string item_type;
    std::string type;
    std::string name;
};

struct PlyElement {
    std::string name;
    int count = 0;
    std::vector<PlyProperty> properties;
};

bool load_ply_mesh(const std::filesystem::path& path, Mesh& mesh, std::string& error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        error = "Could not open PLY mesh: " + path.string();
        return false;
    }
    std::string line;
    std::getline(in, line);
    if (line != "ply") {
        error = "Invalid PLY mesh: " + path.string();
        return false;
    }

    std::string format;
    std::vector<PlyElement> elements;
    PlyElement* current = nullptr;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream ls(line);
        std::string word;
        ls >> word;
        if (word == "format") {
            ls >> format;
        } else if (word == "element") {
            PlyElement element;
            ls >> element.name >> element.count;
            elements.push_back(std::move(element));
            current = &elements.back();
        } else if (word == "property" && current) {
            PlyProperty prop;
            std::string type;
            ls >> type;
            if (type == "list") {
                prop.list = true;
                ls >> prop.count_type >> prop.item_type >> prop.name;
            } else {
                prop.type = type;
                ls >> prop.name;
            }
            current->properties.push_back(std::move(prop));
        } else if (word == "end_header") {
            break;
        }
    }
    if (format == "binary_big_endian") {
        error = "Unsupported big-endian PLY mesh: " + path.string();
        return false;
    }

    const auto read_ascii_scalar = [&](const std::string&) {
        double value = 0.0;
        in >> value;
        return value;
    };
    const bool binary = format == "binary_little_endian";
    for (const PlyElement& element : elements) {
        if (element.name == "vertex") {
            mesh.vertices.reserve(mesh.vertices.size() + static_cast<size_t>(element.count));
            for (int i = 0; i < element.count; ++i) {
                Vec3 p{};
                Vec3 n{};
                Vec2 uv{};
                bool has_n = false;
                bool has_uv = false;
                for (const PlyProperty& prop : element.properties) {
                    if (prop.list) {
                        const int count = static_cast<int>(binary ? read_binary_scalar(in, prop.count_type) : read_ascii_scalar(prop.count_type));
                        for (int j = 0; j < count; ++j) (void)(binary ? read_binary_scalar(in, prop.item_type) : read_ascii_scalar(prop.item_type));
                        continue;
                    }
                    const float v = static_cast<float>(binary ? read_binary_scalar(in, prop.type) : read_ascii_scalar(prop.type));
                    if (prop.name == "x") p.x = v;
                    else if (prop.name == "y") p.y = v;
                    else if (prop.name == "z") p.z = v;
                    else if (prop.name == "nx") { n.x = v; has_n = true; }
                    else if (prop.name == "ny") { n.y = v; has_n = true; }
                    else if (prop.name == "nz") { n.z = v; has_n = true; }
                    else if (prop.name == "u" || prop.name == "s" || prop.name == "texture_u") { uv.x = v; has_uv = true; }
                    else if (prop.name == "v" || prop.name == "t" || prop.name == "texture_v") { uv.y = v; has_uv = true; }
                }
                mesh.vertices.push_back(p);
                if (has_n) mesh.normals.push_back(normalize(n));
                if (has_uv) mesh.texcoords.push_back(uv);
            }
        } else if (element.name == "face") {
            for (int i = 0; i < element.count; ++i) {
                bool consumed_list = false;
                for (const PlyProperty& prop : element.properties) {
                    if (prop.list) {
                        const int count = static_cast<int>(binary ? read_binary_scalar(in, prop.count_type) : read_ascii_scalar(prop.count_type));
                        std::vector<uint32_t> face;
                        face.reserve(static_cast<size_t>(std::max(0, count)));
                        for (int j = 0; j < count; ++j) {
                            face.push_back(static_cast<uint32_t>(binary ? read_binary_scalar(in, prop.item_type) : read_ascii_scalar(prop.item_type)));
                        }
                        if (!consumed_list) triangulate_face(mesh, face);
                        consumed_list = true;
                    } else {
                        (void)(binary ? read_binary_scalar(in, prop.type) : read_ascii_scalar(prop.type));
                    }
                }
            }
        } else {
            for (int i = 0; i < element.count; ++i) {
                for (const PlyProperty& prop : element.properties) {
                    if (prop.list) {
                        const int count = static_cast<int>(binary ? read_binary_scalar(in, prop.count_type) : read_ascii_scalar(prop.count_type));
                        for (int j = 0; j < count; ++j) (void)(binary ? read_binary_scalar(in, prop.item_type) : read_ascii_scalar(prop.item_type));
                    } else {
                        (void)(binary ? read_binary_scalar(in, prop.type) : read_ascii_scalar(prop.type));
                    }
                }
            }
        }
    }
    if (mesh.normals.size() != mesh.vertices.size()) mesh.normals.clear();
    if (mesh.texcoords.size() != mesh.vertices.size()) mesh.texcoords.clear();
    return !mesh.vertices.empty() && !mesh.indices.empty();
}

class PbrtLoader {
public:
    SceneLoadResult load(const std::string& path) {
        scene_.camera = {{0.0f, 1.0f, 4.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, 45.0f};
        scene_.environment.constant = true;
        scene_.materials.push_back(make_material("default", {0.8f, 0.8f, 0.8f}, BrdfModel::Principled, 0.5f, 0.0f));
        named_materials_["default"] = 0;
        state_.material = 0;
        parse_file(std::filesystem::absolute(path));
        if (!error_.empty()) return {scene_, error_};
        if (scene_.meshes.empty()) return {make_default_scene(), "PBRT scene contains no supported meshes: " + path};
        return {scene_, {}};
    }

private:
    Scene scene_;
    ImportState state_;
    std::vector<ImportState> attribute_stack_;
    std::vector<Mat4> transform_stack_;
    std::unordered_map<std::string, int> named_materials_;
    std::unordered_map<std::string, std::vector<Mesh>> objects_;
    std::vector<std::string> object_stack_;
    std::string current_object_;
    std::vector<PendingMesh> pending_meshes_;
    std::unordered_set<std::string> include_stack_;
    std::string error_;

    void parse_file(const std::filesystem::path& path) {
        if (!error_.empty()) return;
        const std::string key = path.lexically_normal().string();
        if (include_stack_.count(key)) return;
        include_stack_.insert(key);

        std::ifstream file(path, std::ios::binary);
        if (!file) {
            error_ = "Could not open PBRT scene: " + path.string();
            return;
        }
        std::ostringstream buffer;
        buffer << file.rdbuf();
        std::vector<Token> tokens = tokenize(buffer.str());
        const std::filesystem::path base = path.parent_path();
        for (size_t pos = 0; pos < tokens.size() && error_.empty();) {
            const Token command = tokens[pos++];
            handle(command, tokens, pos, base);
        }
        flush_pending_meshes();
        include_stack_.erase(key);
    }

    void handle(const Token& command, const std::vector<Token>& tokens, size_t& pos, const std::filesystem::path& base) {
        const std::string& op = command.text;
        if (op == "Include" && pos < tokens.size()) {
            parse_file(resolve_path(base, tokens[pos++].text));
        } else if (op == "AttributeBegin") {
            attribute_stack_.push_back(state_);
        } else if (op == "AttributeEnd") {
            if (!attribute_stack_.empty()) {
                state_ = attribute_stack_.back();
                attribute_stack_.pop_back();
            }
        } else if (op == "TransformBegin") {
            transform_stack_.push_back(state_.transform);
        } else if (op == "TransformEnd") {
            if (!transform_stack_.empty()) {
                state_.transform = transform_stack_.back();
                transform_stack_.pop_back();
            }
        } else if (op == "Identity") {
            state_.transform = Mat4{};
        } else if (op == "Translate" && pos + 2 < tokens.size()) {
            Vec3 t{};
            parse_float(tokens[pos++].text, t.x);
            parse_float(tokens[pos++].text, t.y);
            parse_float(tokens[pos++].text, t.z);
            state_.transform = multiply(state_.transform, translate(t));
        } else if (op == "Scale" && pos + 2 < tokens.size()) {
            Vec3 s{1.0f};
            parse_float(tokens[pos++].text, s.x);
            parse_float(tokens[pos++].text, s.y);
            parse_float(tokens[pos++].text, s.z);
            state_.transform = multiply(state_.transform, scale(s));
        } else if (op == "Rotate" && pos + 3 < tokens.size()) {
            float angle = 0.0f;
            Vec3 axis{};
            parse_float(tokens[pos++].text, angle);
            parse_float(tokens[pos++].text, axis.x);
            parse_float(tokens[pos++].text, axis.y);
            parse_float(tokens[pos++].text, axis.z);
            state_.transform = multiply(state_.transform, rotate(angle, axis));
        } else if ((op == "Transform" || op == "ConcatTransform") && pos < tokens.size()) {
            Mat4 m = parse_matrix(tokens, pos);
            state_.transform = op == "Transform" ? m : multiply(state_.transform, m);
        } else if (op == "LookAt" && pos + 8 < tokens.size()) {
            parse_look_at(tokens, pos);
        } else if (op == "Camera") {
            if (pos < tokens.size()) ++pos;
            const auto params = parse_params(tokens, pos);
            scene_.camera.fov_degrees = std::clamp(float_param(params, "fov", scene_.camera.fov_degrees), 10.0f, 120.0f);
            if (has_non_identity_transform(state_.transform)) {
                apply_camera_transform(state_.transform);
            }
        } else if (op == "Texture") {
            parse_texture(tokens, pos, base);
        } else if (op == "MakeNamedMaterial") {
            parse_named_material(tokens, pos);
        } else if (op == "NamedMaterial" && pos < tokens.size()) {
            const std::string name = tokens[pos++].text;
            const auto it = named_materials_.find(name);
            state_.material = it == named_materials_.end() ? 0 : it->second;
        } else if (op == "Material") {
            parse_inline_material(tokens, pos);
        } else if (op == "AreaLightSource") {
            parse_area_light(tokens, pos);
        } else if (op == "LightSource") {
            parse_light_source(tokens, pos, base);
        } else if (op == "ObjectBegin" && pos < tokens.size()) {
            object_stack_.push_back(current_object_);
            current_object_ = tokens[pos++].text;
            objects_[current_object_].clear();
        } else if (op == "ObjectEnd") {
            flush_pending_meshes();
            current_object_ = object_stack_.empty() ? std::string{} : object_stack_.back();
            if (!object_stack_.empty()) object_stack_.pop_back();
        } else if (op == "ObjectInstance" && pos < tokens.size()) {
            flush_pending_meshes();
            instantiate_object(tokens[pos++].text);
        } else if (op == "Shape") {
            parse_shape(tokens, pos, base);
        } else {
            (void)parse_params(tokens, pos);
        }
    }

    Mat4 parse_matrix(const std::vector<Token>& tokens, size_t& pos) {
        Mat4 m;
        if (tokens[pos].text == "[") ++pos;
        for (int i = 0; i < 16 && pos < tokens.size(); ++i) {
            if (tokens[pos].text == "]") break;
            parse_float(tokens[pos++].text, m.m[i / 4][i % 4]);
        }
        if (pos < tokens.size() && tokens[pos].text == "]") ++pos;
        return m;
    }

    void parse_look_at(const std::vector<Token>& tokens, size_t& pos) {
        Vec3 eye{};
        Vec3 look{};
        Vec3 up{};
        parse_float(tokens[pos++].text, eye.x);
        parse_float(tokens[pos++].text, eye.y);
        parse_float(tokens[pos++].text, eye.z);
        parse_float(tokens[pos++].text, look.x);
        parse_float(tokens[pos++].text, look.y);
        parse_float(tokens[pos++].text, look.z);
        parse_float(tokens[pos++].text, up.x);
        parse_float(tokens[pos++].text, up.y);
        parse_float(tokens[pos++].text, up.z);
        scene_.camera.position = eye;
        scene_.camera.target = look;
        scene_.camera.up = normalize(up);
        scene_.camera.right_sign = -1.0f;
    }

    bool has_non_identity_transform(const Mat4& m) const {
        Mat4 identity;
        for (int y = 0; y < 4; ++y) {
            for (int x = 0; x < 4; ++x) {
                if (std::fabs(m.m[y][x] - identity.m[y][x]) > 1.0e-6f) {
                    return true;
                }
            }
        }
        return false;
    }

    void apply_camera_transform(const Mat4& camera_to_world) {
        const Vec3 origin = transform_point(camera_to_world, {});
        const Vec3 target = transform_point(camera_to_world, {0.0f, 0.0f, 1.0f});
        const Vec3 up = transform_vector(camera_to_world, {0.0f, 1.0f, 0.0f});
        const Vec3 forward = normalize(target - origin);
        if (dot(forward, forward) <= 0.0f || (std::fabs(origin.x) + std::fabs(origin.y) + std::fabs(origin.z)) <= 0.0f) {
            return;
        }
        scene_.camera.position = origin;
        scene_.camera.target = target;
        scene_.camera.up = dot(up, up) > 0.0f ? normalize(up) : Vec3{0.0f, 1.0f, 0.0f};
        scene_.camera.right_sign = -1.0f;
    }

    void parse_texture(const std::vector<Token>& tokens, size_t& pos, const std::filesystem::path& base) {
        if (pos + 2 >= tokens.size()) return;
        const std::string name = tokens[pos++].text;
        ++pos;
        const std::string kind = lower(tokens[pos++].text);
        const auto params = parse_params(tokens, pos);
        if (kind != "imagemap") return;
        const std::string filename = string_param(params, "filename");
        if (filename.empty()) return;
        Texture texture;
        std::string error;
        const std::filesystem::path texture_path = resolve_path(base, filename);
        if (load_texture_file(name, texture_path.string(), texture, error)) {
            texture.path = filename;
            scene_.textures.push_back(std::make_shared<Texture>(std::move(texture)));
        }
    }

    void parse_named_material(const std::vector<Token>& tokens, size_t& pos) {
        if (pos >= tokens.size()) return;
        const std::string name = tokens[pos++].text;
        const auto params = parse_params(tokens, pos);
        named_materials_[name] = add_material(name, params);
    }

    void parse_inline_material(const std::vector<Token>& tokens, size_t& pos) {
        const std::string type = pos < tokens.size() ? tokens[pos++].text : "diffuse";
        auto params = parse_params(tokens, pos);
        params.push_back({"string", "type", {type}});
        state_.material = add_material("material_" + std::to_string(scene_.materials.size()), params);
    }

    int add_material(const std::string& name, const std::vector<Param>& params) {
        const std::string type = lower(string_param(params, "type", "diffuse"));
        const Vec3 color = vec3_param(params, "reflectance", vec3_param(params, "Kd", vec3_param(params, "color", {0.8f, 0.8f, 0.8f})));
        const float roughness = std::clamp(float_param(params, "roughness", float_param(params, "uroughness", 0.5f)), 0.02f, 1.0f);
        std::shared_ptr<Material> material;
        if (type == "dielectric" || type == "thindielectric" || type == "glass") {
            const Vec3 transmission = vec3_param(params, "transmittance", vec3_param(params, "T", {1.0f, 1.0f, 1.0f}));
            auto dielectric = std::make_shared<DielectricMaterial>(name, transmission, std::clamp(float_param(params, "eta", 1.5f), 1.0f, 3.0f));
            dielectric->transmission_tint = transmission;
            material = dielectric;
        } else if (type == "coateddiffuse") {
            auto principled = std::make_shared<PrincipledMaterial>(name, color, 1.0f, 0.0f);
            principled->clearcoat = 1.0f;
            principled->clearcoat_roughness = std::clamp(float_param(params, "interface.roughness", roughness), 0.02f, 1.0f);
            material = principled;
        } else if (type == "conductor") {
            material = make_material(name, color, BrdfModel::Conductor, roughness, 1.0f);
        } else if (type == "coatedconductor") {
            auto principled = std::make_shared<PrincipledMaterial>(name, color, std::clamp(float_param(params, "conductor.roughness", roughness), 0.02f, 1.0f), 1.0f);
            principled->clearcoat = 1.0f;
            principled->clearcoat_roughness = std::clamp(float_param(params, "interface.roughness", 0.02f), 0.02f, 1.0f);
            material = principled;
        } else if (type == "diffuse" || type == "matte") {
            material = make_material(name, color, BrdfModel::Lambertian, roughness, 0.0f);
        } else {
            material = make_material(name, color, BrdfModel::Principled, roughness, std::clamp(float_param(params, "metallic", 0.0f), 0.0f, 1.0f));
        }
        const std::string reflectance_texture = string_param(params, "reflectance", string_param(params, "Kd"));
        if (find_param(params, "reflectance") && find_param(params, "reflectance")->type == "texture") {
            const int texture = find_texture_index(scene_, reflectance_texture);
            material->albedo_texture = texture >= 0 ? scene_.textures[static_cast<size_t>(texture)] : nullptr;
        }
        scene_.materials.push_back(material);
        return static_cast<int>(scene_.materials.size() - 1);
    }

    void parse_area_light(const std::vector<Token>& tokens, size_t& pos) {
        if (pos < tokens.size()) ++pos;
        const auto params = parse_params(tokens, pos);
        const Vec3 l = vec3_param(params, "L", {1.0f, 1.0f, 1.0f});
        const float scale_value = std::max(0.0f, float_param(params, "scale", 1.0f));
        const float intensity = std::max({l.x, l.y, l.z}) * scale_value;
        state_.area_light.enabled = intensity > 0.0f;
        state_.area_light.double_sided = true;
        state_.area_light.intensity = intensity;
        state_.area_light.color = intensity > 0.0f ? l / std::max(1.0e-6f, std::max({l.x, l.y, l.z})) : Vec3{1.0f};
    }

    void parse_light_source(const std::vector<Token>& tokens, size_t& pos, const std::filesystem::path& base) {
        const std::string type = pos < tokens.size() ? lower(tokens[pos++].text) : "";
        const auto params = parse_params(tokens, pos);
        if (type == "infinite") {
            scene_.environment.color = vec3_param(params, "L", {1.0f, 1.0f, 1.0f});
            scene_.environment.strength = std::max(0.0f, float_param(params, "scale", 1.0f));
            scene_.environment.constant = true;
            scene_.environment.mapping = Environment::Mapping::EqualArea;
            scene_.environment.light_from_world_x = inverse_transform_component_basis_assuming_rotation(state_.transform, 0);
            scene_.environment.light_from_world_y = inverse_transform_component_basis_assuming_rotation(state_.transform, 1);
            scene_.environment.light_from_world_z = inverse_transform_component_basis_assuming_rotation(state_.transform, 2);
            const std::string filename = string_param(params, "filename");
            if (!filename.empty()) {
                Texture texture;
                std::string error;
                if (load_texture_file("environment", resolve_path(base, filename).string(), texture, error)) {
                    texture.path = filename;
                    scene_.textures.push_back(std::make_shared<Texture>(std::move(texture)));
                    scene_.environment.texture = scene_.textures.back();
                    scene_.environment.constant = false;
                }
            }
        } else if (type == "point" || type == "spot" || type == "goniometric") {
            const Vec3 color = vec3_param(params, "I", vec3_param(params, "L", {1.0f, 1.0f, 1.0f}));
            const float scale_value = std::max(0.0f, float_param(params, "scale", 1.0f));
            const float intensity = std::max({color.x, color.y, color.z}) * scale_value;
            if (intensity <= 0.0f) return;
            const int mat = add_light_material("point_light_material_" + std::to_string(scene_.materials.size()), color * scale_value);
            Mesh mesh = make_uv_sphere_mesh("point_light_" + std::to_string(scene_.meshes.size()), mat, {}, 0.035f, 12, 6);
            apply_transform(mesh, state_.transform);
            mesh.light.enabled = true;
            mesh.light.double_sided = true;
            mesh.light.intensity = intensity;
            mesh.light.color = color / std::max(1.0e-6f, std::max({color.x, color.y, color.z}));
            scene_.meshes.push_back(std::move(mesh));
        }
    }

    void parse_shape(const std::vector<Token>& tokens, size_t& pos, const std::filesystem::path& base) {
        if (pos >= tokens.size()) return;
        const std::string type = lower(tokens[pos++].text);
        const auto params = parse_params(tokens, pos);
        if (type == "trianglemesh") {
            add_triangle_mesh(params);
        } else if (type == "plymesh") {
            add_ply_mesh(params, base);
        } else if (type == "bilinearmesh") {
            add_bilinear_mesh(params);
        } else if (type == "disk") {
            add_disk(params);
        } else if (type == "sphere") {
            add_sphere(params);
        }
    }

    int add_light_material(const std::string& name, Vec3 emission) {
        auto material = make_material(name, {1.0f, 1.0f, 1.0f}, BrdfModel::Lambertian, 0.5f, 0.0f);
        material->emission = emission;
        scene_.materials.push_back(material);
        return static_cast<int>(scene_.materials.size() - 1);
    }

    void finish_mesh(Mesh& mesh) {
        mesh.material = state_.material;
        mesh.light = state_.area_light;
        append_mesh(current_object_, std::move(mesh));
    }

    void append_mesh(const std::string& object, Mesh&& mesh) {
        if (mesh.texcoords.size() != mesh.vertices.size()) {
            mesh.texcoords.resize(mesh.vertices.size());
            for (size_t i = 0; i < mesh.vertices.size(); ++i) {
                mesh.texcoords[i] = {mesh.vertices[i].x + 0.5f, mesh.vertices[i].z + 0.5f};
            }
        }
        if (!mesh.vertices.empty() && !mesh.indices.empty()) {
            if (!object.empty()) {
                objects_[object].push_back(std::move(mesh));
            } else {
                scene_.meshes.push_back(std::move(mesh));
            }
        }
    }

    void flush_pending_meshes() {
        for (PendingMesh& pending : pending_meshes_) {
            collect_pending_mesh(pending);
        }
        pending_meshes_.clear();
    }

    void collect_pending_mesh(PendingMesh& pending) {
        MeshLoadResult result = pending.future.get();
        if (!result.error.empty() && error_.empty()) {
            error_ = result.error;
        }
        append_mesh(pending.object, std::move(result.mesh));
    }

    void limit_pending_meshes() {
        const unsigned workers = std::max(2u, std::thread::hardware_concurrency());
        if (pending_meshes_.size() < workers) return;
        collect_pending_mesh(pending_meshes_.front());
        pending_meshes_.erase(pending_meshes_.begin());
    }

    void instantiate_object(const std::string& name) {
        const auto it = objects_.find(name);
        if (it == objects_.end()) return;
        for (const Mesh& source : it->second) {
            Mesh mesh = source;
            mesh.name = name + "_" + std::to_string(scene_.meshes.size());
            apply_transform(mesh, state_.transform);
            scene_.meshes.push_back(std::move(mesh));
        }
    }

    void add_triangle_mesh(const std::vector<Param>& params) {
        const std::vector<float> p = float_array_param(params, "P");
        const std::vector<int> indices = int_array_param(params, "indices").empty() ? int_array_param(params, "vertexindices") : int_array_param(params, "indices");
        if (p.size() < 9 || indices.size() < 3) return;
        Mesh mesh;
        mesh.name = "trianglemesh_" + std::to_string(scene_.meshes.size());
        for (size_t i = 0; i + 2 < p.size(); i += 3) {
            mesh.vertices.push_back(transform_point(state_.transform, {p[i], p[i + 1], p[i + 2]}));
        }
        const std::vector<float> n = float_array_param(params, "N");
        for (size_t i = 0; i + 2 < n.size(); i += 3) {
            mesh.normals.push_back(normalize(transform_vector(state_.transform, {n[i], n[i + 1], n[i + 2]})));
        }
        const std::vector<float> uv = float_array_param(params, "uv").empty() ? float_array_param(params, "st") : float_array_param(params, "uv");
        for (size_t i = 0; i + 1 < uv.size(); i += 2) {
            mesh.texcoords.push_back({uv[i], uv[i + 1]});
        }
        for (int index : indices) {
            if (index >= 0 && index < static_cast<int>(mesh.vertices.size())) {
                mesh.indices.push_back(static_cast<uint32_t>(index));
            }
        }
        finish_mesh(mesh);
    }

    void add_ply_mesh(const std::vector<Param>& params, const std::filesystem::path& base) {
        const std::string filename = string_param(params, "filename");
        if (filename.empty()) return;
        const std::filesystem::path path = resolve_path(base, filename);
        const Mat4 transform = state_.transform;
        const int material = state_.material;
        const LightComponent light = state_.area_light;
        limit_pending_meshes();
        PendingMesh pending;
        pending.object = current_object_;
        pending.future = std::async(std::launch::async, [path, filename, transform, material, light]() {
            MeshLoadResult result;
            result.mesh.name = std::filesystem::path(filename).stem().string();
            if (!load_ply_mesh(path, result.mesh, result.error)) {
                result.mesh = {};
                return result;
            }
            result.mesh.material = material;
            result.mesh.light = light;
            apply_transform(result.mesh, transform);
            return result;
        });
        pending_meshes_.push_back(std::move(pending));
    }

    void add_bilinear_mesh(const std::vector<Param>& params) {
        const std::vector<float> p = float_array_param(params, "P");
        if (p.size() < 12) return;
        Mesh mesh;
        mesh.name = "bilinearmesh_" + std::to_string(scene_.meshes.size());
        for (size_t i = 0; i + 2 < p.size(); i += 3) {
            mesh.vertices.push_back(transform_point(state_.transform, {p[i], p[i + 1], p[i + 2]}));
        }
        const std::vector<int> indices = int_array_param(params, "indices");
        if (!indices.empty()) {
            for (size_t i = 0; i + 3 < indices.size(); i += 4) {
                const uint32_t a = static_cast<uint32_t>(indices[i]);
                const uint32_t b = static_cast<uint32_t>(indices[i + 1]);
                const uint32_t c = static_cast<uint32_t>(indices[i + 2]);
                const uint32_t d = static_cast<uint32_t>(indices[i + 3]);
                triangulate_face(mesh, {a, b, c, d});
            }
        } else {
            for (uint32_t i = 0; i + 3 < mesh.vertices.size(); i += 4) {
                triangulate_face(mesh, {i, i + 1, i + 2, i + 3});
            }
        }
        const std::vector<float> uv = float_array_param(params, "uv").empty() ? float_array_param(params, "st") : float_array_param(params, "uv");
        for (size_t i = 0; i + 1 < uv.size(); i += 2) {
            mesh.texcoords.push_back({uv[i], uv[i + 1]});
        }
        finish_mesh(mesh);
    }

    void add_disk(const std::vector<Param>& params) {
        const float radius = std::max(0.0f, float_param(params, "radius", 1.0f));
        const float inner_radius = std::clamp(float_param(params, "innerradius", 0.0f), 0.0f, radius);
        if (radius <= 0.0f) return;
        Mesh mesh;
        mesh.name = "disk_" + std::to_string(scene_.meshes.size());
        constexpr int kSegments = 48;
        if (inner_radius <= 0.0f) {
            mesh.vertices.push_back(transform_point(state_.transform, {}));
            mesh.texcoords.push_back({0.5f, 0.5f});
            for (int i = 0; i < kSegments; ++i) {
                const float a = 2.0f * kPi * static_cast<float>(i) / static_cast<float>(kSegments);
                mesh.vertices.push_back(transform_point(state_.transform, {radius * std::cos(a), radius * std::sin(a), 0.0f}));
                mesh.texcoords.push_back({0.5f + 0.5f * std::cos(a), 0.5f + 0.5f * std::sin(a)});
            }
            for (uint32_t i = 0; i < kSegments; ++i) {
                mesh.indices.push_back(0);
                mesh.indices.push_back(i + 1);
                mesh.indices.push_back(i + 1 == kSegments ? 1 : i + 2);
            }
        } else {
            for (int i = 0; i < kSegments; ++i) {
                const float a = 2.0f * kPi * static_cast<float>(i) / static_cast<float>(kSegments);
                mesh.vertices.push_back(transform_point(state_.transform, {inner_radius * std::cos(a), inner_radius * std::sin(a), 0.0f}));
                mesh.vertices.push_back(transform_point(state_.transform, {radius * std::cos(a), radius * std::sin(a), 0.0f}));
                mesh.texcoords.push_back({0.5f + 0.5f * inner_radius / radius * std::cos(a), 0.5f + 0.5f * inner_radius / radius * std::sin(a)});
                mesh.texcoords.push_back({0.5f + 0.5f * std::cos(a), 0.5f + 0.5f * std::sin(a)});
            }
            for (uint32_t i = 0; i < kSegments; ++i) {
                const uint32_t j = i + 1 == kSegments ? 0 : i + 1;
                triangulate_face(mesh, {i * 2, i * 2 + 1, j * 2 + 1, j * 2});
            }
        }
        finish_mesh(mesh);
    }

    void add_sphere(const std::vector<Param>& params) {
        const float radius = std::max(0.0f, float_param(params, "radius", 1.0f));
        Mesh mesh = make_uv_sphere_mesh("sphere_" + std::to_string(scene_.meshes.size()), state_.material, {}, radius, 32, 16);
        apply_transform(mesh, state_.transform);
        finish_mesh(mesh);
    }
};

} // namespace

SceneLoadResult load_pbrt_scene(const std::string& path) {
    return PbrtLoader{}.load(path);
}

} // namespace lt
