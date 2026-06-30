#include "lt/scene.h"
#include "lt/log.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <variant>

namespace lt {
namespace {

struct Json {
    using Array = std::vector<Json>;
    using Object = std::map<std::string, Json>;
    std::variant<std::nullptr_t, bool, double, std::string, Array, Object> value;

    const Json& operator[](const std::string& key) const {
        static const Json empty;
        if (const auto* object = std::get_if<Object>(&value)) {
            const auto it = object->find(key);
            return it == object->end() ? empty : it->second;
        }
        return empty;
    }

    const Json& operator[](size_t index) const {
        static const Json empty;
        if (const auto* array = std::get_if<Array>(&value)) {
            return index < array->size() ? (*array)[index] : empty;
        }
        return empty;
    }

    bool is_null() const { return std::holds_alternative<std::nullptr_t>(value); }
    bool is_array() const { return std::holds_alternative<Array>(value); }
    bool is_object() const { return std::holds_alternative<Object>(value); }
    const Array& array() const {
        static const Array empty;
        const auto* data = std::get_if<Array>(&value);
        return data ? *data : empty;
    }
    std::string string(const std::string& fallback = {}) const {
        const auto* data = std::get_if<std::string>(&value);
        return data ? *data : fallback;
    }
    int integer(int fallback = 0) const {
        const auto* data = std::get_if<double>(&value);
        return data ? static_cast<int>(*data) : fallback;
    }
    float number(float fallback = 0.0f) const {
        const auto* data = std::get_if<double>(&value);
        return data ? static_cast<float>(*data) : fallback;
    }
    bool boolean(bool fallback = false) const {
        const auto* data = std::get_if<bool>(&value);
        return data ? *data : fallback;
    }
};

class JsonParser {
public:
    explicit JsonParser(std::string text) : text_(std::move(text)) {}

    bool parse(Json& out, std::string& error) {
        skip_ws();
        if (!parse_value(out)) {
            error = "Invalid glTF JSON near byte " + std::to_string(pos_);
            return false;
        }
        skip_ws();
        if (pos_ != text_.size()) {
            error = "Trailing data in glTF JSON";
            return false;
        }
        return true;
    }

private:
    bool parse_value(Json& out) {
        skip_ws();
        if (pos_ >= text_.size()) return false;
        const char c = text_[pos_];
        if (c == 'n' && consume("null")) {
            out.value = nullptr;
            return true;
        }
        if (c == 't' && consume("true")) {
            out.value = true;
            return true;
        }
        if (c == 'f' && consume("false")) {
            out.value = false;
            return true;
        }
        if (c == '"') {
            std::string s;
            if (!parse_string(s)) return false;
            out.value = std::move(s);
            return true;
        }
        if (c == '[') return parse_array(out);
        if (c == '{') return parse_object(out);
        return parse_number(out);
    }

    bool parse_array(Json& out) {
        ++pos_;
        Json::Array array;
        skip_ws();
        if (match(']')) {
            out.value = std::move(array);
            return true;
        }
        while (true) {
            Json item;
            if (!parse_value(item)) return false;
            array.push_back(std::move(item));
            skip_ws();
            if (match(']')) break;
            if (!match(',')) return false;
        }
        out.value = std::move(array);
        return true;
    }

    bool parse_object(Json& out) {
        ++pos_;
        Json::Object object;
        skip_ws();
        if (match('}')) {
            out.value = std::move(object);
            return true;
        }
        while (true) {
            std::string key;
            if (!parse_string(key)) return false;
            skip_ws();
            if (!match(':')) return false;
            Json item;
            if (!parse_value(item)) return false;
            object[std::move(key)] = std::move(item);
            skip_ws();
            if (match('}')) break;
            if (!match(',')) return false;
        }
        out.value = std::move(object);
        return true;
    }

    bool parse_string(std::string& out) {
        if (!match('"')) return false;
        while (pos_ < text_.size()) {
            const char c = text_[pos_++];
            if (c == '"') return true;
            if (c != '\\') {
                out.push_back(c);
                continue;
            }
            if (pos_ >= text_.size()) return false;
            const char e = text_[pos_++];
            if (e == '"' || e == '\\' || e == '/') out.push_back(e);
            else if (e == 'b') out.push_back('\b');
            else if (e == 'f') out.push_back('\f');
            else if (e == 'n') out.push_back('\n');
            else if (e == 'r') out.push_back('\r');
            else if (e == 't') out.push_back('\t');
            else if (e == 'u') {
                if (pos_ + 4 > text_.size()) return false;
                pos_ += 4;
                out.push_back('?');
            } else {
                return false;
            }
        }
        return false;
    }

    bool parse_number(Json& out) {
        const size_t start = pos_;
        if (pos_ < text_.size() && text_[pos_] == '-') ++pos_;
        while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
        if (pos_ < text_.size() && text_[pos_] == '.') {
            ++pos_;
            while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
        }
        if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
            ++pos_;
            if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) ++pos_;
            while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
        }
        if (start == pos_) return false;
        out.value = std::stod(text_.substr(start, pos_ - start));
        return true;
    }

    bool consume(const char* token) {
        const size_t len = std::strlen(token);
        if (text_.compare(pos_, len, token) != 0) return false;
        pos_ += len;
        return true;
    }

    bool match(char c) {
        skip_ws();
        if (pos_ < text_.size() && text_[pos_] == c) {
            ++pos_;
            return true;
        }
        return false;
    }

    void skip_ws() {
        while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_]))) ++pos_;
    }

    std::string text_;
    size_t pos_ = 0;
};

struct BufferView {
    int buffer = -1;
    size_t offset = 0;
    size_t length = 0;
    size_t stride = 0;
};

struct Accessor {
    int view = -1;
    size_t offset = 0;
    int component_type = 0;
    size_t count = 0;
    std::string type;
    bool normalized = false;
};

struct Mat4 {
    float m[16] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    };
};

std::string parent_path(const std::string& path) {
    const size_t slash = path.find_last_of("\\/");
    return slash == std::string::npos ? std::string{} : path.substr(0, slash);
}

bool is_absolute_path(const std::string& path) {
    return (path.size() >= 2 && path[1] == ':') || (!path.empty() && (path[0] == '/' || path[0] == '\\'));
}

std::string join_path(const std::string& parent, const std::string& child) {
    if (parent.empty() || is_absolute_path(child)) return child;
    const char last = parent.back();
    return parent + ((last == '\\' || last == '/') ? "" : "\\") + child;
}

std::string lowercase_extension(const std::string& path) {
    const size_t dot = path.find_last_of('.');
    std::string ext = dot == std::string::npos ? std::string{} : path.substr(dot);
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext;
}

bool read_file_bytes(const std::string& path, std::vector<unsigned char>& bytes) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    in.seekg(0, std::ios::end);
    const std::streamoff size = in.tellg();
    in.seekg(0, std::ios::beg);
    if (size < 0) return false;
    bytes.resize(static_cast<size_t>(size));
    return bytes.empty() || static_cast<bool>(in.read(reinterpret_cast<char*>(bytes.data()), size));
}

uint32_t read_u32(const std::vector<unsigned char>& bytes, size_t offset) {
    if (offset + 4 > bytes.size()) return 0;
    return static_cast<uint32_t>(bytes[offset + 0]) |
        (static_cast<uint32_t>(bytes[offset + 1]) << 8u) |
        (static_cast<uint32_t>(bytes[offset + 2]) << 16u) |
        (static_cast<uint32_t>(bytes[offset + 3]) << 24u);
}

bool load_gltf_document(const std::string& path, Json& root, std::vector<unsigned char>& binary_chunk, std::string& error) {
    std::vector<unsigned char> file;
    if (!read_file_bytes(path, file)) {
        error = "Could not open glTF scene: " + path;
        return false;
    }

    std::string json_text;
    if (lowercase_extension(path) == ".glb") {
        if (file.size() < 20 || read_u32(file, 0) != 0x46546c67u || read_u32(file, 4) != 2u) {
            error = "Invalid GLB header: " + path;
            return false;
        }
        size_t offset = 12;
        while (offset + 8 <= file.size()) {
            const uint32_t chunk_length = read_u32(file, offset);
            const uint32_t chunk_type = read_u32(file, offset + 4);
            offset += 8;
            if (offset + chunk_length > file.size()) {
                error = "GLB chunk extends past end of file";
                return false;
            }
            if (chunk_type == 0x4e4f534au) {
                json_text.assign(reinterpret_cast<const char*>(file.data() + offset), chunk_length);
            } else if (chunk_type == 0x004e4942u) {
                binary_chunk.assign(file.begin() + static_cast<std::ptrdiff_t>(offset), file.begin() + static_cast<std::ptrdiff_t>(offset + chunk_length));
            }
            offset += (chunk_length + 3u) & ~3u;
        }
    } else {
        json_text.assign(reinterpret_cast<const char*>(file.data()), file.size());
    }

    JsonParser parser(std::move(json_text));
    return parser.parse(root, error);
}

int components_for_type(const std::string& type) {
    if (type == "SCALAR") return 1;
    if (type == "VEC2") return 2;
    if (type == "VEC3") return 3;
    if (type == "VEC4") return 4;
    if (type == "MAT4") return 16;
    return 0;
}

size_t component_size(int component_type) {
    if (component_type == 5120 || component_type == 5121) return 1;
    if (component_type == 5122 || component_type == 5123) return 2;
    if (component_type == 5125 || component_type == 5126) return 4;
    return 0;
}

Mat4 multiply(const Mat4& a, const Mat4& b) {
    Mat4 r{};
    for (int c = 0; c < 4; ++c) {
        for (int row = 0; row < 4; ++row) {
            r.m[c * 4 + row] =
                a.m[0 * 4 + row] * b.m[c * 4 + 0] +
                a.m[1 * 4 + row] * b.m[c * 4 + 1] +
                a.m[2 * 4 + row] * b.m[c * 4 + 2] +
                a.m[3 * 4 + row] * b.m[c * 4 + 3];
        }
    }
    return r;
}

Vec3 transform_point(const Mat4& m, Vec3 p) {
    return {
        m.m[0] * p.x + m.m[4] * p.y + m.m[8] * p.z + m.m[12],
        m.m[1] * p.x + m.m[5] * p.y + m.m[9] * p.z + m.m[13],
        m.m[2] * p.x + m.m[6] * p.y + m.m[10] * p.z + m.m[14],
    };
}

Vec3 transform_vector(const Mat4& m, Vec3 p) {
    return {
        m.m[0] * p.x + m.m[4] * p.y + m.m[8] * p.z,
        m.m[1] * p.x + m.m[5] * p.y + m.m[9] * p.z,
        m.m[2] * p.x + m.m[6] * p.y + m.m[10] * p.z,
    };
}

Mat4 local_transform(const Json& node) {
    Mat4 result;
    if (node["matrix"].is_array() && node["matrix"].array().size() >= 16) {
        for (int i = 0; i < 16; ++i) result.m[i] = node["matrix"][static_cast<size_t>(i)].number(result.m[i]);
        return result;
    }

    Vec3 t{};
    Vec3 s{1.0f};
    float qx = 0.0f;
    float qy = 0.0f;
    float qz = 0.0f;
    float qw = 1.0f;
    if (node["translation"].is_array()) {
        t = {node["translation"][0].number(), node["translation"][1].number(), node["translation"][2].number()};
    }
    if (node["scale"].is_array()) {
        s = {node["scale"][0].number(1.0f), node["scale"][1].number(1.0f), node["scale"][2].number(1.0f)};
    }
    if (node["rotation"].is_array()) {
        qx = node["rotation"][0].number();
        qy = node["rotation"][1].number();
        qz = node["rotation"][2].number();
        qw = node["rotation"][3].number(1.0f);
    }

    const float xx = qx * qx;
    const float yy = qy * qy;
    const float zz = qz * qz;
    const float xy = qx * qy;
    const float xz = qx * qz;
    const float yz = qy * qz;
    const float wx = qw * qx;
    const float wy = qw * qy;
    const float wz = qw * qz;
    result.m[0] = (1.0f - 2.0f * (yy + zz)) * s.x;
    result.m[1] = (2.0f * (xy + wz)) * s.x;
    result.m[2] = (2.0f * (xz - wy)) * s.x;
    result.m[4] = (2.0f * (xy - wz)) * s.y;
    result.m[5] = (1.0f - 2.0f * (xx + zz)) * s.y;
    result.m[6] = (2.0f * (yz + wx)) * s.y;
    result.m[8] = (2.0f * (xz + wy)) * s.z;
    result.m[9] = (2.0f * (yz - wx)) * s.z;
    result.m[10] = (1.0f - 2.0f * (xx + yy)) * s.z;
    result.m[12] = t.x;
    result.m[13] = t.y;
    result.m[14] = t.z;
    return result;
}

const unsigned char* accessor_data(
    const std::vector<std::vector<unsigned char>>& buffers,
    const std::vector<BufferView>& views,
    const Accessor& accessor,
    size_t& stride,
    size_t& element_size) {
    if (accessor.view < 0 || accessor.view >= static_cast<int>(views.size())) return nullptr;
    const BufferView& view = views[static_cast<size_t>(accessor.view)];
    if (view.buffer < 0 || view.buffer >= static_cast<int>(buffers.size())) return nullptr;
    const std::vector<unsigned char>& buffer = buffers[static_cast<size_t>(view.buffer)];
    element_size = component_size(accessor.component_type) * static_cast<size_t>(components_for_type(accessor.type));
    stride = view.stride ? view.stride : element_size;
    const size_t start = view.offset + accessor.offset;
    if (element_size == 0 || start + accessor.count * stride > buffer.size() + (stride - element_size)) return nullptr;
    return buffer.data() + start;
}

float read_float_component(const unsigned char* src, int component_type, bool normalized = false) {
    if (component_type == 5126) {
        float v = 0.0f;
        std::memcpy(&v, src, sizeof(float));
        return v;
    }
    if (component_type == 5121) {
        return normalized ? static_cast<float>(*src) / 255.0f : static_cast<float>(*src);
    }
    if (component_type == 5123) {
        uint16_t v = 0;
        std::memcpy(&v, src, sizeof(uint16_t));
        return normalized ? static_cast<float>(v) / 65535.0f : static_cast<float>(v);
    }
    return 0.0f;
}

uint32_t read_index_component(const unsigned char* src, int component_type) {
    if (component_type == 5121) return *src;
    if (component_type == 5123) {
        uint16_t v = 0;
        std::memcpy(&v, src, sizeof(uint16_t));
        return v;
    }
    if (component_type == 5125) {
        uint32_t v = 0;
        std::memcpy(&v, src, sizeof(uint32_t));
        return v;
    }
    return 0;
}

bool read_vec3_accessor(const std::vector<std::vector<unsigned char>>& buffers, const std::vector<BufferView>& views, const Accessor& accessor, std::vector<Vec3>& out) {
    size_t stride = 0;
    size_t element_size = 0;
    const unsigned char* data = accessor_data(buffers, views, accessor, stride, element_size);
    if (!data || accessor.component_type != 5126 || components_for_type(accessor.type) < 3) return false;
    out.resize(accessor.count);
    for (size_t i = 0; i < accessor.count; ++i) {
        const unsigned char* src = data + i * stride;
        out[i] = {
            read_float_component(src + 0, accessor.component_type),
            read_float_component(src + 4, accessor.component_type),
            read_float_component(src + 8, accessor.component_type),
        };
    }
    return true;
}

bool read_vec2_accessor(const std::vector<std::vector<unsigned char>>& buffers, const std::vector<BufferView>& views, const Accessor& accessor, std::vector<Vec2>& out) {
    size_t stride = 0;
    size_t element_size = 0;
    const unsigned char* data = accessor_data(buffers, views, accessor, stride, element_size);
    if (!data || components_for_type(accessor.type) < 2) return false;
    const size_t size = component_size(accessor.component_type);
    if (!(accessor.component_type == 5126 || accessor.component_type == 5121 || accessor.component_type == 5123) || size == 0) return false;
    out.resize(accessor.count);
    for (size_t i = 0; i < accessor.count; ++i) {
        const unsigned char* src = data + i * stride;
        out[i] = {
            read_float_component(src + 0, accessor.component_type, accessor.normalized),
            read_float_component(src + size, accessor.component_type, accessor.normalized),
        };
    }
    return true;
}

bool read_indices(const std::vector<std::vector<unsigned char>>& buffers, const std::vector<BufferView>& views, const Accessor& accessor, std::vector<uint32_t>& out) {
    size_t stride = 0;
    size_t element_size = 0;
    const unsigned char* data = accessor_data(buffers, views, accessor, stride, element_size);
    if (!data || components_for_type(accessor.type) != 1) return false;
    out.resize(accessor.count);
    const size_t size = component_size(accessor.component_type);
    for (size_t i = 0; i < accessor.count; ++i) {
        out[i] = read_index_component(data + i * stride, accessor.component_type);
        if (size == 0) return false;
    }
    return true;
}

std::vector<BufferView> parse_buffer_views(const Json& root) {
    std::vector<BufferView> views;
    for (const Json& item : root["bufferViews"].array()) {
        views.push_back({
            item["buffer"].integer(0),
            static_cast<size_t>(std::max(0, item["byteOffset"].integer(0))),
            static_cast<size_t>(std::max(0, item["byteLength"].integer(0))),
            static_cast<size_t>(std::max(0, item["byteStride"].integer(0))),
        });
    }
    return views;
}

std::vector<Accessor> parse_accessors(const Json& root) {
    std::vector<Accessor> accessors;
    for (const Json& item : root["accessors"].array()) {
        accessors.push_back({
            item["bufferView"].integer(-1),
            static_cast<size_t>(std::max(0, item["byteOffset"].integer(0))),
            item["componentType"].integer(),
            static_cast<size_t>(std::max(0, item["count"].integer(0))),
            item["type"].string(),
            item["normalized"].boolean(false),
        });
    }
    return accessors;
}

std::vector<std::vector<unsigned char>> load_buffers(const Json& root, const std::string& scene_dir, const std::vector<unsigned char>& binary_chunk, std::string& error) {
    std::vector<std::vector<unsigned char>> buffers;
    for (const Json& item : root["buffers"].array()) {
        const std::string uri = item["uri"].string();
        std::vector<unsigned char> data;
        if (uri.empty() && !binary_chunk.empty()) {
            data = binary_chunk;
        } else if (uri.rfind("data:", 0) == 0) {
            error = "glTF base64 data URIs are not supported yet";
            return {};
        } else if (!read_file_bytes(join_path(scene_dir, uri), data)) {
            error = "Could not open glTF buffer: " + uri;
            return {};
        }
        buffers.push_back(std::move(data));
    }
    return buffers;
}

int material_texture_source(const Json& root, int texture_index) {
    if (texture_index < 0 || texture_index >= static_cast<int>(root["textures"].array().size())) return -1;
    return root["textures"][static_cast<size_t>(texture_index)]["source"].integer(-1);
}

std::shared_ptr<Texture> scene_texture_for_gltf_texture(
    const Json& root,
    const std::vector<int>& image_to_texture,
    const Scene& scene,
    int texture_index,
    TextureRole role = TextureRole::Unknown,
    TextureColorSpace color_space = TextureColorSpace::Auto) {
    const int source = material_texture_source(root, texture_index);
    if (source < 0 || source >= static_cast<int>(image_to_texture.size())) {
        return {};
    }
    const int scene_texture = image_to_texture[static_cast<size_t>(source)];
    if (scene_texture < 0 || scene_texture >= static_cast<int>(scene.textures.size())) {
        return {};
    }
    std::shared_ptr<Texture> texture = scene.textures[static_cast<size_t>(scene_texture)];
    if (texture && role != TextureRole::Unknown) {
        apply_texture_role(*texture, role, color_space);
    }
    return texture;
}

std::string embedded_image_extension(const Json& image, const unsigned char* data, size_t size) {
    std::string mime = image["mimeType"].string();
    std::transform(mime.begin(), mime.end(), mime.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (mime == "image/png") {
        return ".png";
    }
    if (mime == "image/jpeg" || mime == "image/jpg") {
        return ".jpg";
    }
    if (size >= 8 &&
        data[0] == 0x89u && data[1] == 0x50u && data[2] == 0x4eu && data[3] == 0x47u &&
        data[4] == 0x0du && data[5] == 0x0au && data[6] == 0x1au && data[7] == 0x0au) {
        return ".png";
    }
    if (size >= 3 && data[0] == 0xffu && data[1] == 0xd8u && data[2] == 0xffu) {
        return ".jpg";
    }
    if (size >= 4 && data[0] == 0x76u && data[1] == 0x2fu && data[2] == 0x31u && data[3] == 0x01u) {
        return ".exr";
    }
    if (size >= 10 && std::memcmp(data, "#?RADIANCE", 10) == 0) {
        return ".hdr";
    }
    return ".png";
}

void load_images(const Json& root, const std::string& scene_dir, const std::vector<std::vector<unsigned char>>& buffers, const std::vector<BufferView>& views, Scene& scene, std::vector<int>& image_to_texture) {
    image_to_texture.assign(root["images"].array().size(), -1);
    for (size_t i = 0; i < root["images"].array().size(); ++i) {
        const Json& image = root["images"][i];
        Texture texture;
        std::string error;
        const std::string name = image["name"].string("image_" + std::to_string(i));
        bool ok = false;
        const std::string uri = image["uri"].string();
        if (!uri.empty() && uri.rfind("data:", 0) != 0) {
            const std::string path = join_path(scene_dir, uri);
            ok = load_texture_file(name, path, texture, error);
            texture.path = uri;
        } else {
            const int view_index = image["bufferView"].integer(-1);
            if (view_index >= 0 && view_index < static_cast<int>(views.size())) {
                const BufferView& view = views[static_cast<size_t>(view_index)];
                if (view.buffer >= 0 && view.buffer < static_cast<int>(buffers.size())) {
                    const std::vector<unsigned char>& buffer = buffers[static_cast<size_t>(view.buffer)];
                    if (view.offset + view.length <= buffer.size()) {
                        const unsigned char* image_data = buffer.data() + view.offset;
                        ok = load_texture_memory(name, image_data, view.length, texture, error);
                        texture.path = name;
                        if (ok) {
                            texture.encoded_bytes.assign(image_data, image_data + view.length);
                            texture.encoded_extension = embedded_image_extension(image, image_data, view.length);
                        }
                    }
                }
            }
        }
        if (ok) {
            image_to_texture[i] = static_cast<int>(scene.textures.size());
            scene.textures.push_back(std::make_shared<Texture>(std::move(texture)));
        } else {
            LT_LOG_WARN("glTF image '{}' was skipped: {}", name, error.empty() ? "unsupported or missing image data" : error);
        }
    }
}

void load_materials(const Json& root, const std::vector<int>& image_to_texture, Scene& scene) {
    if (root["materials"].array().empty()) {
        scene.materials.push_back(make_material("default", {0.8f, 0.8f, 0.8f}, BrdfModel::StandardSurface, 0.5f, 0.0f));
        return;
    }
    for (size_t i = 0; i < root["materials"].array().size(); ++i) {
        const Json& material_json = root["materials"][i];
        const Json& pbr = material_json["pbrMetallicRoughness"];
        Vec3 base{1.0f, 1.0f, 1.0f};
        if (pbr["baseColorFactor"].is_array()) {
            base = {pbr["baseColorFactor"][0].number(1.0f), pbr["baseColorFactor"][1].number(1.0f), pbr["baseColorFactor"][2].number(1.0f)};
        }
        const float roughness = pbr["roughnessFactor"].number(1.0f);
        const float metallic = pbr["metallicFactor"].number(1.0f);
        const Json& extensions = material_json["extensions"];
        const Json& transmission = extensions["KHR_materials_transmission"];
        const bool has_transmission_texture = !transmission["transmissionTexture"].is_null();
        const float transmission_factor = std::clamp(transmission["transmissionFactor"].number(has_transmission_texture ? 1.0f : 0.0f), 0.0f, 1.0f);
        const bool uses_transmission = transmission_factor > 0.0f;
        const float ior = std::clamp(extensions["KHR_materials_ior"]["ior"].number(1.5f), 1.0f, 3.0f);
        Vec3 transmission_tint{1.0f, 1.0f, 1.0f};
        const Json& volume = extensions["KHR_materials_volume"];
        if (volume["attenuationColor"].is_array()) {
            transmission_tint = {
                volume["attenuationColor"][0].number(1.0f),
                volume["attenuationColor"][1].number(1.0f),
                volume["attenuationColor"][2].number(1.0f),
            };
        }
        auto material = std::make_shared<StandardSurfaceMaterial>(
            material_json["name"].string("material_" + std::to_string(i)),
            base,
            std::clamp(roughness, 0.02f, 1.0f),
            metallic);
        material->transmission_weight = transmission_factor;
        material->transmission_color = transmission_tint;
        material->specular_ior = ior;
        material->unsupported_volume = !volume.is_null();
        (void)uses_transmission;
        if (pbr["baseColorFactor"].is_array() && pbr["baseColorFactor"].array().size() >= 4) {
            material->alpha = std::clamp(pbr["baseColorFactor"][3].number(1.0f), 0.0f, 1.0f);
        }
        const std::string alpha_mode = material_json["alphaMode"].string("OPAQUE");
        if (alpha_mode == "MASK") {
            material->alpha_mode = AlphaMode::Mask;
        } else if (alpha_mode == "BLEND") {
            material->alpha_mode = AlphaMode::Blend;
        }
        material->double_sided = material_json["doubleSided"].boolean(false);
        material->alpha_cutoff = std::clamp(material_json["alphaCutoff"].number(0.5f), 0.0f, 1.0f);
        const int base_texture = pbr["baseColorTexture"]["index"].integer(-1);
        material->albedo_texture = scene_texture_for_gltf_texture(root, image_to_texture, scene, base_texture, TextureRole::Color, TextureColorSpace::SRGB);
        if (material->albedo_texture) {
            material->base_color_input.texture = material->albedo_texture;
            material->base_color_input.role = TextureRole::Color;
            material->base_color_input.color_space = TextureColorSpace::SceneLinear;
        }
        const int normal_texture = material_json["normalTexture"]["index"].integer(-1);
        material->normal_texture = scene_texture_for_gltf_texture(root, image_to_texture, scene, normal_texture, TextureRole::Normal, TextureColorSpace::Raw);
        material->normal_scale = material_json["normalTexture"]["scale"].number(1.0f);
        if (material_json["emissiveFactor"].is_array()) {
            material->emission = {
                material_json["emissiveFactor"][0].number(),
                material_json["emissiveFactor"][1].number(),
                material_json["emissiveFactor"][2].number(),
            };
        }
        const float emissive_strength = extensions["KHR_materials_emissive_strength"]["emissiveStrength"].number(1.0f);
        material->emission = material->emission * std::max(0.0f, emissive_strength);
        const int emission_texture = material_json["emissiveTexture"]["index"].integer(-1);
        material->emission_texture = scene_texture_for_gltf_texture(root, image_to_texture, scene, emission_texture, TextureRole::Emission, TextureColorSpace::SRGB);
        if (material->emission_texture && material->emission.x == 0.0f && material->emission.y == 0.0f && material->emission.z == 0.0f) {
            material->emission = Vec3{1.0f, 1.0f, 1.0f} * std::max(0.0f, emissive_strength);
        }
        const int metallic_roughness_texture = pbr["metallicRoughnessTexture"]["index"].integer(-1);
        std::shared_ptr<Texture> mr_texture = scene_texture_for_gltf_texture(root, image_to_texture, scene, metallic_roughness_texture, TextureRole::Data, TextureColorSpace::Raw);
        if (mr_texture) {
            material->roughness = 1.0f;
            material->metalness = 1.0f;
            material->roughness_input.texture = mr_texture;
            material->roughness_input.channel = MaterialInputChannel::G;
            material->roughness_input.role = TextureRole::Data;
            material->roughness_input.color_space = TextureColorSpace::Raw;
            material->metalness_input.texture = mr_texture;
            material->metalness_input.channel = MaterialInputChannel::B;
            material->metalness_input.role = TextureRole::Data;
            material->metalness_input.color_space = TextureColorSpace::Raw;
        }
        const int transmission_texture = transmission["transmissionTexture"]["index"].integer(-1);
        material->transmission_input.texture = scene_texture_for_gltf_texture(root, image_to_texture, scene, transmission_texture, TextureRole::Data, TextureColorSpace::Raw);
        material->transmission_input.channel = MaterialInputChannel::R;
        const Json& sheen = extensions["KHR_materials_sheen"];
        if (sheen["sheenColorFactor"].is_array()) {
            material->sheen_color = {
                sheen["sheenColorFactor"][0].number(),
                sheen["sheenColorFactor"][1].number(),
                sheen["sheenColorFactor"][2].number(),
            };
            material->sheen_weight = 1.0f;
        }
        material->sheen_roughness = std::clamp(sheen["sheenRoughnessFactor"].number(0.0f), 0.0f, 1.0f);
        material->sheen_color_input.texture = scene_texture_for_gltf_texture(root, image_to_texture, scene, sheen["sheenColorTexture"]["index"].integer(-1), TextureRole::Color, TextureColorSpace::SRGB);
        material->sheen_roughness_input.texture = scene_texture_for_gltf_texture(root, image_to_texture, scene, sheen["sheenRoughnessTexture"]["index"].integer(-1), TextureRole::Data, TextureColorSpace::Raw);
        material->sheen_roughness_input.channel = MaterialInputChannel::A;
        const Json& clearcoat = extensions["KHR_materials_clearcoat"];
        material->coat_weight = std::clamp(clearcoat["clearcoatFactor"].number(0.0f), 0.0f, 1.0f);
        material->coat_roughness = std::clamp(clearcoat["clearcoatRoughnessFactor"].number(0.0f), 0.0f, 1.0f);
        material->coat_input.texture = scene_texture_for_gltf_texture(root, image_to_texture, scene, clearcoat["clearcoatTexture"]["index"].integer(-1), TextureRole::Data, TextureColorSpace::Raw);
        material->coat_input.channel = MaterialInputChannel::R;
        material->coat_roughness_input.texture = scene_texture_for_gltf_texture(root, image_to_texture, scene, clearcoat["clearcoatRoughnessTexture"]["index"].integer(-1), TextureRole::Data, TextureColorSpace::Raw);
        material->coat_roughness_input.channel = MaterialInputChannel::G;
        scene.materials.push_back(material);
    }
}

void import_primitive(
    const Json& primitive,
    const std::string& name,
    const Mat4& transform,
    const std::vector<std::vector<unsigned char>>& buffers,
    const std::vector<BufferView>& views,
    const std::vector<Accessor>& accessors,
    Scene& scene) {
    if (primitive["mode"].integer(4) != 4) return;
    const int position_accessor = primitive["attributes"]["POSITION"].integer(-1);
    if (position_accessor < 0 || position_accessor >= static_cast<int>(accessors.size())) return;

    std::vector<Vec3> positions;
    if (!read_vec3_accessor(buffers, views, accessors[static_cast<size_t>(position_accessor)], positions)) return;

    Mesh mesh;
    mesh.name = name;
    mesh.material = std::clamp(primitive["material"].integer(0), 0, std::max(0, static_cast<int>(scene.materials.size()) - 1));
    mesh.vertices.reserve(positions.size());
    for (Vec3 p : positions) {
        mesh.vertices.push_back(transform_point(transform, p));
    }

    const int normal_accessor = primitive["attributes"]["NORMAL"].integer(-1);
    if (normal_accessor >= 0 && normal_accessor < static_cast<int>(accessors.size())) {
        std::vector<Vec3> normals;
        if (read_vec3_accessor(buffers, views, accessors[static_cast<size_t>(normal_accessor)], normals) && normals.size() == positions.size()) {
            mesh.normals.reserve(normals.size());
            for (Vec3 n : normals) {
                mesh.normals.push_back(normalize(transform_vector(transform, n)));
            }
        }
    }

    const int texcoord_accessor = primitive["attributes"]["TEXCOORD_0"].integer(-1);
    if (texcoord_accessor >= 0 && texcoord_accessor < static_cast<int>(accessors.size())) {
        if (read_vec2_accessor(buffers, views, accessors[static_cast<size_t>(texcoord_accessor)], mesh.texcoords)) {
            for (Vec2& uv : mesh.texcoords) {
                uv.y = 1.0f - uv.y;
            }
        }
    }

    const int indices_accessor = primitive["indices"].integer(-1);
    if (indices_accessor >= 0 && indices_accessor < static_cast<int>(accessors.size())) {
        if (!read_indices(buffers, views, accessors[static_cast<size_t>(indices_accessor)], mesh.indices)) {
            return;
        }
    } else {
        mesh.indices.resize(mesh.vertices.size());
        for (uint32_t i = 0; i < mesh.indices.size(); ++i) mesh.indices[i] = i;
    }
    const bool indices_valid = std::all_of(mesh.indices.begin(), mesh.indices.end(), [&](uint32_t index) {
        return index < mesh.vertices.size();
    });
    if (mesh.indices.size() >= 3 && mesh.indices.size() % 3 == 0 && indices_valid) {
        scene.meshes.push_back(std::move(mesh));
    }
}

void import_node(
    const Json& root,
    int node_index,
    const Mat4& parent,
    const std::vector<std::vector<unsigned char>>& buffers,
    const std::vector<BufferView>& views,
    const std::vector<Accessor>& accessors,
    Scene& scene) {
    if (node_index < 0 || node_index >= static_cast<int>(root["nodes"].array().size())) return;
    const Json& node = root["nodes"][static_cast<size_t>(node_index)];
    const Mat4 world = multiply(parent, local_transform(node));
    const int mesh_index = node["mesh"].integer(-1);
    if (mesh_index >= 0 && mesh_index < static_cast<int>(root["meshes"].array().size())) {
        const Json& gltf_mesh = root["meshes"][static_cast<size_t>(mesh_index)];
        const std::string base_name = node["name"].string(gltf_mesh["name"].string("mesh_" + std::to_string(mesh_index)));
        for (size_t i = 0; i < gltf_mesh["primitives"].array().size(); ++i) {
            import_primitive(gltf_mesh["primitives"][i], base_name + "_" + std::to_string(i), world, buffers, views, accessors, scene);
        }
    }
    for (const Json& child : node["children"].array()) {
        import_node(root, child.integer(-1), world, buffers, views, accessors, scene);
    }
}

void load_camera(const Json& root, Scene& scene) {
    for (const Json& node : root["nodes"].array()) {
        const int camera_index = node["camera"].integer(-1);
        if (camera_index < 0 || camera_index >= static_cast<int>(root["cameras"].array().size())) continue;
        const Mat4 transform = local_transform(node);
        scene.camera.position = transform_point(transform, {});
        scene.camera.target = transform_point(transform, {0.0f, 0.0f, -1.0f});
        const Json& camera = root["cameras"][static_cast<size_t>(camera_index)];
        const float yfov = camera["perspective"]["yfov"].number(45.0f * kPi / 180.0f);
        scene.camera.fov_degrees = std::clamp(yfov * 180.0f / kPi, 10.0f, 120.0f);
        return;
    }
}

} // namespace

SceneLoadResult load_gltf_scene(const std::string& path) {
    LT_LOG_INFO("Loading glTF scene '{}'", path);
    Json root;
    std::vector<unsigned char> binary_chunk;
    std::string error;
    if (!load_gltf_document(path, root, binary_chunk, error)) {
        LT_LOG_ERROR("Failed to load glTF document '{}': {}", path, error);
        return {make_default_scene(), error};
    }

    const std::string scene_dir = parent_path(path);
    const std::vector<BufferView> views = parse_buffer_views(root);
    const std::vector<Accessor> accessors = parse_accessors(root);
    std::vector<std::vector<unsigned char>> buffers = load_buffers(root, scene_dir, binary_chunk, error);
    if (!error.empty()) {
        LT_LOG_ERROR("Failed to load glTF buffers '{}': {}", path, error);
        return {make_default_scene(), error};
    }

    Scene scene;
    scene.camera = {{0.0f, 1.0f, 4.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, 45.0f};
    std::vector<int> image_to_texture;
    load_images(root, scene_dir, buffers, views, scene, image_to_texture);
    load_materials(root, image_to_texture, scene);
    load_camera(root, scene);

    const int scene_index = root["scene"].integer(0);
    const Json& gltf_scene = root["scenes"][static_cast<size_t>(std::max(0, scene_index))];
    for (const Json& node : gltf_scene["nodes"].array()) {
        import_node(root, node.integer(-1), Mat4{}, buffers, views, accessors, scene);
    }

    if (scene.meshes.empty()) {
        LT_LOG_ERROR("glTF scene '{}' contains no supported triangle meshes", path);
        return {make_default_scene(), "glTF scene contains no supported triangle meshes: " + path};
    }
    LT_LOG_INFO("Loaded glTF scene '{}' (meshes={}, materials={}, textures={})",
        path, scene.meshes.size(), scene.materials.size(), scene.textures.size());
    return {std::move(scene), {}};
}

} // namespace lt
