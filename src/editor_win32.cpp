#include "lt/renderer.h"
#include "lt/log.h"
#include "lt/materialx_adapter.h"
#include "editor/editor_platform.h"
#include "editor/editor_state.h"
#include "editor/solid_preview.h"

#include <windows.h>
#include <commdlg.h>
#include <shellapi.h>
#include <d3d11.h>
#include <psapi.h>

#if LT_HAS_CUDA
#include <cuda_runtime.h>
#endif

#include "imgui.h"
#include "imgui_impl_dx11.h"
#include "imgui_impl_win32.h"

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <ctime>
#include <cctype>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <future>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam);

namespace {

using namespace lt::editor;

void upload_preview_texture();
void apply_scene_render_settings(const lt::Scene& scene);
void discard_pending_render_task();
void reset_renderer_caches();

void clear_modal_input_state() {
    if (!ImGui::GetCurrentContext()) {
        return;
    }
    ImGuiIO& io = ImGui::GetIO();
    io.ClearEventsQueue();
    io.ClearInputKeys();
    io.ClearInputMouse();
}

int show_modal_message_box(HWND hwnd, LPCWSTR text, LPCWSTR caption, UINT type) {
    const int result = MessageBoxW(hwnd, text, caption, type);
    clear_modal_input_state();
    return result;
}

bool show_open_file_dialog(OPENFILENAMEW& ofn) {
    const BOOL result = GetOpenFileNameW(&ofn);
    clear_modal_input_state();
    return result != FALSE;
}

bool show_save_file_dialog(OPENFILENAMEW& ofn) {
    const BOOL result = GetSaveFileNameW(&ofn);
    clear_modal_input_state();
    return result != FALSE;
}

lt::LogObserverHandle g_editor_log_observer = 0;
constexpr size_t kMaxEditorLogs = 2000;

const char* tool_mode_name(ToolMode mode) {
    switch (mode) {
    case ToolMode::Select: return "Select";
    case ToolMode::Move: return "Move";
    case ToolMode::Rotate: return "Rotate";
    case ToolMode::Scale: return "Scale";
    default: return "Unknown";
    }
}

const char* tool_mode_icon(ToolMode mode) {
    switch (mode) {
    case ToolMode::Select: return "S";
    case ToolMode::Move: return "M";
    case ToolMode::Rotate: return "R";
    case ToolMode::Scale: return "K";
    default: return "?";
    }
}

enum class EditorIcon {
    Select,
    Move,
    Rotate,
    Scale,
    LocalSpace,
    WorldSpace,
    Mesh,
    Sphere,
    Cube,
    Plane,
    Light,
    Material,
    Texture,
    Bake,
    Volume,
    Lightmap,
    Environment,
    Camera,
    Render,
    Object,
    Add,
    Clear,
    Duplicate,
    Delete,
    Cpu,
    Gpu,
    RenderedPreview,
    MaterialPreview,
    SolidPreview,
    WireframePreview,
    Fullscreen,
    ExitFullscreen,
};

struct EditorIconInfo {
    const char* fallback;
    const char* blender_source;
};

const EditorIconInfo& editor_icon_info(EditorIcon icon) {
    static const EditorIconInfo infos[] = {
        {"S", "ops.generic.select.dat"},
        {"M", "ops.transform.translate.dat"},
        {"R", "ops.transform.rotate.dat"},
        {"K", "ops.transform.resize.dat"},
        {"L", nullptr},
        {"W", nullptr},
        {"M", nullptr},    // Mesh — procedural
        {"O", nullptr},    // Sphere — procedural
        {"C", nullptr},    // Cube — procedural
        {"P", nullptr},    // Plane — procedural
        {"L", nullptr},
        {"M", nullptr},
        {"T", "brush.paint_texture.fill.dat"},
        {"B", nullptr},
        {"V", nullptr},
        {"U", nullptr},
        {"E", nullptr},
        {"C", nullptr},
        {"R", nullptr},
        {"O", nullptr},
        {"+", nullptr},
        {"X", nullptr},
        {"D", nullptr},
        {"Del", nullptr},
        {"C", nullptr},
        {"G", nullptr},
        {"R", nullptr},
        {"M", nullptr},
        {"S", nullptr},
        {"W", nullptr},
        {"F", nullptr},
        {"X", nullptr},
    };
    return infos[static_cast<int>(icon)];
}

EditorIcon tool_mode_editor_icon(ToolMode mode) {
    switch (mode) {
    case ToolMode::Select: return EditorIcon::Select;
    case ToolMode::Move: return EditorIcon::Move;
    case ToolMode::Rotate: return EditorIcon::Rotate;
    case ToolMode::Scale: return EditorIcon::Scale;
    default: return EditorIcon::Select;
    }
}

void format_bytes(size_t bytes, char* buf, size_t buf_size) {
    if (bytes >= 1024ull * 1024ull * 1024ull)
        std::snprintf(buf, buf_size, "%.2f GiB", static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
    else if (bytes >= 1024ull * 1024ull)
        std::snprintf(buf, buf_size, "%.2f MiB", static_cast<double>(bytes) / (1024.0 * 1024.0));
    else if (bytes >= 1024ull)
        std::snprintf(buf, buf_size, "%.2f KiB", static_cast<double>(bytes) / 1024.0);
    else
        std::snprintf(buf, buf_size, "%zu B", bytes);
}

void enqueue_editor_log(const lt::LogRecord& record) {
    std::lock_guard lock(g_editor.log_mutex);
    if (g_editor.pending_logs.size() >= 512) {
        g_editor.pending_logs.erase(g_editor.pending_logs.begin());
    }
    g_editor.pending_logs.push_back(record);
}

void drain_editor_logs() {
    std::vector<lt::LogRecord> pending;
    {
        std::lock_guard lock(g_editor.log_mutex);
        pending.swap(g_editor.pending_logs);
    }
    if (pending.empty()) {
        return;
    }
    g_editor.logs.insert(g_editor.logs.end(), std::make_move_iterator(pending.begin()), std::make_move_iterator(pending.end()));
    if (g_editor.logs.size() > kMaxEditorLogs) {
        g_editor.logs.erase(g_editor.logs.begin(), g_editor.logs.begin() + static_cast<std::ptrdiff_t>(g_editor.logs.size() - kMaxEditorLogs));
    }
}

std::string log_time_text(const lt::LogRecord& record) {
    const std::time_t time = std::chrono::system_clock::to_time_t(record.timestamp);
    std::tm local_time{};
    localtime_s(&local_time, &time);
    char buffer[32] = {};
    std::strftime(buffer, sizeof(buffer), "%H:%M:%S", &local_time);
    return buffer;
}

ImVec4 log_level_color(lt::LogLevel level) {
    switch (level) {
    case lt::LogLevel::Trace: return {0.58f, 0.60f, 0.64f, 1.0f};
    case lt::LogLevel::Debug: return {0.50f, 0.70f, 0.95f, 1.0f};
    case lt::LogLevel::Info: return {0.86f, 0.86f, 0.84f, 1.0f};
    case lt::LogLevel::Warn: return {1.0f, 0.73f, 0.30f, 1.0f};
    case lt::LogLevel::Error: return {1.0f, 0.38f, 0.34f, 1.0f};
    case lt::LogLevel::Critical: return {1.0f, 0.18f, 0.20f, 1.0f};
    case lt::LogLevel::Off: return {0.86f, 0.86f, 0.84f, 1.0f};
    }
    return {0.86f, 0.86f, 0.84f, 1.0f};
}

bool passes_log_filter(const lt::LogRecord& record) {
    return static_cast<int>(record.level) >= static_cast<int>(g_editor.log_filter_level);
}

void initialize_editor_logging() {
    lt::LogConfig config;
    config.logger_name = "lt_editor";
    config.enable_file = true;
    config.file_path = "logs/lt_editor.log";
    config.file_level = lt::LogLevel::Debug;
    config.enable_console = false;
    lt::initialize_logging(config);
    g_editor_log_observer = lt::add_log_observer(enqueue_editor_log);
    LT_LOG_INFO("Editor logging initialized");
}

void shutdown_editor_logging() {
    LT_LOG_INFO("Editor shutting down");
    lt::remove_log_observer(g_editor_log_observer);
    g_editor_log_observer = 0;
    lt::flush_logs();
    lt::shutdown_logging();
}

struct EditorLoggingScope {
    ~EditorLoggingScope() {
        shutdown_editor_logging();
    }
};

void set_renderer(bool use_cuda) {
    const bool can_use_cuda = use_cuda &&
        g_editor.cuda.available() &&
        !lt::stylized_rendering_enabled(g_editor.settings, g_editor.scene);
    lt::IRenderer* next = can_use_cuda ? static_cast<lt::IRenderer*>(&g_editor.cuda) : static_cast<lt::IRenderer*>(&g_editor.cpu);
    if (use_cuda && !can_use_cuda) {
        LT_LOG_WARN("CUDA renderer request fell back to CPU");
    }
    if (g_editor.renderer != next) {
        g_editor.renderer = next;
        LT_LOG_INFO("Editor renderer changed to {}", g_editor.renderer->name());
        reset_accumulation(lt::RenderDirty::All);
    }
}

void select_mesh(int index) {
    if (index >= 0 && index < static_cast<int>(g_editor.scene.meshes.size())) {
        g_editor.selection_kind = SelectionKind::Mesh;
        g_editor.selected_mesh = index;
        g_editor.selected_sphere = -1;
        g_editor.selected_directional_light = -1;
        g_editor.selected_point_light = -1;
    } else {
        g_editor.selection_kind = SelectionKind::None;
        g_editor.selected_mesh = -1;
        g_editor.selected_sphere = -1;
        g_editor.selected_directional_light = -1;
        g_editor.selected_point_light = -1;
    }
}

void select_sphere(int index) {
    if (index >= 0 && index < static_cast<int>(g_editor.scene.spheres.size())) {
        g_editor.selection_kind = SelectionKind::Sphere;
        g_editor.selected_sphere = index;
        g_editor.selected_mesh = -1;
        g_editor.selected_directional_light = -1;
        g_editor.selected_point_light = -1;
    } else {
        g_editor.selection_kind = SelectionKind::None;
        g_editor.selected_mesh = -1;
        g_editor.selected_sphere = -1;
        g_editor.selected_directional_light = -1;
        g_editor.selected_point_light = -1;
    }
}

void select_directional_light(int index) {
    if (index >= 0 && index < static_cast<int>(g_editor.scene.directional_lights.size())) {
        g_editor.selection_kind = SelectionKind::DirectionalLight;
        g_editor.selected_directional_light = index;
        g_editor.selected_mesh = -1;
        g_editor.selected_sphere = -1;
        g_editor.selected_point_light = -1;
    } else {
        g_editor.selection_kind = SelectionKind::None;
        g_editor.selected_mesh = -1;
        g_editor.selected_sphere = -1;
        g_editor.selected_directional_light = -1;
        g_editor.selected_point_light = -1;
    }
}

void select_point_light(int index) {
    if (index >= 0 && index < static_cast<int>(g_editor.scene.point_lights.size())) {
        g_editor.selection_kind = SelectionKind::PointLight;
        g_editor.selected_point_light = index;
        g_editor.selected_mesh = -1;
        g_editor.selected_sphere = -1;
        g_editor.selected_directional_light = -1;
    } else {
        g_editor.selection_kind = SelectionKind::None;
        g_editor.selected_mesh = -1;
        g_editor.selected_sphere = -1;
        g_editor.selected_directional_light = -1;
        g_editor.selected_point_light = -1;
    }
}

bool has_selection() {
    if (g_editor.selection_kind == SelectionKind::Mesh) {
        return g_editor.selected_mesh >= 0 && g_editor.selected_mesh < static_cast<int>(g_editor.scene.meshes.size());
    }
    if (g_editor.selection_kind == SelectionKind::Sphere) {
        return g_editor.selected_sphere >= 0 && g_editor.selected_sphere < static_cast<int>(g_editor.scene.spheres.size());
    }
    if (g_editor.selection_kind == SelectionKind::DirectionalLight) {
        return g_editor.selected_directional_light >= 0 && g_editor.selected_directional_light < static_cast<int>(g_editor.scene.directional_lights.size());
    }
    if (g_editor.selection_kind == SelectionKind::PointLight) {
        return g_editor.selected_point_light >= 0 && g_editor.selected_point_light < static_cast<int>(g_editor.scene.point_lights.size());
    }
    return false;
}

bool has_mesh_selection() {
    return g_editor.selection_kind == SelectionKind::Mesh &&
        g_editor.selected_mesh >= 0 &&
        g_editor.selected_mesh < static_cast<int>(g_editor.scene.meshes.size());
}

bool has_sphere_selection() {
    return g_editor.selection_kind == SelectionKind::Sphere &&
        g_editor.selected_sphere >= 0 &&
        g_editor.selected_sphere < static_cast<int>(g_editor.scene.spheres.size());
}

bool has_directional_light_selection() {
    return g_editor.selection_kind == SelectionKind::DirectionalLight &&
        g_editor.selected_directional_light >= 0 &&
        g_editor.selected_directional_light < static_cast<int>(g_editor.scene.directional_lights.size());
}

bool has_point_light_selection() {
    return g_editor.selection_kind == SelectionKind::PointLight &&
        g_editor.selected_point_light >= 0 &&
        g_editor.selected_point_light < static_cast<int>(g_editor.scene.point_lights.size());
}

void invalidate_mesh_bounds_cache() {
    g_bounds_cache.scene_generation = 0;
    g_bounds_cache.local_min.clear();
    g_bounds_cache.local_max.clear();
    g_pick_cache.scene_generation = 0;
    g_pick_cache.render_scene = {};
}

void discard_pending_render_task() {
    if (!g_render_future.valid()) {
        return;
    }
    g_render_future.wait();
    (void)g_render_future.get();
}

void reset_renderer_caches() {
    discard_pending_render_task();
    g_editor.cpu.reset();
    g_editor.cuda.reset();
}

void set_scene(lt::Scene scene, const std::string& path) {
    reset_renderer_caches();
    g_editor.scene = std::move(scene);
    g_editor.scene_path = path;
    apply_scene_render_settings(g_editor.scene);
    invalidate_mesh_bounds_cache();
    select_mesh(-1);
    LT_LOG_INFO("Editor scene set to '{}' (meshes={}, spheres={}, dir_lights={}, point_lights={}, materials={}, textures={})",
        g_editor.scene_path,
        g_editor.scene.meshes.size(),
        g_editor.scene.spheres.size(),
        g_editor.scene.directional_lights.size(),
        g_editor.scene.point_lights.size(),
        g_editor.scene.materials.size(),
        g_editor.scene.textures.size());
    reset_accumulation(lt::RenderDirty::All);
}

bool scene_load_in_progress() {
    return g_load_task.future.valid();
}

void load_scene_file(const std::string& path) {
    if (scene_load_in_progress()) {
        LT_LOG_WARN("Scene load skipped while another scene is loading: '{}'", path);
        show_modal_message_box(g_hwnd, L"A scene is already loading. Please wait for it to finish.", L"Scene load in progress", MB_OK | MB_ICONINFORMATION);
        return;
    }
    LT_LOG_INFO("Starting async scene load '{}'", path);
    g_load_task.generation += 1;
    g_load_task.path = path;
    g_load_task.started = std::chrono::steady_clock::now();
    g_load_task.future = std::async(std::launch::async, [path]() {
        return lt::load_scene(path);
    });
}

std::string resolve_startup_scene_path(const std::string& path) {
    const auto is_url = [](const std::string& value) {
        return value.rfind("http://", 0) == 0 || value.rfind("https://", 0) == 0;
    };
    const auto resolve_from_ancestors = [](std::filesystem::path base, const std::filesystem::path& relative) -> std::string {
        std::error_code error;
        base = std::filesystem::absolute(base, error);
        if (error) {
            return {};
        }
        for (;;) {
            const std::filesystem::path candidate = base / relative;
            if (std::filesystem::exists(candidate)) {
                return candidate.string();
            }
            const std::filesystem::path parent = base.parent_path();
            if (parent.empty() || parent == base) {
                break;
            }
            base = parent;
        }
        return {};
    };
    if (is_url(path)) {
        return path;
    }
    std::filesystem::path scene(path);
    if (scene.is_absolute()) {
        return scene.string();
    }
    if (std::filesystem::exists(scene)) {
        return std::filesystem::absolute(scene).string();
    }
    if (const std::string resolved = resolve_from_ancestors(std::filesystem::current_path(), scene); !resolved.empty()) {
        return resolved;
    }
    char exe_path[MAX_PATH] = {};
    if (GetModuleFileNameA(nullptr, exe_path, MAX_PATH) > 0) {
        const std::filesystem::path exe_dir = std::filesystem::path(exe_path).parent_path();
        if (const std::string resolved = resolve_from_ancestors(exe_dir, scene); !resolved.empty()) {
            return resolved;
        }
    }
    return path;
}

void poll_scene_load_result() {
    if (!g_load_task.future.valid() || g_load_task.future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return;
    lt::SceneLoadResult loaded = g_load_task.future.get();
    const std::string path = g_load_task.path;
    g_load_task = {};
    if (!loaded.error.empty()) {
        LT_LOG_WARN("Scene load warning for '{}': {}", path, loaded.error);
        if (loaded.scene.meshes.empty() && loaded.scene.spheres.empty()) {
            show_modal_message_box(g_hwnd, widen(loaded.error).c_str(), L"Scene load warning", MB_OK | MB_ICONWARNING);
        }
        if (loaded.scene.meshes.empty() && loaded.scene.spheres.empty()) {
            LT_LOG_ERROR("Scene load produced no renderable geometry: '{}'", path);
            return;
        }
    }
    set_scene(std::move(loaded.scene), path);
}

void new_scene() {
    lt::Scene scene = lt::make_default_scene();
    scene.uses_builtin_default_meshes = false;
    set_scene(std::move(scene), "");
}

void open_scene_dialog() {
    wchar_t filename[MAX_PATH] = {};
    OPENFILENAMEW ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = g_hwnd;
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrFilter = L"Scene files (*.lt;*.fbx;*.pyscene;*.glb;*.gltf;*.pbrt)\0*.lt;*.fbx;*.pyscene;*.glb;*.gltf;*.pbrt\0LightTransport scenes (*.lt)\0*.lt\0FBX scenes (*.fbx;*.pyscene)\0*.fbx;*.pyscene\0glTF scenes (*.glb;*.gltf)\0*.glb;*.gltf\0PBRT scenes (*.pbrt)\0*.pbrt\0All files (*.*)\0*.*\0";
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;
    if (show_open_file_dialog(ofn)) load_scene_file(narrow(filename));
}

std::string texture_name_from_path(const std::string& path) {
    size_t begin = path.find_last_of("\\/");
    begin = begin == std::string::npos ? 0 : begin + 1;
    size_t end = path.find_last_of('.');
    if (end == std::string::npos || end < begin) {
        end = path.size();
    }
    std::string name = path.substr(begin, end - begin);
    return name.empty() ? "texture" : name;
}

std::string lowercase_extension(const std::string& path) {
    const size_t dot = path.find_last_of('.');
    std::string ext = dot == std::string::npos ? std::string{} : path.substr(dot);
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext;
}

template <size_t N>
void copy_setting_text(char (&target)[N], const std::string& value) {
    std::snprintf(target, N, "%s", value.c_str());
}

bool is_url_path(const std::string& path) {
    return path.rfind("http://", 0) == 0 || path.rfind("https://", 0) == 0;
}

std::string default_irradiance_volume_cache_path() {
    if (g_editor.scene_path.empty()) {
        return "cache/irradiance_volume/untitled.ivol";
    }
    if (is_url_path(g_editor.scene_path)) {
        std::string safe = g_editor.scene_path;
        for (char& c : safe) {
            if (!std::isalnum(static_cast<unsigned char>(c))) {
                c = '_';
            }
        }
        return (std::filesystem::path("cache") / "irradiance_volume" / (safe + ".ivol")).string();
    }
    return g_editor.scene_path + ".ivol";
}

std::string default_lightmap_cache_path() {
    if (g_editor.scene_path.empty()) {
        return "cache/lightmap/untitled.lmap";
    }
    if (is_url_path(g_editor.scene_path)) {
        std::string safe = g_editor.scene_path;
        for (char& c : safe) {
            if (!std::isalnum(static_cast<unsigned char>(c))) {
                c = '_';
            }
        }
        return (std::filesystem::path("cache") / "lightmap" / (safe + ".lmap")).string();
    }
    return g_editor.scene_path + ".lmap";
}

std::string ensure_lt_extension(const std::string& path) {
    if (lowercase_extension(path) == ".lt") {
        return path;
    }
    std::filesystem::path scene_path(path);
    scene_path.replace_extension(".lt");
    return scene_path.string();
}

bool same_scene_file(const std::string& left, const std::string& right) {
    if (left.empty() || right.empty() || is_url_path(left) || is_url_path(right)) {
        return false;
    }
    std::error_code error;
    std::filesystem::path left_path = std::filesystem::absolute(left, error);
    if (error) {
        return false;
    }
    left_path = left_path.lexically_normal();
    std::filesystem::path right_path = std::filesystem::absolute(right, error);
    if (error) {
        return false;
    }
    right_path = right_path.lexically_normal();
#if defined(_WIN32)
    std::string left_text = left_path.string();
    std::string right_text = right_path.string();
    std::transform(left_text.begin(), left_text.end(), left_text.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    std::transform(right_text.begin(), right_text.end(), right_text.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return left_text == right_text;
#else
    return left_path == right_path;
#endif
}

std::string sanitize_asset_name(std::string name) {
    if (name.empty()) {
        name = "texture";
    }
    for (char& c : name) {
        const unsigned char value = static_cast<unsigned char>(c);
        if (!std::isalnum(value) && c != '-' && c != '_') {
            c = '_';
        }
    }
    if (name.empty()) {
        return "texture";
    }
    return name;
}

std::string normalized_extension(std::string extension, const char* fallback = ".png") {
    if (extension.empty()) {
        return fallback;
    }
    if (extension.front() != '.') {
        extension.insert(extension.begin(), '.');
    }
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (extension == ".jpeg") {
        return ".jpg";
    }
    return extension;
}

bool texture_needs_hdr_sidecar(const lt::Texture& texture) {
    return std::any_of(texture.pixels.begin(), texture.pixels.end(), [](lt::Vec3 value) {
        return value.x < 0.0f || value.x > 1.0f ||
            value.y < 0.0f || value.y > 1.0f ||
            value.z < 0.0f || value.z > 1.0f;
    });
}

std::filesystem::path resolve_texture_path_for_scene(const std::string& texture_path, const std::string& source_scene_path) {
    std::filesystem::path path(texture_path);
    if (path.is_absolute() || source_scene_path.empty() || is_url_path(source_scene_path)) {
        return path;
    }
    return std::filesystem::path(source_scene_path).parent_path() / path;
}

std::string path_relative_to_scene(const std::filesystem::path& asset_path, const std::filesystem::path& scene_path) {
    const std::filesystem::path scene_dir = scene_path.parent_path();
    std::error_code error;
    const std::filesystem::path absolute_asset = std::filesystem::absolute(asset_path, error);
    if (error) {
        return asset_path.string();
    }
    const std::filesystem::path absolute_scene_dir = std::filesystem::absolute(scene_dir.empty() ? std::filesystem::path(".") : scene_dir, error);
    if (error) {
        return absolute_asset.string();
    }
    const std::filesystem::path relative = std::filesystem::relative(absolute_asset, absolute_scene_dir, error);
    return error ? absolute_asset.string() : relative.generic_string();
}

std::filesystem::path unique_asset_path(
    const std::filesystem::path& assets_dir,
    const std::string& base_name,
    const std::string& extension,
    std::unordered_set<std::string>& used_names) {
    const std::string safe_base = sanitize_asset_name(base_name);
    const std::string safe_extension = normalized_extension(extension);
    for (int suffix = 0; suffix < 100000; ++suffix) {
        const std::string filename = suffix == 0
            ? safe_base + safe_extension
            : safe_base + "_" + std::to_string(suffix) + safe_extension;
        if (used_names.insert(filename).second) {
            return assets_dir / filename;
        }
    }
    return assets_dir / (safe_base + "_" + std::to_string(used_names.size()) + safe_extension);
}

bool write_bytes_file(const std::filesystem::path& path, const std::vector<unsigned char>& bytes, std::string& error) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        error = "Could not write texture asset: " + path.string();
        return false;
    }
    output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!output) {
        error = "Could not finish writing texture asset: " + path.string();
        return false;
    }
    return true;
}

bool export_texture_sidecar(
    lt::Scene& scene,
    const std::shared_ptr<lt::Texture>& texture,
    const std::string& target_scene_path,
    std::unordered_set<std::string>& used_asset_names,
    std::string& error) {
    if (!texture) {
        return true;
    }
    const std::filesystem::path target_scene(target_scene_path);
    const std::string scene_stem = target_scene.stem().string().empty() ? "scene" : sanitize_asset_name(target_scene.stem().string());
    const std::filesystem::path assets_dir = target_scene.parent_path() / (scene_stem + "_assets");
    std::error_code fs_error;
    std::filesystem::create_directories(assets_dir, fs_error);
    if (fs_error) {
        error = "Could not create texture asset directory: " + assets_dir.string();
        return false;
    }

    std::string extension;
    if (!texture->encoded_bytes.empty()) {
        extension = normalized_extension(texture->encoded_extension);
    } else {
        const bool prefer_hdr = scene.environment.texture == texture || texture_needs_hdr_sidecar(*texture);
        extension = prefer_hdr ? ".hdr" : ".png";
    }

    const std::filesystem::path asset_path = unique_asset_path(assets_dir, texture->name, extension, used_asset_names);
    if (!texture->encoded_bytes.empty()) {
        if (!write_bytes_file(asset_path, texture->encoded_bytes, error)) {
            return false;
        }
    } else if (extension == ".hdr") {
        if (!lt::write_texture_hdr(*texture, asset_path.string(), error)) {
            return false;
        }
    } else if (!lt::write_texture_png(*texture, asset_path.string(), error)) {
        return false;
    }
    texture->path = path_relative_to_scene(asset_path, target_scene);
    return true;
}

bool rewrite_texture_paths_for_save_as(lt::Scene& scene, const std::string& source_scene_path, const std::string& target_scene_path, std::string& error) {
    std::unordered_set<std::string> used_asset_names;
    const std::filesystem::path target_scene(target_scene_path);
    for (const std::shared_ptr<lt::Texture>& texture : scene.textures) {
        if (!texture) {
            continue;
        }
        if (!texture->encoded_bytes.empty()) {
            if (!export_texture_sidecar(scene, texture, target_scene_path, used_asset_names, error)) {
                return false;
            }
            continue;
        }
        if (!texture->path.empty() && !is_url_path(texture->path)) {
            const std::filesystem::path source_texture = resolve_texture_path_for_scene(texture->path, source_scene_path);
            std::error_code exists_error;
            if (std::filesystem::exists(source_texture, exists_error) && !exists_error) {
                texture->path = path_relative_to_scene(source_texture, target_scene);
                continue;
            }
        }
        if (!texture->pixels.empty()) {
            if (!export_texture_sidecar(scene, texture, target_scene_path, used_asset_names, error)) {
                return false;
            }
            continue;
        }
        error = "Texture has no usable file path or pixels: " + texture->name;
        return false;
    }
    return true;
}

lt::SceneRenderSettings capture_scene_render_settings() {
    lt::SceneRenderSettings settings;
    settings.sampling_mode = static_cast<int>(g_editor.settings.sampling_mode);
    settings.mis_heuristic = static_cast<int>(g_editor.settings.mis_heuristic);
    settings.stylized_samples = g_editor.settings.stylized_samples;
    settings.stylized_max_depth = g_editor.settings.stylized_max_depth;
    settings.use_irradiance_volume = g_editor.settings.use_irradiance_volume;
    settings.irradiance_volume_grid_resolution = g_editor.settings.irradiance_volume_grid_resolution;
    settings.irradiance_volume_subgrid_resolution = g_editor.settings.irradiance_volume_subgrid_resolution;
    settings.irradiance_volume_direction_resolution = g_editor.settings.irradiance_volume_direction_resolution;
    settings.irradiance_volume_bake_samples = g_editor.settings.irradiance_volume_bake_samples;
    settings.irradiance_volume_bake_bounces = g_editor.settings.irradiance_volume_bake_bounces;
    settings.irradiance_volume_bounds_inset = g_editor.settings.irradiance_volume_bounds_inset;
    settings.irradiance_volume_principled_gi = g_editor.settings.irradiance_volume_principled_gi;
    settings.irradiance_volume_debug_probes = g_editor.settings.irradiance_volume_debug_probes;
    settings.irradiance_volume_debug_probe_radius_scale = g_editor.settings.irradiance_volume_debug_probe_radius_scale;
    settings.irradiance_volume_cache_enabled = g_editor.settings.irradiance_volume_cache_enabled;
    settings.irradiance_volume_auto_update = g_editor.settings.irradiance_volume_auto_update;
    settings.irradiance_volume_manual_bounds = g_editor.settings.irradiance_volume_manual_bounds;
    settings.irradiance_volume_bounds_min = g_editor.settings.irradiance_volume_bounds_min;
    settings.irradiance_volume_bounds_max = g_editor.settings.irradiance_volume_bounds_max;
    settings.irradiance_volume_bake_backend = static_cast<int>(g_editor.settings.irradiance_volume_bake_backend);
    settings.use_lightmap = g_editor.settings.use_lightmap;
    settings.lightmap_resolution = g_editor.settings.lightmap_resolution;
    settings.lightmap_padding = g_editor.settings.lightmap_padding;
    settings.lightmap_dilation = g_editor.settings.lightmap_dilation;
    settings.lightmap_bake_samples = g_editor.settings.lightmap_bake_samples;
    settings.lightmap_bake_bounces = g_editor.settings.lightmap_bake_bounces;
    settings.lightmap_principled_gi = g_editor.settings.lightmap_principled_gi;
    settings.lightmap_cache_enabled = g_editor.settings.lightmap_cache_enabled;
    settings.lightmap_auto_update = g_editor.settings.lightmap_auto_update;
    settings.lightmap_bake_backend = static_cast<int>(g_editor.settings.lightmap_bake_backend);
    return settings;
}

void sync_editor_settings_to_scene(lt::Scene& scene) {
    scene.render_settings = capture_scene_render_settings();
    scene.has_render_settings = true;
}

void apply_scene_render_settings(const lt::Scene& scene) {
    if (!scene.has_render_settings) {
        return;
    }
    const lt::SceneRenderSettings& settings = scene.render_settings;
    g_editor.settings.sampling_mode = static_cast<lt::PathSamplingMode>(settings.sampling_mode);
    g_editor.settings.mis_heuristic = static_cast<lt::MisHeuristic>(settings.mis_heuristic);
    g_editor.settings.stylized_samples = settings.stylized_samples;
    g_editor.settings.stylized_max_depth = settings.stylized_max_depth;
    g_editor.settings.use_irradiance_volume = settings.use_irradiance_volume;
    g_editor.settings.irradiance_volume_grid_resolution = settings.irradiance_volume_grid_resolution;
    g_editor.settings.irradiance_volume_subgrid_resolution = settings.irradiance_volume_subgrid_resolution;
    g_editor.settings.irradiance_volume_direction_resolution = settings.irradiance_volume_direction_resolution;
    g_editor.settings.irradiance_volume_bake_samples = settings.irradiance_volume_bake_samples;
    g_editor.settings.irradiance_volume_bake_bounces = settings.irradiance_volume_bake_bounces;
    g_editor.settings.irradiance_volume_bounds_inset = settings.irradiance_volume_bounds_inset;
    g_editor.settings.irradiance_volume_principled_gi = settings.irradiance_volume_principled_gi;
    g_editor.settings.irradiance_volume_debug_probes = settings.irradiance_volume_debug_probes;
    g_editor.settings.irradiance_volume_debug_probe_radius_scale = settings.irradiance_volume_debug_probe_radius_scale;
    g_editor.settings.irradiance_volume_cache_enabled = settings.irradiance_volume_cache_enabled;
    g_editor.settings.irradiance_volume_auto_update = settings.irradiance_volume_auto_update;
    g_editor.settings.irradiance_volume_manual_bounds = settings.irradiance_volume_manual_bounds;
    g_editor.settings.irradiance_volume_bounds_min = settings.irradiance_volume_bounds_min;
    g_editor.settings.irradiance_volume_bounds_max = settings.irradiance_volume_bounds_max;
    g_editor.settings.irradiance_volume_bake_backend = static_cast<lt::IrradianceVolumeBakeBackend>(settings.irradiance_volume_bake_backend);
    g_editor.settings.use_lightmap = settings.use_lightmap;
    g_editor.settings.lightmap_resolution = settings.lightmap_resolution;
    g_editor.settings.lightmap_padding = settings.lightmap_padding;
    g_editor.settings.lightmap_dilation = settings.lightmap_dilation;
    g_editor.settings.lightmap_bake_samples = settings.lightmap_bake_samples;
    g_editor.settings.lightmap_bake_bounces = settings.lightmap_bake_bounces;
    g_editor.settings.lightmap_principled_gi = settings.lightmap_principled_gi;
    g_editor.settings.lightmap_cache_enabled = settings.lightmap_cache_enabled;
    g_editor.settings.lightmap_auto_update = settings.lightmap_auto_update;
    g_editor.settings.lightmap_bake_backend = static_cast<lt::LightmapBakeBackend>(settings.lightmap_bake_backend);
}

void load_texture_dialog(lt::Material& material) {
    wchar_t filename[MAX_PATH] = {};
    OPENFILENAMEW ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = g_hwnd;
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrFilter = L"Image textures (*.ppm;*.png;*.jpg;*.jpeg;*.hdr;*.exr)\0*.ppm;*.png;*.jpg;*.jpeg;*.hdr;*.exr\0All files (*.*)\0*.*\0";
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;
    if (!show_open_file_dialog(ofn)) return;

    const std::string path = narrow(filename);
    const std::string name = texture_name_from_path(path);
    lt::Texture texture;
    std::string error;
    if (!lt::load_texture_file(name, path, texture, error)) {
        LT_LOG_ERROR("Texture load failed '{}': {}", path, error);
        show_modal_message_box(g_hwnd, widen(error).c_str(), L"Texture load failed", MB_OK | MB_ICONERROR);
        return;
    }
    auto shared = std::make_shared<lt::Texture>(std::move(texture));
    g_editor.scene.textures.push_back(shared);
    material.albedo_texture = shared;
    LT_LOG_INFO("Loaded material texture '{}'", path);
    reset_accumulation(lt::RenderDirty::Texture | lt::RenderDirty::Material);
}

void load_environment_texture_dialog() {
    wchar_t filename[MAX_PATH] = {};
    OPENFILENAMEW ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = g_hwnd;
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrFilter = L"HDRI environment (*.hdr;*.exr)\0*.hdr;*.exr\0Image textures (*.hdr;*.exr;*.ppm;*.png;*.jpg;*.jpeg)\0*.hdr;*.exr;*.ppm;*.png;*.jpg;*.jpeg\0All files (*.*)\0*.*\0";
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;
    if (!show_open_file_dialog(ofn)) return;

    const std::string path = narrow(filename);
    const std::string name = texture_name_from_path(path);
    lt::Texture texture;
    std::string error;
    if (!lt::load_texture_file(name, path, texture, error)) {
        LT_LOG_ERROR("Environment load failed '{}': {}", path, error);
        show_modal_message_box(g_hwnd, widen(error).c_str(), L"Environment load failed", MB_OK | MB_ICONERROR);
        return;
    }
    auto shared = std::make_shared<lt::Texture>(std::move(texture));
    g_editor.scene.textures.push_back(shared);
    g_editor.scene.environment.texture = shared;
    g_editor.scene.environment.constant = false;
    g_editor.scene.environment.mapping = lt::Environment::Mapping::Equirectangular;
    g_editor.scene.environment.light_from_world_x = {1.0f, 0.0f, 0.0f};
    g_editor.scene.environment.light_from_world_y = {0.0f, 1.0f, 0.0f};
    g_editor.scene.environment.light_from_world_z = {0.0f, 0.0f, 1.0f};
    LT_LOG_INFO("Loaded environment texture '{}'", path);
    reset_accumulation(lt::RenderDirty::Texture | lt::RenderDirty::Environment);
}

void restore_texture_paths(const std::vector<std::pair<std::shared_ptr<lt::Texture>, std::string>>& original_paths) {
    for (const auto& original : original_paths) {
        if (original.first) {
            original.first->path = original.second;
        }
    }
}

bool save_scene_to_path(const std::string& requested_path, bool rewrite_assets) {
    const std::string target_path = ensure_lt_extension(requested_path);
    sync_editor_settings_to_scene(g_editor.scene);

    std::vector<std::pair<std::shared_ptr<lt::Texture>, std::string>> original_paths;
    if (rewrite_assets) {
        original_paths.reserve(g_editor.scene.textures.size());
        for (const std::shared_ptr<lt::Texture>& texture : g_editor.scene.textures) {
            if (texture) {
                original_paths.push_back({texture, texture->path});
            }
        }
    }

    std::string error;
    if (rewrite_assets && !rewrite_texture_paths_for_save_as(g_editor.scene, g_editor.scene_path, target_path, error)) {
        restore_texture_paths(original_paths);
        LT_LOG_ERROR("Save As asset export failed '{}': {}", target_path, error);
        show_modal_message_box(g_hwnd, widen(error).c_str(), L"Save As failed", MB_OK | MB_ICONERROR);
        return false;
    }

    if (!lt::save_scene(g_editor.scene, target_path, error)) {
        if (rewrite_assets) {
            restore_texture_paths(original_paths);
        }
        LT_LOG_ERROR("Save failed '{}': {}", target_path, error);
        show_modal_message_box(g_hwnd, widen(error).c_str(), L"Save failed", MB_OK | MB_ICONERROR);
        return false;
    }

    g_editor.scene_path = target_path;
    if (rewrite_assets) {
        reset_renderer_caches();
        reset_accumulation(lt::RenderDirty::All);
    }
    LT_LOG_INFO("Saved scene '{}'", g_editor.scene_path);
    return true;
}

bool save_scene_as_dialog() {
    wchar_t filename[MAX_PATH] = {};
    std::filesystem::path suggested = is_url_path(g_editor.scene_path) || g_editor.scene_path.empty()
        ? std::filesystem::path("scene.lt")
        : std::filesystem::path(g_editor.scene_path);
    suggested.replace_extension(".lt");
    const std::wstring initial = widen(suggested.string());
    const size_t count = std::min<size_t>(initial.size(), MAX_PATH - 1u);
    for (size_t i = 0; i < count; ++i) {
        filename[i] = initial[i];
    }

    OPENFILENAMEW ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = g_hwnd;
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrFilter = L"LightTransport scenes (*.lt)\0*.lt\0All files (*.*)\0*.*\0";
    ofn.lpstrDefExt = L"lt";
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT;
    if (!show_save_file_dialog(ofn)) {
        return false;
    }
    const std::string target_path = ensure_lt_extension(narrow(filename));
    const bool source_is_lt = lowercase_extension(g_editor.scene_path) == ".lt";
    const bool rewrite_assets = !source_is_lt || !same_scene_file(g_editor.scene_path, target_path);
    return save_scene_to_path(target_path, rewrite_assets);
}

void save_scene() {
    if (lowercase_extension(g_editor.scene_path) != ".lt") {
        LT_LOG_INFO("Save requested for imported scene '{}'; opening Save As", g_editor.scene_path);
        save_scene_as_dialog();
        return;
    }
    const std::wstring message = L"Save changes to this scene?\n\n" + widen(g_editor.scene_path);
    if (show_modal_message_box(g_hwnd, message.c_str(), L"Confirm Save", MB_YESNO | MB_ICONQUESTION | MB_DEFBUTTON2) != IDYES) {
        return;
    }
    save_scene_to_path(g_editor.scene_path, false);
}

int default_material(const char* preferred) {
    int material = lt::find_material(g_editor.scene, preferred);
    if (material < 0) material = lt::find_material(g_editor.scene, "white");
    if (material >= 0) return material;
    const bool is_light = std::string(preferred) == "light";
    g_editor.scene.materials.push_back(lt::make_material(
        is_light ? "light" : "white",
        is_light ? lt::Vec3{1.0f, 0.9f, 0.72f} : lt::Vec3{0.78f, 0.78f, 0.72f},
        lt::BrdfModel::Lambertian,
        0.65f,
        0.0f));
    return static_cast<int>(g_editor.scene.materials.size()) - 1;
}

bool material_name_exists(const std::string& name, int ignore_index = -1) {
    for (int i = 0; i < static_cast<int>(g_editor.scene.materials.size()); ++i) {
        if (i == ignore_index) continue;
        const std::shared_ptr<lt::Material>& material = g_editor.scene.materials[static_cast<size_t>(i)];
        if (material && material->name == name) return true;
    }
    return false;
}

std::string unique_material_name(std::string base, int ignore_index = -1) {
    if (base.empty()) {
        base = "Material";
    }
    if (!material_name_exists(base, ignore_index)) {
        return base;
    }
    for (int suffix = 1; suffix < 100000; ++suffix) {
        const std::string candidate = base + "_" + std::to_string(suffix);
        if (!material_name_exists(candidate, ignore_index)) {
            return candidate;
        }
    }
    return base + "_" + std::to_string(static_cast<int>(g_editor.scene.materials.size()));
}

void add_material() {
    g_editor.scene.materials.push_back(lt::make_material(
        unique_material_name("Material"),
        {0.8f, 0.8f, 0.8f},
        lt::BrdfModel::Lambertian,
        0.5f,
        0.0f));
    reset_accumulation(lt::RenderDirty::Material);
}

bool object_name_exists(const std::string& name, SelectionKind ignore_kind = SelectionKind::None, int ignore_index = -1) {
    for (int i = 0; i < static_cast<int>(g_editor.scene.meshes.size()); ++i) {
        if (ignore_kind == SelectionKind::Mesh && i == ignore_index) continue;
        if (g_editor.scene.meshes[static_cast<size_t>(i)].name == name) return true;
    }
    for (int i = 0; i < static_cast<int>(g_editor.scene.spheres.size()); ++i) {
        if (ignore_kind == SelectionKind::Sphere && i == ignore_index) continue;
        if (g_editor.scene.spheres[static_cast<size_t>(i)].name == name) return true;
    }
    for (int i = 0; i < static_cast<int>(g_editor.scene.directional_lights.size()); ++i) {
        if (ignore_kind == SelectionKind::DirectionalLight && i == ignore_index) continue;
        if (g_editor.scene.directional_lights[static_cast<size_t>(i)].name == name) return true;
    }
    for (int i = 0; i < static_cast<int>(g_editor.scene.point_lights.size()); ++i) {
        if (ignore_kind == SelectionKind::PointLight && i == ignore_index) continue;
        if (g_editor.scene.point_lights[static_cast<size_t>(i)].name == name) return true;
    }
    return false;
}

std::string unique_object_name(std::string base, SelectionKind ignore_kind = SelectionKind::None, int ignore_index = -1) {
    if (base.empty()) {
        base = "Object";
    }
    if (!object_name_exists(base, ignore_kind, ignore_index)) {
        return base;
    }
    for (int suffix = 1; suffix < 100000; ++suffix) {
        const std::string candidate = base + "_" + std::to_string(suffix);
        if (!object_name_exists(candidate, ignore_kind, ignore_index)) {
            return candidate;
        }
    }
    return base + "_" + std::to_string(static_cast<int>(g_editor.scene.meshes.size() + g_editor.scene.spheres.size() + g_editor.scene.directional_lights.size() + g_editor.scene.point_lights.size()));
}

lt::Vec3 spawn_position(float distance = 1.5f) {
    lt::Vec3 forward = lt::normalize(g_editor.scene.camera.target - g_editor.scene.camera.position);
    if (lt::dot(forward, forward) <= 0.0f) {
        forward = {0.0f, 0.0f, -1.0f};
    }
    return g_editor.scene.camera.position + forward * distance;
}

void add_cube_mesh() {
    const int material = default_material("white");
    g_editor.scene.meshes.push_back(lt::make_cube_mesh(unique_object_name("Cube"), material, spawn_position(), 0.35f));
    g_editor.scene.uses_builtin_default_meshes = false;
    invalidate_mesh_bounds_cache();
    select_mesh(static_cast<int>(g_editor.scene.meshes.size()) - 1);
    reset_accumulation(lt::RenderDirty::Geometry | lt::RenderDirty::Material);
}

void add_plane_mesh() {
    const int material = default_material("white");
    lt::Mesh plane = lt::make_quad_mesh(unique_object_name("Plane"), material,
        {-0.5f, 0.0f, -0.5f},
        {0.5f, 0.0f, -0.5f},
        {0.5f, 0.0f, 0.5f},
        {-0.5f, 0.0f, 0.5f});
    plane.translation = spawn_position();
    g_editor.scene.meshes.push_back(std::move(plane));
    g_editor.scene.uses_builtin_default_meshes = false;
    invalidate_mesh_bounds_cache();
    select_mesh(static_cast<int>(g_editor.scene.meshes.size()) - 1);
    reset_accumulation(lt::RenderDirty::Geometry | lt::RenderDirty::Material);
}

void add_uv_sphere_mesh() {
    const int material = default_material("white");
    g_editor.scene.meshes.push_back(lt::make_uv_sphere_mesh(unique_object_name("UV_Sphere"), material, spawn_position(), 0.35f, 32, 16));
    g_editor.scene.uses_builtin_default_meshes = false;
    invalidate_mesh_bounds_cache();
    select_mesh(static_cast<int>(g_editor.scene.meshes.size()) - 1);
    reset_accumulation(lt::RenderDirty::Geometry | lt::RenderDirty::Material);
}

void add_analytic_sphere() {
    lt::Sphere sphere;
    sphere.name = unique_object_name("Sphere");
    sphere.material = default_material("white");
    sphere.center = spawn_position();
    sphere.radius = 0.35f;
    g_editor.scene.spheres.push_back(std::move(sphere));
    g_editor.scene.uses_builtin_default_meshes = false;
    select_sphere(static_cast<int>(g_editor.scene.spheres.size()) - 1);
    reset_accumulation(lt::RenderDirty::Geometry | lt::RenderDirty::Material);
}

void add_area_light_mesh() {
    const int material = default_material("light");
    lt::Mesh light = lt::make_quad_mesh(unique_object_name("Area_Light"), material,
        {-0.35f, 0.0f, -0.35f},
        {0.35f, 0.0f, -0.35f},
        {0.35f, 0.0f, 0.35f},
        {-0.35f, 0.0f, 0.35f});
    light.translation = spawn_position(1.2f);
    light.light = {true, false, {1.0f, 0.888889f, 0.666667f}, 9.0f};
    g_editor.scene.meshes.push_back(std::move(light));
    g_editor.scene.uses_builtin_default_meshes = false;
    invalidate_mesh_bounds_cache();
    select_mesh(static_cast<int>(g_editor.scene.meshes.size()) - 1);
    reset_accumulation(lt::RenderDirty::Geometry | lt::RenderDirty::Material);
}

void add_directional_light() {
    lt::DirectionalLight light;
    light.name = unique_object_name("Directional_Light");
    light.position = spawn_position(1.2f);
    light.direction = lt::normalize(lt::Vec3{0.2f, -1.0f, -0.3f});
    light.color = {1.0f, 0.95f, 0.85f};
    light.intensity = 1.5f;
    g_editor.scene.directional_lights.push_back(std::move(light));
    g_editor.scene.uses_builtin_default_meshes = false;
    select_directional_light(static_cast<int>(g_editor.scene.directional_lights.size()) - 1);
    reset_accumulation(lt::RenderDirty::Geometry);
}

void add_point_light() {
    lt::PointLight light;
    light.name = unique_object_name("Point_Light");
    light.position = spawn_position(1.2f);
    light.color = {1.0f, 0.888889f, 0.666667f};
    light.intensity = 10.0f;
    g_editor.scene.point_lights.push_back(std::move(light));
    g_editor.scene.uses_builtin_default_meshes = false;
    select_point_light(static_cast<int>(g_editor.scene.point_lights.size()) - 1);
    reset_accumulation(lt::RenderDirty::Geometry);
}

void add_mesh() {
    add_cube_mesh();
}

void duplicate_selected() {
    if (!has_selection()) return;
    if (has_mesh_selection()) {
        lt::Mesh copy = g_editor.scene.meshes[static_cast<size_t>(g_editor.selected_mesh)];
        copy.name = unique_object_name(copy.name + "_copy");
        copy.translation.x += std::max({copy.scale.x, copy.scale.y, copy.scale.z}) * 1.2f;
        g_editor.scene.meshes.push_back(copy);
        select_mesh(static_cast<int>(g_editor.scene.meshes.size()) - 1);
    } else if (has_sphere_selection()) {
        lt::Sphere copy = g_editor.scene.spheres[static_cast<size_t>(g_editor.selected_sphere)];
        copy.name = unique_object_name(copy.name + "_copy");
        copy.center.x += copy.radius * 2.2f;
        g_editor.scene.spheres.push_back(copy);
        select_sphere(static_cast<int>(g_editor.scene.spheres.size()) - 1);
    } else if (has_directional_light_selection()) {
        lt::DirectionalLight copy = g_editor.scene.directional_lights[static_cast<size_t>(g_editor.selected_directional_light)];
        copy.name = unique_object_name(copy.name + "_copy");
        copy.position.x += 0.5f;
        g_editor.scene.directional_lights.push_back(copy);
        select_directional_light(static_cast<int>(g_editor.scene.directional_lights.size()) - 1);
    } else if (has_point_light_selection()) {
        lt::PointLight copy = g_editor.scene.point_lights[static_cast<size_t>(g_editor.selected_point_light)];
        copy.name = unique_object_name(copy.name + "_copy");
        copy.position.x += 0.5f;
        g_editor.scene.point_lights.push_back(copy);
        select_point_light(static_cast<int>(g_editor.scene.point_lights.size()) - 1);
    }
    g_editor.scene.uses_builtin_default_meshes = false;
    invalidate_mesh_bounds_cache();
    reset_accumulation(lt::RenderDirty::Geometry);
}

void delete_selected() {
    if (!has_selection()) return;
    if (has_mesh_selection()) {
        const int old_index = g_editor.selected_mesh;
        g_editor.scene.meshes.erase(g_editor.scene.meshes.begin() + old_index);
        if (!g_editor.scene.meshes.empty()) {
            select_mesh(std::min(old_index, static_cast<int>(g_editor.scene.meshes.size()) - 1));
        } else {
            select_sphere(g_editor.scene.spheres.empty() ? -1 : 0);
        }
    } else if (has_sphere_selection()) {
        const int old_index = g_editor.selected_sphere;
        g_editor.scene.spheres.erase(g_editor.scene.spheres.begin() + old_index);
        if (!g_editor.scene.spheres.empty()) {
            select_sphere(std::min(old_index, static_cast<int>(g_editor.scene.spheres.size()) - 1));
        } else {
            select_mesh(g_editor.scene.meshes.empty() ? -1 : 0);
        }
    } else if (has_directional_light_selection()) {
        const int old_index = g_editor.selected_directional_light;
        g_editor.scene.directional_lights.erase(g_editor.scene.directional_lights.begin() + old_index);
        if (!g_editor.scene.directional_lights.empty()) {
            select_directional_light(std::min(old_index, static_cast<int>(g_editor.scene.directional_lights.size()) - 1));
        } else if (!g_editor.scene.point_lights.empty()) {
            select_point_light(0);
        } else {
            select_mesh(g_editor.scene.meshes.empty() ? -1 : 0);
        }
    } else if (has_point_light_selection()) {
        const int old_index = g_editor.selected_point_light;
        g_editor.scene.point_lights.erase(g_editor.scene.point_lights.begin() + old_index);
        if (!g_editor.scene.point_lights.empty()) {
            select_point_light(std::min(old_index, static_cast<int>(g_editor.scene.point_lights.size()) - 1));
        } else if (!g_editor.scene.directional_lights.empty()) {
            select_directional_light(0);
        } else {
            select_mesh(g_editor.scene.meshes.empty() ? -1 : 0);
        }
    }
    g_editor.scene.uses_builtin_default_meshes = false;
    invalidate_mesh_bounds_cache();
    reset_accumulation(lt::RenderDirty::Geometry);
}

void recreate_preview_texture(int width, int height) {
    if (g_preview.width == width && g_preview.height == height && g_preview.texture) return;
    release_preview_texture();
    D3D11_TEXTURE2D_DESC desc{};
    desc.Width = static_cast<UINT>(width);
    desc.Height = static_cast<UINT>(height);
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DYNAMIC;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    if (g_device->CreateTexture2D(&desc, nullptr, &g_preview.texture) == S_OK) {
        g_device->CreateShaderResourceView(g_preview.texture, nullptr, &g_preview.srv);
        g_preview.width = width;
        g_preview.height = height;
    }
}

void upload_preview_texture() {
    if (!g_preview.texture || g_editor.framebuffer.rgba.empty()) return;
    D3D11_MAPPED_SUBRESOURCE mapped{};
    if (g_context->Map(g_preview.texture, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped) != S_OK) return;
    const uint8_t* src = reinterpret_cast<const uint8_t*>(g_editor.framebuffer.rgba.data());
    uint8_t* dst = reinterpret_cast<uint8_t*>(mapped.pData);
    const size_t src_pitch = static_cast<size_t>(g_preview.width) * sizeof(uint32_t);
    for (int y = 0; y < g_preview.height; ++y) {
        std::memcpy(dst + static_cast<size_t>(y) * mapped.RowPitch, src + static_cast<size_t>(y) * src_pitch, src_pitch);
    }
    g_context->Unmap(g_preview.texture, 0);
}

void poll_render_result() {
    if (!g_render_future.valid() || g_render_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return;
    RenderResult result = g_render_future.get();
    const bool current_result = result.generation == g_editor.render_generation;
    const bool compatible_stale_result =
        result.content_generation == g_editor.content_generation &&
        result.framebuffer.width == g_editor.framebuffer.width &&
        result.framebuffer.height == g_editor.framebuffer.height;
    if (!current_result && !compatible_stale_result) return;
    const bool preserve_svgf_stale_history =
        !current_result &&
        compatible_stale_result &&
        lt::svgf_denoising_enabled(g_editor.settings);
    if (current_result || preserve_svgf_stale_history) {
        g_editor.framebuffer = std::move(result.framebuffer);
        g_editor.frame_index = result.completed_frame + 1u;
    } else {
        g_editor.framebuffer.rgba = std::move(result.framebuffer.rgba);
        g_editor.framebuffer.clear_accumulation();
        g_editor.frame_index = 0;
    }
    g_editor.last_sample_ms = result.elapsed_ms;
    upload_preview_texture();
}

void launch_render_task() {
    if (g_editor.paused || scene_load_in_progress() || g_render_future.valid()) return;
    if (g_editor.viewport_preview_mode != ViewportPreviewMode::Rendered) return;
    lt::Scene scene = g_editor.scene;
    lt::RenderSettings settings = g_editor.settings;
    settings.frame_index = g_editor.frame_index;
    settings.dirty = g_editor.dirty;
    if (lt::temporal_jitter_enabled(settings)) {
        const lt::Vec2 jitter = lt::temporal_jitter(settings.frame_index);
        settings.camera_jitter_x = jitter.x;
        settings.camera_jitter_y = jitter.y;
    } else {
        settings.camera_jitter_x = 0.0f;
        settings.camera_jitter_y = 0.0f;
    }
    lt::Framebuffer framebuffer = g_editor.framebuffer;
    lt::IRenderer* renderer = lt::stylized_rendering_enabled(settings, scene)
        ? static_cast<lt::IRenderer*>(&g_editor.cpu)
        : g_editor.renderer;
    lt::RasterizedGBufferInterop gbuffer_interop;
    framebuffer.aov.rasterized = false;
    if (lt::svgf_denoising_enabled(settings) && settings.svgf_rasterized_gbuffer) {
        if (renderer == static_cast<lt::IRenderer*>(&g_editor.cuda)) {
            render_svgf_gbuffer_interop(g_solid_preview, scene, g_editor.viewport_size, settings, gbuffer_interop);
        } else {
            render_svgf_gbuffer(g_solid_preview, scene, g_editor.viewport_size, settings, framebuffer);
        }
    }
    const uint64_t generation = g_editor.render_generation;
    const uint64_t content_generation = g_editor.content_generation;
    copy_setting_text(settings.irradiance_volume_cache_key, g_editor.scene_path);
    copy_setting_text(settings.irradiance_volume_cache_path, default_irradiance_volume_cache_path());
    settings.irradiance_volume_bake_progress = g_editor.irradiance_volume_bake_progress.get();
    copy_setting_text(settings.lightmap_cache_key, g_editor.scene_path);
    copy_setting_text(settings.lightmap_cache_path, default_lightmap_cache_path());
    settings.lightmap_bake_progress = g_editor.lightmap_bake_progress.get();
    const bool consume_force_rebake = settings.irradiance_volume_force_rebake;
    const bool consume_lightmap_force_rebake = settings.lightmap_force_rebake;
    g_editor.dirty = lt::RenderDirty::None;
    g_render_future = std::async(std::launch::async, [
        scene = std::move(scene),
        settings,
        framebuffer = std::move(framebuffer),
        generation,
        content_generation,
        renderer,
        cuda_renderer = &g_editor.cuda,
        gbuffer_interop
    ]() mutable {
        const auto begin = std::chrono::steady_clock::now();
        if (gbuffer_interop.valid && renderer == static_cast<lt::IRenderer*>(cuda_renderer)) {
            cuda_renderer->render_with_rasterized_gbuffer_interop(scene, settings, framebuffer, gbuffer_interop);
        } else {
            renderer->render(scene, settings, framebuffer);
        }
        release_svgf_gbuffer_interop(gbuffer_interop);
        const auto end = std::chrono::steady_clock::now();
        const double elapsed_ms = std::chrono::duration<double, std::milli>(end - begin).count();
        return RenderResult{generation, content_generation, settings.frame_index, elapsed_ms, std::move(framebuffer)};
    });
    if (consume_force_rebake) {
        g_editor.settings.irradiance_volume_force_rebake = false;
    }
    if (consume_lightmap_force_rebake) {
        g_editor.settings.lightmap_force_rebake = false;
    }
}

void prepare_render_preview() {
    const int width = std::max(64, static_cast<int>(g_editor.viewport_size.x));
    const int height = std::max(64, static_cast<int>(g_editor.viewport_size.y));
    if (width != g_editor.settings.width || height != g_editor.settings.height) {
        g_editor.settings.width = width;
        g_editor.settings.height = height;
        g_editor.framebuffer.resize(width, height);
        reset_accumulation();
    }
    recreate_preview_texture(width, height);
    poll_render_result();
}

ViewTransform make_view_transform(bool apply_camera_handedness = true) {
    const float aspect = g_editor.viewport_size.x / std::max(1.0f, g_editor.viewport_size.y);
    const float half_height = std::tan(g_editor.scene.camera.fov_degrees * lt::kPi / 360.0f);
    const float half_width = aspect * half_height;
    const lt::Vec3 forward = lt::normalize(g_editor.scene.camera.target - g_editor.scene.camera.position);
    const float right_sign = apply_camera_handedness && g_editor.scene.camera.right_sign < 0.0f ? -1.0f : 1.0f;
    const lt::Vec3 right = lt::normalize(lt::cross(forward, g_editor.scene.camera.up)) * right_sign;
    const lt::Vec3 up = lt::cross(right, forward) * right_sign;
    return {forward, right, up, half_width, half_height};
}

lt::Ray make_view_ray_from_screen(ImVec2 mouse) {
    const ViewTransform view = make_view_transform(true);
    const ImVec2 min = g_editor.viewport_image_min;
    const ImVec2 size = {std::max(1.0f, g_editor.viewport_image_max.x - min.x), std::max(1.0f, g_editor.viewport_image_max.y - min.y)};
    const float u = ((mouse.x - min.x) / size.x * 2.0f - 1.0f) * view.half_width;
    const float v = (1.0f - (mouse.y - min.y) / size.y * 2.0f) * view.half_height;
    return {g_editor.scene.camera.position, lt::normalize(view.forward + view.right * u + view.up * v)};
}

bool project_point(lt::Vec3 point, ImVec2& screen, float* depth = nullptr) {
    const ViewTransform view = make_view_transform(true);
    const lt::Vec3 to_point = point - g_editor.scene.camera.position;
    const float z = lt::dot(to_point, view.forward);
    if (z <= 0.001f) return false;
    const float u = lt::dot(to_point, view.right) / (z * view.half_width);
    const float v = lt::dot(to_point, view.up) / (z * view.half_height);
    const ImVec2 size = {g_editor.viewport_image_max.x - g_editor.viewport_image_min.x, g_editor.viewport_image_max.y - g_editor.viewport_image_min.y};
    screen = {g_editor.viewport_image_min.x + (u * 0.5f + 0.5f) * size.x, g_editor.viewport_image_min.y + (0.5f - v * 0.5f) * size.y};
    if (depth) *depth = z;
    return true;
}

bool intersect_triangle(const lt::Triangle& tri, const lt::Ray& ray, float& t) {
    const lt::Vec3 e1 = tri.v1 - tri.v0;
    const lt::Vec3 e2 = tri.v2 - tri.v0;
    const lt::Vec3 p = lt::cross(ray.direction, e2);
    const float det = lt::dot(e1, p);
    if (std::fabs(det) < 1.0e-7f) return false;
    const float inv_det = 1.0f / det;
    const lt::Vec3 s = ray.origin - tri.v0;
    const float u = inv_det * lt::dot(s, p);
    if (u < 0.0f || u > 1.0f) return false;
    const lt::Vec3 q = lt::cross(s, e1);
    const float v = inv_det * lt::dot(ray.direction, q);
    if (v < 0.0f || u + v > 1.0f) return false;
    t = inv_det * lt::dot(e2, q);
    return t > 0.001f;
}

bool intersect_sphere_pick(const lt::RenderSphere& sphere, const lt::Ray& ray, float& t) {
    const lt::Vec3 oc = ray.origin - sphere.center;
    const float a = lt::dot(ray.direction, ray.direction);
    const float half_b = lt::dot(oc, ray.direction);
    const float c = lt::dot(oc, oc) - sphere.radius * sphere.radius;
    const float discriminant = half_b * half_b - a * c;
    if (discriminant < 0.0f) {
        return false;
    }
    const float root = std::sqrt(discriminant);
    t = (-half_b - root) / a;
    if (t <= 0.001f) {
        t = (-half_b + root) / a;
    }
    return t > 0.001f;
}

bool pick_object_cpu(ImVec2 mouse, SelectionKind& kind, int& object_index) {
    const lt::Ray ray = make_view_ray_from_screen(mouse);
    if (g_pick_cache.scene_generation != g_editor.render_generation) {
        g_pick_cache.render_scene = lt::build_render_scene(g_editor.scene);
        g_pick_cache.scene_generation = g_editor.render_generation;
    }
    float best_t = lt::kInfinity;
    kind = SelectionKind::None;
    object_index = -1;
    for (const lt::Triangle& tri : g_pick_cache.render_scene.triangles) {
        float t = 0.0f;
        if (intersect_triangle(tri, ray, t) && t < best_t) {
            best_t = t;
            kind = SelectionKind::Mesh;
            object_index = tri.mesh;
        }
    }
    for (const lt::RenderSphere& sphere : g_pick_cache.render_scene.spheres) {
        float t = 0.0f;
        if (intersect_sphere_pick(sphere, ray, t) && t < best_t) {
            best_t = t;
            kind = SelectionKind::Sphere;
            object_index = sphere.sphere;
        }
    }
    return kind != SelectionKind::None && object_index >= 0;
}

bool pick_object(ImVec2 mouse, SelectionKind& kind, int& object_index) {
    const GpuPickResult gpu_pick = pick_solid_preview_object(
        g_solid_preview,
        g_editor.scene,
        g_editor.viewport_size,
        g_editor.viewport_image_min,
        g_editor.viewport_image_max,
        mouse,
        kind,
        object_index);
    if (gpu_pick == GpuPickResult::Hit) {
        return true;
    }
    if (gpu_pick == GpuPickResult::Miss) {
        return false;
    }
    return pick_object_cpu(mouse, kind, object_index);
}

void mesh_center_radius(const lt::Mesh& mesh, lt::Vec3& center, float& radius) {
    center = mesh.translation;
    if (mesh.vertices.empty()) {
        radius = std::max({mesh.scale.x, mesh.scale.y, mesh.scale.z});
        return;
    }
    radius = 0.0f;
    for (lt::Vec3 v : mesh.vertices) {
        radius = std::max(radius, lt::length(v * mesh.scale));
    }
    radius = std::max(radius, 0.05f);
}

lt::Vec3 rotate_xyz_editor(lt::Vec3 p, lt::Vec3 r) {
    const float cx = std::cos(r.x), sx = std::sin(r.x);
    const float cy = std::cos(r.y), sy = std::sin(r.y);
    const float cz = std::cos(r.z), sz = std::sin(r.z);
    p = {p.x, p.y * cx - p.z * sx, p.y * sx + p.z * cx};
    p = {p.x * cy + p.z * sy, p.y, -p.x * sy + p.z * cy};
    p = {p.x * cz - p.y * sz, p.x * sz + p.y * cz, p.z};
    return p;
}

lt::Vec3 transform_mesh_point(const lt::Mesh& mesh, lt::Vec3 p) {
    return mesh.translation + rotate_xyz_editor(p * mesh.scale, mesh.rotation);
}

void ensure_mesh_bounds_cache() {
    if (g_bounds_cache.scene_generation == g_editor.render_generation &&
        g_bounds_cache.local_min.size() == g_editor.scene.meshes.size() &&
        g_bounds_cache.local_max.size() == g_editor.scene.meshes.size()) {
        return;
    }
    g_bounds_cache.local_min.resize(g_editor.scene.meshes.size());
    g_bounds_cache.local_max.resize(g_editor.scene.meshes.size());
    for (size_t i = 0; i < g_editor.scene.meshes.size(); ++i) {
        const lt::Mesh& mesh = g_editor.scene.meshes[i];
        lt::Vec3 local_min{lt::kInfinity, lt::kInfinity, lt::kInfinity};
        lt::Vec3 local_max{-lt::kInfinity, -lt::kInfinity, -lt::kInfinity};
        for (lt::Vec3 v : mesh.vertices) {
            local_min = lt::min(local_min, v);
            local_max = lt::max(local_max, v);
        }
        if (mesh.vertices.empty()) {
            local_min = {-0.5f, -0.5f, -0.5f};
            local_max = {0.5f, 0.5f, 0.5f};
        }
        g_bounds_cache.local_min[i] = local_min;
        g_bounds_cache.local_max[i] = local_max;
    }
    g_bounds_cache.scene_generation = g_editor.render_generation;
}

void move_camera(float right_delta, float up_delta, float forward_delta) {
    lt::Camera& camera = g_editor.scene.camera;
    const ViewTransform view = make_view_transform(true);
    const lt::Vec3 delta = view.right * right_delta + view.up * up_delta + view.forward * forward_delta;
    camera.position += delta;
    camera.target += delta;
}

void rotate_camera(float yaw, float pitch) {
    lt::Camera& camera = g_editor.scene.camera;
    const float rotate_sign = camera.right_sign < 0.0f ? -1.0f : 1.0f;
    lt::Vec3 dir = lt::normalize(camera.target - camera.position);
    const float radius = lt::length(camera.target - camera.position);
    float current_yaw = std::atan2(dir.x, dir.z);
    float current_pitch = std::asin(std::clamp(dir.y, -0.99f, 0.99f));
    current_yaw += yaw * rotate_sign;
    current_pitch = std::clamp(current_pitch + pitch, -1.45f, 1.45f);
    camera.target = camera.position + lt::Vec3{std::sin(current_yaw) * std::cos(current_pitch), std::sin(current_pitch), std::cos(current_yaw) * std::cos(current_pitch)} * radius;
}

bool move_selected_object(lt::Vec3 delta) {
    if (has_mesh_selection()) {
        lt::Mesh& mesh = g_editor.scene.meshes[static_cast<size_t>(g_editor.selected_mesh)];
        mesh.translation += delta;
        g_editor.scene.uses_builtin_default_meshes = false;
        return true;
    }
    if (has_sphere_selection()) {
        lt::Sphere& sphere = g_editor.scene.spheres[static_cast<size_t>(g_editor.selected_sphere)];
        sphere.center += delta;
        g_editor.scene.uses_builtin_default_meshes = false;
        return true;
    }
    if (has_point_light_selection()) {
        lt::PointLight& light = g_editor.scene.point_lights[static_cast<size_t>(g_editor.selected_point_light)];
        light.position += delta;
        g_editor.scene.uses_builtin_default_meshes = false;
        return true;
    }
    if (has_directional_light_selection()) {
        lt::DirectionalLight& light = g_editor.scene.directional_lights[static_cast<size_t>(g_editor.selected_directional_light)];
        light.position += delta;
        g_editor.scene.uses_builtin_default_meshes = false;
        return true;
    }
    return false;
}

lt::Vec3 handle_axis(GizmoHandle handle) {
    switch (handle) {
    case GizmoHandle::AxisX: return {1.0f, 0.0f, 0.0f};
    case GizmoHandle::AxisY: return {0.0f, 1.0f, 0.0f};
    case GizmoHandle::AxisZ: return {0.0f, 0.0f, 1.0f};
    default: return {};
    }
}

lt::Vec3 local_axis(const lt::Mesh& mesh, lt::Vec3 axis) {
    return lt::normalize(rotate_xyz_editor(axis, mesh.rotation));
}

lt::Vec3 transform_axis(const lt::Mesh& mesh, lt::Vec3 axis) {
    if (g_editor.tool_mode == ToolMode::Move && g_editor.move_space == TransformSpace::World) {
        return axis;
    }
    return local_axis(mesh, axis);
}

lt::Vec3 handle_axis(const lt::Mesh& mesh, GizmoHandle handle) {
    return transform_axis(mesh, handle_axis(handle));
}

void handle_plane_axes(const lt::Mesh& mesh, GizmoHandle handle, lt::Vec3& a, lt::Vec3& b) {
    if (handle == GizmoHandle::PlaneXY) {
        a = transform_axis(mesh, {1.0f, 0.0f, 0.0f});
        b = transform_axis(mesh, {0.0f, 1.0f, 0.0f});
    } else if (handle == GizmoHandle::PlaneYZ) {
        a = transform_axis(mesh, {0.0f, 1.0f, 0.0f});
        b = transform_axis(mesh, {0.0f, 0.0f, 1.0f});
    } else {
        a = transform_axis(mesh, {0.0f, 0.0f, 1.0f});
        b = transform_axis(mesh, {1.0f, 0.0f, 0.0f});
    }
}

void handle_plane_axes(GizmoHandle handle, lt::Vec3& a, lt::Vec3& b) {
    if (handle == GizmoHandle::PlaneXY) {
        a = {1.0f, 0.0f, 0.0f};
        b = {0.0f, 1.0f, 0.0f};
    } else if (handle == GizmoHandle::PlaneYZ) {
        a = {0.0f, 1.0f, 0.0f};
        b = {0.0f, 0.0f, 1.0f};
    } else {
        a = {0.0f, 0.0f, 1.0f};
        b = {1.0f, 0.0f, 0.0f};
    }
}

float distance_to_segment(ImVec2 p, ImVec2 a, ImVec2 b) {
    const ImVec2 ab{b.x - a.x, b.y - a.y};
    const ImVec2 ap{p.x - a.x, p.y - a.y};
    const float ab2 = ab.x * ab.x + ab.y * ab.y;
    const float t = ab2 > 0.0f ? std::clamp((ap.x * ab.x + ap.y * ab.y) / ab2, 0.0f, 1.0f) : 0.0f;
    const ImVec2 q{a.x + ab.x * t, a.y + ab.y * t};
    const float dx = p.x - q.x;
    const float dy = p.y - q.y;
    return std::sqrt(dx * dx + dy * dy);
}

float distance(ImVec2 a, ImVec2 b) {
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

bool point_in_triangle(ImVec2 p, ImVec2 a, ImVec2 b, ImVec2 c) {
    const auto sign = [](ImVec2 p1, ImVec2 p2, ImVec2 p3) {
        return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
    };
    const float d1 = sign(p, a, b);
    const float d2 = sign(p, b, c);
    const float d3 = sign(p, c, a);
    const bool has_neg = d1 < 0.0f || d2 < 0.0f || d3 < 0.0f;
    const bool has_pos = d1 > 0.0f || d2 > 0.0f || d3 > 0.0f;
    return !(has_neg && has_pos);
}

float world_per_pixel_at(lt::Vec3 world_point) {
    const lt::Vec3 to_point = world_point - g_editor.scene.camera.position;
    const lt::Vec3 forward = lt::normalize(g_editor.scene.camera.target - g_editor.scene.camera.position);
    const float depth = std::max(0.1f, lt::dot(to_point, forward));
    const float half_height = std::tan(g_editor.scene.camera.fov_degrees * lt::kPi / 360.0f);
    return depth * half_height * 2.0f / std::max(1.0f, g_editor.viewport_size.y);
}

GizmoHandle pick_move_gizmo_axes(lt::Vec3 center3, lt::Vec3 ax, lt::Vec3 ay, lt::Vec3 az, ImVec2 mouse, bool allow_uniform) {
    ImVec2 c{}, x{}, y{}, z{}, xy{}, yz{}, zx{};
    if (!project_point(center3, c)) return GizmoHandle::None;
    if (allow_uniform && distance(mouse, c) <= 10.0f) {
        return GizmoHandle::Uniform;
    }

    const float ps = world_per_pixel_at(center3);
    const float arrow_len = 80.0f * ps;
    const float plane_ofs = 40.0f * ps;

    project_point(center3 + ax * arrow_len, x);
    project_point(center3 + ay * arrow_len, y);
    project_point(center3 + az * arrow_len, z);
    project_point(center3 + (ax + ay) * plane_ofs, xy);
    project_point(center3 + (ay + az) * plane_ofs, yz);
    project_point(center3 + (az + ax) * plane_ofs, zx);
    float best = 12.0f;
    GizmoHandle result = GizmoHandle::None;
    const float dx = distance_to_segment(mouse, c, x);
    if (dx < best) { best = dx; result = GizmoHandle::AxisX; }
    const float dy = distance_to_segment(mouse, c, y);
    if (dy < best) { best = dy; result = GizmoHandle::AxisY; }
    const float dz = distance_to_segment(mouse, c, z);
    if (dz < best) { best = dz; result = GizmoHandle::AxisZ; }
    if (result != GizmoHandle::None) return result;
    if (point_in_triangle(mouse, c, x, xy) || point_in_triangle(mouse, c, y, xy)) return GizmoHandle::PlaneXY;
    if (point_in_triangle(mouse, c, y, yz) || point_in_triangle(mouse, c, z, yz)) return GizmoHandle::PlaneYZ;
    if (point_in_triangle(mouse, c, z, zx) || point_in_triangle(mouse, c, x, zx)) return GizmoHandle::PlaneZX;
    return result;
}

GizmoHandle pick_move_gizmo(const lt::Mesh& mesh, lt::Vec3 center3, ImVec2 mouse, bool allow_uniform) {
    return pick_move_gizmo_axes(
        center3,
        transform_axis(mesh, {1.0f, 0.0f, 0.0f}),
        transform_axis(mesh, {0.0f, 1.0f, 0.0f}),
        transform_axis(mesh, {0.0f, 0.0f, 1.0f}),
        mouse,
        allow_uniform);
}

GizmoHandle pick_sphere_move_gizmo(lt::Vec3 center3, ImVec2 mouse) {
    return pick_move_gizmo_axes(
        center3,
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f},
        mouse,
        false);
}

GizmoHandle pick_rotate_gizmo(const lt::Mesh& mesh, lt::Vec3 center3, ImVec2 mouse) {
    ImVec2 c{}, px{}, py{}, pz{};
    if (!project_point(center3, c)) return GizmoHandle::None;
    const float ps = world_per_pixel_at(center3);
    const float rot_r = 60.0f * ps;
    project_point(center3 + local_axis(mesh, {1.0f, 0.0f, 0.0f}) * rot_r, px);
    project_point(center3 + local_axis(mesh, {0.0f, 1.0f, 0.0f}) * rot_r, py);
    project_point(center3 + local_axis(mesh, {0.0f, 0.0f, 1.0f}) * rot_r, pz);
    const ImVec2 ax{px.x - c.x, px.y - c.y};
    const ImVec2 ay{py.x - c.x, py.y - c.y};
    const ImVec2 az{pz.x - c.x, pz.y - c.y};
    const auto ellipse_error = [&](ImVec2 a, ImVec2 b) {
        const float det = a.x * b.y - a.y * b.x;
        if (std::fabs(det) < 1.0e-5f) return 1.0e9f;
        const ImVec2 p{mouse.x - c.x, mouse.y - c.y};
        const float u = (p.x * b.y - p.y * b.x) / det;
        const float v = (a.x * p.y - a.y * p.x) / det;
        return std::fabs(std::sqrt(u * u + v * v) - 1.0f);
    };
    float best = 0.28f;
    GizmoHandle result = GizmoHandle::None;
    const float ex = ellipse_error(ay, az);
    if (ex < best) { best = ex; result = GizmoHandle::AxisX; }
    const float ey = ellipse_error(az, ax);
    if (ey < best) { best = ey; result = GizmoHandle::AxisY; }
    const float ez = ellipse_error(ax, ay);
    if (ez < best) { best = ez; result = GizmoHandle::AxisZ; }
    return result;
}

float ellipse_angle(ImVec2 mouse, ImVec2 center, ImVec2 axis_a, ImVec2 axis_b) {
    const float det = axis_a.x * axis_b.y - axis_a.y * axis_b.x;
    if (std::fabs(det) < 1.0e-5f) {
        return std::atan2(mouse.y - center.y, mouse.x - center.x);
    }
    const ImVec2 p{mouse.x - center.x, mouse.y - center.y};
    const float u = (p.x * axis_b.y - p.y * axis_b.x) / det;
    const float v = (axis_a.x * p.y - axis_a.y * p.x) / det;
    return std::atan2(v, u);
}

void setup_drag_projection(const lt::Mesh& mesh, lt::Vec3 center3, GizmoHandle handle) {
    project_point(center3, g_editor.drag_start_center_screen, &g_editor.drag_start_depth);
    if (g_editor.tool_mode == ToolMode::Rotate) {
        ImVec2 px{}, py{}, pz{};
        project_point(center3 + local_axis(mesh, {1.0f, 0.0f, 0.0f}) * 0.7f, px);
        project_point(center3 + local_axis(mesh, {0.0f, 1.0f, 0.0f}) * 0.7f, py);
        project_point(center3 + local_axis(mesh, {0.0f, 0.0f, 1.0f}) * 0.7f, pz);
        const ImVec2 ax{px.x - g_editor.drag_start_center_screen.x, px.y - g_editor.drag_start_center_screen.y};
        const ImVec2 ay{py.x - g_editor.drag_start_center_screen.x, py.y - g_editor.drag_start_center_screen.y};
        const ImVec2 az{pz.x - g_editor.drag_start_center_screen.x, pz.y - g_editor.drag_start_center_screen.y};
        if (handle == GizmoHandle::AxisX) {
            g_editor.drag_start_axis_a_screen = {g_editor.drag_start_center_screen.x + ay.x, g_editor.drag_start_center_screen.y + ay.y};
            g_editor.drag_start_axis_b_screen = {g_editor.drag_start_center_screen.x + az.x, g_editor.drag_start_center_screen.y + az.y};
        } else if (handle == GizmoHandle::AxisY) {
            g_editor.drag_start_axis_a_screen = {g_editor.drag_start_center_screen.x + az.x, g_editor.drag_start_center_screen.y + az.y};
            g_editor.drag_start_axis_b_screen = {g_editor.drag_start_center_screen.x + ax.x, g_editor.drag_start_center_screen.y + ax.y};
        } else {
            g_editor.drag_start_axis_a_screen = {g_editor.drag_start_center_screen.x + ax.x, g_editor.drag_start_center_screen.y + ax.y};
            g_editor.drag_start_axis_b_screen = {g_editor.drag_start_center_screen.x + ay.x, g_editor.drag_start_center_screen.y + ay.y};
        }
    } else if (handle == GizmoHandle::AxisX || handle == GizmoHandle::AxisY || handle == GizmoHandle::AxisZ) {
        const lt::Vec3 axis = handle_axis(mesh, handle);
        project_point(center3 + axis, g_editor.drag_start_axis_a_screen);
        g_editor.drag_start_axis_b_screen = g_editor.drag_start_center_screen;
    } else if (handle == GizmoHandle::Uniform) {
        g_editor.drag_start_axis_a_screen = g_editor.drag_start_center_screen;
        g_editor.drag_start_axis_b_screen = g_editor.drag_start_center_screen;
    } else {
        lt::Vec3 a{}, b{};
        handle_plane_axes(mesh, handle, a, b);
        project_point(center3 + a, g_editor.drag_start_axis_a_screen);
        project_point(center3 + b, g_editor.drag_start_axis_b_screen);
    }
    const ImVec2 axis_a{
        g_editor.drag_start_axis_a_screen.x - g_editor.drag_start_center_screen.x,
        g_editor.drag_start_axis_a_screen.y - g_editor.drag_start_center_screen.y,
    };
    const ImVec2 axis_b{
        g_editor.drag_start_axis_b_screen.x - g_editor.drag_start_center_screen.x,
        g_editor.drag_start_axis_b_screen.y - g_editor.drag_start_center_screen.y,
    };
    g_editor.drag_start_angle = ellipse_angle(g_editor.drag_start_mouse, g_editor.drag_start_center_screen, axis_a, axis_b);
}

void setup_sphere_drag_projection(lt::Vec3 center3, GizmoHandle handle) {
    project_point(center3, g_editor.drag_start_center_screen, &g_editor.drag_start_depth);
    if (handle == GizmoHandle::AxisX || handle == GizmoHandle::AxisY || handle == GizmoHandle::AxisZ) {
        project_point(center3 + handle_axis(handle), g_editor.drag_start_axis_a_screen);
        g_editor.drag_start_axis_b_screen = g_editor.drag_start_center_screen;
        return;
    }

    lt::Vec3 a{}, b{};
    handle_plane_axes(handle, a, b);
    project_point(center3 + a, g_editor.drag_start_axis_a_screen);
    project_point(center3 + b, g_editor.drag_start_axis_b_screen);
}

void handle_sphere_gizmo_drag() {
    if (g_editor.tool_mode != ToolMode::Move) {
        g_editor.hovered_gizmo = GizmoHandle::None;
        if (g_editor.drag_start_sphere_index >= 0) {
            g_editor.gizmo_dragging = false;
            g_editor.active_gizmo = GizmoHandle::None;
            g_editor.drag_start_sphere_index = -1;
        }
        return;
    }

    lt::Sphere& sphere = g_editor.scene.spheres[static_cast<size_t>(g_editor.selected_sphere)];
    ImVec2 center{};
    float depth = 1.0f;
    if (!project_point(sphere.center, center, &depth)) {
        g_editor.hovered_gizmo = GizmoHandle::None;
        return;
    }

    const ImVec2 mouse = ImGui::GetIO().MousePos;
    const GizmoHandle picked = pick_sphere_move_gizmo(sphere.center, mouse);
    g_editor.hovered_gizmo = picked;
    if (g_editor.viewport_hovered && picked != GizmoHandle::None && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        g_editor.gizmo_dragging = true;
        g_editor.drag_start_mouse = mouse;
        g_editor.drag_start_mesh_index = -1;
        g_editor.drag_start_sphere = sphere;
        g_editor.drag_start_sphere_index = g_editor.selected_sphere;
        g_editor.active_gizmo = picked;
        setup_sphere_drag_projection(sphere.center, picked);
    }
    if (!g_editor.gizmo_dragging) return;
    if (!ImGui::IsMouseDown(ImGuiMouseButton_Left) ||
        g_editor.selection_kind != SelectionKind::Sphere ||
        g_editor.drag_start_sphere_index != g_editor.selected_sphere) {
        g_editor.gizmo_dragging = false;
        g_editor.active_gizmo = GizmoHandle::None;
        g_editor.drag_start_sphere_index = -1;
        return;
    }

    const ImVec2 delta{mouse.x - g_editor.drag_start_mouse.x, mouse.y - g_editor.drag_start_mouse.y};
    const ViewTransform view = make_view_transform(true);
    const float world_per_pixel = g_editor.drag_start_depth * view.half_height * 2.0f / std::max(1.0f, g_editor.viewport_size.y);
    if (g_editor.active_gizmo == GizmoHandle::AxisX || g_editor.active_gizmo == GizmoHandle::AxisY || g_editor.active_gizmo == GizmoHandle::AxisZ) {
        const lt::Vec3 axis = handle_axis(g_editor.active_gizmo);
        const ImVec2 screen_axis{
            g_editor.drag_start_axis_a_screen.x - g_editor.drag_start_center_screen.x,
            g_editor.drag_start_axis_a_screen.y - g_editor.drag_start_center_screen.y,
        };
        const float len = std::sqrt(screen_axis.x * screen_axis.x + screen_axis.y * screen_axis.y);
        const float amount = len > 0.0f ? (delta.x * screen_axis.x + delta.y * screen_axis.y) / len * world_per_pixel : 0.0f;
        sphere.center = g_editor.drag_start_sphere.center + axis * amount;
    } else {
        lt::Vec3 a{}, b{};
        handle_plane_axes(g_editor.active_gizmo, a, b);
        const ImVec2 sa{
            g_editor.drag_start_axis_a_screen.x - g_editor.drag_start_center_screen.x,
            g_editor.drag_start_axis_a_screen.y - g_editor.drag_start_center_screen.y,
        };
        const ImVec2 sb{
            g_editor.drag_start_axis_b_screen.x - g_editor.drag_start_center_screen.x,
            g_editor.drag_start_axis_b_screen.y - g_editor.drag_start_center_screen.y,
        };
        const float det = sa.x * sb.y - sa.y * sb.x;
        if (std::fabs(det) > 80.0f) {
            const float amount_a = (delta.x * sb.y - delta.y * sb.x) / det;
            const float amount_b = (sa.x * delta.y - sa.y * delta.x) / det;
            sphere.center = g_editor.drag_start_sphere.center + a * amount_a + b * amount_b;
        } else {
            const ImVec2 combined{sa.x + sb.x, sa.y + sb.y};
            const float len = std::sqrt(combined.x * combined.x + combined.y * combined.y);
            const float amount = len > 1.0f ? (delta.x * combined.x + delta.y * combined.y) / len * world_per_pixel : 0.0f;
            sphere.center = g_editor.drag_start_sphere.center + lt::normalize(a + b) * amount;
        }
    }
    g_editor.scene.uses_builtin_default_meshes = false;
    reset_accumulation(lt::RenderDirty::Transform);
}

void handle_point_light_gizmo_drag() {
    if (g_editor.tool_mode != ToolMode::Move) {
        g_editor.hovered_gizmo = GizmoHandle::None;
        if (g_editor.drag_start_point_light_index >= 0) {
            g_editor.gizmo_dragging = false;
            g_editor.active_gizmo = GizmoHandle::None;
            g_editor.drag_start_point_light_index = -1;
        }
        return;
    }

    lt::PointLight& light = g_editor.scene.point_lights[static_cast<size_t>(g_editor.selected_point_light)];
    ImVec2 center{};
    float depth = 1.0f;
    if (!project_point(light.position, center, &depth)) {
        g_editor.hovered_gizmo = GizmoHandle::None;
        return;
    }

    const ImVec2 mouse = ImGui::GetIO().MousePos;
    const GizmoHandle picked = pick_sphere_move_gizmo(light.position, mouse);
    g_editor.hovered_gizmo = picked;
    if (g_editor.viewport_hovered && picked != GizmoHandle::None && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        g_editor.gizmo_dragging = true;
        g_editor.drag_start_mouse = mouse;
        g_editor.drag_start_mesh_index = -1;
        g_editor.drag_start_sphere_index = -1;
        g_editor.drag_start_point_light = light;
        g_editor.drag_start_point_light_index = g_editor.selected_point_light;
        g_editor.active_gizmo = picked;
        setup_sphere_drag_projection(light.position, picked);
    }
    if (!g_editor.gizmo_dragging) return;
    if (!ImGui::IsMouseDown(ImGuiMouseButton_Left) ||
        g_editor.selection_kind != SelectionKind::PointLight ||
        g_editor.drag_start_point_light_index != g_editor.selected_point_light) {
        g_editor.gizmo_dragging = false;
        g_editor.active_gizmo = GizmoHandle::None;
        g_editor.drag_start_point_light_index = -1;
        return;
    }

    const ImVec2 delta{mouse.x - g_editor.drag_start_mouse.x, mouse.y - g_editor.drag_start_mouse.y};
    const ViewTransform view = make_view_transform(true);
    const float world_per_pixel = g_editor.drag_start_depth * view.half_height * 2.0f / std::max(1.0f, g_editor.viewport_size.y);
    if (g_editor.active_gizmo == GizmoHandle::AxisX || g_editor.active_gizmo == GizmoHandle::AxisY || g_editor.active_gizmo == GizmoHandle::AxisZ) {
        const lt::Vec3 axis = handle_axis(g_editor.active_gizmo);
        const ImVec2 screen_axis{
            g_editor.drag_start_axis_a_screen.x - g_editor.drag_start_center_screen.x,
            g_editor.drag_start_axis_a_screen.y - g_editor.drag_start_center_screen.y,
        };
        const float len = std::sqrt(screen_axis.x * screen_axis.x + screen_axis.y * screen_axis.y);
        const float amount = len > 0.0f ? (delta.x * screen_axis.x + delta.y * screen_axis.y) / len * world_per_pixel : 0.0f;
        light.position = g_editor.drag_start_point_light.position + axis * amount;
    } else {
        lt::Vec3 a{}, b{};
        handle_plane_axes(g_editor.active_gizmo, a, b);
        const ImVec2 sa{
            g_editor.drag_start_axis_a_screen.x - g_editor.drag_start_center_screen.x,
            g_editor.drag_start_axis_a_screen.y - g_editor.drag_start_center_screen.y,
        };
        const ImVec2 sb{
            g_editor.drag_start_axis_b_screen.x - g_editor.drag_start_center_screen.x,
            g_editor.drag_start_axis_b_screen.y - g_editor.drag_start_center_screen.y,
        };
        const float det = sa.x * sb.y - sa.y * sb.x;
        if (std::fabs(det) > 80.0f) {
            const float amount_a = (delta.x * sb.y - delta.y * sb.x) / det;
            const float amount_b = (sa.x * delta.y - sa.y * delta.x) / det;
            light.position = g_editor.drag_start_point_light.position + a * amount_a + b * amount_b;
        } else {
            const ImVec2 combined{sa.x + sb.x, sa.y + sb.y};
            const float len = std::sqrt(combined.x * combined.x + combined.y * combined.y);
            const float amount = len > 1.0f ? (delta.x * combined.x + delta.y * combined.y) / len * world_per_pixel : 0.0f;
            light.position = g_editor.drag_start_point_light.position + lt::normalize(a + b) * amount;
        }
    }
    g_editor.scene.uses_builtin_default_meshes = false;
    reset_accumulation(lt::RenderDirty::Transform);
}

void handle_directional_light_gizmo_drag() {
    lt::DirectionalLight& light = g_editor.scene.directional_lights[static_cast<size_t>(g_editor.selected_directional_light)];

    if (g_editor.tool_mode == ToolMode::Move) {
        // Use the same move gizmo logic as sphere
        ImVec2 center{};
        float depth = 1.0f;
        if (!project_point(light.position, center, &depth)) {
            g_editor.hovered_gizmo = GizmoHandle::None;
            return;
        }
        const ImVec2 mouse = ImGui::GetIO().MousePos;
        const GizmoHandle picked = pick_move_gizmo_axes(
            light.position,
            {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f},
            mouse, false);
        g_editor.hovered_gizmo = picked;
        if (g_editor.viewport_hovered && picked != GizmoHandle::None && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            g_editor.gizmo_dragging = true;
            g_editor.drag_start_mouse = mouse;
            g_editor.drag_start_mesh_index = -1;
            g_editor.drag_start_sphere_index = -1;
            g_editor.drag_start_point_light_index = -1;
            g_editor.drag_start_directional_light = light;
            g_editor.drag_start_directional_light_index = g_editor.selected_directional_light;
            g_editor.active_gizmo = picked;
            setup_sphere_drag_projection(light.position, picked);
        }
        if (!g_editor.gizmo_dragging) return;
        if (!ImGui::IsMouseDown(ImGuiMouseButton_Left) ||
            g_editor.selection_kind != SelectionKind::DirectionalLight ||
            g_editor.drag_start_directional_light_index != g_editor.selected_directional_light) {
            g_editor.gizmo_dragging = false;
            g_editor.active_gizmo = GizmoHandle::None;
            g_editor.drag_start_directional_light_index = -1;
            return;
        }
        const ImVec2 delta{mouse.x - g_editor.drag_start_mouse.x, mouse.y - g_editor.drag_start_mouse.y};
        const ViewTransform view = make_view_transform(true);
        const float world_per_pixel = g_editor.drag_start_depth * view.half_height * 2.0f / std::max(1.0f, g_editor.viewport_size.y);
        if (g_editor.active_gizmo == GizmoHandle::AxisX || g_editor.active_gizmo == GizmoHandle::AxisY || g_editor.active_gizmo == GizmoHandle::AxisZ) {
            const lt::Vec3 axis = handle_axis(g_editor.active_gizmo);
            const ImVec2 screen_axis{
                g_editor.drag_start_axis_a_screen.x - g_editor.drag_start_center_screen.x,
                g_editor.drag_start_axis_a_screen.y - g_editor.drag_start_center_screen.y,
            };
            const float len = std::sqrt(screen_axis.x * screen_axis.x + screen_axis.y * screen_axis.y);
            const float amount = len > 0.0f ? (delta.x * screen_axis.x + delta.y * screen_axis.y) / len * world_per_pixel : 0.0f;
            light.position = g_editor.drag_start_directional_light.position + axis * amount;
        } else {
            lt::Vec3 a{}, b{};
            handle_plane_axes(g_editor.active_gizmo, a, b);
            const ImVec2 sa{
                g_editor.drag_start_axis_a_screen.x - g_editor.drag_start_center_screen.x,
                g_editor.drag_start_axis_a_screen.y - g_editor.drag_start_center_screen.y,
            };
            const ImVec2 sb{
                g_editor.drag_start_axis_b_screen.x - g_editor.drag_start_center_screen.x,
                g_editor.drag_start_axis_b_screen.y - g_editor.drag_start_center_screen.y,
            };
            const float det = sa.x * sb.y - sa.y * sb.x;
            if (std::fabs(det) > 80.0f) {
                const float amount_a = (delta.x * sb.y - delta.y * sb.x) / det;
                const float amount_b = (sa.x * delta.y - sa.y * delta.x) / det;
                light.position = g_editor.drag_start_directional_light.position + a * amount_a + b * amount_b;
            } else {
                const ImVec2 combined{sa.x + sb.x, sa.y + sb.y};
                const float len = std::sqrt(combined.x * combined.x + combined.y * combined.y);
                const float amount = len > 1.0f ? (delta.x * combined.x + delta.y * combined.y) / len * world_per_pixel : 0.0f;
                light.position = g_editor.drag_start_directional_light.position + lt::normalize(a + b) * amount;
            }
        }
        g_editor.scene.uses_builtin_default_meshes = false;
        reset_accumulation(lt::RenderDirty::Transform);

    } else if (g_editor.tool_mode == ToolMode::Rotate) {
        // Use the same rotate gizmo pick logic as mesh (world axes, no local transform)
        const ImVec2 mouse = ImGui::GetIO().MousePos;
        ImVec2 c{}, px{}, py{}, pz{};
        if (!project_point(light.position, c)) { g_editor.hovered_gizmo = GizmoHandle::None; return; }
        const float ps = world_per_pixel_at(light.position);
        const float rot_r = 60.0f * ps;
        project_point(light.position + lt::Vec3{1.0f, 0.0f, 0.0f} * rot_r, px);
        project_point(light.position + lt::Vec3{0.0f, 1.0f, 0.0f} * rot_r, py);
        project_point(light.position + lt::Vec3{0.0f, 0.0f, 1.0f} * rot_r, pz);
        const ImVec2 ax{px.x - c.x, px.y - c.y};
        const ImVec2 ay{py.x - c.x, py.y - c.y};
        const ImVec2 az{pz.x - c.x, pz.y - c.y};

        // Pick nearest ring (same logic as pick_rotate_gizmo for mesh)
        const auto ellipse_err = [&](ImVec2 a, ImVec2 b) {
            const float det = a.x * b.y - a.y * b.x;
            if (std::fabs(det) < 1.0e-5f) return 1.0e9f;
            const ImVec2 p{mouse.x - c.x, mouse.y - c.y};
            const float u = (p.x * b.y - p.y * b.x) / det;
            const float v = (a.x * p.y - a.y * p.x) / det;
            return std::fabs(std::sqrt(u * u + v * v) - 1.0f);
        };
        float best = 0.28f;
        GizmoHandle picked = GizmoHandle::None;
        int pick_axis = -1; // 0=X, 1=Y, 2=Z
        const float ex = ellipse_err(ay, az);
        if (ex < best) { best = ex; picked = GizmoHandle::AxisX; pick_axis = 0; }
        const float ey = ellipse_err(az, ax);
        if (ey < best) { best = ey; picked = GizmoHandle::AxisY; pick_axis = 1; }
        const float ez = ellipse_err(ax, ay);
        if (ez < best) { best = ez; picked = GizmoHandle::AxisZ; pick_axis = 2; }
        g_editor.hovered_gizmo = picked;

        if (g_editor.viewport_hovered && picked != GizmoHandle::None && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            g_editor.gizmo_dragging = true;
            g_editor.drag_start_mouse = mouse;
            g_editor.drag_start_mesh_index = -1;
            g_editor.drag_start_sphere_index = -1;
            g_editor.drag_start_point_light_index = -1;
            g_editor.drag_start_directional_light_index = -1;
            g_editor.active_gizmo = picked;
            g_editor.drag_start_directional_light = light;
            g_editor.drag_start_directional_light_index = g_editor.selected_directional_light;
            // Store initial ellipse angle for incremental rotation
            const lt::Vec3 wx = {1.0f, 0.0f, 0.0f}, wy = {0.0f, 1.0f, 0.0f}, wz = {0.0f, 0.0f, 1.0f};
            if (picked == GizmoHandle::AxisX) {
                g_editor.drag_start_axis_a_screen = py; g_editor.drag_start_axis_b_screen = pz;
            } else if (picked == GizmoHandle::AxisY) {
                g_editor.drag_start_axis_a_screen = pz; g_editor.drag_start_axis_b_screen = px;
            } else {
                g_editor.drag_start_axis_a_screen = px; g_editor.drag_start_axis_b_screen = py;
            }
            g_editor.drag_start_center_screen = c;
            g_editor.drag_start_angle = ellipse_angle(mouse, c,
                {g_editor.drag_start_axis_a_screen.x - c.x, g_editor.drag_start_axis_a_screen.y - c.y},
                {g_editor.drag_start_axis_b_screen.x - c.x, g_editor.drag_start_axis_b_screen.y - c.y});
        }

        if (!g_editor.gizmo_dragging) return;
        if (!ImGui::IsMouseDown(ImGuiMouseButton_Left) ||
            g_editor.selection_kind != SelectionKind::DirectionalLight ||
            g_editor.drag_start_directional_light_index != g_editor.selected_directional_light) {
            g_editor.gizmo_dragging = false;
            g_editor.active_gizmo = GizmoHandle::None;
            g_editor.drag_start_directional_light_index = -1;
            return;
        }

        // Incremental ellipse rotation — same pattern as mesh rotation
        const ImVec2 ea{
            g_editor.drag_start_axis_a_screen.x - g_editor.drag_start_center_screen.x,
            g_editor.drag_start_axis_a_screen.y - g_editor.drag_start_center_screen.y,
        };
        const ImVec2 eb{
            g_editor.drag_start_axis_b_screen.x - g_editor.drag_start_center_screen.x,
            g_editor.drag_start_axis_b_screen.y - g_editor.drag_start_center_screen.y,
        };
        float signed_amount = ellipse_angle(mouse, g_editor.drag_start_center_screen, ea, eb) - g_editor.drag_start_angle;
        if (signed_amount > lt::kPi) signed_amount -= 2.0f * lt::kPi;
        if (signed_amount < -lt::kPi) signed_amount += 2.0f * lt::kPi;
        if (std::fabs(signed_amount) < 1.0e-6f) return;

        // Rotation axis in world space
        lt::Vec3 rot_axis;
        if (g_editor.active_gizmo == GizmoHandle::AxisX) rot_axis = {1.0f, 0.0f, 0.0f};
        else if (g_editor.active_gizmo == GizmoHandle::AxisY) rot_axis = {0.0f, 1.0f, 0.0f};
        else rot_axis = {0.0f, 0.0f, 1.0f};

        // Apply rotation to the initial direction
        lt::Vec3 init_dir = g_editor.drag_start_directional_light.direction;
        float c_ = std::cos(signed_amount), s_ = std::sin(signed_amount);
        lt::Vec3 rotated = init_dir * c_ + lt::cross(rot_axis, init_dir) * s_ + rot_axis * lt::dot(rot_axis, init_dir) * (1.0f - c_);
        if (lt::dot(rotated, rotated) > 1.0e-8f) {
            light.direction = lt::normalize(rotated);
        }

        g_editor.scene.uses_builtin_default_meshes = false;
        reset_accumulation(lt::RenderDirty::Geometry);

    } else {
        g_editor.hovered_gizmo = GizmoHandle::None;
        g_editor.gizmo_dragging = false;
        g_editor.active_gizmo = GizmoHandle::None;
    }
}

void handle_gizmo_drag() {
    if (has_sphere_selection()) {
        handle_sphere_gizmo_drag();
        return;
    }
    if (has_point_light_selection()) {
        handle_point_light_gizmo_drag();
        return;
    }
    if (has_directional_light_selection()) {
        handle_directional_light_gizmo_drag();
        return;
    }
    if (!has_mesh_selection()) {
        g_editor.hovered_gizmo = GizmoHandle::None;
        g_editor.gizmo_dragging = false;
        g_editor.active_gizmo = GizmoHandle::None;
        g_editor.drag_start_mesh_index = -1;
        g_editor.drag_start_sphere_index = -1;
        g_editor.drag_start_point_light_index = -1;
        return;
    }
    lt::Mesh& mesh = g_editor.scene.meshes[static_cast<size_t>(g_editor.selected_mesh)];
    lt::Vec3 center3{};
    float radius = 1.0f;
    mesh_center_radius(mesh, center3, radius);
    ImVec2 center{};
    float depth = 1.0f;
    if (!project_point(center3, center, &depth)) {
        g_editor.hovered_gizmo = GizmoHandle::None;
        return;
    }

    const ImVec2 mouse = ImGui::GetIO().MousePos;
    GizmoHandle picked = GizmoHandle::Uniform;
    if (g_editor.tool_mode == ToolMode::Move || g_editor.tool_mode == ToolMode::Scale) {
        picked = pick_move_gizmo(mesh, center3, mouse, g_editor.tool_mode == ToolMode::Scale);
    } else if (g_editor.tool_mode == ToolMode::Rotate) {
        picked = pick_rotate_gizmo(mesh, center3, mouse);
    }
    g_editor.hovered_gizmo = picked;
    if (g_editor.viewport_hovered && picked != GizmoHandle::None && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && g_editor.tool_mode != ToolMode::Select) {
        g_editor.gizmo_dragging = true;
        g_editor.drag_start_mouse = mouse;
        g_editor.drag_start_mesh = mesh;
        g_editor.drag_start_mesh_index = g_editor.selected_mesh;
        g_editor.drag_start_sphere_index = -1;
        g_editor.drag_start_point_light_index = -1;
        g_editor.active_gizmo = picked;
        setup_drag_projection(mesh, center3, picked);
    }
    if (!g_editor.gizmo_dragging) return;
    if (!ImGui::IsMouseDown(ImGuiMouseButton_Left) || g_editor.drag_start_mesh_index != g_editor.selected_mesh) {
        g_editor.gizmo_dragging = false;
        g_editor.active_gizmo = GizmoHandle::None;
        return;
    }
    const ImVec2 delta{mouse.x - g_editor.drag_start_mouse.x, mouse.y - g_editor.drag_start_mouse.y};
    const ViewTransform view = make_view_transform(true);
    const float world_per_pixel = g_editor.drag_start_depth * view.half_height * 2.0f / std::max(1.0f, g_editor.viewport_size.y);
    if (g_editor.tool_mode == ToolMode::Move) {
        if (g_editor.active_gizmo == GizmoHandle::AxisX || g_editor.active_gizmo == GizmoHandle::AxisY || g_editor.active_gizmo == GizmoHandle::AxisZ) {
            const lt::Vec3 axis = handle_axis(g_editor.drag_start_mesh, g_editor.active_gizmo);
            const ImVec2 screen_axis{
                g_editor.drag_start_axis_a_screen.x - g_editor.drag_start_center_screen.x,
                g_editor.drag_start_axis_a_screen.y - g_editor.drag_start_center_screen.y,
            };
            const float len = std::sqrt(screen_axis.x * screen_axis.x + screen_axis.y * screen_axis.y);
            const float amount = len > 0.0f ? (delta.x * screen_axis.x + delta.y * screen_axis.y) / len * world_per_pixel : 0.0f;
            mesh.translation = g_editor.drag_start_mesh.translation + axis * amount;
        } else {
            lt::Vec3 a{}, b{};
            handle_plane_axes(g_editor.drag_start_mesh, g_editor.active_gizmo, a, b);
            const ImVec2 sa{
                g_editor.drag_start_axis_a_screen.x - g_editor.drag_start_center_screen.x,
                g_editor.drag_start_axis_a_screen.y - g_editor.drag_start_center_screen.y,
            };
            const ImVec2 sb{
                g_editor.drag_start_axis_b_screen.x - g_editor.drag_start_center_screen.x,
                g_editor.drag_start_axis_b_screen.y - g_editor.drag_start_center_screen.y,
            };
            const float det = sa.x * sb.y - sa.y * sb.x;
            float amount_a = 0.0f;
            float amount_b = 0.0f;
            if (std::fabs(det) > 80.0f) {
                amount_a = (delta.x * sb.y - delta.y * sb.x) / det;
                amount_b = (sa.x * delta.y - sa.y * delta.x) / det;
            } else {
                const ImVec2 combined{sa.x + sb.x, sa.y + sb.y};
                const float len = std::sqrt(combined.x * combined.x + combined.y * combined.y);
                const float amount = len > 1.0f ? (delta.x * combined.x + delta.y * combined.y) / len * world_per_pixel : 0.0f;
                mesh.translation = g_editor.drag_start_mesh.translation + lt::normalize(a + b) * amount;
                reset_accumulation(lt::RenderDirty::Transform);
                return;
            }
            mesh.translation = g_editor.drag_start_mesh.translation + a * amount_a + b * amount_b;
        }
    } else if (g_editor.tool_mode == ToolMode::Rotate) {
        const ImVec2 axis_a{
            g_editor.drag_start_axis_a_screen.x - g_editor.drag_start_center_screen.x,
            g_editor.drag_start_axis_a_screen.y - g_editor.drag_start_center_screen.y,
        };
        const ImVec2 axis_b{
            g_editor.drag_start_axis_b_screen.x - g_editor.drag_start_center_screen.x,
            g_editor.drag_start_axis_b_screen.y - g_editor.drag_start_center_screen.y,
        };
        float signed_amount = ellipse_angle(mouse, g_editor.drag_start_center_screen, axis_a, axis_b) - g_editor.drag_start_angle;
        if (signed_amount > lt::kPi) signed_amount -= 2.0f * lt::kPi;
        if (signed_amount < -lt::kPi) signed_amount += 2.0f * lt::kPi;
        mesh.rotation = g_editor.drag_start_mesh.rotation;
        if (g_editor.active_gizmo == GizmoHandle::AxisX) mesh.rotation.x += signed_amount;
        else if (g_editor.active_gizmo == GizmoHandle::AxisY) mesh.rotation.y += signed_amount;
        else if (g_editor.active_gizmo == GizmoHandle::AxisZ) mesh.rotation.z += signed_amount;
    } else if (g_editor.tool_mode == ToolMode::Scale) {
        const auto projected_pixels = [&](ImVec2 direction) {
            const float len = std::sqrt(direction.x * direction.x + direction.y * direction.y);
            return len > 1.0f ? (delta.x * direction.x + delta.y * direction.y) / len : 0.0f;
        };

        float amount = 0.0f;
        if (g_editor.active_gizmo == GizmoHandle::Uniform) {
            amount = std::fabs(delta.x) > std::fabs(delta.y) ? delta.x : -delta.y;
        } else if (g_editor.active_gizmo == GizmoHandle::AxisX || g_editor.active_gizmo == GizmoHandle::AxisY || g_editor.active_gizmo == GizmoHandle::AxisZ) {
            const ImVec2 direction{
                g_editor.drag_start_axis_a_screen.x - g_editor.drag_start_center_screen.x,
                g_editor.drag_start_axis_a_screen.y - g_editor.drag_start_center_screen.y,
            };
            amount = projected_pixels(direction);
        } else {
            const ImVec2 direction{
                g_editor.drag_start_axis_a_screen.x + g_editor.drag_start_axis_b_screen.x - g_editor.drag_start_center_screen.x * 2.0f,
                g_editor.drag_start_axis_a_screen.y + g_editor.drag_start_axis_b_screen.y - g_editor.drag_start_center_screen.y * 2.0f,
            };
            amount = projected_pixels(direction);
        }

        const float factor = std::max(0.05f, 1.0f + amount * 0.01f);
        mesh.scale = g_editor.drag_start_mesh.scale;
        if (g_editor.active_gizmo == GizmoHandle::AxisX) mesh.scale.x = g_editor.drag_start_mesh.scale.x * factor;
        else if (g_editor.active_gizmo == GizmoHandle::AxisY) mesh.scale.y = g_editor.drag_start_mesh.scale.y * factor;
        else if (g_editor.active_gizmo == GizmoHandle::AxisZ) mesh.scale.z = g_editor.drag_start_mesh.scale.z * factor;
        else if (g_editor.active_gizmo == GizmoHandle::PlaneXY) {
            mesh.scale.x = g_editor.drag_start_mesh.scale.x * factor;
            mesh.scale.y = g_editor.drag_start_mesh.scale.y * factor;
        } else if (g_editor.active_gizmo == GizmoHandle::PlaneYZ) {
            mesh.scale.y = g_editor.drag_start_mesh.scale.y * factor;
            mesh.scale.z = g_editor.drag_start_mesh.scale.z * factor;
        } else if (g_editor.active_gizmo == GizmoHandle::PlaneZX) {
            mesh.scale.z = g_editor.drag_start_mesh.scale.z * factor;
            mesh.scale.x = g_editor.drag_start_mesh.scale.x * factor;
        } else {
            mesh.scale = g_editor.drag_start_mesh.scale * factor;
        }
        mesh.scale.x = std::max(0.01f, mesh.scale.x);
        mesh.scale.y = std::max(0.01f, mesh.scale.y);
        mesh.scale.z = std::max(0.01f, mesh.scale.z);
    }
    reset_accumulation(lt::RenderDirty::Transform);
}

void draw_arrow_2d(ImDrawList* draw_list, ImVec2 start, ImVec2 end, ImU32 color) {
    draw_list->AddLine(start, end, color, 3.0f);
    const float dx = end.x - start.x;
    const float dy = end.y - start.y;
    const float len = std::sqrt(dx * dx + dy * dy);
    if (len <= 1.0f) return;
    const float nx = dx / len;
    const float ny = dy / len;
    draw_list->AddTriangleFilled(end, {end.x - nx * 12.0f - ny * 6.0f, end.y - ny * 12.0f + nx * 6.0f}, {end.x - nx * 12.0f + ny * 6.0f, end.y - ny * 12.0f - nx * 6.0f}, color);
}

bool handle_is_hot(GizmoHandle handle) {
    return g_editor.active_gizmo == handle || (!g_editor.gizmo_dragging && g_editor.hovered_gizmo == handle);
}

ImU32 hot_color(GizmoHandle handle, ImU32 normal) {
    return handle_is_hot(handle) ? IM_COL32(255, 235, 80, 255) : normal;
}

float hot_thickness(GizmoHandle handle, float normal) {
    return handle_is_hot(handle) ? normal + 2.0f : normal;
}

void draw_arrow_2d_handle(ImDrawList* draw_list, ImVec2 start, ImVec2 end, ImU32 color, GizmoHandle handle) {
    const ImU32 c = hot_color(handle, color);
    const float thickness = hot_thickness(handle, 3.0f);
    draw_list->AddLine({start.x + 1.0f, start.y + 1.0f}, {end.x + 1.0f, end.y + 1.0f}, IM_COL32(0, 0, 0, 125), thickness + 1.0f);
    draw_list->AddLine(start, end, c, thickness);
    const float dx = end.x - start.x;
    const float dy = end.y - start.y;
    const float len = std::sqrt(dx * dx + dy * dy);
    if (len <= 1.0f) return;
    const float nx = dx / len;
    const float ny = dy / len;
    const float head = handle_is_hot(handle) ? 15.0f : 12.0f;
    const float wing = handle_is_hot(handle) ? 8.0f : 6.0f;
    draw_list->AddTriangleFilled({end.x + 1.0f, end.y + 1.0f}, {end.x - nx * head - ny * wing + 1.0f, end.y - ny * head + nx * wing + 1.0f}, {end.x - nx * head + ny * wing + 1.0f, end.y - ny * head - nx * wing + 1.0f}, IM_COL32(0, 0, 0, 125));
    draw_list->AddTriangleFilled(end, {end.x - nx * head - ny * wing, end.y - ny * head + nx * wing}, {end.x - nx * head + ny * wing, end.y - ny * head - nx * wing}, c);
}

void draw_axis_label(ImDrawList* draw_list, ImVec2 p, const char* label, ImU32 color) {
    const ImVec2 text_size = ImGui::CalcTextSize(label);
    const ImVec2 pad{5.0f, 3.0f};
    const ImVec2 min{p.x - text_size.x * 0.5f - pad.x, p.y - text_size.y * 0.5f - pad.y};
    const ImVec2 max{p.x + text_size.x * 0.5f + pad.x, p.y + text_size.y * 0.5f + pad.y};
    draw_list->AddRectFilled(min, max, IM_COL32(18, 18, 20, 210), 4.0f);
    draw_list->AddRect(min, max, color, 4.0f, 0, 1.2f);
    draw_list->AddText({p.x - text_size.x * 0.5f, p.y - text_size.y * 0.5f}, color, label);
}

void draw_gizmo_center(ImDrawList* draw_list, ImVec2 center) {
    const bool hot = handle_is_hot(GizmoHandle::Uniform);
    const float radius = hot ? 8.5f : 6.5f;
    const ImU32 fill = hot ? IM_COL32(255, 218, 96, 245) : IM_COL32(232, 232, 226, 230);
    draw_list->AddCircleFilled({center.x + 1.0f, center.y + 1.0f}, radius + 1.5f, IM_COL32(0, 0, 0, 120), 32);
    draw_list->AddCircleFilled(center, radius, fill, 32);
    draw_list->AddCircle(center, radius + 3.5f, IM_COL32(255, 255, 255, hot ? 190 : 95), 40, 1.2f);
}

void draw_polyline_ellipse(ImDrawList* draw_list, ImVec2 center, ImVec2 axis_a, ImVec2 axis_b, ImU32 color, float thickness) {
    constexpr int segments = 96;
    ImVec2 prev{center.x + axis_a.x, center.y + axis_a.y};
    for (int i = 1; i <= segments; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(segments) * 2.0f * lt::kPi;
        const ImVec2 p{
            center.x + std::cos(t) * axis_a.x + std::sin(t) * axis_b.x,
            center.y + std::cos(t) * axis_a.y + std::sin(t) * axis_b.y,
        };
        draw_list->AddLine(prev, p, color, thickness);
        prev = p;
    }
}

void draw_mesh_outline(ImDrawList* draw_list) {
    if (!has_mesh_selection()) return;
    const lt::Mesh& mesh = g_editor.scene.meshes[static_cast<size_t>(g_editor.selected_mesh)];
    constexpr ImU32 orange = IM_COL32(255, 150, 35, 255);
    for (size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
        ImVec2 p0{}, p1{}, p2{};
        const lt::Vec3 v0 = transform_mesh_point(mesh, mesh.vertices[static_cast<size_t>(mesh.indices[i])]);
        const lt::Vec3 v1 = transform_mesh_point(mesh, mesh.vertices[static_cast<size_t>(mesh.indices[i + 1])]);
        const lt::Vec3 v2 = transform_mesh_point(mesh, mesh.vertices[static_cast<size_t>(mesh.indices[i + 2])]);
        if (project_point(v0, p0) && project_point(v1, p1) && project_point(v2, p2)) {
            draw_list->AddLine(p0, p1, orange, 2.0f);
            draw_list->AddLine(p1, p2, orange, 2.0f);
            draw_list->AddLine(p2, p0, orange, 2.0f);
        }
    }
}

void draw_mesh_bounds_outline(ImDrawList* draw_list, int mesh_index, ImU32 color, float thickness) {
    if (mesh_index < 0 || mesh_index >= static_cast<int>(g_editor.scene.meshes.size())) return;
    ensure_mesh_bounds_cache();
    const lt::Mesh& mesh = g_editor.scene.meshes[static_cast<size_t>(mesh_index)];
    const lt::Vec3 local_min = g_bounds_cache.local_min[static_cast<size_t>(mesh_index)];
    const lt::Vec3 local_max = g_bounds_cache.local_max[static_cast<size_t>(mesh_index)];
    const lt::Vec3 corners[8] = {
        {local_min.x, local_min.y, local_min.z},
        {local_max.x, local_min.y, local_min.z},
        {local_max.x, local_max.y, local_min.z},
        {local_min.x, local_max.y, local_min.z},
        {local_min.x, local_min.y, local_max.z},
        {local_max.x, local_min.y, local_max.z},
        {local_max.x, local_max.y, local_max.z},
        {local_min.x, local_max.y, local_max.z},
    };
    ImVec2 projected[8]{};
    bool visible[8]{};
    for (int i = 0; i < 8; ++i) {
        visible[i] = project_point(transform_mesh_point(mesh, corners[i]), projected[i]);
    }
    constexpr int edges[12][2] = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0},
        {4, 5}, {5, 6}, {6, 7}, {7, 4},
        {0, 4}, {1, 5}, {2, 6}, {3, 7},
    };
    for (const auto& edge : edges) {
        if (visible[edge[0]] && visible[edge[1]]) {
            draw_list->AddLine(projected[edge[0]], projected[edge[1]], color, thickness);
        }
    }
}

void draw_sphere_outline(ImDrawList* draw_list, int sphere_index, ImU32 color, float thickness) {
    if (sphere_index < 0 || sphere_index >= static_cast<int>(g_editor.scene.spheres.size())) return;
    const lt::Sphere& sphere = g_editor.scene.spheres[static_cast<size_t>(sphere_index)];
    ImVec2 center{};
    float depth = 1.0f;
    if (!project_point(sphere.center, center, &depth)) return;
    const ViewTransform view = make_view_transform(true);
    const int segments = 72;
    ImVec2 previous{};
    bool has_previous = false;
    for (int i = 0; i <= segments; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(segments) * 2.0f * lt::kPi;
        const lt::Vec3 p = sphere.center + view.right * (std::cos(t) * sphere.radius) + view.up * (std::sin(t) * sphere.radius);
        ImVec2 projected{};
        if (project_point(p, projected)) {
            if (has_previous) {
                draw_list->AddLine(previous, projected, color, thickness);
            }
            previous = projected;
            has_previous = true;
        } else {
            has_previous = false;
        }
    }
}

void draw_move_gizmo_handles(ImDrawList* draw_list, lt::Vec3 center3, lt::Vec3 ax, lt::Vec3 ay, lt::Vec3 az, bool draw_uniform) {
    ImVec2 center{};
    if (!project_point(center3, center)) return;
    ImVec2 px{}, py{}, pz{}, pxy{}, pyz{}, pzx{};

    const float ps = world_per_pixel_at(center3);
    const float arrow_len = 80.0f * ps;
    const float plane_ofs = 40.0f * ps;

    const bool okx = project_point(center3 + ax * arrow_len, px);
    const bool oky = project_point(center3 + ay * arrow_len, py);
    const bool okz = project_point(center3 + az * arrow_len, pz);
    const bool okxy = project_point(center3 + (ax + ay) * plane_ofs, pxy);
    const bool okyz = project_point(center3 + (ay + az) * plane_ofs, pyz);
    const bool okzx = project_point(center3 + (az + ax) * plane_ofs, pzx);
    if (okx && oky && okxy) {
        const ImU32 fill = handle_is_hot(GizmoHandle::PlaneXY) ? IM_COL32(255, 235, 80, 115) : IM_COL32(235, 200, 70, 70);
        const ImU32 edge = hot_color(GizmoHandle::PlaneXY, IM_COL32(235, 200, 70, 210));
        const ImVec2 a{center.x + (px.x - center.x) * 0.28f + (py.x - center.x) * 0.28f, center.y + (px.y - center.y) * 0.28f + (py.y - center.y) * 0.28f};
        const ImVec2 b{center.x + (px.x - center.x) * 0.48f + (py.x - center.x) * 0.28f, center.y + (px.y - center.y) * 0.48f + (py.y - center.y) * 0.28f};
        const ImVec2 c{center.x + (px.x - center.x) * 0.48f + (py.x - center.x) * 0.48f, center.y + (px.y - center.y) * 0.48f + (py.y - center.y) * 0.48f};
        const ImVec2 d{center.x + (px.x - center.x) * 0.28f + (py.x - center.x) * 0.48f, center.y + (px.y - center.y) * 0.28f + (py.y - center.y) * 0.48f};
        draw_list->AddQuadFilled(a, b, c, d, fill);
        draw_list->AddQuad(a, b, c, d, edge, hot_thickness(GizmoHandle::PlaneXY, 1.5f));
    }
    if (oky && okz && okyz) {
        const ImU32 fill = handle_is_hot(GizmoHandle::PlaneYZ) ? IM_COL32(255, 235, 80, 105) : IM_COL32(70, 180, 235, 65);
        const ImU32 edge = hot_color(GizmoHandle::PlaneYZ, IM_COL32(70, 180, 235, 200));
        const ImVec2 a{center.x + (py.x - center.x) * 0.28f + (pz.x - center.x) * 0.28f, center.y + (py.y - center.y) * 0.28f + (pz.y - center.y) * 0.28f};
        const ImVec2 b{center.x + (py.x - center.x) * 0.48f + (pz.x - center.x) * 0.28f, center.y + (py.y - center.y) * 0.48f + (pz.y - center.y) * 0.28f};
        const ImVec2 c{center.x + (py.x - center.x) * 0.48f + (pz.x - center.x) * 0.48f, center.y + (py.y - center.y) * 0.48f + (pz.y - center.y) * 0.48f};
        const ImVec2 d{center.x + (py.x - center.x) * 0.28f + (pz.x - center.x) * 0.48f, center.y + (py.y - center.y) * 0.28f + (pz.y - center.y) * 0.48f};
        draw_list->AddQuadFilled(a, b, c, d, fill);
        draw_list->AddQuad(a, b, c, d, edge, hot_thickness(GizmoHandle::PlaneYZ, 1.5f));
    }
    if (okz && okx && okzx) {
        const ImU32 fill = handle_is_hot(GizmoHandle::PlaneZX) ? IM_COL32(255, 235, 80, 105) : IM_COL32(235, 90, 180, 65);
        const ImU32 edge = hot_color(GizmoHandle::PlaneZX, IM_COL32(235, 90, 180, 200));
        const ImVec2 a{center.x + (pz.x - center.x) * 0.28f + (px.x - center.x) * 0.28f, center.y + (pz.y - center.y) * 0.28f + (px.y - center.y) * 0.28f};
        const ImVec2 b{center.x + (pz.x - center.x) * 0.48f + (px.x - center.x) * 0.28f, center.y + (pz.y - center.y) * 0.48f + (px.y - center.y) * 0.28f};
        const ImVec2 c{center.x + (pz.x - center.x) * 0.48f + (px.x - center.x) * 0.48f, center.y + (pz.y - center.y) * 0.48f + (px.y - center.y) * 0.48f};
        const ImVec2 d{center.x + (pz.x - center.x) * 0.28f + (px.x - center.x) * 0.48f, center.y + (pz.y - center.y) * 0.28f + (px.y - center.y) * 0.48f};
        draw_list->AddQuadFilled(a, b, c, d, fill);
        draw_list->AddQuad(a, b, c, d, edge, hot_thickness(GizmoHandle::PlaneZX, 1.5f));
    }
    if (okx) draw_arrow_2d_handle(draw_list, center, px, IM_COL32(235, 80, 80, 255), GizmoHandle::AxisX);
    if (oky) draw_arrow_2d_handle(draw_list, center, py, IM_COL32(80, 210, 105, 255), GizmoHandle::AxisY);
    if (okz) draw_arrow_2d_handle(draw_list, center, pz, IM_COL32(90, 140, 255, 255), GizmoHandle::AxisZ);
    if (okx) draw_axis_label(draw_list, {px.x + 18.0f, px.y}, "X", IM_COL32(245, 92, 88, 255));
    if (oky) draw_axis_label(draw_list, {py.x + 18.0f, py.y}, "Y", IM_COL32(100, 230, 130, 255));
    if (okz) draw_axis_label(draw_list, {pz.x + 18.0f, pz.y}, "Z", IM_COL32(116, 160, 255, 255));
    if (draw_uniform) {
        draw_gizmo_center(draw_list, center);
    }
}

void draw_scene_reference_outlines(ImDrawList* draw_list) {
    const ImU32 mesh_color = IM_COL32(150, 170, 190, 130);
    for (int i = 0; i < static_cast<int>(g_editor.scene.meshes.size()); ++i) {
        const bool selected = g_editor.selection_kind == SelectionKind::Mesh && i == g_editor.selected_mesh;
        if (!selected) {
            draw_mesh_bounds_outline(draw_list, i, mesh_color, 1.0f);
        }
    }
    for (int i = 0; i < static_cast<int>(g_editor.scene.spheres.size()); ++i) {
        const bool selected = g_editor.selection_kind == SelectionKind::Sphere && i == g_editor.selected_sphere;
        if (!selected) {
            draw_sphere_outline(draw_list, i, mesh_color, 1.0f);
        }
    }
}

void draw_gizmo_overlay() {
    if (!has_selection()) return;
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    const bool draw_selection_reference = false;
    if (has_sphere_selection()) {
        const lt::Sphere& sphere = g_editor.scene.spheres[static_cast<size_t>(g_editor.selected_sphere)];
        if (draw_selection_reference && !g_editor.hide_dirty_wireframes)
            draw_sphere_outline(draw_list, g_editor.selected_sphere, IM_COL32(255, 150, 35, 255), 2.0f);
        if (g_editor.tool_mode == ToolMode::Move) {
            draw_move_gizmo_handles(
                draw_list,
                sphere.center,
                {1.0f, 0.0f, 0.0f},
                {0.0f, 1.0f, 0.0f},
                {0.0f, 0.0f, 1.0f},
                false);
        }
        return;
    }
    if (has_point_light_selection()) {
        const lt::PointLight& light = g_editor.scene.point_lights[static_cast<size_t>(g_editor.selected_point_light)];
        if (g_editor.tool_mode == ToolMode::Move) {
            draw_move_gizmo_handles(
                draw_list,
                light.position,
                {1.0f, 0.0f, 0.0f},
                {0.0f, 1.0f, 0.0f},
                {0.0f, 0.0f, 1.0f},
                false);
        }
        return;
    }
    if (has_directional_light_selection()) {
        const lt::DirectionalLight& light = g_editor.scene.directional_lights[static_cast<size_t>(g_editor.selected_directional_light)];
        if (g_editor.tool_mode == ToolMode::Move) {
            draw_move_gizmo_handles(
                draw_list,
                light.position,
                {1.0f, 0.0f, 0.0f},
                {0.0f, 1.0f, 0.0f},
                {0.0f, 0.0f, 1.0f},
                false);
        } else if (g_editor.tool_mode == ToolMode::Rotate) {
            ImVec2 px{}, py{}, pz{};
            ImVec2 center{};
            if (!project_point(light.position, center)) return;
            const float ps = world_per_pixel_at(light.position);
            const float rot_r = 60.0f * ps;
            project_point(light.position + lt::Vec3{1.0f, 0.0f, 0.0f} * rot_r, px);
            project_point(light.position + lt::Vec3{0.0f, 1.0f, 0.0f} * rot_r, py);
            project_point(light.position + lt::Vec3{0.0f, 0.0f, 1.0f} * rot_r, pz);
            const ImVec2 ax{px.x - center.x, px.y - center.y};
            const ImVec2 ay{py.x - center.x, py.y - center.y};
            const ImVec2 az{pz.x - center.x, pz.y - center.y};
            draw_polyline_ellipse(draw_list, center, ay, az, hot_color(GizmoHandle::AxisX, IM_COL32(235, 80, 80, 255)), hot_thickness(GizmoHandle::AxisX, 2.0f));
            draw_polyline_ellipse(draw_list, center, az, ax, hot_color(GizmoHandle::AxisY, IM_COL32(80, 210, 105, 255)), hot_thickness(GizmoHandle::AxisY, 2.0f));
            draw_polyline_ellipse(draw_list, center, ax, ay, hot_color(GizmoHandle::AxisZ, IM_COL32(90, 140, 255, 255)), hot_thickness(GizmoHandle::AxisZ, 2.0f));
        }
        // Always draw direction arrow as visual indicator
        // direction points surface→light; arrow shows light→surface (where light shines)
        {
            const lt::Vec3 dir = lt::normalize(light.direction);
            if (lt::dot(dir, dir) > 0.0f) {
                const float ps = world_per_pixel_at(light.position);
                const float arrow_world = 80.0f * ps;
                const lt::Vec3 tip = light.position - dir * arrow_world;
                ImVec2 s_pos{}, s_tip{};
                if (project_point(light.position, s_pos) && project_point(tip, s_tip)) {
                    const ImU32 arrow_color = IM_COL32(255, 200, 70, 200);
                    draw_list->AddLine(s_pos, s_tip, arrow_color, 2.0f);
                    const ImVec2 dv{s_tip.x - s_pos.x, s_tip.y - s_pos.y};
                    const float len = std::sqrt(dv.x * dv.x + dv.y * dv.y);
                    if (len > 4.0f) {
                        const ImVec2 nd{dv.x / len, dv.y / len};
                        const ImVec2 perp{-nd.y * 7.0f, nd.x * 7.0f};
                        draw_list->AddTriangleFilled(
                            s_tip,
                            {s_tip.x - nd.x * 12.0f + perp.x, s_tip.y - nd.y * 12.0f + perp.y},
                            {s_tip.x - nd.x * 12.0f - perp.x, s_tip.y - nd.y * 12.0f - perp.y},
                            arrow_color);
                    }
                }
            }
        }
        return;
    }
    lt::Vec3 center3{};
    float radius3 = 1.0f;
    mesh_center_radius(g_editor.scene.meshes[static_cast<size_t>(g_editor.selected_mesh)], center3, radius3);
    ImVec2 center{};
    if (!project_point(center3, center)) return;
    if (g_editor.tool_mode == ToolMode::Select) {
        if (draw_selection_reference && !g_editor.hide_dirty_wireframes)
            draw_mesh_bounds_outline(draw_list, g_editor.selected_mesh, IM_COL32(255, 150, 35, 255), 2.0f);
    }
    if (g_editor.tool_mode == ToolMode::Move || g_editor.tool_mode == ToolMode::Scale) {
        const lt::Mesh& mesh = g_editor.scene.meshes[static_cast<size_t>(g_editor.selected_mesh)];
        const lt::Vec3 ax = transform_axis(mesh, {1.0f, 0.0f, 0.0f});
        const lt::Vec3 ay = transform_axis(mesh, {0.0f, 1.0f, 0.0f});
        const lt::Vec3 az = transform_axis(mesh, {0.0f, 0.0f, 1.0f});
        draw_move_gizmo_handles(draw_list, center3, ax, ay, az, g_editor.tool_mode == ToolMode::Scale);
    } else if (g_editor.tool_mode == ToolMode::Rotate) {
        const lt::Mesh& mesh = g_editor.scene.meshes[static_cast<size_t>(g_editor.selected_mesh)];
        ImVec2 px{}, py{}, pz{};
        const float ps = world_per_pixel_at(center3);
        const float rot_r = 60.0f * ps;
        project_point(center3 + local_axis(mesh, {1.0f, 0.0f, 0.0f}) * rot_r, px);
        project_point(center3 + local_axis(mesh, {0.0f, 1.0f, 0.0f}) * rot_r, py);
        project_point(center3 + local_axis(mesh, {0.0f, 0.0f, 1.0f}) * rot_r, pz);
        const ImVec2 ax{px.x - center.x, px.y - center.y};
        const ImVec2 ay{py.x - center.x, py.y - center.y};
        const ImVec2 az{pz.x - center.x, pz.y - center.y};
        draw_polyline_ellipse(draw_list, center, ay, az, hot_color(GizmoHandle::AxisX, IM_COL32(235, 80, 80, 255)), hot_thickness(GizmoHandle::AxisX, 2.0f));
        draw_polyline_ellipse(draw_list, center, az, ax, hot_color(GizmoHandle::AxisY, IM_COL32(80, 210, 105, 255)), hot_thickness(GizmoHandle::AxisY, 2.0f));
        draw_polyline_ellipse(draw_list, center, ax, ay, hot_color(GizmoHandle::AxisZ, IM_COL32(90, 140, 255, 255)), hot_thickness(GizmoHandle::AxisZ, 2.0f));
        draw_gizmo_center(draw_list, center);
        draw_axis_label(draw_list, {px.x + 16.0f, px.y}, "X", IM_COL32(245, 92, 88, 255));
        draw_axis_label(draw_list, {py.x + 16.0f, py.y}, "Y", IM_COL32(100, 230, 130, 255));
        draw_axis_label(draw_list, {pz.x + 16.0f, pz.y}, "Z", IM_COL32(116, 160, 255, 255));
    }
}

bool edit_vec3(const char* label, lt::Vec3& value, float speed = 0.02f, lt::RenderDirty dirty = lt::RenderDirty::Render) {
    float data[3] = {value.x, value.y, value.z};
    if (ImGui::DragFloat3(label, data, speed, -1000.0f, 1000.0f, "%.3f")) {
        value = {data[0], data[1], data[2]};
        reset_accumulation(dirty);
        return true;
    }
    return false;
}

lt::Vec3 mesh_world_center(const lt::Mesh& mesh) {
    if (mesh.vertices.empty()) {
        return mesh.translation;
    }

    lt::Aabb bounds;
    for (lt::Vec3 v : mesh.vertices) {
        const lt::Vec3 p = transform_mesh_point(mesh, v);
        bounds.min = lt::min(bounds.min, p);
        bounds.max = lt::max(bounds.max, p);
    }
    return (bounds.min + bounds.max) * 0.5f;
}

bool edit_location_world(const char* label, lt::Mesh& mesh) {
    lt::Vec3 location = mesh_world_center(mesh);
    float data[3] = {location.x, location.y, location.z};
    if (ImGui::DragFloat3(label, data, 0.02f, -1000.0f, 1000.0f, "%.3f")) {
        const lt::Vec3 next{data[0], data[1], data[2]};
        mesh.translation += next - location;
        reset_accumulation(lt::RenderDirty::Transform);
        return true;
    }
    return false;
}

bool edit_rotation_degrees(const char* label, lt::Vec3& radians) {
    constexpr float kRadToDeg = 180.0f / lt::kPi;
    constexpr float kDegToRad = lt::kPi / 180.0f;
    float data[3] = {radians.x * kRadToDeg, radians.y * kRadToDeg, radians.z * kRadToDeg};
    if (ImGui::DragFloat3(label, data, 0.2f, -36000.0f, 36000.0f, "%.2f deg")) {
        radians = {data[0] * kDegToRad, data[1] * kDegToRad, data[2] * kDegToRad};
        reset_accumulation(lt::RenderDirty::Transform);
        return true;
    }
    return false;
}

bool color_edit(const char* label, lt::Vec3& value, lt::RenderDirty dirty = lt::RenderDirty::Render) {
    float data[3] = {value.x, value.y, value.z};
    if (ImGui::ColorEdit3(label, data, ImGuiColorEditFlags_Float)) {
        value = {data[0], data[1], data[2]};
        reset_accumulation(dirty);
        return true;
    }
    return false;
}

bool color_edit_hdr(const char* label, lt::Vec3& value, lt::RenderDirty dirty = lt::RenderDirty::Render) {
    float data[3] = {value.x, value.y, value.z};
    if (ImGui::ColorEdit3(label, data, ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR)) {
        value = {std::max(0.0f, data[0]), std::max(0.0f, data[1]), std::max(0.0f, data[2])};
        reset_accumulation(dirty);
        return true;
    }
    return false;
}

void mark_npr_dirty() {
    if (lt::scene_has_npr_styles(g_editor.scene)) {
        set_renderer(false);
    }
    reset_accumulation(lt::RenderDirty::Material);
}

void draw_npr_controls(lt::Material& material) {
    ImGui::SeparatorText("NPR");
    const char* npr_styles[] = {"None", "Color Map", "X-Toon", "Cross Hatching"};
    int npr_style = static_cast<int>(material.npr.style);
    if (ImGui::Combo("NPR Style", &npr_style, npr_styles, IM_ARRAYSIZE(npr_styles))) {
        material.npr.style = static_cast<lt::NprStyle>(npr_style);
        mark_npr_dirty();
    }

    if (material.npr.style == lt::NprStyle::ColorMap) {
        float range[2] = {material.npr.value_min, material.npr.value_max};
        if (ImGui::DragFloat2("Brightness Range", range, 0.01f, -100.0f, 100.0f, "%.2f")) {
            material.npr.value_min = range[0];
            material.npr.value_max = std::max(range[1], range[0] + 1.0e-4f);
            mark_npr_dirty();
        }
    } else if (material.npr.style == lt::NprStyle::XToon) {
        const char* detail_modes[] = {"Constant", "Depth", "Near Silhouette", "Highlight"};
        int detail_mode = static_cast<int>(material.npr.xtoon_detail_mode);
        if (ImGui::Combo("Detail Mode", &detail_mode, detail_modes, IM_ARRAYSIZE(detail_modes))) {
            material.npr.xtoon_detail_mode = static_cast<lt::XToonDetailMode>(detail_mode);
            mark_npr_dirty();
        }
        if (ImGui::DragInt("Tone Steps", &material.npr.xtoon_steps, 1.0f, 1, 8)) {
            material.npr.xtoon_steps = std::clamp(material.npr.xtoon_steps, 1, 8);
            mark_npr_dirty();
        }
        color_edit_hdr("Shadow", material.npr.xtoon_shadow, lt::RenderDirty::Material);
        color_edit_hdr("Mid", material.npr.xtoon_mid, lt::RenderDirty::Material);
        color_edit_hdr("Lit", material.npr.xtoon_lit, lt::RenderDirty::Material);
        color_edit_hdr("Accent", material.npr.xtoon_accent, lt::RenderDirty::Material);
        if (ImGui::DragFloat("Detail Strength", &material.npr.xtoon_detail_strength, 0.01f, 0.0f, 1.0f, "%.2f")) {
            material.npr.xtoon_detail_strength = std::clamp(material.npr.xtoon_detail_strength, 0.0f, 1.0f);
            mark_npr_dirty();
        }
        if (ImGui::DragFloat("Detail Threshold", &material.npr.xtoon_detail_threshold, 0.01f, 0.0f, 1.0f, "%.2f")) {
            material.npr.xtoon_detail_threshold = std::clamp(material.npr.xtoon_detail_threshold, 0.0f, 1.0f);
            mark_npr_dirty();
        }
        if (ImGui::DragFloat("Detail Power", &material.npr.xtoon_detail_power, 1.0f, 1.0f, 256.0f, "%.0f")) {
            material.npr.xtoon_detail_power = std::clamp(material.npr.xtoon_detail_power, 1.0f, 256.0f);
            mark_npr_dirty();
        }
        if (material.npr.xtoon_detail_mode == lt::XToonDetailMode::Depth) {
            if (ImGui::DragFloat("Depth Near", &material.npr.xtoon_depth_near, 0.05f, 0.001f, 1000.0f, "%.3f")) {
                material.npr.xtoon_depth_near = std::max(0.001f, material.npr.xtoon_depth_near);
                material.npr.xtoon_depth_far = std::max(material.npr.xtoon_depth_far, material.npr.xtoon_depth_near + 0.001f);
                mark_npr_dirty();
            }
            if (ImGui::DragFloat("Depth Far", &material.npr.xtoon_depth_far, 0.05f, 0.001f, 1000.0f, "%.3f")) {
                material.npr.xtoon_depth_far = std::max(material.npr.xtoon_depth_far, material.npr.xtoon_depth_near + 0.001f);
                mark_npr_dirty();
            }
        }
    } else if (material.npr.style == lt::NprStyle::CrossHatching) {
        if (ImGui::DragInt("Hatch Sets", &material.npr.hatch_sets, 1.0f, 1, 8)) {
            material.npr.hatch_sets = std::clamp(material.npr.hatch_sets, 1, 8);
            mark_npr_dirty();
        }
        if (ImGui::DragFloat("Spacing", &material.npr.hatch_spacing, 0.002f, 0.001f, 10.0f, "%.4f")) {
            material.npr.hatch_spacing = std::max(0.001f, material.npr.hatch_spacing);
            material.npr.hatch_width = std::min(material.npr.hatch_width, material.npr.hatch_spacing);
            mark_npr_dirty();
        }
        if (ImGui::DragFloat("Line Width", &material.npr.hatch_width, 0.001f, 0.0001f, 10.0f, "%.4f")) {
            material.npr.hatch_width = std::clamp(material.npr.hatch_width, 0.0001f, material.npr.hatch_spacing);
            mark_npr_dirty();
        }
        if (ImGui::SliderAngle("Angle", &material.npr.hatch_angle, -180.0f, 180.0f)) {
            mark_npr_dirty();
        }
        float range[2] = {material.npr.hatch_value_min, material.npr.hatch_value_max};
        if (ImGui::DragFloat2("Brightness Range", range, 0.01f, -100.0f, 100.0f, "%.2f")) {
            material.npr.hatch_value_min = range[0];
            material.npr.hatch_value_max = std::max(range[1], range[0] + 1.0e-4f);
            mark_npr_dirty();
        }
        color_edit_hdr("Ink", material.npr.hatch_ink, lt::RenderDirty::Material);
        color_edit_hdr("Paper", material.npr.hatch_paper, lt::RenderDirty::Material);
        if (ImGui::Checkbox("Passthrough", &material.npr.hatch_passthrough)) {
            mark_npr_dirty();
        }
        if (ImGui::Checkbox("Shadow Only", &material.npr.hatch_shadow_only)) {
            mark_npr_dirty();
        }
    }
}

void apply_blender_style() {
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 2.0f;
    style.ChildRounding = 2.0f;
    style.FrameRounding = 3.0f;
    style.PopupRounding = 3.0f;
    style.ScrollbarRounding = 3.0f;
    style.GrabRounding = 3.0f;
    style.TabRounding = 3.0f;
    style.WindowBorderSize = 1.0f;
    style.FrameBorderSize = 1.0f;
    style.WindowPadding = {8.0f, 7.0f};
    style.FramePadding = {7.0f, 4.0f};
    style.ItemSpacing = {7.0f, 6.0f};
    style.ItemInnerSpacing = {6.0f, 4.0f};
    ImVec4* c = style.Colors;
    c[ImGuiCol_Text] = ImVec4(0.86f, 0.86f, 0.84f, 1.0f);
    c[ImGuiCol_TextDisabled] = ImVec4(0.45f, 0.45f, 0.43f, 1.0f);
    c[ImGuiCol_WindowBg] = ImVec4(0.105f, 0.108f, 0.112f, 1.0f);
    c[ImGuiCol_ChildBg] = ImVec4(0.128f, 0.132f, 0.138f, 1.0f);
    c[ImGuiCol_PopupBg] = ImVec4(0.12f, 0.125f, 0.132f, 0.98f);
    c[ImGuiCol_Border] = ImVec4(0.25f, 0.255f, 0.26f, 1.0f);
    c[ImGuiCol_FrameBg] = ImVec4(0.165f, 0.17f, 0.18f, 1.0f);
    c[ImGuiCol_FrameBgHovered] = ImVec4(0.23f, 0.235f, 0.245f, 1.0f);
    c[ImGuiCol_FrameBgActive] = ImVec4(0.30f, 0.31f, 0.32f, 1.0f);
    c[ImGuiCol_TitleBg] = ImVec4(0.095f, 0.098f, 0.102f, 1.0f);
    c[ImGuiCol_TitleBgActive] = ImVec4(0.145f, 0.15f, 0.158f, 1.0f);
    c[ImGuiCol_MenuBarBg] = ImVec4(0.09f, 0.092f, 0.096f, 1.0f);
    c[ImGuiCol_Header] = ImVec4(0.95f, 0.50f, 0.16f, 0.28f);
    c[ImGuiCol_HeaderHovered] = ImVec4(0.95f, 0.50f, 0.16f, 0.42f);
    c[ImGuiCol_HeaderActive] = ImVec4(0.95f, 0.50f, 0.16f, 0.62f);
    c[ImGuiCol_Button] = ImVec4(0.18f, 0.185f, 0.195f, 1.0f);
    c[ImGuiCol_ButtonHovered] = ImVec4(0.27f, 0.275f, 0.285f, 1.0f);
    c[ImGuiCol_ButtonActive] = ImVec4(0.95f, 0.50f, 0.16f, 0.78f);
    c[ImGuiCol_Tab] = ImVec4(0.15f, 0.155f, 0.162f, 1.0f);
    c[ImGuiCol_TabHovered] = ImVec4(0.32f, 0.25f, 0.20f, 1.0f);
    c[ImGuiCol_TabActive] = ImVec4(0.23f, 0.20f, 0.17f, 1.0f);
    c[ImGuiCol_CheckMark] = ImVec4(0.95f, 0.55f, 0.18f, 1.0f);
    c[ImGuiCol_SliderGrab] = ImVec4(0.95f, 0.50f, 0.16f, 0.72f);
    c[ImGuiCol_SliderGrabActive] = ImVec4(1.0f, 0.62f, 0.25f, 1.0f);
}

// ---- Blender .dat icon loader ----

struct IconTriangle {
    uint8_t x0, y0, x1, y1, x2, y2;
    uint8_t r, g, b, a;
};

struct LoadedIcon {
    std::vector<IconTriangle> triangles;
};

std::unordered_map<std::string, LoadedIcon> g_blender_icons;

bool load_dat_icon(const std::string& path, const std::string& name) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;
    std::vector<char> raw((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (raw.size() < 8) return false;
    if (raw[0] != 'V' || raw[1] != 'C' || raw[2] != 'O') return false;

    const uint8_t* body = reinterpret_cast<const uint8_t*>(raw.data() + 8);
    const size_t body_size = raw.size() - 8;
    const size_t count = body_size / 18;

    if (count == 0) return false;

    LoadedIcon icon;
    icon.triangles.resize(count);
    const uint8_t* coords = body;
    const uint8_t* colors = body + count * 6;

    for (size_t i = 0; i < count; ++i) {
        IconTriangle& tri = icon.triangles[i];
        tri.x0 = coords[i * 6 + 0]; tri.y0 = coords[i * 6 + 1];
        tri.x1 = coords[i * 6 + 2]; tri.y1 = coords[i * 6 + 3];
        tri.x2 = coords[i * 6 + 4]; tri.y2 = coords[i * 6 + 5];
        tri.r = colors[i * 12 + 0]; tri.g = colors[i * 12 + 1];
        tri.b = colors[i * 12 + 2]; tri.a = colors[i * 12 + 3];
    }

    g_blender_icons[name] = std::move(icon);
    LT_LOG_DEBUG("Loaded Blender icon '{}': {} triangles", name, count);
    return true;
}

void load_blender_icons() {
    // Search paths for Blender icon directory
    std::vector<std::string> search;
    if (const char* env = std::getenv("BLENDER_ICONS_DIR")) search.push_back(env);
    search.push_back("F:/Steam/steamapps/common/Blender/4.5/datafiles/icons");
    search.push_back("C:/Program Files/Blender Foundation/Blender 4.5/datafiles/icons");
    search.push_back("./icons");

    // Also try to locate relative to executable
    wchar_t exe_path[MAX_PATH] = {};
    if (GetModuleFileNameW(nullptr, exe_path, MAX_PATH)) {
        std::filesystem::path exe_dir = std::filesystem::path(exe_path).parent_path();
        search.push_back((exe_dir / "icons").string());
    }

    std::string icon_dir;
    for (const auto& p : search) {
        if (std::filesystem::exists(p)) { icon_dir = p; break; }
    }
    if (icon_dir.empty()) {
        LT_LOG_INFO("No Blender icon directory found; using procedural fallback icons");
        return;
    }

    int loaded = 0;
    for (int i = 0; i < static_cast<int>(EditorIcon::ExitFullscreen) + 1; ++i) {
        const EditorIconInfo& info = editor_icon_info(static_cast<EditorIcon>(i));
        if (!info.blender_source) continue;
        std::string path = icon_dir + "/" + info.blender_source;
        if (load_dat_icon(path, info.blender_source)) ++loaded;
    }

    LT_LOG_INFO("Loaded {} Blender icons from '{}'", loaded, icon_dir);
    if (loaded == 0) LT_LOG_WARN("No Blender icons were loaded; verify the icon directory contains .dat files");
}

void draw_blender_icon(ImDrawList* draw_list, const LoadedIcon& icon, ImVec2 min, float size, ImU32 tint) {
    const float scale = size / 255.0f;
    const int tr = static_cast<int>((tint >> IM_COL32_R_SHIFT) & 0xFF);
    const int tg = static_cast<int>((tint >> IM_COL32_G_SHIFT) & 0xFF);
    const int tb = static_cast<int>((tint >> IM_COL32_B_SHIFT) & 0xFF);
    const int ta = static_cast<int>((tint >> IM_COL32_A_SHIFT) & 0xFF);

    for (const auto& tri : icon.triangles) {
        const ImVec2 p0{min.x + tri.x0 * scale, min.y + (255 - tri.y0) * scale};
        const ImVec2 p1{min.x + tri.x1 * scale, min.y + (255 - tri.y1) * scale};
        const ImVec2 p2{min.x + tri.x2 * scale, min.y + (255 - tri.y2) * scale};
        const ImU32 col = IM_COL32(
            (tri.r * tr) / 255,
            (tri.g * tg) / 255,
            (tri.b * tb) / 255,
            (tri.a * ta) / 255);
        draw_list->AddTriangleFilled(p0, p1, p2, col);
    }
}

void draw_editor_icon(ImDrawList* draw_list, EditorIcon icon, ImVec2 min, ImVec2 max, ImU32 color, ImU32 accent = 0) {
    if (!draw_list) return;

    // Prefer loaded Blender icon
    const EditorIconInfo& info = editor_icon_info(icon);
    if (info.blender_source) {
        auto it = g_blender_icons.find(info.blender_source);
        if (it != g_blender_icons.end()) {
            const float size = std::min(max.x - min.x, max.y - min.y);
            draw_blender_icon(draw_list, it->second, min, size, color);
            return;
        }
    }

    // Procedural fallback
    const float w = max.x - min.x;
    const float h = max.y - min.y;
    const float s = std::min(w, h);
    const ImVec2 c{(min.x + max.x) * 0.5f, (min.y + max.y) * 0.5f};
    const float lw = std::max(1.3f, s * 0.09f);
    if (accent == 0) accent = ImGui::GetColorU32(ImGuiCol_CheckMark);

    auto p = [&](float x, float y) { return ImVec2{min.x + x * w, min.y + y * h}; };
    auto line = [&](float ax, float ay, float bx, float by, ImU32 col = 0, float thick = 0.0f) {
        draw_list->AddLine(p(ax, ay), p(bx, by), col ? col : color, thick > 0.0f ? thick : lw);
    };
    auto rect = [&](float ax, float ay, float bx, float by, ImU32 col = 0) {
        draw_list->AddRect(p(ax, ay), p(bx, by), col ? col : color, 1.5f, 0, lw);
    };
    auto circle = [&](float x, float y, float r, ImU32 col = 0, int segments = 24) {
        draw_list->AddCircle(p(x, y), r * s, col ? col : color, segments, lw);
    };
    auto filled_circle = [&](float x, float y, float r, ImU32 col) {
        draw_list->AddCircleFilled(p(x, y), r * s, col, 24);
    };

    switch (icon) {
    case EditorIcon::Select:
        draw_list->AddTriangleFilled(p(0.24f, 0.14f), p(0.76f, 0.48f), p(0.45f, 0.56f), color);
        line(0.46f, 0.55f, 0.64f, 0.84f, color);
        line(0.55f, 0.68f, 0.72f, 0.61f, color);
        break;
    case EditorIcon::Move:
        line(0.18f, 0.50f, 0.82f, 0.50f);
        line(0.50f, 0.18f, 0.50f, 0.82f);
        draw_list->AddTriangleFilled(p(0.82f, 0.50f), p(0.66f, 0.40f), p(0.66f, 0.60f), accent);
        draw_list->AddTriangleFilled(p(0.18f, 0.50f), p(0.34f, 0.40f), p(0.34f, 0.60f), color);
        draw_list->AddTriangleFilled(p(0.50f, 0.18f), p(0.40f, 0.34f), p(0.60f, 0.34f), color);
        draw_list->AddTriangleFilled(p(0.50f, 0.82f), p(0.40f, 0.66f), p(0.60f, 0.66f), color);
        break;
    case EditorIcon::Rotate:
        draw_list->PathArcTo(c, s * 0.34f, -2.30f, 0.70f, 28);
        draw_list->PathStroke(color, 0, lw);
        draw_list->AddTriangleFilled(p(0.75f, 0.36f), p(0.88f, 0.55f), p(0.66f, 0.56f), accent);
        draw_list->PathArcTo(c, s * 0.24f, 0.95f, 3.40f, 22);
        draw_list->PathStroke(color, 0, lw);
        break;
    case EditorIcon::Scale:
        rect(0.24f, 0.24f, 0.76f, 0.76f);
        line(0.24f, 0.76f, 0.08f, 0.92f, accent);
        line(0.76f, 0.24f, 0.92f, 0.08f, accent);
        draw_list->AddRectFilled(p(0.05f, 0.85f), p(0.18f, 0.98f), accent, 1.5f);
        draw_list->AddRectFilled(p(0.82f, 0.02f), p(0.95f, 0.15f), accent, 1.5f);
        break;
    case EditorIcon::LocalSpace:
        line(0.24f, 0.76f, 0.76f, 0.24f, accent);
        line(0.24f, 0.76f, 0.57f, 0.76f);
        line(0.24f, 0.76f, 0.24f, 0.43f);
        break;
    case EditorIcon::WorldSpace:
        circle(0.50f, 0.50f, 0.34f);
        line(0.16f, 0.50f, 0.84f, 0.50f, accent);
        line(0.50f, 0.16f, 0.50f, 0.84f);
        break;
    case EditorIcon::Mesh:
    case EditorIcon::Cube:
        line(0.26f, 0.34f, 0.52f, 0.20f);
        line(0.52f, 0.20f, 0.76f, 0.35f);
        line(0.76f, 0.35f, 0.51f, 0.50f);
        line(0.51f, 0.50f, 0.26f, 0.34f);
        line(0.26f, 0.34f, 0.26f, 0.64f);
        line(0.51f, 0.50f, 0.51f, 0.80f, accent);
        line(0.76f, 0.35f, 0.76f, 0.65f);
        line(0.26f, 0.64f, 0.51f, 0.80f);
        line(0.51f, 0.80f, 0.76f, 0.65f);
        break;
    case EditorIcon::Plane:
        line(0.18f, 0.64f, 0.40f, 0.32f);
        line(0.40f, 0.32f, 0.86f, 0.38f);
        line(0.86f, 0.38f, 0.62f, 0.72f);
        line(0.62f, 0.72f, 0.18f, 0.64f);
        break;
    case EditorIcon::Sphere:
        circle(0.50f, 0.50f, 0.34f);
        circle(0.50f, 0.50f, 0.20f, accent);
        line(0.18f, 0.50f, 0.82f, 0.50f, accent, lw * 0.8f);
        break;
    case EditorIcon::Light:
        filled_circle(0.50f, 0.40f, 0.15f, accent);
        line(0.50f, 0.58f, 0.50f, 0.82f);
        line(0.34f, 0.82f, 0.66f, 0.82f);
        line(0.24f, 0.40f, 0.12f, 0.40f, accent, lw * 0.8f);
        line(0.76f, 0.40f, 0.88f, 0.40f, accent, lw * 0.8f);
        line(0.34f, 0.24f, 0.24f, 0.14f, accent, lw * 0.8f);
        line(0.66f, 0.24f, 0.76f, 0.14f, accent, lw * 0.8f);
        break;
    case EditorIcon::Material:
        filled_circle(0.40f, 0.42f, 0.17f, ImGui::GetColorU32(ImVec4(0.92f, 0.48f, 0.16f, 1.0f)));
        filled_circle(0.60f, 0.42f, 0.17f, ImGui::GetColorU32(ImVec4(0.25f, 0.58f, 0.95f, 1.0f)));
        filled_circle(0.50f, 0.61f, 0.17f, ImGui::GetColorU32(ImVec4(0.42f, 0.78f, 0.40f, 1.0f)));
        circle(0.50f, 0.50f, 0.34f, color);
        break;
    case EditorIcon::Texture:
        rect(0.22f, 0.24f, 0.78f, 0.76f);
        line(0.22f, 0.38f, 0.78f, 0.38f, accent, lw * 0.75f);
        line(0.38f, 0.24f, 0.38f, 0.76f, accent, lw * 0.75f);
        line(0.58f, 0.24f, 0.58f, 0.76f, accent, lw * 0.75f);
        break;
    case EditorIcon::Bake:
    case EditorIcon::Volume:
        rect(0.22f, 0.28f, 0.78f, 0.78f);
        line(0.30f, 0.28f, 0.42f, 0.13f, accent);
        line(0.50f, 0.28f, 0.62f, 0.13f, accent);
        line(0.70f, 0.28f, 0.82f, 0.13f, accent);
        break;
    case EditorIcon::Lightmap:
        rect(0.18f, 0.20f, 0.82f, 0.80f);
        line(0.18f, 0.50f, 0.82f, 0.50f, accent, lw * 0.8f);
        line(0.50f, 0.20f, 0.50f, 0.80f, accent, lw * 0.8f);
        filled_circle(0.34f, 0.36f, 0.07f, accent);
        break;
    case EditorIcon::Environment:
        circle(0.50f, 0.50f, 0.34f);
        line(0.18f, 0.55f, 0.36f, 0.40f, accent);
        line(0.36f, 0.40f, 0.52f, 0.58f, accent);
        line(0.52f, 0.58f, 0.78f, 0.34f, accent);
        break;
    case EditorIcon::Camera:
        rect(0.18f, 0.34f, 0.62f, 0.68f);
        draw_list->AddTriangle(p(0.62f, 0.42f), p(0.86f, 0.30f), p(0.86f, 0.76f), color, lw);
        filled_circle(0.38f, 0.51f, 0.08f, accent);
        break;
    case EditorIcon::Render:
        rect(0.18f, 0.24f, 0.82f, 0.76f);
        draw_list->AddTriangleFilled(p(0.43f, 0.36f), p(0.43f, 0.64f), p(0.66f, 0.50f), accent);
        break;
    case EditorIcon::Object:
        circle(0.50f, 0.50f, 0.34f);
        line(0.30f, 0.30f, 0.70f, 0.70f, accent);
        line(0.70f, 0.30f, 0.30f, 0.70f, accent);
        break;
    case EditorIcon::Add:
        line(0.50f, 0.20f, 0.50f, 0.80f, accent, lw * 1.2f);
        line(0.20f, 0.50f, 0.80f, 0.50f, accent, lw * 1.2f);
        break;
    case EditorIcon::Clear:
    case EditorIcon::Delete:
        line(0.26f, 0.26f, 0.74f, 0.74f, accent, lw * 1.1f);
        line(0.74f, 0.26f, 0.26f, 0.74f, accent, lw * 1.1f);
        break;
    case EditorIcon::Duplicate:
        rect(0.18f, 0.30f, 0.58f, 0.70f);
        rect(0.38f, 0.18f, 0.78f, 0.58f, accent);
        break;
    case EditorIcon::Cpu:
    case EditorIcon::Gpu:
        rect(0.24f, 0.24f, 0.76f, 0.76f);
        line(0.34f, 0.14f, 0.34f, 0.24f);
        line(0.50f, 0.14f, 0.50f, 0.24f);
        line(0.66f, 0.14f, 0.66f, 0.24f);
        line(0.34f, 0.76f, 0.34f, 0.86f);
        line(0.50f, 0.76f, 0.50f, 0.86f);
        line(0.66f, 0.76f, 0.66f, 0.86f);
        if (icon == EditorIcon::Gpu) line(0.34f, 0.50f, 0.66f, 0.50f, accent);
        break;
    case EditorIcon::RenderedPreview:
        filled_circle(0.50f, 0.50f, 0.32f, ImGui::GetColorU32(ImVec4(0.36f, 0.58f, 0.92f, 1.0f)));
        circle(0.50f, 0.50f, 0.32f, color);
        filled_circle(0.39f, 0.37f, 0.08f, ImGui::GetColorU32(ImVec4(0.88f, 0.94f, 1.0f, 0.92f)));
        break;
    case EditorIcon::MaterialPreview:
        filled_circle(0.50f, 0.50f, 0.32f, ImGui::GetColorU32(ImVec4(0.92f, 0.62f, 0.25f, 1.0f)));
        draw_list->PathArcTo(p(0.50f, 0.50f), s * 0.32f, -2.70f, 0.35f, 18);
        draw_list->PathLineTo(p(0.50f, 0.50f));
        draw_list->PathFillConvex(ImGui::GetColorU32(ImVec4(0.25f, 0.42f, 0.78f, 1.0f)));
        draw_list->PathArcTo(p(0.50f, 0.50f), s * 0.32f, 0.35f, 2.55f, 18);
        draw_list->PathLineTo(p(0.50f, 0.50f));
        draw_list->PathFillConvex(ImGui::GetColorU32(ImVec4(0.82f, 0.86f, 0.90f, 1.0f)));
        circle(0.50f, 0.50f, 0.32f, color);
        filled_circle(0.38f, 0.35f, 0.07f, ImGui::GetColorU32(ImVec4(1.0f, 0.96f, 0.78f, 0.95f)));
        break;
    case EditorIcon::SolidPreview:
        filled_circle(0.50f, 0.50f, 0.32f, ImGui::GetColorU32(ImVec4(0.58f, 0.60f, 0.64f, 1.0f)));
        draw_list->PathArcTo(p(0.50f, 0.50f), s * 0.32f, -1.57f, 1.57f, 18);
        draw_list->PathLineTo(p(0.50f, 0.82f));
        draw_list->PathLineTo(p(0.50f, 0.18f));
        draw_list->PathFillConvex(ImGui::GetColorU32(ImVec4(0.34f, 0.35f, 0.38f, 1.0f)));
        circle(0.50f, 0.50f, 0.32f, color);
        break;
    case EditorIcon::WireframePreview:
        circle(0.50f, 0.50f, 0.32f, color);
        circle(0.50f, 0.50f, 0.18f, accent);
        line(0.18f, 0.50f, 0.82f, 0.50f, accent, lw * 0.75f);
        line(0.50f, 0.18f, 0.50f, 0.82f, accent, lw * 0.75f);
        break;
    case EditorIcon::Fullscreen:
        line(0.22f, 0.40f, 0.22f, 0.22f);
        line(0.22f, 0.22f, 0.40f, 0.22f);
        line(0.60f, 0.22f, 0.78f, 0.22f);
        line(0.78f, 0.22f, 0.78f, 0.40f);
        line(0.78f, 0.60f, 0.78f, 0.78f);
        line(0.78f, 0.78f, 0.60f, 0.78f);
        line(0.40f, 0.78f, 0.22f, 0.78f);
        line(0.22f, 0.78f, 0.22f, 0.60f);
        break;
    case EditorIcon::ExitFullscreen:
        line(0.18f, 0.34f, 0.34f, 0.34f);
        line(0.34f, 0.34f, 0.34f, 0.18f);
        line(0.66f, 0.18f, 0.66f, 0.34f);
        line(0.66f, 0.34f, 0.82f, 0.34f);
        line(0.82f, 0.66f, 0.66f, 0.66f);
        line(0.66f, 0.66f, 0.66f, 0.82f);
        line(0.34f, 0.82f, 0.34f, 0.66f);
        line(0.34f, 0.66f, 0.18f, 0.66f);
        break;
    }
}

void draw_editor_icon(EditorIcon icon, ImVec2 min, float size, ImU32 color, ImU32 accent = 0) {
    draw_editor_icon(ImGui::GetWindowDrawList(), icon, min, {min.x + size, min.y + size}, color, accent);
}

bool icon_button(const char* id, EditorIcon icon, const char* tooltip, ImVec2 size, bool active = false) {
    ImGui::PushID(id);
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    const bool clicked = ImGui::InvisibleButton("##icon_button", size);
    const bool hovered = ImGui::IsItemHovered();
    const bool held = ImGui::IsItemActive();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    const ImU32 bg = ImGui::GetColorU32(active || held ? ImGuiCol_ButtonActive : hovered ? ImGuiCol_ButtonHovered : ImGuiCol_Button);
    const ImU32 border = ImGui::GetColorU32(ImGuiCol_Border);
    const ImU32 color = ImGui::GetColorU32(active ? ImVec4(1.0f, 0.93f, 0.82f, 1.0f) : ImGui::GetStyleColorVec4(ImGuiCol_Text));
    draw_list->AddRectFilled(pos, {pos.x + size.x, pos.y + size.y}, bg, 4.0f);
    draw_list->AddRect(pos, {pos.x + size.x, pos.y + size.y}, border, 4.0f);
    const float icon_size = std::min(size.x, size.y) - 12.0f;
    const ImVec2 icon_min{pos.x + (size.x - icon_size) * 0.5f, pos.y + (size.y - icon_size) * 0.5f};
    draw_editor_icon(draw_list, icon, icon_min, {icon_min.x + icon_size, icon_min.y + icon_size}, color);
    if (hovered && tooltip && tooltip[0]) {
        const EditorIconInfo& info = editor_icon_info(icon);
        if (info.blender_source) {
            ImGui::SetTooltip("%s\nBlender icon: %s", tooltip, info.blender_source);
        } else {
            ImGui::SetTooltip("%s", tooltip);
        }
    }
    ImGui::PopID();
    return clicked;
}

bool icon_text_button(const char* label, EditorIcon icon, ImVec2 size = {0.0f, 0.0f}) {
    const std::string text = std::string("     ") + label;
    const bool clicked = ImGui::Button(text.c_str(), size);
    const ImVec2 item_min = ImGui::GetItemRectMin();
    const ImVec2 item_size = ImGui::GetItemRectSize();
    const float icon_size = std::max(12.0f, ImGui::GetFrameHeight() - 8.0f);
    draw_editor_icon(icon, {item_min.x + 7.0f, item_min.y + (item_size.y - icon_size) * 0.5f}, icon_size, ImGui::GetColorU32(ImGuiCol_Text));
    return clicked;
}

void draw_menu_icon(EditorIcon icon) {
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    const float icon_size = ImGui::GetFrameHeight() - 7.0f;
    draw_editor_icon(icon, {pos.x + 2.0f, pos.y + 3.5f}, icon_size, ImGui::GetColorU32(ImGuiCol_Text));
    ImGui::Dummy({icon_size + 6.0f, ImGui::GetFrameHeight()});
    ImGui::SameLine(0.0f, 0.0f);
}

bool icon_menu_item(EditorIcon icon, const char* label, const char* shortcut = nullptr, bool enabled = true) {
    draw_menu_icon(icon);
    return ImGui::MenuItem(label, shortcut, false, enabled);
}

bool begin_icon_tab_item(const char* label, EditorIcon icon) {
    const std::string tab_label = std::string("      ") + label;
    const bool open = ImGui::BeginTabItem(tab_label.c_str());
    const ImVec2 item_min = ImGui::GetItemRectMin();
    const ImVec2 item_size = ImGui::GetItemRectSize();
    const float icon_size = std::max(12.0f, ImGui::GetFrameHeight() - 8.0f);
    draw_editor_icon(icon, {item_min.x + 7.0f, item_min.y + (item_size.y - icon_size) * 0.5f}, icon_size, ImGui::GetColorU32(ImGuiCol_Text));
    return open;
}

enum class PropertiesPanel { Object, Material, Camera, Render };

const char* properties_panel_name(PropertiesPanel panel) {
    switch (panel) {
    case PropertiesPanel::Object: return "Object";
    case PropertiesPanel::Material: return "Material";
    case PropertiesPanel::Camera: return "Camera";
    case PropertiesPanel::Render: return "Render";
    default: return "Properties";
    }
}

constexpr float kPropertiesIconBarWidth = 40.0f;
constexpr float kPropertiesIconButtonSize = 28.0f;
constexpr float kPropertiesIconBarPaddingX = (kPropertiesIconBarWidth - kPropertiesIconButtonSize) * 0.5f;

EditorIcon properties_panel_icon(PropertiesPanel panel) {
    switch (panel) {
    case PropertiesPanel::Object: return EditorIcon::Object;
    case PropertiesPanel::Material: return EditorIcon::Material;
    case PropertiesPanel::Camera: return EditorIcon::Camera;
    case PropertiesPanel::Render: return EditorIcon::Render;
    default: return EditorIcon::Object;
    }
}

void properties_panel_button(PropertiesPanel panel, PropertiesPanel& active_panel) {
    const bool active = active_panel == panel;
    if (icon_button(properties_panel_name(panel), properties_panel_icon(panel), properties_panel_name(panel), ImVec2(kPropertiesIconButtonSize, kPropertiesIconButtonSize), active)) {
        active_panel = panel;
    }
}

bool icon_tree_node(EditorIcon icon, const char* label, ImGuiTreeNodeFlags flags) {
    const std::string tree_label = std::string("   ") + label;
    const bool open = ImGui::TreeNodeEx(tree_label.c_str(), flags);
    const ImVec2 item_min = ImGui::GetItemRectMin();
    const ImVec2 item_size = ImGui::GetItemRectSize();
    const float icon_size = std::max(12.0f, ImGui::GetFrameHeight() - 8.0f);
    draw_editor_icon(icon, {item_min.x + 20.0f, item_min.y + (item_size.y - icon_size) * 0.5f}, icon_size, ImGui::GetColorU32(ImGuiCol_Text));
    return open;
}

bool icon_selectable(EditorIcon icon, const char* label, bool selected, bool light_badge = false) {
    const std::string item_label = std::string("     ") + label;
    const bool clicked = ImGui::Selectable(item_label.c_str(), selected);
    const ImVec2 item_min = ImGui::GetItemRectMin();
    const ImVec2 item_size = ImGui::GetItemRectSize();
    const float icon_size = std::max(12.0f, ImGui::GetFrameHeight() - 8.0f);
    draw_editor_icon(icon, {item_min.x + 5.0f, item_min.y + (item_size.y - icon_size) * 0.5f}, icon_size, ImGui::GetColorU32(ImGuiCol_Text));
    if (light_badge) {
        draw_editor_icon(EditorIcon::Light, {item_min.x + item_size.x - icon_size - 6.0f, item_min.y + (item_size.y - icon_size) * 0.5f}, icon_size, ImGui::GetColorU32(ImGuiCol_TextDisabled));
    }
    return clicked;
}

bool icon_collapsing_header(EditorIcon icon, const char* label, ImGuiTreeNodeFlags flags = 0) {
    const std::string header_label = std::string("   ") + label;
    const bool open = ImGui::CollapsingHeader(header_label.c_str(), flags);
    const ImVec2 item_min = ImGui::GetItemRectMin();
    const ImVec2 item_size = ImGui::GetItemRectSize();
    const float icon_size = std::max(12.0f, ImGui::GetFrameHeight() - 8.0f);
    draw_editor_icon(icon, {item_min.x + 21.0f, item_min.y + (item_size.y - icon_size) * 0.5f}, icon_size, ImGui::GetColorU32(ImGuiCol_Text));
    return open;
}

bool toolbar_button(ToolMode mode) {
    const bool active = g_editor.tool_mode == mode;
    const bool clicked = icon_button(tool_mode_name(mode), tool_mode_editor_icon(mode), tool_mode_name(mode), ImVec2(36.0f, 34.0f), active);
    if (clicked) {
        g_editor.tool_mode = mode;
    }
    return clicked;
}

float draw_top_bar() {
    const float height = ImGui::GetFrameHeight();
    if (!ImGui::BeginMainMenuBar()) return height;
    if (ImGui::BeginMenu("File")) {
        if (ImGui::MenuItem("New", "Ctrl+N")) new_scene();
        if (ImGui::MenuItem("Open")) open_scene_dialog();
        if (ImGui::MenuItem("Save", "Ctrl+S")) save_scene();
        if (ImGui::MenuItem("Save As...", "Ctrl+Shift+S")) save_scene_as_dialog();
        if (ImGui::MenuItem("Exit", "Esc")) PostQuitMessage(0);
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Object")) {
        if (ImGui::BeginMenu("Add Mesh")) {
            if (icon_menu_item(EditorIcon::Cube, "Cube")) add_cube_mesh();
            if (icon_menu_item(EditorIcon::Plane, "Plane")) add_plane_mesh();
            if (icon_menu_item(EditorIcon::Sphere, "UV Sphere")) add_uv_sphere_mesh();
            if (icon_menu_item(EditorIcon::Sphere, "Analytic Sphere")) add_analytic_sphere();
            ImGui::Separator();
            if (icon_menu_item(EditorIcon::Light, "Area Light")) add_area_light_mesh();
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Add Light")) {
            if (icon_menu_item(EditorIcon::Light, "Directional Light")) add_directional_light();
            if (icon_menu_item(EditorIcon::Light, "Point Light")) add_point_light();
            ImGui::EndMenu();
        }
        if (icon_menu_item(EditorIcon::Duplicate, "Duplicate", nullptr, has_selection())) duplicate_selected();
        if (icon_menu_item(EditorIcon::Delete, "Delete", "Del", has_selection())) delete_selected();
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Render")) {
        if (ImGui::MenuItem(g_editor.paused ? "Resume" : "Pause", "Space")) g_editor.paused = !g_editor.paused;
        if (ImGui::MenuItem("Reset Accumulation", "Ctrl+R")) reset_accumulation();
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Window")) {
        ImGui::MenuItem("Log", nullptr, &g_editor.show_log_panel);
        ImGui::MenuItem("Statistics", nullptr, &g_editor.show_statistics_panel);
        ImGui::EndMenu();
    }
    ImGui::Separator();
    const lt::IRenderer* effective_renderer =
        lt::stylized_rendering_enabled(g_editor.settings, g_editor.scene)
            ? static_cast<const lt::IRenderer*>(&g_editor.cpu)
            : g_editor.renderer;
    ImGui::Text("%s | %s | samples %u | %.2f ms | %s",
                tool_mode_name(g_editor.tool_mode),
                effective_renderer->name(),
                g_editor.frame_index,
                g_editor.last_sample_ms,
                g_editor.scene_path.c_str());
    ImGui::EndMainMenuBar();
    return height;
}

void draw_toolbar() {
    ImGui::Begin("Toolbar", nullptr, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4.0f, 6.0f));
    toolbar_button(ToolMode::Select);
    toolbar_button(ToolMode::Move);
    toolbar_button(ToolMode::Rotate);
    toolbar_button(ToolMode::Scale);
    ImGui::PopStyleVar();
    ImGui::Separator();
    const bool world = g_editor.move_space == TransformSpace::World;
    if (icon_button("TransformSpace", world ? EditorIcon::WorldSpace : EditorIcon::LocalSpace, world ? "World move gizmo space" : "Local move gizmo space", ImVec2(36.0f, 30.0f), world)) {
        g_editor.move_space = world ? TransformSpace::Local : TransformSpace::World;
    }
    ImGui::End();
}

void draw_outliner() {
    ImGui::Begin("Scene Collection", nullptr, ImGuiWindowFlags_NoCollapse);
    if (icon_tree_node(EditorIcon::Mesh, "Meshes", ImGuiTreeNodeFlags_DefaultOpen)) {
        for (int i = 0; i < static_cast<int>(g_editor.scene.meshes.size()); ++i) {
            const bool selected = g_editor.selection_kind == SelectionKind::Mesh && g_editor.selected_mesh == i;
            const lt::Mesh& mesh = g_editor.scene.meshes[static_cast<size_t>(i)];
            ImGui::PushID(i);
            if (icon_selectable(mesh.light.enabled ? EditorIcon::Light : EditorIcon::Mesh, mesh.name.c_str(), selected, mesh.light.enabled)) select_mesh(i);
            ImGui::PopID();
        }
        ImGui::TreePop();
    }
    if (icon_tree_node(EditorIcon::Sphere, "Spheres", ImGuiTreeNodeFlags_DefaultOpen)) {
        for (int i = 0; i < static_cast<int>(g_editor.scene.spheres.size()); ++i) {
            const bool selected = g_editor.selection_kind == SelectionKind::Sphere && g_editor.selected_sphere == i;
            const lt::Sphere& sphere = g_editor.scene.spheres[static_cast<size_t>(i)];
            ImGui::PushID(i);
            if (icon_selectable(EditorIcon::Sphere, sphere.name.c_str(), selected)) select_sphere(i);
            ImGui::PopID();
        }
        ImGui::TreePop();
    }
    if (icon_tree_node(EditorIcon::Light, "Lights", ImGuiTreeNodeFlags_DefaultOpen)) {
        for (int i = 0; i < static_cast<int>(g_editor.scene.directional_lights.size()); ++i) {
            const bool selected = g_editor.selection_kind == SelectionKind::DirectionalLight && g_editor.selected_directional_light == i;
            const lt::DirectionalLight& light = g_editor.scene.directional_lights[static_cast<size_t>(i)];
            ImGui::PushID(i);
            if (icon_selectable(EditorIcon::Light, light.name.c_str(), selected)) select_directional_light(i);
            ImGui::PopID();
        }
        for (int i = 0; i < static_cast<int>(g_editor.scene.point_lights.size()); ++i) {
            const bool selected = g_editor.selection_kind == SelectionKind::PointLight && g_editor.selected_point_light == i;
            const lt::PointLight& light = g_editor.scene.point_lights[static_cast<size_t>(i)];
            ImGui::PushID(i);
            if (icon_selectable(EditorIcon::Light, light.name.c_str(), selected)) select_point_light(i);
            ImGui::PopID();
        }
        ImGui::TreePop();
    }
    ImGui::End();
}

void draw_properties() {
    ImGui::Begin("Properties", nullptr, ImGuiWindowFlags_NoCollapse);
    static PropertiesPanel active_panel = PropertiesPanel::Object;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(kPropertiesIconBarPaddingX, 6.0f));
    ImGui::BeginChild("PropertiesPanelTabs", ImVec2(kPropertiesIconBarWidth, 0.0f), true, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 5.0f));
    properties_panel_button(PropertiesPanel::Object, active_panel);
    properties_panel_button(PropertiesPanel::Material, active_panel);
    properties_panel_button(PropertiesPanel::Camera, active_panel);
    properties_panel_button(PropertiesPanel::Render, active_panel);
    ImGui::PopStyleVar();
    ImGui::EndChild();
    ImGui::PopStyleVar();
    ImGui::SameLine(0.0f, 8.0f);
    ImGui::BeginChild("PropertiesPanelContent", ImVec2(0.0f, 0.0f), false);
    ImGui::SeparatorText(properties_panel_name(active_panel));
    if (active_panel == PropertiesPanel::Object) {
            if (has_mesh_selection()) {
                lt::Mesh& mesh = g_editor.scene.meshes[static_cast<size_t>(g_editor.selected_mesh)];
                char name[128] = {};
                std::strncpy(name, mesh.name.c_str(), sizeof(name) - 1);
                if (ImGui::InputText("Name", name, sizeof(name))) {
                    mesh.name = name;
                    g_editor.scene.uses_builtin_default_meshes = false;
                }
                if (ImGui::IsItemDeactivatedAfterEdit()) {
                    const std::string unique = unique_object_name(mesh.name, SelectionKind::Mesh, g_editor.selected_mesh);
                    if (mesh.name != unique) {
                        mesh.name = unique;
                        g_editor.scene.uses_builtin_default_meshes = false;
                    }
                }
                edit_location_world("Location", mesh);
                edit_rotation_degrees("Rotation", mesh.rotation);
                if (edit_vec3("Scale", mesh.scale, 0.01f, lt::RenderDirty::Transform)) {
                    mesh.scale.x = std::max(0.01f, mesh.scale.x);
                    mesh.scale.y = std::max(0.01f, mesh.scale.y);
                    mesh.scale.z = std::max(0.01f, mesh.scale.z);
                }
                if (ImGui::Checkbox("Exclude from IV Bake", &mesh.exclude_from_irradiance_volume_bake)) {
                    g_editor.scene.uses_builtin_default_meshes = false;
                    reset_accumulation(lt::RenderDirty::IrradianceVolume);
                }
                if (ImGui::Combo("Material", &mesh.material, [](void* data, int idx, const char** out) {
                    const auto* scene = static_cast<const lt::Scene*>(data);
                    *out = scene->materials[static_cast<size_t>(idx)]->name.c_str();
                    return true;
                }, &g_editor.scene, static_cast<int>(g_editor.scene.materials.size()))) reset_accumulation(lt::RenderDirty::Geometry);
                if (ImGui::Checkbox("Light", &mesh.light.enabled)) {
                    if (mesh.light.enabled) {
                        mesh.light.double_sided = false;
                        if (lt::dot(mesh.light.color, mesh.light.color) <= 0.0f) {
                            mesh.light.color = {1.0f, 0.888889f, 0.666667f};
                        }
                        if (mesh.light.intensity <= 1.0f) {
                            mesh.light.intensity = 9.0f;
                        }
                    }
                    g_editor.scene.uses_builtin_default_meshes = false;
                    reset_accumulation(lt::RenderDirty::Geometry | lt::RenderDirty::Material);
                }
                if (mesh.light.enabled) {
                    if (ImGui::Checkbox("Double-sided Light", &mesh.light.double_sided)) {
                        g_editor.scene.uses_builtin_default_meshes = false;
                        reset_accumulation(lt::RenderDirty::Geometry);
                    }
                    const lt::Vec3 old_color = mesh.light.color;
                    color_edit("Light Color", mesh.light.color, lt::RenderDirty::Geometry);
                    if (old_color.x != mesh.light.color.x || old_color.y != mesh.light.color.y || old_color.z != mesh.light.color.z) {
                        g_editor.scene.uses_builtin_default_meshes = false;
                    }
                    if (ImGui::DragFloat("Intensity", &mesh.light.intensity, 0.1f, 0.0f, 1000.0f, "%.2f")) {
                        mesh.light.intensity = std::max(0.0f, mesh.light.intensity);
                        g_editor.scene.uses_builtin_default_meshes = false;
                        reset_accumulation(lt::RenderDirty::Geometry);
                    }
                }
                ImGui::Text("Vertices: %d", static_cast<int>(mesh.vertices.size()));
                ImGui::Text("Triangles: %d", static_cast<int>(mesh.indices.size() / 3));
            } else if (has_sphere_selection()) {
                lt::Sphere& sphere = g_editor.scene.spheres[static_cast<size_t>(g_editor.selected_sphere)];
                char name[128] = {};
                std::strncpy(name, sphere.name.c_str(), sizeof(name) - 1);
                if (ImGui::InputText("Name", name, sizeof(name))) {
                    sphere.name = name;
                    g_editor.scene.uses_builtin_default_meshes = false;
                }
                if (ImGui::IsItemDeactivatedAfterEdit()) {
                    const std::string unique = unique_object_name(sphere.name, SelectionKind::Sphere, g_editor.selected_sphere);
                    if (sphere.name != unique) {
                        sphere.name = unique;
                        g_editor.scene.uses_builtin_default_meshes = false;
                    }
                }
                if (edit_vec3("Location", sphere.center, 0.02f, lt::RenderDirty::Transform)) {
                    g_editor.scene.uses_builtin_default_meshes = false;
                }
                if (ImGui::Checkbox("Exclude from IV Bake", &sphere.exclude_from_irradiance_volume_bake)) {
                    g_editor.scene.uses_builtin_default_meshes = false;
                    reset_accumulation(lt::RenderDirty::IrradianceVolume);
                }
                if (ImGui::DragFloat("Radius", &sphere.radius, 0.01f, 0.001f, 1000.0f, "%.3f")) {
                    sphere.radius = std::max(0.001f, sphere.radius);
                    g_editor.scene.uses_builtin_default_meshes = false;
                    reset_accumulation(lt::RenderDirty::Geometry);
                }
                if (ImGui::Combo("Material", &sphere.material, [](void* data, int idx, const char** out) {
                    const auto* scene = static_cast<const lt::Scene*>(data);
                    *out = scene->materials[static_cast<size_t>(idx)]->name.c_str();
                    return true;
                }, &g_editor.scene, static_cast<int>(g_editor.scene.materials.size()))) {
                    g_editor.scene.uses_builtin_default_meshes = false;
                    reset_accumulation(lt::RenderDirty::Geometry);
                }
                ImGui::TextDisabled("Analytic sphere");
            } else if (has_directional_light_selection()) {
                lt::DirectionalLight& light = g_editor.scene.directional_lights[static_cast<size_t>(g_editor.selected_directional_light)];
                char name[128] = {};
                std::strncpy(name, light.name.c_str(), sizeof(name) - 1);
                if (ImGui::InputText("Name", name, sizeof(name))) {
                    light.name = name;
                    g_editor.scene.uses_builtin_default_meshes = false;
                }
                if (ImGui::IsItemDeactivatedAfterEdit()) {
                    const std::string unique = unique_object_name(light.name, SelectionKind::DirectionalLight, g_editor.selected_directional_light);
                    if (light.name != unique) {
                        light.name = unique;
                        g_editor.scene.uses_builtin_default_meshes = false;
                    }
                }
                if (edit_vec3("Position", light.position, 0.02f, lt::RenderDirty::Transform)) {
                    g_editor.scene.uses_builtin_default_meshes = false;
                }
                // Direction as yaw/pitch in degrees (more intuitive than raw xyz)
                lt::Vec3 dir = lt::normalize(light.direction);
                if (lt::dot(dir, dir) <= 0.0f) dir = {0.0f, -1.0f, 0.0f};
                float yaw_deg = std::atan2(dir.x, dir.z) * 180.0f / lt::kPi;
                float pitch_deg = std::asin(std::clamp(dir.y, -1.0f, 1.0f)) * 180.0f / lt::kPi;
                bool dir_changed = false;
                if (ImGui::DragFloat("Yaw", &yaw_deg, 1.0f, -180.0f, 180.0f, "%.1f°")) dir_changed = true;
                if (ImGui::DragFloat("Pitch", &pitch_deg, 1.0f, -89.9f, 89.9f, "%.1f°")) dir_changed = true;
                if (dir_changed) {
                    const float yr = yaw_deg * lt::kPi / 180.0f;
                    const float pr = pitch_deg * lt::kPi / 180.0f;
                    light.direction = {std::cos(pr) * std::sin(yr), std::sin(pr), std::cos(pr) * std::cos(yr)};
                    g_editor.scene.uses_builtin_default_meshes = false;
                    reset_accumulation(lt::RenderDirty::Geometry);
                }
                color_edit("Color", light.color, lt::RenderDirty::Geometry);
                if (ImGui::DragFloat("Intensity", &light.intensity, 0.1f, 0.0f, 1000.0f, "%.2f")) {
                    light.intensity = std::max(0.0f, light.intensity);
                    g_editor.scene.uses_builtin_default_meshes = false;
                    reset_accumulation(lt::RenderDirty::Geometry);
                }
            } else if (has_point_light_selection()) {
                lt::PointLight& light = g_editor.scene.point_lights[static_cast<size_t>(g_editor.selected_point_light)];
                char name[128] = {};
                std::strncpy(name, light.name.c_str(), sizeof(name) - 1);
                if (ImGui::InputText("Name", name, sizeof(name))) {
                    light.name = name;
                    g_editor.scene.uses_builtin_default_meshes = false;
                }
                if (ImGui::IsItemDeactivatedAfterEdit()) {
                    const std::string unique = unique_object_name(light.name, SelectionKind::PointLight, g_editor.selected_point_light);
                    if (light.name != unique) {
                        light.name = unique;
                        g_editor.scene.uses_builtin_default_meshes = false;
                    }
                }
                if (edit_vec3("Position", light.position, 0.02f, lt::RenderDirty::Geometry)) {
                    g_editor.scene.uses_builtin_default_meshes = false;
                }
                color_edit("Color", light.color, lt::RenderDirty::Geometry);
                if (ImGui::DragFloat("Intensity", &light.intensity, 0.1f, 0.0f, 100000.0f, "%.2f")) {
                    light.intensity = std::max(0.0f, light.intensity);
                    g_editor.scene.uses_builtin_default_meshes = false;
                    reset_accumulation(lt::RenderDirty::Geometry);
                }
            } else {
                ImGui::TextDisabled("No object selected");
            }
    }
    if (active_panel == PropertiesPanel::Material) {
            if (icon_text_button("Add Material", EditorIcon::Add)) {
                add_material();
            }
            ImGui::Separator();
            for (int material_index = 0; material_index < static_cast<int>(g_editor.scene.materials.size()); ++material_index) {
                std::shared_ptr<lt::Material>& material = g_editor.scene.materials[static_cast<size_t>(material_index)];
                if (!material) {
                    continue;
                }
                ImGui::PushID(material_index);
                if (ImGui::TreeNodeEx(material->name.c_str())) {
                    color_edit("Base Color", material->albedo, lt::RenderDirty::Material);
                    int texture_index = 0;
                    if (material->albedo_texture) {
                        for (int i = 0; i < static_cast<int>(g_editor.scene.textures.size()); ++i) {
                            if (g_editor.scene.textures[static_cast<size_t>(i)] == material->albedo_texture) {
                                texture_index = i + 1;
                                break;
                            }
                        }
                    }
                    const auto texture_label = [](void* data, int idx, const char** out) {
                        const lt::Scene* scene = static_cast<const lt::Scene*>(data);
                        if (idx == 0) {
                            *out = "None";
                            return true;
                        }
                        const int texture_idx = idx - 1;
                        if (texture_idx < 0 || texture_idx >= static_cast<int>(scene->textures.size()) || !scene->textures[static_cast<size_t>(texture_idx)]) {
                            return false;
                        }
                        *out = scene->textures[static_cast<size_t>(texture_idx)]->name.c_str();
                        return true;
                    };
                    if (ImGui::Combo("Base Texture", &texture_index, texture_label, &g_editor.scene, static_cast<int>(g_editor.scene.textures.size()) + 1)) {
                        material->albedo_texture = texture_index > 0 ? g_editor.scene.textures[static_cast<size_t>(texture_index - 1)] : nullptr;
                        if (auto* standard = dynamic_cast<lt::StandardSurfaceMaterial*>(material.get())) {
                            standard->base_color_input.texture = material->albedo_texture;
                            standard->base_color_input.role = lt::TextureRole::Color;
                            standard->base_color_input.color_space = lt::TextureColorSpace::SceneLinear;
                        }
                        reset_accumulation(lt::RenderDirty::Material);
                    }
                    if (icon_text_button("Load Texture", EditorIcon::Texture)) {
                        load_texture_dialog(*material);
                    }
                    if (ImGui::Checkbox("Double-sided Material", &material->double_sided)) {
                        reset_accumulation(lt::RenderDirty::Material);
                    }
                    const lt::MaterialSystemStatus material_system = lt::material_system_status();
                    ImGui::TextDisabled("MaterialX %s | OIIO %s | OCIO %s",
                        material_system.materialx_available ? "on" : "fallback",
                        material_system.openimageio_available ? "on" : "fallback",
                        material_system.opencolorio_available ? "on" : "fallback");
                    const char* models[] = {"Lambertian", "Principled", "Mirror", "Dielectric", "Conductor", "Standard Surface", "Diffuse Transmission"};
                    int model = static_cast<int>(material->model());
                    if (ImGui::Combo("BRDF", &model, models, IM_ARRAYSIZE(models))) {
                        const std::string name = material->name;
                        const lt::Vec3 albedo = material->albedo;
                        float roughness = 0.5f;
                        float metallic = 0.0f;
                        if (const auto* principled = dynamic_cast<const lt::PrincipledMaterial*>(material.get())) {
                            roughness = principled->roughness;
                            metallic = principled->metallic;
                        } else if (const auto* dielectric = dynamic_cast<const lt::DielectricMaterial*>(material.get())) {
                            roughness = dielectric->ior;
                        } else if (const auto* standard = dynamic_cast<const lt::StandardSurfaceMaterial*>(material.get())) {
                            roughness = standard->roughness;
                            metallic = standard->metalness;
                        }
                        const auto next_model = static_cast<lt::BrdfModel>(model);
                        if (next_model == lt::BrdfModel::Dielectric && material->model() != lt::BrdfModel::Dielectric) {
                            roughness = 1.5f;
                        } else if (next_model != lt::BrdfModel::Dielectric && material->model() == lt::BrdfModel::Dielectric) {
                            roughness = 0.5f;
                        }
                        std::shared_ptr<lt::Texture> base_texture = material->albedo_texture;
                        std::shared_ptr<lt::Texture> normal_texture = material->normal_texture;
                        std::shared_ptr<lt::Texture> emission_texture = material->emission_texture;
                        const float alpha = material->alpha;
                        const float alpha_cutoff = material->alpha_cutoff;
                        const lt::AlphaMode alpha_mode = material->alpha_mode;
                        const bool double_sided = material->double_sided;
                        const float normal_scale = material->normal_scale;
                        const lt::Vec3 emission = material->emission;
                        const lt::NprSettings npr = material->npr;
                        lt::Vec3 sheen_color;
                        float sheen_roughness = 0.0f;
                        std::shared_ptr<lt::Texture> sheen_color_texture;
                        std::shared_ptr<lt::Texture> sheen_roughness_texture;
                        float clearcoat = 0.0f;
                        float clearcoat_roughness = 0.0f;
                        std::shared_ptr<lt::Texture> clearcoat_texture;
                        std::shared_ptr<lt::Texture> clearcoat_roughness_texture;
                        if (const auto* principled = dynamic_cast<const lt::PrincipledMaterial*>(material.get())) {
                            sheen_color = principled->sheen_color;
                            sheen_roughness = principled->sheen_roughness;
                            sheen_color_texture = principled->sheen_color_texture;
                            sheen_roughness_texture = principled->sheen_roughness_texture;
                            clearcoat = principled->clearcoat;
                            clearcoat_roughness = principled->clearcoat_roughness;
                            clearcoat_texture = principled->clearcoat_texture;
                            clearcoat_roughness_texture = principled->clearcoat_roughness_texture;
                        }
                        material = lt::make_material(name, albedo, next_model, roughness, metallic);
                        material->albedo_texture = std::move(base_texture);
                        material->normal_texture = std::move(normal_texture);
                        material->emission_texture = std::move(emission_texture);
                        material->alpha = alpha;
                        material->alpha_cutoff = alpha_cutoff;
                        material->alpha_mode = alpha_mode;
                        material->double_sided = double_sided;
                        material->normal_scale = normal_scale;
                        material->emission = emission;
                        material->npr = npr;
                        if (auto* principled = dynamic_cast<lt::PrincipledMaterial*>(material.get())) {
                            principled->sheen_color = sheen_color;
                            principled->sheen_roughness = sheen_roughness;
                            principled->sheen_color_texture = std::move(sheen_color_texture);
                            principled->sheen_roughness_texture = std::move(sheen_roughness_texture);
                            principled->clearcoat = clearcoat;
                            principled->clearcoat_roughness = clearcoat_roughness;
                            principled->clearcoat_texture = std::move(clearcoat_texture);
                            principled->clearcoat_roughness_texture = std::move(clearcoat_roughness_texture);
                        }
                        reset_accumulation(lt::RenderDirty::Material);
                    }
                    if (material->model() == lt::BrdfModel::Mirror) {
                        ImGui::TextDisabled("Mirror is an untinted ideal specular reflector.");
                    }
                    if (material->model() == lt::BrdfModel::Conductor) {
                        ImGui::TextDisabled("Conductor uses Base Color as ideal metal reflectance tint.");
                    }
                    if (auto* principled = dynamic_cast<lt::PrincipledMaterial*>(material.get())) {
                        if (ImGui::DragFloat("Roughness", &principled->roughness, 0.01f, 0.02f, 1.0f, "%.2f")) {
                            principled->roughness = std::clamp(principled->roughness, 0.02f, 1.0f);
                            reset_accumulation(lt::RenderDirty::Material);
                        }
                        if (ImGui::DragFloat("Metallic", &principled->metallic, 0.01f, 0.0f, 1.0f, "%.2f")) {
                            principled->metallic = std::clamp(principled->metallic, 0.0f, 1.0f);
                            reset_accumulation(lt::RenderDirty::Material);
                        }
                        if (ImGui::TreeNode("Sheen")) {
                            if (color_edit("Sheen Color", principled->sheen_color, lt::RenderDirty::Material)) {
                                principled->sheen_color = lt::clamp(principled->sheen_color);
                            }
                            if (ImGui::DragFloat("Sheen Roughness", &principled->sheen_roughness, 0.01f, 0.0f, 1.0f, "%.2f")) {
                                principled->sheen_roughness = std::clamp(principled->sheen_roughness, 0.0f, 1.0f);
                                reset_accumulation(lt::RenderDirty::Material);
                            }
                            int sheen_color_texture = 0;
                            int sheen_roughness_texture = 0;
                            for (int i = 0; i < static_cast<int>(g_editor.scene.textures.size()); ++i) {
                                if (g_editor.scene.textures[static_cast<size_t>(i)] == principled->sheen_color_texture) {
                                    sheen_color_texture = i + 1;
                                }
                                if (g_editor.scene.textures[static_cast<size_t>(i)] == principled->sheen_roughness_texture) {
                                    sheen_roughness_texture = i + 1;
                                }
                            }
                            if (ImGui::Combo("Sheen Color Texture", &sheen_color_texture, texture_label, &g_editor.scene, static_cast<int>(g_editor.scene.textures.size()) + 1)) {
                                principled->sheen_color_texture = sheen_color_texture > 0 ? g_editor.scene.textures[static_cast<size_t>(sheen_color_texture - 1)] : nullptr;
                                reset_accumulation(lt::RenderDirty::Material);
                            }
                            if (ImGui::Combo("Sheen Roughness Texture", &sheen_roughness_texture, texture_label, &g_editor.scene, static_cast<int>(g_editor.scene.textures.size()) + 1)) {
                                principled->sheen_roughness_texture = sheen_roughness_texture > 0 ? g_editor.scene.textures[static_cast<size_t>(sheen_roughness_texture - 1)] : nullptr;
                                reset_accumulation(lt::RenderDirty::Material);
                            }
                            ImGui::TreePop();
                        }
                        if (ImGui::TreeNode("Clearcoat")) {
                            if (ImGui::DragFloat("Clearcoat", &principled->clearcoat, 0.01f, 0.0f, 1.0f, "%.2f")) {
                                principled->clearcoat = std::clamp(principled->clearcoat, 0.0f, 1.0f);
                                reset_accumulation(lt::RenderDirty::Material);
                            }
                            if (ImGui::DragFloat("Clearcoat Roughness", &principled->clearcoat_roughness, 0.01f, 0.0f, 1.0f, "%.2f")) {
                                principled->clearcoat_roughness = std::clamp(principled->clearcoat_roughness, 0.0f, 1.0f);
                                reset_accumulation(lt::RenderDirty::Material);
                            }
                            int clearcoat_texture = 0;
                            int clearcoat_roughness_texture = 0;
                            for (int i = 0; i < static_cast<int>(g_editor.scene.textures.size()); ++i) {
                                if (g_editor.scene.textures[static_cast<size_t>(i)] == principled->clearcoat_texture) {
                                    clearcoat_texture = i + 1;
                                }
                                if (g_editor.scene.textures[static_cast<size_t>(i)] == principled->clearcoat_roughness_texture) {
                                    clearcoat_roughness_texture = i + 1;
                                }
                            }
                            if (ImGui::Combo("Clearcoat Texture", &clearcoat_texture, texture_label, &g_editor.scene, static_cast<int>(g_editor.scene.textures.size()) + 1)) {
                                principled->clearcoat_texture = clearcoat_texture > 0 ? g_editor.scene.textures[static_cast<size_t>(clearcoat_texture - 1)] : nullptr;
                                reset_accumulation(lt::RenderDirty::Material);
                            }
                            if (ImGui::Combo("Clearcoat Roughness Texture", &clearcoat_roughness_texture, texture_label, &g_editor.scene, static_cast<int>(g_editor.scene.textures.size()) + 1)) {
                                principled->clearcoat_roughness_texture = clearcoat_roughness_texture > 0 ? g_editor.scene.textures[static_cast<size_t>(clearcoat_roughness_texture - 1)] : nullptr;
                                reset_accumulation(lt::RenderDirty::Material);
                            }
                            ImGui::TreePop();
                        }
                    }
                    if (auto* dielectric = dynamic_cast<lt::DielectricMaterial*>(material.get())) {
                        if (ImGui::DragFloat("IOR", &dielectric->ior, 0.01f, 1.0f, 3.0f, "%.2f")) {
                            dielectric->ior = std::clamp(dielectric->ior, 1.0f, 3.0f);
                            reset_accumulation(lt::RenderDirty::Material);
                        }
                    }
                    if (auto* standard = dynamic_cast<lt::StandardSurfaceMaterial*>(material.get())) {
                        if (ImGui::DragFloat("Roughness", &standard->roughness, 0.01f, 0.02f, 1.0f, "%.2f")) {
                            standard->roughness = std::clamp(standard->roughness, 0.02f, 1.0f);
                            reset_accumulation(lt::RenderDirty::Material);
                        }
                        if (ImGui::DragFloat("Metalness", &standard->metalness, 0.01f, 0.0f, 1.0f, "%.2f")) {
                            standard->metalness = std::clamp(standard->metalness, 0.0f, 1.0f);
                            reset_accumulation(lt::RenderDirty::Material);
                        }
                        if (ImGui::DragFloat("Specular Weight", &standard->specular_weight, 0.01f, 0.0f, 2.0f, "%.2f")) {
                            standard->specular_weight = std::max(0.0f, standard->specular_weight);
                            reset_accumulation(lt::RenderDirty::Material);
                        }
                        if (ImGui::DragFloat("Transmission", &standard->transmission_weight, 0.01f, 0.0f, 1.0f, "%.2f")) {
                            standard->transmission_weight = std::clamp(standard->transmission_weight, 0.0f, 1.0f);
                            reset_accumulation(lt::RenderDirty::Material);
                        }
                        if (ImGui::DragFloat("Specular IOR", &standard->specular_ior, 0.01f, 1.0f, 3.0f, "%.2f")) {
                            standard->specular_ior = std::clamp(standard->specular_ior, 1.0f, 3.0f);
                            reset_accumulation(lt::RenderDirty::Material);
                        }
                        if (color_edit("Transmission Color", standard->transmission_color, lt::RenderDirty::Material)) {
                            standard->transmission_color = lt::clamp(standard->transmission_color);
                        }
                        auto material_input_texture_combo = [&](const char* label, lt::MaterialInput& input, lt::MaterialInputChannel channel) {
                            int texture = 0;
                            for (int i = 0; i < static_cast<int>(g_editor.scene.textures.size()); ++i) {
                                if (g_editor.scene.textures[static_cast<size_t>(i)] == input.texture) {
                                    texture = i + 1;
                                    break;
                                }
                            }
                            if (ImGui::Combo(label, &texture, texture_label, &g_editor.scene, static_cast<int>(g_editor.scene.textures.size()) + 1)) {
                                input.texture = texture > 0 ? g_editor.scene.textures[static_cast<size_t>(texture - 1)] : nullptr;
                                input.channel = channel;
                                input.role = lt::TextureRole::Data;
                                input.color_space = lt::TextureColorSpace::Raw;
                                if (input.texture) {
                                    lt::apply_texture_role(*input.texture, input.role, input.color_space);
                                }
                                reset_accumulation(lt::RenderDirty::Material);
                            }
                        };
                        material_input_texture_combo("Roughness Texture", standard->roughness_input, lt::MaterialInputChannel::R);
                        material_input_texture_combo("Metalness Texture", standard->metalness_input, lt::MaterialInputChannel::R);
                        material_input_texture_combo("Specular Texture", standard->specular_weight_input, lt::MaterialInputChannel::R);
                        if (ImGui::TreeNode("Coat")) {
                            if (ImGui::DragFloat("Coat Weight", &standard->coat_weight, 0.01f, 0.0f, 1.0f, "%.2f")) {
                                standard->coat_weight = std::clamp(standard->coat_weight, 0.0f, 1.0f);
                                reset_accumulation(lt::RenderDirty::Material);
                            }
                            if (ImGui::DragFloat("Coat Roughness", &standard->coat_roughness, 0.01f, 0.0f, 1.0f, "%.2f")) {
                                standard->coat_roughness = std::clamp(standard->coat_roughness, 0.0f, 1.0f);
                                reset_accumulation(lt::RenderDirty::Material);
                            }
                            ImGui::TreePop();
                        }
                        if (ImGui::TreeNode("Sheen")) {
                            if (color_edit("Sheen Color", standard->sheen_color, lt::RenderDirty::Material)) {
                                standard->sheen_color = lt::clamp(standard->sheen_color);
                            }
                            if (ImGui::DragFloat("Sheen Weight", &standard->sheen_weight, 0.01f, 0.0f, 1.0f, "%.2f")) {
                                standard->sheen_weight = std::clamp(standard->sheen_weight, 0.0f, 1.0f);
                                reset_accumulation(lt::RenderDirty::Material);
                            }
                            if (ImGui::DragFloat("Sheen Roughness", &standard->sheen_roughness, 0.01f, 0.0f, 1.0f, "%.2f")) {
                                standard->sheen_roughness = std::clamp(standard->sheen_roughness, 0.0f, 1.0f);
                                reset_accumulation(lt::RenderDirty::Material);
                            }
                            ImGui::TreePop();
                        }
                    }
                    draw_npr_controls(*material);
                    ImGui::TreePop();
                }
                ImGui::PopID();
            }
    }
    if (active_panel == PropertiesPanel::Camera) {
            edit_vec3("Position", g_editor.scene.camera.position, 0.02f, lt::RenderDirty::Camera);
            edit_vec3("Target", g_editor.scene.camera.target, 0.02f, lt::RenderDirty::Camera);
            if (ImGui::DragFloat("FOV", &g_editor.scene.camera.fov_degrees, 0.2f, 10.0f, 120.0f, "%.1f")) {
                g_editor.scene.camera.fov_degrees = std::clamp(g_editor.scene.camera.fov_degrees, 10.0f, 120.0f);
                reset_accumulation(lt::RenderDirty::Camera);
            }
    }
    if (active_panel == PropertiesPanel::Render) {
            const bool using_cuda = g_editor.renderer == &g_editor.cuda;
            const char* renderer_name = using_cuda ? "CUDA Path Tracer" : "CPU Path Tracer";
            if (ImGui::BeginCombo("Renderer", renderer_name)) {
                if (icon_selectable(EditorIcon::Cpu, "CPU Path Tracer", !using_cuda)) {
                    set_renderer(false);
                }
                const bool cuda_available = g_editor.cuda.available();
                const bool cuda_disabled = !cuda_available ||
                    lt::stylized_rendering_enabled(g_editor.settings, g_editor.scene);
                ImGui::BeginDisabled(cuda_disabled);
                if (icon_selectable(EditorIcon::Gpu, "CUDA Path Tracer", using_cuda)) {
                    set_renderer(true);
                }
                ImGui::EndDisabled();
                ImGui::EndCombo();
            }
            const char* sampling_modes[] = {"UNI", "NEE", "MIS"};
            int sampling_mode = static_cast<int>(g_editor.settings.sampling_mode);
            if (ImGui::Combo("Sampling", &sampling_mode, sampling_modes, IM_ARRAYSIZE(sampling_modes))) {
                g_editor.settings.sampling_mode = static_cast<lt::PathSamplingMode>(sampling_mode);
                reset_accumulation();
            }
            ImGui::BeginDisabled(!lt::uses_multiple_importance_sampling(g_editor.settings));
            const char* heuristics[] = {"Balance", "Power"};
            int heuristic = static_cast<int>(g_editor.settings.mis_heuristic);
            if (ImGui::Combo("MIS heuristic", &heuristic, heuristics, IM_ARRAYSIZE(heuristics))) {
                g_editor.settings.mis_heuristic = heuristic == 0 ? lt::MisHeuristic::Balance : lt::MisHeuristic::Power;
                reset_accumulation();
            }
            ImGui::EndDisabled();
            if (ImGui::DragInt("Samples / frame", &g_editor.settings.samples_per_pixel, 1.0f, 1, 64)) {
                g_editor.settings.samples_per_pixel = std::clamp(g_editor.settings.samples_per_pixel, 1, 64);
                reset_accumulation();
            }
            if (ImGui::DragInt("Max bounces", &g_editor.settings.max_bounces, 1.0f, 1, 32)) {
                g_editor.settings.max_bounces = std::clamp(g_editor.settings.max_bounces, 1, 32);
                reset_accumulation();
            }
            const char* denoisers[] = {"Off", "SVGF"};
            int denoiser = static_cast<int>(g_editor.settings.denoiser_mode);
            if (ImGui::Combo("Denoiser", &denoiser, denoisers, IM_ARRAYSIZE(denoisers))) {
                g_editor.settings.denoiser_mode = static_cast<lt::DenoiserMode>(denoiser);
                if (lt::svgf_denoising_enabled(g_editor.settings)) {
                    g_editor.settings.max_bounces = 2;
                }
                reset_accumulation();
            }
            if (lt::svgf_denoising_enabled(g_editor.settings) && icon_collapsing_header(EditorIcon::Render, "SVGF")) {
                if (ImGui::Checkbox("Rasterized G-Buffer", &g_editor.settings.svgf_rasterized_gbuffer)) {
                    reset_accumulation();
                }
                const char* aa_modes[] = {"Off", "Stable Post AA", "TAA"};
                int aa_mode = static_cast<int>(g_editor.settings.antialiasing_mode);
                aa_mode = std::clamp(aa_mode, 0, IM_ARRAYSIZE(aa_modes) - 1);
                if (ImGui::Combo("Antialiasing", &aa_mode, aa_modes, IM_ARRAYSIZE(aa_modes))) {
                    g_editor.settings.antialiasing_mode = static_cast<lt::AntialiasingMode>(aa_mode);
                    reset_accumulation();
                }
                const char* debug_views[] = {"Final", "Raw", "Albedo", "Normal", "Depth", "Variance", "History"};
                int debug_view = static_cast<int>(g_editor.settings.svgf_debug_view);
                if (ImGui::Combo("Debug View", &debug_view, debug_views, IM_ARRAYSIZE(debug_views))) {
                    g_editor.settings.svgf_debug_view = static_cast<lt::DenoiserDebugView>(debug_view);
                    reset_accumulation();
                }
                if (ImGui::DragInt("SVGF Iterations", &g_editor.settings.svgf_iterations, 1.0f, 0, 8)) {
                    g_editor.settings.svgf_iterations = std::clamp(g_editor.settings.svgf_iterations, 0, 8);
                    reset_accumulation();
                }
                if (ImGui::DragFloat("Alpha", &g_editor.settings.svgf_alpha, 0.002f, 0.0f, 1.0f, "%.3f")) {
                    g_editor.settings.svgf_alpha = std::clamp(g_editor.settings.svgf_alpha, 0.0f, 1.0f);
                    reset_accumulation();
                }
                if (ImGui::DragFloat("Moments Alpha", &g_editor.settings.svgf_moments_alpha, 0.002f, 0.0f, 1.0f, "%.3f")) {
                    g_editor.settings.svgf_moments_alpha = std::clamp(g_editor.settings.svgf_moments_alpha, 0.0f, 1.0f);
                    reset_accumulation();
                }
                if (ImGui::DragFloat("Phi Color", &g_editor.settings.svgf_phi_color, 0.05f, 0.001f, 100.0f, "%.3f")) {
                    g_editor.settings.svgf_phi_color = std::max(0.001f, g_editor.settings.svgf_phi_color);
                    reset_accumulation();
                }
                if (ImGui::DragFloat("Phi Normal", &g_editor.settings.svgf_phi_normal, 1.0f, 0.001f, 512.0f, "%.3f")) {
                    g_editor.settings.svgf_phi_normal = std::max(0.001f, g_editor.settings.svgf_phi_normal);
                    reset_accumulation();
                }
                if (ImGui::DragFloat("Phi Depth", &g_editor.settings.svgf_phi_depth, 0.02f, 0.001f, 16.0f, "%.3f")) {
                    g_editor.settings.svgf_phi_depth = std::max(0.001f, g_editor.settings.svgf_phi_depth);
                    reset_accumulation();
                }
            }
            ImGui::Checkbox("Hide Wireframes", &g_editor.hide_dirty_wireframes);
            const char* accel_modes[] = {"Auto", "Flat BVH", "Two-level BVH"};
            int accel_mode = static_cast<int>(g_editor.settings.acceleration_structure);
            if (ImGui::Combo("Acceleration", &accel_mode, accel_modes, IM_ARRAYSIZE(accel_modes))) {
                g_editor.settings.acceleration_structure = static_cast<lt::AccelerationStructure>(accel_mode);
                reset_accumulation(lt::RenderDirty::Geometry);
            }
            if (icon_collapsing_header(EditorIcon::Volume, "Irradiance Volume")) {
                if (ImGui::Checkbox("Enable Irradiance Volume", &g_editor.settings.use_irradiance_volume)) {
                    reset_accumulation(lt::RenderDirty::IrradianceVolume);
                }
                ImGui::BeginDisabled(!g_editor.settings.use_irradiance_volume);
                if (ImGui::Checkbox("Disk Cache", &g_editor.settings.irradiance_volume_cache_enabled)) {
                    reset_accumulation(lt::RenderDirty::IrradianceVolume);
                }
                if (ImGui::Checkbox("Auto Update", &g_editor.settings.irradiance_volume_auto_update)) {
                    reset_accumulation(lt::RenderDirty::IrradianceVolume);
                }
                if (icon_text_button("Update Volume", EditorIcon::Bake)) {
                    g_editor.settings.irradiance_volume_force_rebake = true;
                    reset_accumulation(lt::RenderDirty::IrradianceVolume);
                }
                const std::string cache_path = default_irradiance_volume_cache_path();
                ImGui::TextWrapped("Cache: %s", cache_path.c_str());
                if (ImGui::DragInt("Volume Grid", &g_editor.settings.irradiance_volume_grid_resolution, 1.0f, 2, 16)) {
                    g_editor.settings.irradiance_volume_grid_resolution = std::clamp(g_editor.settings.irradiance_volume_grid_resolution, 2, 64);
                    reset_accumulation(lt::RenderDirty::IrradianceVolume);
                }
                if (ImGui::DragInt("Subgrid", &g_editor.settings.irradiance_volume_subgrid_resolution, 1.0f, 2, 8)) {
                    g_editor.settings.irradiance_volume_subgrid_resolution = std::clamp(g_editor.settings.irradiance_volume_subgrid_resolution, 2, 32);
                    reset_accumulation(lt::RenderDirty::IrradianceVolume);
                }
                if (ImGui::DragInt("Direction Grid", &g_editor.settings.irradiance_volume_direction_resolution, 1.0f, 1, 16)) {
                    g_editor.settings.irradiance_volume_direction_resolution = std::clamp(g_editor.settings.irradiance_volume_direction_resolution, 1, 32);
                    reset_accumulation(lt::RenderDirty::IrradianceVolume);
                }
                if (ImGui::DragInt("Bake Samples", &g_editor.settings.irradiance_volume_bake_samples, 1.0f, 1, 16)) {
                    g_editor.settings.irradiance_volume_bake_samples = std::clamp(g_editor.settings.irradiance_volume_bake_samples, 1, 256);
                    reset_accumulation(lt::RenderDirty::IrradianceVolume);
                }
                if (ImGui::DragInt("Bake Bounces", &g_editor.settings.irradiance_volume_bake_bounces, 1.0f, 1, 16)) {
                    g_editor.settings.irradiance_volume_bake_bounces = std::clamp(g_editor.settings.irradiance_volume_bake_bounces, 1, 32);
                    reset_accumulation(lt::RenderDirty::IrradianceVolume);
                }
                int bake_backend = static_cast<int>(g_editor.settings.irradiance_volume_bake_backend);
                if (ImGui::Combo("Bake Backend", &bake_backend, "GPU\0CPU\0\0")) {
                    g_editor.settings.irradiance_volume_bake_backend = static_cast<lt::IrradianceVolumeBakeBackend>(bake_backend);
                    reset_accumulation(lt::RenderDirty::IrradianceVolume);
                }
                if (ImGui::DragFloat("Bounds Inset", &g_editor.settings.irradiance_volume_bounds_inset, 0.002f, 0.0f, 0.45f, "%.3f")) {
                    g_editor.settings.irradiance_volume_bounds_inset = std::clamp(g_editor.settings.irradiance_volume_bounds_inset, 0.0f, 0.45f);
                    reset_accumulation(lt::RenderDirty::IrradianceVolume);
                }
                if (ImGui::Checkbox("Principled GI", &g_editor.settings.irradiance_volume_principled_gi)) {
                    reset_accumulation();
                }
                if (ImGui::Checkbox("Debug Probes", &g_editor.settings.irradiance_volume_debug_probes)) {
                    if (g_editor.settings.irradiance_volume_debug_probes) {
                        g_editor.settings.use_irradiance_volume = true;
                    }
                    reset_accumulation();
                }
                if (ImGui::DragFloat("Probe Radius", &g_editor.settings.irradiance_volume_debug_probe_radius_scale, 0.01f, 0.0f, 2.0f, "%.2f")) {
                    g_editor.settings.irradiance_volume_debug_probe_radius_scale =
                        std::clamp(g_editor.settings.irradiance_volume_debug_probe_radius_scale, 0.0f, 2.0f);
                    reset_accumulation();
                }
                if (ImGui::Checkbox("Manual Bounds", &g_editor.settings.irradiance_volume_manual_bounds)) {
                    reset_accumulation(lt::RenderDirty::IrradianceVolume);
                }
                ImGui::BeginDisabled(!g_editor.settings.irradiance_volume_manual_bounds);
                float bounds_min[3] = {
                    g_editor.settings.irradiance_volume_bounds_min.x,
                    g_editor.settings.irradiance_volume_bounds_min.y,
                    g_editor.settings.irradiance_volume_bounds_min.z,
                };
                if (ImGui::DragFloat3("Bounds Min", bounds_min, 0.02f, -1000.0f, 1000.0f, "%.3f")) {
                    g_editor.settings.irradiance_volume_bounds_min = {bounds_min[0], bounds_min[1], bounds_min[2]};
                    reset_accumulation(lt::RenderDirty::IrradianceVolume);
                }
                float bounds_max[3] = {
                    g_editor.settings.irradiance_volume_bounds_max.x,
                    g_editor.settings.irradiance_volume_bounds_max.y,
                    g_editor.settings.irradiance_volume_bounds_max.z,
                };
                if (ImGui::DragFloat3("Bounds Max", bounds_max, 0.02f, -1000.0f, 1000.0f, "%.3f")) {
                    g_editor.settings.irradiance_volume_bounds_max = {bounds_max[0], bounds_max[1], bounds_max[2]};
                    reset_accumulation(lt::RenderDirty::IrradianceVolume);
                }
                ImGui::EndDisabled();
                ImGui::EndDisabled();
            }
            if (icon_collapsing_header(EditorIcon::Lightmap, "Lightmap")) {
                if (ImGui::Checkbox("Enable Lightmap", &g_editor.settings.use_lightmap)) {
                    reset_accumulation(lt::RenderDirty::Lightmap);
                }
                ImGui::BeginDisabled(!g_editor.settings.use_lightmap);
                if (ImGui::Checkbox("Lightmap Disk Cache", &g_editor.settings.lightmap_cache_enabled)) {
                    reset_accumulation(lt::RenderDirty::Lightmap);
                }
                if (ImGui::Checkbox("Lightmap Auto Update", &g_editor.settings.lightmap_auto_update)) {
                    reset_accumulation(lt::RenderDirty::Lightmap);
                }
                if (icon_text_button("Update Lightmap", EditorIcon::Bake)) {
                    g_editor.settings.lightmap_force_rebake = true;
                    reset_accumulation(lt::RenderDirty::Lightmap);
                }
                const std::string lightmap_cache_path = default_lightmap_cache_path();
                ImGui::TextWrapped("Lightmap Cache: %s", lightmap_cache_path.c_str());
                if (ImGui::DragInt("Lightmap Resolution", &g_editor.settings.lightmap_resolution, 16.0f, 16, 4096)) {
                    g_editor.settings.lightmap_resolution = std::clamp(g_editor.settings.lightmap_resolution, 16, 16384);
                    reset_accumulation(lt::RenderDirty::Lightmap);
                }
                if (ImGui::DragInt("Lightmap Padding", &g_editor.settings.lightmap_padding, 1.0f, 0, 32)) {
                    g_editor.settings.lightmap_padding = std::clamp(g_editor.settings.lightmap_padding, 0, 64);
                    reset_accumulation(lt::RenderDirty::Lightmap);
                }
                if (ImGui::DragInt("Lightmap Dilation", &g_editor.settings.lightmap_dilation, 1.0f, 0, 32)) {
                    g_editor.settings.lightmap_dilation = std::clamp(g_editor.settings.lightmap_dilation, 0, 64);
                    reset_accumulation(lt::RenderDirty::Lightmap);
                }
                if (ImGui::DragInt("Lightmap Samples", &g_editor.settings.lightmap_bake_samples, 1.0f, 1, 1024)) {
                    g_editor.settings.lightmap_bake_samples = std::clamp(g_editor.settings.lightmap_bake_samples, 1, 1024);
                    reset_accumulation(lt::RenderDirty::Lightmap);
                }
                if (ImGui::DragInt("Lightmap Bounces", &g_editor.settings.lightmap_bake_bounces, 1.0f, 1, 16)) {
                    g_editor.settings.lightmap_bake_bounces = std::clamp(g_editor.settings.lightmap_bake_bounces, 1, 32);
                    reset_accumulation(lt::RenderDirty::Lightmap);
                }
                if (ImGui::Checkbox("Lightmap Principled GI", &g_editor.settings.lightmap_principled_gi)) {
                    reset_accumulation();
                }
                int lm_bake_backend = static_cast<int>(g_editor.settings.lightmap_bake_backend);
                if (ImGui::Combo("Lightmap Bake Backend", &lm_bake_backend, "GPU\0CPU\0\0")) {
                    g_editor.settings.lightmap_bake_backend = static_cast<lt::LightmapBakeBackend>(lm_bake_backend);
                    reset_accumulation(lt::RenderDirty::Lightmap);
                }
                ImGui::EndDisabled();
            }
            if (icon_collapsing_header(EditorIcon::Render, "NPR Sampling")) {
                if (ImGui::DragInt("Style Samples", &g_editor.settings.stylized_samples, 1.0f, 1, 128)) {
                    g_editor.settings.stylized_samples = std::clamp(g_editor.settings.stylized_samples, 1, 128);
                    if (lt::stylized_rendering_enabled(g_editor.settings, g_editor.scene)) {
                        set_renderer(false);
                    }
                    reset_accumulation();
                }
                if (ImGui::DragInt("Style Depth", &g_editor.settings.stylized_max_depth, 1.0f, 0, 32)) {
                    g_editor.settings.stylized_max_depth = std::clamp(g_editor.settings.stylized_max_depth, 0, 32);
                    if (lt::stylized_rendering_enabled(g_editor.settings, g_editor.scene)) {
                        set_renderer(false);
                    }
                    reset_accumulation();
                }
            }
            if (icon_collapsing_header(EditorIcon::Environment, "Environment")) {
                color_edit("Environment Color", g_editor.scene.environment.color, lt::RenderDirty::Environment);
                if (ImGui::DragFloat("Strength", &g_editor.scene.environment.strength, 0.05f, 0.0f, 100.0f, "%.2f")) {
                    g_editor.scene.environment.strength = std::max(0.0f, g_editor.scene.environment.strength);
                    reset_accumulation(lt::RenderDirty::Environment);
                }
                bool constant_environment = g_editor.scene.environment.constant && !g_editor.scene.environment.texture;
                if (ImGui::Checkbox("Constant Environment", &constant_environment)) {
                    g_editor.scene.environment.constant = constant_environment;
                    if (constant_environment) {
                        g_editor.scene.environment.texture = nullptr;
                    }
                    reset_accumulation(lt::RenderDirty::Environment);
                }
                ImGui::Text("HDRI: %s", g_editor.scene.environment.texture ? g_editor.scene.environment.texture->name.c_str() : "None");
                if (icon_text_button("Load HDRI", EditorIcon::Environment)) {
                    load_environment_texture_dialog();
                }
                ImGui::SameLine();
                if (icon_text_button("Clear HDRI", EditorIcon::Clear)) {
                    g_editor.scene.environment.texture = nullptr;
                    g_editor.scene.environment.constant = false;
                    reset_accumulation(lt::RenderDirty::Environment);
                }
            }
            ImGui::Checkbox("Pause", &g_editor.paused);
    }
    ImGui::EndChild();
    ImGui::End();
}

void handle_viewport_input() {
    ImGuiIO& io = ImGui::GetIO();
    if (!g_editor.viewport_hovered && !g_editor.viewport_focused) return;
    if (g_editor.gizmo_dragging) return;
    float right_axis = 0.0f;
    float up_axis = 0.0f;
    float forward_axis = 0.0f;
    if (ImGui::IsKeyDown(ImGuiKey_W)) forward_axis += 1.0f;
    if (ImGui::IsKeyDown(ImGuiKey_S)) forward_axis -= 1.0f;
    if (ImGui::IsKeyDown(ImGuiKey_A)) right_axis -= 1.0f;
    if (ImGui::IsKeyDown(ImGuiKey_D)) right_axis += 1.0f;
    if (ImGui::IsKeyDown(ImGuiKey_Q)) up_axis -= 1.0f;
    if (ImGui::IsKeyDown(ImGuiKey_E)) up_axis += 1.0f;
    const bool movement_requested = right_axis != 0.0f || up_axis != 0.0f || forward_axis != 0.0f;
    const bool right_mouse_down = ImGui::IsMouseDown(ImGuiMouseButton_Right);
    bool camera_changed = false;
    bool object_changed = false;
    if (movement_requested) {
        if (g_editor.tool_mode == ToolMode::Move && has_selection() && !right_mouse_down) {
            const float step = io.KeyShift ? 0.012f : 0.06f;
            const float right_delta = right_axis * step;
            const float up_delta = up_axis * step;
            const float forward_delta = forward_axis * step;
            const ViewTransform view = make_view_transform(true);
            const lt::Vec3 delta = view.right * right_delta + view.up * up_delta + view.forward * forward_delta;
            object_changed = move_selected_object(delta);
        } else {
            const float step = io.KeyShift ? 0.012f : 0.06f;
            const float right_delta = right_axis * step;
            const float up_delta = up_axis * step;
            const float forward_delta = forward_axis * step;
            move_camera(right_delta, up_delta, forward_delta);
            camera_changed = true;
        }
    }
    if (ImGui::IsMouseDragging(ImGuiMouseButton_Right)) {
        const ImVec2 delta = io.MouseDelta;
        if (delta.x != 0.0f || delta.y != 0.0f) {
            rotate_camera(delta.x * -0.004f, delta.y * -0.004f);
            camera_changed = true;
        }
    }
    if (camera_changed) {
        reset_accumulation(lt::RenderDirty::Camera);
    } else if (object_changed) {
        reset_accumulation(lt::RenderDirty::Transform);
    }
}

bool point_in_rect(ImVec2 p, ImVec2 min, ImVec2 max) {
    return p.x >= min.x && p.x <= max.x && p.y >= min.y && p.y <= max.y;
}

// Helper overlay: draw subtle bounding-box outlines for all objects.
void draw_solid_overlay_outlines(ImDrawList* draw_list) {
    const ImU32 mesh_color = IM_COL32(100, 108, 118, 90);
    for (int i = 0; i < static_cast<int>(g_editor.scene.meshes.size()); ++i) {
        // Skip selection; it is drawn separately in draw_solid_selection_outline.
        if (g_editor.selection_kind == SelectionKind::Mesh && i == g_editor.selected_mesh) continue;
        draw_mesh_bounds_outline(draw_list, i, mesh_color, 0.8f);
    }
    for (int i = 0; i < static_cast<int>(g_editor.scene.spheres.size()); ++i) {
        if (g_editor.selection_kind == SelectionKind::Sphere && i == g_editor.selected_sphere) continue;
        draw_sphere_outline(draw_list, i, mesh_color, 0.8f);
    }
}

// Helper overlay: highlight area-light meshes and show directional-light indicators.
void draw_solid_light_helpers(ImDrawList* draw_list) {
    const ImU32 area_light_color = IM_COL32(255, 210, 90, 180);
    const ImU32 dir_light_color = IM_COL32(255, 200, 70, 200);

    // Area light meshes: highlight their bounding boxes
    for (int i = 0; i < static_cast<int>(g_editor.scene.meshes.size()); ++i) {
        if (g_editor.scene.meshes[static_cast<size_t>(i)].light.enabled) {
            draw_mesh_bounds_outline(draw_list, i, area_light_color, 1.5f);
        }
    }

    // Directional lights: draw arrow showing where the light shines (opposite of direction)
    for (const auto& dl : g_editor.scene.directional_lights) {
        const lt::Vec3 dir = lt::normalize(dl.direction);
        if (lt::dot(dir, dir) <= 0.0f) continue;
        const float ps = world_per_pixel_at(dl.position);
        const float arrow_len = 80.0f * ps;
        const lt::Vec3 tip = dl.position - dir * arrow_len;
        ImVec2 s_origin{}, s_tip{};
        if (project_point(dl.position, s_origin) && project_point(tip, s_tip)) {
            draw_list->AddLine(s_origin, s_tip, dir_light_color, 1.8f);
            // Arrowhead
            const ImVec2 dir = {s_tip.x - s_origin.x, s_tip.y - s_origin.y};
            const float len = std::sqrt(dir.x * dir.x + dir.y * dir.y);
            if (len > 0.0f) {
                const ImVec2 nd = {dir.x / len, dir.y / len};
                const ImVec2 perp = {-nd.y * 8.0f, nd.x * 8.0f};
                draw_list->AddLine(s_tip, {s_tip.x - nd.x * 14.0f + perp.x, s_tip.y - nd.y * 14.0f + perp.y}, dir_light_color, 1.8f);
                draw_list->AddLine(s_tip, {s_tip.x - nd.x * 14.0f - perp.x, s_tip.y - nd.y * 14.0f - perp.y}, dir_light_color, 1.8f);
            }
        }
    }
}

// Helper overlay: draw selection outline (orange bounds or sphere circle).
void draw_solid_selection_outline(ImDrawList* draw_list) {
    const ImU32 sel_color = IM_COL32(255, 150, 35, 240);
    if (g_editor.selection_kind == SelectionKind::Mesh) {
        draw_mesh_bounds_outline(draw_list, g_editor.selected_mesh, sel_color, 2.2f);
    } else if (g_editor.selection_kind == SelectionKind::Sphere) {
        draw_sphere_outline(draw_list, g_editor.selected_sphere, sel_color, 2.2f);
    } else if (g_editor.selection_kind == SelectionKind::PointLight) {
        const lt::PointLight& light = g_editor.scene.point_lights[static_cast<size_t>(g_editor.selected_point_light)];
        ImVec2 center{};
        if (project_point(light.position, center)) {
            const float r = 8.0f;
            draw_list->AddCircle(center, r, sel_color, 0, 2.2f);
            draw_list->AddLine({center.x - r - 3.0f, center.y}, {center.x + r + 3.0f, center.y}, sel_color, 2.2f);
            draw_list->AddLine({center.x, center.y - r - 3.0f}, {center.x, center.y + r + 3.0f}, sel_color, 2.2f);
        }
    } else if (g_editor.selection_kind == SelectionKind::DirectionalLight) {
        const lt::DirectionalLight& light = g_editor.scene.directional_lights[static_cast<size_t>(g_editor.selected_directional_light)];
        ImVec2 center{};
        if (project_point(light.position, center)) {
            const float r = 7.0f;
            draw_list->AddCircle(center, r, sel_color, 0, 2.2f);
            // Cross mark for directional light
            draw_list->AddLine({center.x - r - 2.0f, center.y}, {center.x + r + 2.0f, center.y}, sel_color, 2.2f);
            draw_list->AddLine({center.x, center.y - r - 2.0f}, {center.x, center.y + r + 2.0f}, sel_color, 2.2f);
        }
    }
}

void draw_viewport() {
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("Viewport", nullptr, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    ImVec2 available = ImGui::GetContentRegionAvail();
    available.x = std::max(64.0f, available.x);
    available.y = std::max(64.0f, available.y);
    g_editor.viewport_size = available;

    auto handle_selection_click = []() {
        if (g_editor.viewport_hovered && g_editor.tool_mode == ToolMode::Select && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            SelectionKind kind = SelectionKind::None;
            int index = -1;
            if (pick_object(ImGui::GetIO().MousePos, kind, index)) {
                kind == SelectionKind::Sphere ? select_sphere(index) : select_mesh(index);
            } else {
                select_mesh(-1);
            }
        }
    };

    const bool realtime_preview =
        g_editor.viewport_preview_mode == ViewportPreviewMode::Solid ||
        g_editor.viewport_preview_mode == ViewportPreviewMode::MaterialPreview ||
        g_editor.viewport_preview_mode == ViewportPreviewMode::Wireframe;
    bool handled_input_early = false;

    if (realtime_preview) {
        g_editor.viewport_image_min = ImGui::GetCursorScreenPos();
        g_editor.viewport_image_max = {g_editor.viewport_image_min.x + available.x, g_editor.viewport_image_min.y + available.y};
        g_editor.viewport_hovered = ImGui::IsWindowHovered() && point_in_rect(ImGui::GetIO().MousePos, g_editor.viewport_image_min, g_editor.viewport_image_max);
        g_editor.viewport_focused = ImGui::IsWindowFocused();

        handle_selection_click();
        handle_gizmo_drag();
        handle_viewport_input();
        handled_input_early = true;

        // D3D11 real-time preview. Camera-only motion updates constants, while
        // geometry buffers rebuild only when geometry_generation changes.
        const int w = std::max(64, static_cast<int>(available.x));
        const int h = std::max(64, static_cast<int>(available.y));
        init_solid_preview(g_solid_preview);
        resize_solid_preview(g_solid_preview, w, h);
        update_solid_preview_buffers(g_solid_preview, g_editor.scene);
        render_solid_preview(g_solid_preview, g_editor.scene, available, g_editor.viewport_preview_mode);
        if (g_solid_preview.srv) {
            ImGui::Image(reinterpret_cast<ImTextureID>(g_solid_preview.srv), available);
        } else {
            ImGui::Dummy(available);
        }
    } else {
        // Rendered mode: path tracer preview
        prepare_render_preview();
        if (g_preview.srv) {
            g_editor.viewport_image_min = ImGui::GetCursorScreenPos();
            ImGui::Image(reinterpret_cast<ImTextureID>(g_preview.srv), available);
            g_editor.viewport_image_max = {g_editor.viewport_image_min.x + available.x, g_editor.viewport_image_min.y + available.y};
        } else {
            g_editor.viewport_image_min = ImGui::GetCursorScreenPos();
            ImGui::Dummy(available);
            g_editor.viewport_image_max = {g_editor.viewport_image_min.x + available.x, g_editor.viewport_image_min.y + available.y};
        }
    }

    g_editor.viewport_hovered = ImGui::IsItemHovered();
    g_editor.viewport_focused = ImGui::IsWindowFocused();
    if (!handled_input_early) {
        handle_selection_click();
        handle_gizmo_drag();
    }

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    if (g_editor.viewport_preview_mode == ViewportPreviewMode::Rendered) {
        // Rendered mode: dirty reference outlines only when accumulating.
        if (!g_editor.hide_dirty_wireframes && g_editor.dirty != lt::RenderDirty::None) {
            draw_scene_reference_outlines(draw_list);
        }
    }

    bool selection_outline_drawn = false;
    if (has_selection() && !g_editor.hide_dirty_wireframes) {
        ID3D11ShaderResourceView* outline_srv = render_selection_outline(
            g_solid_preview,
            g_editor.scene,
            available,
            g_editor.selection_kind,
            g_editor.selection_kind == SelectionKind::Sphere ? g_editor.selected_sphere : g_editor.selected_mesh);
        if (outline_srv) {
            draw_list->AddImage(
                reinterpret_cast<ImTextureID>(outline_srv),
                g_editor.viewport_image_min,
                g_editor.viewport_image_max);
            selection_outline_drawn = true;
        }
    }
    if (has_selection() && !selection_outline_drawn && !g_editor.hide_dirty_wireframes) {
        draw_solid_selection_outline(draw_list);
    }

    draw_gizmo_overlay();
    if (g_editor.viewport_hovered) {
        if (g_editor.tool_mode == ToolMode::Move) ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeAll);
        if (g_editor.tool_mode == ToolMode::Rotate) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        if (g_editor.tool_mode == ToolMode::Scale) ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNWSE);
    }
    if (!handled_input_early) {
        handle_viewport_input();
    }
    const char* hint = g_editor.tool_mode == ToolMode::Move && has_selection()
        ? "LMB drag gizmo | WASD/QE move object | RMB look"
        : "LMB select/drag | RMB look | WASD/QE fly";
    const float button_size = 28.0f;
    const float padding = 10.0f;

    const float gap = 4.0f;
    const float y = g_editor.viewport_image_min.y + padding;
    float x = g_editor.viewport_image_max.x - padding - button_size;
    auto draw_overlay_icon_button = [&](EditorIcon icon, const char* tooltip, bool active, auto&& on_click) {
        const ImVec2 min = {x, y};
        const ImVec2 max = {x + button_size, y + button_size};
        const bool hovered = point_in_rect(ImGui::GetIO().MousePos, min, max);
        const ImU32 fill = active ? IM_COL32(75, 105, 170, 235) : IM_COL32(18, 18, 20, 210);
        const ImU32 border = hovered || active ? IM_COL32(145, 165, 210, 240) : IM_COL32(95, 100, 110, 230);
        const ImU32 icon_color = IM_COL32(226, 229, 235, 255);
        draw_list->AddRectFilled(min, max, fill, 4.0f);
        draw_list->AddRect(min, max, border, 4.0f);
        const float icon_size = button_size - 10.0f;
        const ImVec2 icon_min = {min.x + (button_size - icon_size) * 0.5f, min.y + (button_size - icon_size) * 0.5f};
        draw_editor_icon(draw_list, icon, icon_min, {icon_min.x + icon_size, icon_min.y + icon_size}, icon_color);
        if (hovered) {
            ImGui::SetTooltip("%s", tooltip);
            ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
            if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                on_click();
            }
        }
        x -= button_size + gap;
    };

    draw_overlay_icon_button(
        g_editor.viewport_fullscreen ? EditorIcon::ExitFullscreen : EditorIcon::Fullscreen,
        g_editor.viewport_fullscreen ? "Exit fullscreen viewport" : "Fullscreen viewport",
        false,
        [&]() { g_editor.viewport_fullscreen = !g_editor.viewport_fullscreen; });
    draw_overlay_icon_button(
        EditorIcon::WireframePreview,
        "Wireframe-only viewport preview",
        g_editor.viewport_preview_mode == ViewportPreviewMode::Wireframe,
        [&]() { set_viewport_preview_mode(ViewportPreviewMode::Wireframe); });
    draw_overlay_icon_button(
        EditorIcon::SolidPreview,
        "Solid viewport preview",
        g_editor.viewport_preview_mode == ViewportPreviewMode::Solid,
        [&]() { set_viewport_preview_mode(ViewportPreviewMode::Solid); });
    draw_overlay_icon_button(
        EditorIcon::MaterialPreview,
        "Material preview",
        g_editor.viewport_preview_mode == ViewportPreviewMode::MaterialPreview,
        [&]() { set_viewport_preview_mode(ViewportPreviewMode::MaterialPreview); });
    draw_overlay_icon_button(
        EditorIcon::RenderedPreview,
        "Rendered path tracer preview",
        g_editor.viewport_preview_mode == ViewportPreviewMode::Rendered,
        [&]() { set_viewport_preview_mode(ViewportPreviewMode::Rendered); });

    const ImVec2 text_size = ImGui::CalcTextSize(hint);
    const ImVec2 text_pos = {g_editor.viewport_image_min.x + 8.0f, g_editor.viewport_image_max.y - text_size.y - 8.0f};
    draw_list->AddRectFilled({text_pos.x - 6.0f, text_pos.y - 4.0f}, {text_pos.x + text_size.x + 6.0f, text_pos.y + text_size.y + 4.0f}, IM_COL32(18, 18, 20, 180), 3.0f);
    draw_list->AddText(text_pos, IM_COL32(220, 222, 225, 230), hint);
    ImGui::End();
    ImGui::PopStyleVar();
}

void draw_status_bar() {
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos({viewport->Pos.x, viewport->Pos.y + viewport->Size.y - 26.0f});
    ImGui::SetNextWindowSize({viewport->Size.x, 26.0f});
    ImGui::Begin("Status", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings);
    const int width = g_editor.framebuffer.width > 0
        ? g_editor.framebuffer.width
        : std::max(0, static_cast<int>(g_editor.viewport_size.x));
    const int height = g_editor.framebuffer.height > 0
        ? g_editor.framebuffer.height
        : std::max(0, static_cast<int>(g_editor.viewport_size.y));
    char resolution_text[32] = {};
    std::snprintf(resolution_text, sizeof(resolution_text), "%d x %d", width, height);
    const float bar_height = ImGui::GetWindowHeight();
    const ImVec2 text_size = ImGui::CalcTextSize(resolution_text);
    const ImVec2 bar_min = ImGui::GetWindowPos();
    const ImVec2 text_pos = {
        bar_min.x + 8.0f,
        bar_min.y + std::max(0.0f, std::floor((bar_height - text_size.y) * 0.5f))
    };
    ImGui::GetWindowDrawList()->AddText(text_pos, ImGui::GetColorU32(ImGuiCol_TextDisabled), resolution_text);
    ImGui::End();
}

void draw_layout_splitters(ImVec2 layout_pos, ImVec2 layout_size, float top, float content_height, float viewport_width, float outliner_height) {
    enum class ActiveSplitter { None, Toolbar, Properties, Outliner };
    static ActiveSplitter active = ActiveSplitter::None;

    constexpr float grab = 8.0f;
    const float content_top = layout_pos.y + top;
    const float toolbar_x = layout_pos.x + g_editor.toolbar_width;
    const float properties_x = layout_pos.x + g_editor.toolbar_width + viewport_width;
    const float outliner_y = content_top + outliner_height;
    const ImVec2 mouse = ImGui::GetIO().MousePos;

    const ImVec2 toolbar_min{toolbar_x - grab * 0.5f, content_top};
    const ImVec2 toolbar_max{toolbar_x + grab * 0.5f, content_top + content_height};
    const ImVec2 properties_min{properties_x - grab * 0.5f, content_top};
    const ImVec2 properties_max{properties_x + grab * 0.5f, content_top + content_height};
    const ImVec2 outliner_min{properties_x, outliner_y - grab * 0.5f};
    const ImVec2 outliner_max{layout_pos.x + layout_size.x, outliner_y + grab * 0.5f};

    const bool hover_toolbar = point_in_rect(mouse, toolbar_min, toolbar_max);
    const bool hover_properties = point_in_rect(mouse, properties_min, properties_max);
    const bool hover_outliner = point_in_rect(mouse, outliner_min, outliner_max);
    if (active == ActiveSplitter::None && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        if (hover_toolbar) {
            active = ActiveSplitter::Toolbar;
        } else if (hover_properties) {
            active = ActiveSplitter::Properties;
        } else if (hover_outliner) {
            active = ActiveSplitter::Outliner;
        }
    }
    if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        active = ActiveSplitter::None;
    }

    if (hover_toolbar || hover_properties || active == ActiveSplitter::Toolbar || active == ActiveSplitter::Properties) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
    }
    if (hover_outliner || active == ActiveSplitter::Outliner) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
    }

    const ImVec2 delta = ImGui::GetIO().MouseDelta;
    if (active == ActiveSplitter::Toolbar) {
        const float total = g_editor.toolbar_width + viewport_width;
        g_editor.toolbar_width = std::clamp(g_editor.toolbar_width + delta.x, 56.0f, total - 160.0f);
    } else if (active == ActiveSplitter::Properties) {
        const float max_properties = layout_size.x - g_editor.toolbar_width - 160.0f;
        g_editor.properties_width = std::clamp(g_editor.properties_width - delta.x, 220.0f, std::max(220.0f, max_properties));
    } else if (active == ActiveSplitter::Outliner) {
        g_editor.outliner_fraction = std::clamp(g_editor.outliner_fraction + delta.y / std::max(1.0f, content_height), 0.18f, 0.75f);
    }

    ImDrawList* draw_list = ImGui::GetForegroundDrawList();
    const ImU32 line_color = IM_COL32(78, 82, 88, 190);
    draw_list->AddLine({toolbar_x, content_top}, {toolbar_x, content_top + content_height}, line_color, 1.0f);
    draw_list->AddLine({properties_x, content_top}, {properties_x, content_top + content_height}, line_color, 1.0f);
    draw_list->AddLine({properties_x, outliner_y}, {layout_pos.x + layout_size.x, outliner_y}, line_color, 1.0f);
}

void draw_loading_overlay() {
    if (!scene_load_in_progress()) return;
    const auto now = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(now - g_load_task.started).count();
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos({viewport->Pos.x + viewport->Size.x - 320.0f, viewport->Pos.y + 44.0f}, ImGuiCond_Always);
    ImGui::SetNextWindowSize({292.0f, 86.0f}, ImGuiCond_Always);
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 6.0f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.10f, 0.11f, 0.12f, 0.94f));
    if (ImGui::Begin("SceneLoadOverlay", nullptr, flags)) {
        ImGui::TextUnformatted("Loading scene");
        ImGui::TextDisabled("%s", g_load_task.path.c_str());
        const float phase = static_cast<float>(std::fmod(elapsed * 0.45, 1.0));
        ImGui::ProgressBar(phase, {-1.0f, 8.0f}, "");
        ImGui::TextDisabled("%.1f s", elapsed);
    }
    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar();
}

void draw_irradiance_volume_bake_overlay() {
    if (scene_load_in_progress() || !g_editor.irradiance_volume_bake_progress) {
        return;
    }
    lt::IrradianceVolumeBakeProgress& progress = *g_editor.irradiance_volume_bake_progress;
    const auto phase = static_cast<lt::IrradianceVolumeBakePhase>(progress.phase.load(std::memory_order_relaxed));
    if (phase != lt::IrradianceVolumeBakePhase::LoadingCache &&
        phase != lt::IrradianceVolumeBakePhase::Baking &&
        phase != lt::IrradianceVolumeBakePhase::SavingCache) {
        return;
    }

    const char* title = "Baking irradiance volume";
    if (phase == lt::IrradianceVolumeBakePhase::LoadingCache) {
        title = "Loading irradiance volume";
    } else if (phase == lt::IrradianceVolumeBakePhase::SavingCache) {
        title = "Saving irradiance volume";
    }
    const uint64_t total_samples = progress.total_samples.load(std::memory_order_relaxed);
    const uint64_t completed_samples = progress.completed_samples.load(std::memory_order_relaxed);
    const uint64_t total_rays = progress.total_rays.load(std::memory_order_relaxed);
    const uint64_t traced_rays = progress.traced_rays.load(std::memory_order_relaxed);
    const int direction_count = progress.direction_count.load(std::memory_order_relaxed);
    const double elapsed_ms = progress.elapsed_ms.load(std::memory_order_relaxed);

    const auto now = std::chrono::steady_clock::now();
    const double animated = std::fmod(std::chrono::duration<double>(now.time_since_epoch()).count() * 0.45, 1.0);
    float fraction = static_cast<float>(animated);
    if (phase == lt::IrradianceVolumeBakePhase::Baking && total_samples > 0) {
        fraction = std::clamp(static_cast<float>(completed_samples) / static_cast<float>(total_samples), 0.0f, 1.0f);
    }

    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos({viewport->Pos.x + viewport->Size.x - 340.0f, viewport->Pos.y + 44.0f}, ImGuiCond_Always);
    ImGui::SetNextWindowSize({312.0f, 108.0f}, ImGuiCond_Always);
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 6.0f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.10f, 0.11f, 0.12f, 0.94f));
    if (ImGui::Begin("IrradianceVolumeBakeOverlay", nullptr, flags)) {
        ImGui::TextUnformatted(title);
        ImGui::ProgressBar(fraction, {-1.0f, 8.0f}, "");
        ImGui::TextDisabled(
            "Samples %llu / %llu  dirs %d",
            static_cast<unsigned long long>(completed_samples),
            static_cast<unsigned long long>(total_samples),
            direction_count);
        if (total_rays > 0) {
            ImGui::TextDisabled(
                "Rays %llu / %llu",
                static_cast<unsigned long long>(traced_rays),
                static_cast<unsigned long long>(total_rays));
        } else if (elapsed_ms > 0.0) {
            ImGui::TextDisabled("%.1f ms", elapsed_ms);
        }
    }
    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar();
}

void draw_lightmap_bake_overlay() {
    if (scene_load_in_progress() || !g_editor.lightmap_bake_progress) {
        return;
    }
    lt::LightmapBakeProgress& progress = *g_editor.lightmap_bake_progress;
    const auto phase = static_cast<lt::LightmapBakePhase>(progress.phase.load(std::memory_order_relaxed));
    if (phase != lt::LightmapBakePhase::LoadingCache &&
        phase != lt::LightmapBakePhase::Unwrapping &&
        phase != lt::LightmapBakePhase::Baking &&
        phase != lt::LightmapBakePhase::SavingCache) {
        return;
    }

    const char* title = "Baking lightmap";
    if (phase == lt::LightmapBakePhase::LoadingCache) {
        title = "Loading lightmap";
    } else if (phase == lt::LightmapBakePhase::Unwrapping) {
        title = "Unwrapping lightmap";
    } else if (phase == lt::LightmapBakePhase::SavingCache) {
        title = "Saving lightmap";
    }
    const uint64_t total_texels = progress.total_texels.load(std::memory_order_relaxed);
    const uint64_t completed_texels = progress.completed_texels.load(std::memory_order_relaxed);
    const uint64_t total_rays = progress.total_rays.load(std::memory_order_relaxed);
    const uint64_t traced_rays = progress.traced_rays.load(std::memory_order_relaxed);
    const int width = progress.width.load(std::memory_order_relaxed);
    const int height = progress.height.load(std::memory_order_relaxed);
    const double elapsed_ms = progress.elapsed_ms.load(std::memory_order_relaxed);
    const bool gpu_bake_in_progress =
        phase == lt::LightmapBakePhase::Baking &&
        g_editor.settings.lightmap_bake_backend == lt::LightmapBakeBackend::Gpu;

    const auto now = std::chrono::steady_clock::now();
    const double animated = std::fmod(std::chrono::duration<double>(now.time_since_epoch()).count() * 0.45, 1.0);
    float fraction = static_cast<float>(animated);
    if (!gpu_bake_in_progress && phase == lt::LightmapBakePhase::Baking && total_texels > 0) {
        fraction = std::clamp(static_cast<float>(completed_texels) / static_cast<float>(total_texels), 0.0f, 1.0f);
    }

    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos({viewport->Pos.x + viewport->Size.x - 340.0f, viewport->Pos.y + 158.0f}, ImGuiCond_Always);
    ImGui::SetNextWindowSize({312.0f, 108.0f}, ImGuiCond_Always);
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 6.0f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.10f, 0.11f, 0.12f, 0.94f));
    if (ImGui::Begin("LightmapBakeOverlay", nullptr, flags)) {
        ImGui::TextUnformatted(title);
        ImGui::ProgressBar(fraction, {-1.0f, 8.0f}, "");
        if (gpu_bake_in_progress) {
            ImGui::TextDisabled(
                "GPU texels queued %llu  %dx%d",
                static_cast<unsigned long long>(total_texels),
                width,
                height);
        } else {
            ImGui::TextDisabled(
                "Texels %llu / %llu  %dx%d",
                static_cast<unsigned long long>(completed_texels),
                static_cast<unsigned long long>(total_texels),
                width,
                height);
        }
        if (gpu_bake_in_progress && total_rays > 0) {
            ImGui::TextDisabled(
                "Estimated rays %llu",
                static_cast<unsigned long long>(total_rays));
        } else if (total_rays > 0) {
            ImGui::TextDisabled(
                "Rays %llu / %llu",
                static_cast<unsigned long long>(traced_rays),
                static_cast<unsigned long long>(total_rays));
        } else if (elapsed_ms > 0.0) {
            ImGui::TextDisabled("%.1f ms", elapsed_ms);
        }
    }
    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar();
}

void draw_statistics_panel() {
    if (!g_editor.show_statistics_panel) return;

    ImGui::SetNextWindowSize({460.0f, 520.0f}, ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Statistics", &g_editor.show_statistics_panel)) {
        ImGui::End();
        return;
    }

    char buf[128];
    const lt::Scene& scene = g_editor.scene;

    // Helpers
    auto section = [](const char* title) {
        ImGui::SeparatorText(title);
    };
    auto stat_row = [&](const char* label, const char* fmt, ...) {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted(label);
        ImGui::TableSetColumnIndex(1);
        va_list args;
        va_start(args, fmt);
        std::vsnprintf(buf, sizeof(buf), fmt, args);
        va_end(args);
        ImGui::TextUnformatted(buf);
    };
    auto begin_stats_table = []() {
        ImGui::BeginTable("##stats", 2, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV);
        ImGui::TableSetupColumn("##label", ImGuiTableColumnFlags_WidthFixed, 200.0f);
        ImGui::TableSetupColumn("##value");
    };
    auto end_stats_table = []() {
        ImGui::EndTable();
    };

    // --- Geometry ---
    section("Geometry");

    size_t total_verts = 0;
    size_t total_tris = 0;
    size_t total_light_tris = 0;
    for (const lt::Mesh& m : scene.meshes) {
        total_verts += m.vertices.size();
        total_tris += m.indices.size() / 3;
        if (m.light.enabled)
            total_light_tris += m.indices.size() / 3;
    }

    begin_stats_table();
    stat_row("Meshes", "%d", static_cast<int>(scene.meshes.size()));
    stat_row("Spheres", "%d", static_cast<int>(scene.spheres.size()));
    stat_row("Materials", "%d", static_cast<int>(scene.materials.size()));
    stat_row("Textures", "%d", static_cast<int>(scene.textures.size()));
    stat_row("Vertices", "%zu", total_verts);
    stat_row("Triangles", "%zu", total_tris);
    stat_row("Light Triangles", "%zu", total_light_tris);
    stat_row("Directional Lights", "%d", static_cast<int>(scene.directional_lights.size()));
    stat_row("Point Lights", "%d", static_cast<int>(scene.point_lights.size()));
    end_stats_table();

    // Refresh pick cache for RenderScene stats if stale (once per generation)
    if (g_pick_cache.scene_generation != g_editor.render_generation) {
        g_pick_cache.render_scene = lt::build_render_scene(g_editor.scene);
        g_pick_cache.scene_generation = g_editor.render_generation;
    }
    const lt::RenderScene& rs = g_pick_cache.render_scene;
    const bool rs_valid = !rs.triangles.empty() || !rs.spheres.empty() ||
        !rs.bvh_nodes.empty();

    if (rs_valid) {
        ImGui::Spacing();
        ImGui::TextDisabled("Render Scene (acceleration data)");
        begin_stats_table();
        stat_row("Render Triangles", "%zu", rs.triangles.size());
        stat_row("BVH Nodes", "%zu", rs.bvh_nodes.size());
        stat_row("TLAS Nodes", "%zu", rs.tlas_nodes.size());
        stat_row("Mesh Instances", "%zu", rs.mesh_instances.size());
        stat_row("Flat BVH Nodes", "%zu", rs.flat_bvh_nodes.size());
        end_stats_table();
    }

    // --- Resolution / Renderer ---
    section("Renderer");
    begin_stats_table();
    stat_row("Renderer", "%s", g_editor.renderer->name());
    stat_row("Samples", "%u", g_editor.frame_index);
    stat_row("Viewport", "%dx%d",
        static_cast<int>(g_editor.viewport_size.x),
        static_cast<int>(g_editor.viewport_size.y));
    stat_row("Render Resolution", "%dx%d",
        g_editor.settings.width, g_editor.settings.height);
    stat_row("Sample Time", "%.2f ms", g_editor.last_sample_ms);
    stat_row("Max Bounces", "%d", g_editor.settings.max_bounces);
    stat_row("Samples/Frame", "%d", g_editor.settings.samples_per_pixel);
    stat_row("Denoiser", "%s", lt::svgf_denoising_enabled(g_editor.settings) ? "SVGF" : "Off");
    if (lt::svgf_denoising_enabled(g_editor.settings)) {
        stat_row("SVGF G-Buffer", "%s", g_editor.settings.svgf_rasterized_gbuffer ? "Rasterized" : "Manual Trace");
    }
    stat_row("Acceleration", "%s",
        g_editor.settings.acceleration_structure == lt::AccelerationStructure::Auto ? "Auto" :
        g_editor.settings.acceleration_structure == lt::AccelerationStructure::Flat ? "Flat BVH" :
        "Two-level BVH");
    end_stats_table();

    // --- System RAM ---
    section("System Memory");
    {
        PROCESS_MEMORY_COUNTERS_EX pmc{};
        pmc.cb = sizeof(pmc);
        bool ram_ok = GetProcessMemoryInfo(GetCurrentProcess(),
            reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof(pmc));
        if (ram_ok) {
            begin_stats_table();
            format_bytes(pmc.WorkingSetSize, buf, sizeof(buf));
            stat_row("Working Set", "%s", buf);
            format_bytes(pmc.PrivateUsage, buf, sizeof(buf));
            stat_row("Private Usage", "%s", buf);
            format_bytes(pmc.PeakWorkingSetSize, buf, sizeof(buf));
            stat_row("Peak Working Set", "%s", buf);
            format_bytes(pmc.PagefileUsage, buf, sizeof(buf));
            stat_row("Commit Size", "%s", buf);
            end_stats_table();
        } else {
            ImGui::TextDisabled("Process memory query failed");
        }
    }

    // --- GPU VRAM ---
    section("GPU Memory");
#if LT_HAS_CUDA
    if (g_editor.cuda.available()) {
        size_t vram_free = 0;
        size_t vram_total = 0;
        cudaError_t err = cudaMemGetInfo(&vram_free, &vram_total);
        if (err == cudaSuccess) {
            size_t vram_used = vram_total - vram_free;
            begin_stats_table();
            format_bytes(vram_used, buf, sizeof(buf));
            stat_row("VRAM Used", "%s", buf);
            format_bytes(vram_free, buf, sizeof(buf));
            stat_row("VRAM Free", "%s", buf);
            format_bytes(vram_total, buf, sizeof(buf));
            stat_row("VRAM Total", "%s", buf);
            std::snprintf(buf, sizeof(buf), "%.1f%%", vram_total > 0 ?
                100.0 * static_cast<double>(vram_used) / static_cast<double>(vram_total) : 0.0);
            stat_row("VRAM Usage", "%s", buf);
            end_stats_table();
        } else {
            ImGui::TextDisabled("cudaMemGetInfo failed: %s",
                cudaGetErrorString(err));
        }
    } else {
        ImGui::TextDisabled("CUDA device not available (using CPU renderer)");
    }
#else
    ImGui::TextDisabled("CUDA backend not built (CPU renderer only)");
#endif

    // --- Estimated Scene Memory ---
    section("Estimated Scene Memory");
    {
        size_t mesh_vertex_bytes = 0;
        size_t mesh_index_bytes = 0;
        size_t mesh_normal_bytes = 0;
        size_t mesh_uv_bytes = 0;
        for (const lt::Mesh& m : scene.meshes) {
            mesh_vertex_bytes += m.vertices.size() * sizeof(lt::Vec3);
            mesh_index_bytes += m.indices.size() * sizeof(uint32_t);
            mesh_normal_bytes += m.normals.size() * sizeof(lt::Vec3);
            mesh_uv_bytes += m.texcoords.size() * sizeof(lt::Vec2);
        }
        size_t mesh_total = mesh_vertex_bytes + mesh_index_bytes +
            mesh_normal_bytes + mesh_uv_bytes;

        size_t tex_pixel_bytes = 0;
        size_t tex_alpha_bytes = 0;
        size_t tex_mip_bytes = 0;
        size_t tex_encoded_bytes = 0;
        for (const auto& tex : scene.textures) {
            if (!tex) continue;
            tex_pixel_bytes += tex->pixels.size() * sizeof(lt::Vec3);
            tex_alpha_bytes += tex->alpha.size() * sizeof(float);
            tex_encoded_bytes += tex->encoded_bytes.size();
            for (const auto& mip : tex->mip_pixels)
                tex_mip_bytes += mip.size() * sizeof(lt::Vec3);
        }
        size_t tex_total = tex_pixel_bytes + tex_alpha_bytes +
            tex_mip_bytes + tex_encoded_bytes;

        size_t framebuffer_bytes = static_cast<size_t>(g_editor.settings.width) *
            static_cast<size_t>(g_editor.settings.height) * 4 * sizeof(float);

        size_t mat_estimate = scene.materials.size() * 512; // rough per-material overhead
        size_t sphere_bytes = scene.spheres.size() * sizeof(lt::Sphere);

        size_t scene_total = mesh_total + tex_total + framebuffer_bytes +
            mat_estimate + sphere_bytes;

        begin_stats_table();
        format_bytes(mesh_vertex_bytes, buf, sizeof(buf));
        stat_row("  Vertices", "%s", buf);
        format_bytes(mesh_index_bytes, buf, sizeof(buf));
        stat_row("  Indices", "%s", buf);
        format_bytes(mesh_normal_bytes, buf, sizeof(buf));
        stat_row("  Normals", "%s", buf);
        format_bytes(mesh_uv_bytes, buf, sizeof(buf));
        stat_row("  UVs", "%s", buf);
        format_bytes(mesh_total, buf, sizeof(buf));
        stat_row("Mesh Subtotal", "%s", buf);
        ImGui::TableNextRow(); ImGui::TableSetColumnIndex(0);
        ImGui::TableSetColumnIndex(1); // spacer
        format_bytes(tex_pixel_bytes, buf, sizeof(buf));
        stat_row("  Pixels", "%s", buf);
        format_bytes(tex_alpha_bytes, buf, sizeof(buf));
        stat_row("  Alpha", "%s", buf);
        format_bytes(tex_mip_bytes, buf, sizeof(buf));
        stat_row("  Mips", "%s", buf);
        format_bytes(tex_encoded_bytes, buf, sizeof(buf));
        stat_row("  Encoded", "%s", buf);
        format_bytes(tex_total, buf, sizeof(buf));
        stat_row("Texture Subtotal", "%s", buf);
        ImGui::TableNextRow(); ImGui::TableSetColumnIndex(0);
        ImGui::TableSetColumnIndex(1); // spacer
        format_bytes(framebuffer_bytes, buf, sizeof(buf));
        stat_row("Framebuffer (RGBA32F)", "%s", buf);
        format_bytes(mat_estimate, buf, sizeof(buf));
        stat_row("Materials (est.)", "%s", buf);
        format_bytes(sphere_bytes, buf, sizeof(buf));
        stat_row("Spheres", "%s", buf);
        ImGui::Separator();
        format_bytes(scene_total, buf, sizeof(buf));
        stat_row("Scene Total (est.)", "%s", buf);
        end_stats_table();
    }

    // --- Estimated Acceleration / Cache Memory ---
    section("Estimated Cache Memory");
    {
        size_t render_tri_bytes = 0;
        size_t render_idx_bytes = 0;
        size_t render_flat_idx_bytes = 0;
        size_t render_bvh_bytes = 0;
        size_t render_flat_bvh_bytes = 0;
        size_t render_tlas_bytes = 0;
        size_t render_instance_bytes = 0;
        size_t render_light_idx_bytes = 0;
        if (rs_valid) {
            render_tri_bytes = rs.triangles.size() * sizeof(lt::Triangle);
            render_idx_bytes = rs.triangle_indices.size() * sizeof(int);
            render_flat_idx_bytes = rs.flat_triangle_indices.size() * sizeof(int);
            render_bvh_bytes = rs.bvh_nodes.size() * sizeof(lt::BvhNode);
            render_flat_bvh_bytes = rs.flat_bvh_nodes.size() * sizeof(lt::BvhNode);
            render_tlas_bytes = rs.tlas_nodes.size() * sizeof(lt::BvhNode);
            render_instance_bytes = rs.mesh_instances.size() * sizeof(lt::RenderScene::MeshInstance);
            render_light_idx_bytes = rs.light_triangle_indices.size() * sizeof(int);
        }
        size_t render_total = render_tri_bytes + render_idx_bytes +
            render_flat_idx_bytes + render_bvh_bytes + render_flat_bvh_bytes +
            render_tlas_bytes + render_instance_bytes + render_light_idx_bytes;

        // Lightmap estimate (RGBA32F per texel, from settings)
        size_t lm_est = 0;
        if (g_editor.settings.use_lightmap) {
            lm_est = static_cast<size_t>(g_editor.settings.lightmap_resolution) *
                static_cast<size_t>(g_editor.settings.lightmap_resolution) *
                4 * sizeof(float);
        }

        // Irradiance volume estimate
        size_t iv_est = 0;
        if (g_editor.settings.use_irradiance_volume) {
            const int g = g_editor.settings.irradiance_volume_grid_resolution;
            const int s = g_editor.settings.irradiance_volume_subgrid_resolution;
            const int d = g_editor.settings.irradiance_volume_direction_resolution;
            const size_t cells = static_cast<size_t>(g) * g * g;
            const size_t probes = static_cast<size_t>(s) * s * s;
            const size_t dirs = static_cast<size_t>(d) * d;
            // Each probe stores directions (Vec3) + irradiance (Vec3)
            iv_est = cells * probes * dirs * 2 * sizeof(lt::Vec3);
        }

        begin_stats_table();
        if (rs_valid) {
            format_bytes(render_tri_bytes, buf, sizeof(buf));
            stat_row("  Triangles", "%s", buf);
            format_bytes(render_idx_bytes, buf, sizeof(buf));
            stat_row("  Triangle Indices", "%s", buf);
            format_bytes(render_flat_idx_bytes, buf, sizeof(buf));
            stat_row("  Flat Indices", "%s", buf);
            format_bytes(render_bvh_bytes, buf, sizeof(buf));
            stat_row("  BVH Nodes", "%s", buf);
            format_bytes(render_flat_bvh_bytes, buf, sizeof(buf));
            stat_row("  Flat BVH Nodes", "%s", buf);
            format_bytes(render_tlas_bytes, buf, sizeof(buf));
            stat_row("  TLAS Nodes", "%s", buf);
            format_bytes(render_instance_bytes, buf, sizeof(buf));
            stat_row("  Mesh Instances", "%s", buf);
            format_bytes(render_light_idx_bytes, buf, sizeof(buf));
            stat_row("  Light Indices", "%s", buf);
            format_bytes(render_total, buf, sizeof(buf));
            stat_row("RenderScene Subtotal", "%s", buf);
            ImGui::TableNextRow(); ImGui::TableSetColumnIndex(0);
            ImGui::TableSetColumnIndex(1); // spacer
        } else {
            ImGui::TextDisabled("(pick cache not yet populated)");
        }
        format_bytes(lm_est, buf, sizeof(buf));
        stat_row("Lightmap (est., RGBA32F)", "%s%s",
            buf, g_editor.settings.use_lightmap ? "" : " (disabled)");
        format_bytes(iv_est, buf, sizeof(buf));
        stat_row("Irradiance Volume (est.)", "%s%s",
            buf, g_editor.settings.use_irradiance_volume ? "" : " (disabled)");
        size_t cache_total = render_total + lm_est + iv_est;
        ImGui::Separator();
        format_bytes(cache_total, buf, sizeof(buf));
        stat_row("Cache Total (est.)", "%s", buf);
        end_stats_table();
    }

    // --- Viewport Preview ---
    section("Viewport Preview");
    {
        begin_stats_table();

        const char* mode_name = "Unknown";
        switch (g_editor.viewport_preview_mode) {
            case ViewportPreviewMode::Rendered:  mode_name = "Rendered"; break;
            case ViewportPreviewMode::MaterialPreview: mode_name = "Material Preview"; break;
            case ViewportPreviewMode::Solid:     mode_name = "Solid"; break;
            case ViewportPreviewMode::Wireframe: mode_name = "Wireframe"; break;
        }
        stat_row("Preview Mode", "%s", mode_name);

        stat_row("Preview Texture",
            "%s (%dx%d)",
            g_preview.texture ? "allocated" : "released",
            g_preview.width, g_preview.height);

        stat_row("Solid Preview Verts", "%u", g_solid_preview.vertex_count);
        stat_row("Solid Preview Indices", "%u", g_solid_preview.index_count);

        stat_row("CPU Cache", "%s", g_editor.cpu.has_cached_data() ? "loaded" : "empty");
        stat_row("CUDA Cache", "%s", g_editor.cuda.has_cached_data() ? "loaded" : "empty");

        end_stats_table();
    }

    ImGui::End();
}

void draw_log_panel() {
    if (!g_editor.show_log_panel) {
        return;
    }
    ImGui::SetNextWindowSize({760.0f, 320.0f}, ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Log", &g_editor.show_log_panel)) {
        const char* levels[] = {"Trace", "Debug", "Info", "Warn", "Error", "Critical"};
        int filter = std::clamp(static_cast<int>(g_editor.log_filter_level), 0, static_cast<int>(IM_ARRAYSIZE(levels)) - 1);
        if (ImGui::Combo("Min Level", &filter, levels, IM_ARRAYSIZE(levels))) {
            g_editor.log_filter_level = static_cast<lt::LogLevel>(filter);
        }
        ImGui::SameLine();
        ImGui::Checkbox("Auto-scroll", &g_editor.log_auto_scroll);
        ImGui::SameLine();
        if (ImGui::Button("Clear")) {
            g_editor.logs.clear();
        }
        ImGui::SameLine();
        ImGui::TextDisabled("%d records", static_cast<int>(g_editor.logs.size()));
        ImGui::Separator();

        if (ImGui::BeginChild("LogRecords", {0.0f, 0.0f}, true, ImGuiWindowFlags_HorizontalScrollbar)) {
            for (const lt::LogRecord& record : g_editor.logs) {
                if (!passes_log_filter(record)) {
                    continue;
                }
                std::ostringstream line_builder;
                line_builder << '[' << log_time_text(record) << "] ["
                    << lt::log_level_name(record.level) << "] [t"
                    << record.thread_id << "] " << record.message;
                const std::string line = line_builder.str();
                ImGui::PushStyleColor(ImGuiCol_Text, log_level_color(record.level));
                ImGui::TextUnformatted(line.c_str());
                ImGui::PopStyleColor();
                if (ImGui::IsItemHovered() && !record.file.empty()) {
                    ImGui::SetTooltip("%s:%d\n%s", record.file.c_str(), record.line, record.function.c_str());
                }
            }
            if (g_editor.log_auto_scroll) {
                ImGui::SetScrollHereY(1.0f);
            }
        }
        ImGui::EndChild();
    }
    ImGui::End();
}

void draw_ui() {
    drain_editor_logs();
    const float top = draw_top_bar();
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    const float bottom = 26.0f;
    const ImVec2 layout_pos = viewport->Pos;
    const ImVec2 layout_size = viewport->Size;
    if (g_window_minimized || IsIconic(g_hwnd) || layout_size.x <= 480.0f || layout_size.y <= 240.0f) {
        draw_log_panel();
        draw_statistics_panel();
        return;
    }
    const float content_height = std::max(64.0f, layout_size.y - top - bottom);
    const float max_toolbar = std::max(56.0f, layout_size.x - g_editor.properties_width - 160.0f);
    g_editor.toolbar_width = std::clamp(g_editor.toolbar_width, 56.0f, max_toolbar);
    const float max_properties = std::max(220.0f, layout_size.x - g_editor.toolbar_width - 160.0f);
    g_editor.properties_width = std::clamp(g_editor.properties_width, 220.0f, max_properties);
    float viewport_width = std::max(64.0f, layout_size.x - g_editor.toolbar_width - g_editor.properties_width);
    const float outliner_height = content_height * g_editor.outliner_fraction;
    if (g_editor.viewport_fullscreen) {
        ImGui::SetNextWindowPos({layout_pos.x, layout_pos.y + top}, ImGuiCond_Always);
        ImGui::SetNextWindowSize({layout_size.x, content_height}, ImGuiCond_Always);
        draw_viewport();
        draw_loading_overlay();
        draw_irradiance_volume_bake_overlay();
        draw_lightmap_bake_overlay();
        draw_log_panel();
        draw_statistics_panel();
        draw_status_bar();
        launch_render_task();
        return;
    }
    ImGui::SetNextWindowPos({layout_pos.x, layout_pos.y + top}, ImGuiCond_Always);
    ImGui::SetNextWindowSize({g_editor.toolbar_width, content_height}, ImGuiCond_Always);
    draw_toolbar();
    ImGui::SetNextWindowPos({layout_pos.x + g_editor.toolbar_width, layout_pos.y + top}, ImGuiCond_Always);
    ImGui::SetNextWindowSize({viewport_width, content_height}, ImGuiCond_Always);
    draw_viewport();
    ImGui::SetNextWindowPos({layout_pos.x + g_editor.toolbar_width + viewport_width, layout_pos.y + top}, ImGuiCond_Always);
    ImGui::SetNextWindowSize({g_editor.properties_width, outliner_height}, ImGuiCond_Always);
    draw_outliner();
    ImGui::SetNextWindowPos({layout_pos.x + g_editor.toolbar_width + viewport_width, layout_pos.y + top + outliner_height}, ImGuiCond_Always);
    ImGui::SetNextWindowSize({g_editor.properties_width, content_height - outliner_height}, ImGuiCond_Always);
    draw_properties();
    draw_layout_splitters(layout_pos, layout_size, top, content_height, viewport_width, outliner_height);
    draw_loading_overlay();
    draw_irradiance_volume_bake_overlay();
    draw_lightmap_bake_overlay();
    draw_log_panel();
    draw_statistics_panel();
    draw_status_bar();
    launch_render_task();
}

void handle_global_shortcuts() {
    ImGuiIO& io = ImGui::GetIO();
    if (ImGui::IsKeyPressed(ImGuiKey_Escape)) PostQuitMessage(0);
    if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_N)) new_scene();
    static bool save_shortcut_waiting_for_release = false;
    const bool save_shortcut_down = io.KeyCtrl && ImGui::IsKeyDown(ImGuiKey_S);
    if (!save_shortcut_down) {
        save_shortcut_waiting_for_release = false;
    }
    if (!save_shortcut_waiting_for_release && io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_S)) {
        save_shortcut_waiting_for_release = true;
        if (io.KeyShift) {
            save_scene_as_dialog();
        } else {
            save_scene();
        }
    }
    if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_R)) reset_accumulation();
    if (ImGui::IsKeyPressed(ImGuiKey_Delete)) delete_selected();
    if (ImGui::IsKeyPressed(ImGuiKey_Space)) g_editor.paused = !g_editor.paused;
    if (ImGui::IsKeyPressed(ImGuiKey_F11)) g_editor.viewport_fullscreen = !g_editor.viewport_fullscreen;
}

LRESULT CALLBACK window_proc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam) {
    if (ImGui_ImplWin32_WndProcHandler(hwnd, msg, wparam, lparam)) return true;
    switch (msg) {
    case WM_SIZE:
        g_window_minimized = wparam == SIZE_MINIMIZED;
        if (g_window_minimized) {
            return 0;
        }
        if (g_device && wparam != SIZE_MINIMIZED) {
            const UINT width = LOWORD(lparam);
            const UINT height = HIWORD(lparam);
            if (width == 0 || height == 0 || (width == g_swap_chain_width && height == g_swap_chain_height)) {
                return 0;
            }
            cleanup_render_target();
            g_swap_chain->ResizeBuffers(0, width, height, DXGI_FORMAT_UNKNOWN, 0);
            create_render_target();
            g_swap_chain_width = width;
            g_swap_chain_height = height;
        }
        return 0;
    case WM_SYSCOMMAND:
        if ((wparam & 0xfff0) == SC_KEYMENU) return 0;
        break;
    case WM_DESTROY:
        g_shutting_down = true;
        PostQuitMessage(0);
        return 0;
    default:
        break;
    }
    return DefWindowProcW(hwnd, msg, wparam, lparam);
}

std::string command_line_scene() {
    int argc = 0;
    wchar_t** argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    std::string scene = "scenes/cornell.lt";
    if (argv && argc > 1) scene = narrow(argv[1]);
    if (argv) LocalFree(argv);
    return scene;
}

void load_editor_fonts(ImGuiIO& io) {
    const char* candidates[] = {
        "C:\\Windows\\Fonts\\segoeui.ttf",
        "C:\\Windows\\Fonts\\segoeuib.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
    };
    for (const char* path : candidates) {
        if (GetFileAttributesA(path) != INVALID_FILE_ATTRIBUTES) {
            io.Fonts->AddFontFromFileTTF(path, 15.5f);
            return;
        }
    }
    io.Fonts->AddFontDefault();
}

} // namespace

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR, int show_cmd) {
    initialize_editor_logging();
    EditorLoggingScope logging_scope;
    g_editor.scene_path = resolve_startup_scene_path(command_line_scene());
    LT_LOG_INFO("Starting LightTransport Editor with scene '{}'", g_editor.scene_path);
    g_editor.settings.samples_per_pixel = 1;
    g_editor.settings.max_bounces = 5;
#if LT_HAS_CUDA
    if (g_editor.cuda.available()) {
        g_editor.renderer = &g_editor.cuda;
        LT_LOG_INFO("CUDA renderer is available; editor will start with CUDA");
    } else {
        LT_LOG_WARN("CUDA runtime device is unavailable; editor will start with CPU");
    }
#else
    LT_LOG_WARN("CUDA backend is not built; editor will start with CPU");
#endif

    WNDCLASSW wc{};
    wc.lpfnWndProc = window_proc;
    wc.hInstance = instance;
    wc.lpszClassName = L"LightTransportEditor";
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    RegisterClassW(&wc);
    g_hwnd = CreateWindowExW(0, wc.lpszClassName, L"LightTransport Editor", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, 1440, 900, nullptr, nullptr, instance, nullptr);
    if (!create_device(g_hwnd)) {
        LT_LOG_CRITICAL("Failed to create editor D3D11 device");
        cleanup_device();
        UnregisterClassW(wc.lpszClassName, instance);
        return 1;
    }
    ShowWindow(g_hwnd, show_cmd);
    UpdateWindow(g_hwnd);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigWindowsMoveFromTitleBarOnly = true;
    io.IniFilename = nullptr;
    load_editor_fonts(io);
    apply_blender_style();
    load_blender_icons();
    ImGui_ImplWin32_Init(g_hwnd);
    ImGui_ImplDX11_Init(g_device, g_context);
    load_scene_file(g_editor.scene_path);

    bool running = true;
    while (running) {
        MSG msg{};
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
            if (msg.message == WM_QUIT) running = false;
        }
        if (!running) break;
        ImGui_ImplDX11_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();
        poll_scene_load_result();
        handle_global_shortcuts();
        draw_ui();
        ImGui::Render();
        const float clear_color[4] = {0.10f, 0.10f, 0.10f, 1.0f};
        g_context->OMSetRenderTargets(1, &g_main_rtv, nullptr);
        g_context->ClearRenderTargetView(g_main_rtv, clear_color);
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
        g_swap_chain->Present(1, 0);
    }

    release_editor_memory();

    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();
    cleanup_device();
    DestroyWindow(g_hwnd);
    UnregisterClassW(wc.lpszClassName, instance);
    return 0;
}
