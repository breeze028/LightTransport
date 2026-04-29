#include "lt/renderer.h"

#include <windows.h>
#include <commdlg.h>
#include <shellapi.h>
#include <d3d11.h>

#include "imgui.h"
#include "imgui_impl_dx11.h"
#include "imgui_impl_win32.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstring>
#include <future>
#include <memory>
#include <string>
#include <utility>

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam);

namespace {

enum class ToolMode { Select, Move, Rotate, Scale };
enum class TransformSpace { Local, World };
enum class GizmoHandle { None, AxisX, AxisY, AxisZ, PlaneXY, PlaneYZ, PlaneZX, Uniform };

struct GpuPreview {
    ID3D11Texture2D* texture = nullptr;
    ID3D11ShaderResourceView* srv = nullptr;
    int width = 0;
    int height = 0;
};

struct Editor {
    lt::Scene scene = lt::make_default_scene();
    lt::Framebuffer framebuffer;
    lt::RenderSettings settings;
    lt::CpuPathTracer cpu;
    lt::CudaPathTracer cuda;
    lt::IRenderer* renderer = &cpu;
    lt::RenderDirty dirty = lt::RenderDirty::All;
    std::string scene_path = "scenes/cornell.lt";
    uint32_t frame_index = 0;
    uint64_t render_generation = 1;
    int selected_mesh = 0;
    bool viewport_focused = false;
    bool viewport_hovered = false;
    bool paused = false;
    bool gizmo_dragging = false;
    ToolMode tool_mode = ToolMode::Select;
    TransformSpace move_space = TransformSpace::Local;
    ImVec2 viewport_size = {960.0f, 540.0f};
    ImVec2 viewport_image_min = {};
    ImVec2 viewport_image_max = {};
    ImVec2 drag_start_mouse = {};
    lt::Mesh drag_start_mesh = {};
    int drag_start_mesh_index = -1;
    GizmoHandle hovered_gizmo = GizmoHandle::None;
    GizmoHandle active_gizmo = GizmoHandle::None;
    ImVec2 drag_start_center_screen = {};
    ImVec2 drag_start_axis_a_screen = {};
    ImVec2 drag_start_axis_b_screen = {};
    float drag_start_depth = 1.0f;
    float drag_start_angle = 0.0f;
    double last_sample_ms = 0.0;
};

struct RenderResult {
    uint64_t generation = 0;
    uint32_t completed_frame = 0;
    double elapsed_ms = 0.0;
    lt::Framebuffer framebuffer;
};

struct ViewTransform {
    lt::Vec3 forward;
    lt::Vec3 right;
    lt::Vec3 up;
    float half_width = 1.0f;
    float half_height = 1.0f;
};

Editor g_editor;
GpuPreview g_preview;
HWND g_hwnd = nullptr;
ID3D11Device* g_device = nullptr;
ID3D11DeviceContext* g_context = nullptr;
IDXGISwapChain* g_swap_chain = nullptr;
ID3D11RenderTargetView* g_main_rtv = nullptr;
std::future<RenderResult> g_render_future;

void upload_preview_texture();

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

std::wstring widen(const std::string& text) {
    const int count = MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, nullptr, 0);
    std::wstring out(static_cast<size_t>(std::max(1, count)), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, out.data(), count);
    if (!out.empty() && out.back() == L'\0') out.pop_back();
    return out;
}

std::string narrow(const wchar_t* text) {
    const int count = WideCharToMultiByte(CP_UTF8, 0, text, -1, nullptr, 0, nullptr, nullptr);
    std::string out(static_cast<size_t>(std::max(1, count)), '\0');
    WideCharToMultiByte(CP_UTF8, 0, text, -1, out.data(), count, nullptr, nullptr);
    if (!out.empty() && out.back() == '\0') out.pop_back();
    return out;
}

void release_preview_texture() {
    if (g_preview.srv) g_preview.srv->Release();
    if (g_preview.texture) g_preview.texture->Release();
    g_preview = {};
}

void create_render_target() {
    ID3D11Texture2D* back_buffer = nullptr;
    g_swap_chain->GetBuffer(0, IID_PPV_ARGS(&back_buffer));
    g_device->CreateRenderTargetView(back_buffer, nullptr, &g_main_rtv);
    back_buffer->Release();
}

void cleanup_render_target() {
    if (g_main_rtv) {
        g_main_rtv->Release();
        g_main_rtv = nullptr;
    }
}

bool create_device(HWND hwnd) {
    DXGI_SWAP_CHAIN_DESC sd{};
    sd.BufferCount = 2;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = hwnd;
    sd.SampleDesc.Count = 1;
    sd.Windowed = TRUE;
    sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
    D3D_FEATURE_LEVEL feature_level{};
    const D3D_FEATURE_LEVEL requested[] = {D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_0};
    if (D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, requested, 2, D3D11_SDK_VERSION, &sd, &g_swap_chain, &g_device, &feature_level, &g_context) != S_OK) {
        return false;
    }
    create_render_target();
    return true;
}

void cleanup_device() {
    release_preview_texture();
    cleanup_render_target();
    if (g_swap_chain) g_swap_chain->Release();
    if (g_context) g_context->Release();
    if (g_device) g_device->Release();
}

void reset_accumulation(lt::RenderDirty dirty = lt::RenderDirty::Render) {
    g_editor.dirty = g_editor.dirty | dirty | lt::RenderDirty::Render;
    ++g_editor.render_generation;
    g_editor.frame_index = 0;
    g_editor.framebuffer.clear();
    upload_preview_texture();
}

void set_renderer(bool use_cuda) {
    lt::IRenderer* next = use_cuda && g_editor.cuda.available() ? static_cast<lt::IRenderer*>(&g_editor.cuda) : static_cast<lt::IRenderer*>(&g_editor.cpu);
    if (g_editor.renderer != next) {
        g_editor.renderer = next;
        reset_accumulation(lt::RenderDirty::All);
    }
}

void select_mesh(int index) {
    g_editor.selected_mesh = (index >= 0 && index < static_cast<int>(g_editor.scene.meshes.size())) ? index : -1;
}

bool has_selection() {
    return g_editor.selected_mesh >= 0 && g_editor.selected_mesh < static_cast<int>(g_editor.scene.meshes.size());
}

void set_scene(lt::Scene scene, const std::string& path) {
    g_editor.scene = std::move(scene);
    g_editor.scene_path = path;
    select_mesh(g_editor.scene.meshes.empty() ? -1 : 0);
    reset_accumulation(lt::RenderDirty::All);
}

void load_scene_file(const std::string& path) {
    lt::SceneLoadResult loaded = lt::load_scene(path);
    if (!loaded.error.empty()) {
        MessageBoxW(g_hwnd, widen(loaded.error).c_str(), L"Scene load warning", MB_OK | MB_ICONWARNING);
    }
    set_scene(loaded.scene, path);
}

void open_scene_dialog() {
    wchar_t filename[MAX_PATH] = {};
    OPENFILENAMEW ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = g_hwnd;
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrFilter = L"Scene files (*.lt;*.glb;*.gltf)\0*.lt;*.glb;*.gltf\0LightTransport scenes (*.lt)\0*.lt\0glTF scenes (*.glb;*.gltf)\0*.glb;*.gltf\0All files (*.*)\0*.*\0";
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;
    if (GetOpenFileNameW(&ofn)) load_scene_file(narrow(filename));
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

void load_texture_dialog(lt::Material& material) {
    wchar_t filename[MAX_PATH] = {};
    OPENFILENAMEW ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = g_hwnd;
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrFilter = L"Image textures (*.ppm;*.png;*.jpg;*.jpeg;*.hdr)\0*.ppm;*.png;*.jpg;*.jpeg;*.hdr\0All files (*.*)\0*.*\0";
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;
    if (!GetOpenFileNameW(&ofn)) return;

    const std::string path = narrow(filename);
    const std::string name = texture_name_from_path(path);
    lt::Texture texture;
    std::string error;
    if (!lt::load_texture_file(name, path, texture, error)) {
        MessageBoxW(g_hwnd, widen(error).c_str(), L"Texture load failed", MB_OK | MB_ICONERROR);
        return;
    }
    auto shared = std::make_shared<lt::Texture>(std::move(texture));
    g_editor.scene.textures.push_back(shared);
    material.albedo_texture = shared;
    reset_accumulation(lt::RenderDirty::Texture | lt::RenderDirty::Material);
}

void load_environment_texture_dialog() {
    wchar_t filename[MAX_PATH] = {};
    OPENFILENAMEW ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = g_hwnd;
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrFilter = L"HDRI environment (*.hdr)\0*.hdr\0Image textures (*.hdr;*.ppm;*.png;*.jpg;*.jpeg)\0*.hdr;*.ppm;*.png;*.jpg;*.jpeg\0All files (*.*)\0*.*\0";
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;
    if (!GetOpenFileNameW(&ofn)) return;

    const std::string path = narrow(filename);
    const std::string name = texture_name_from_path(path);
    lt::Texture texture;
    std::string error;
    if (!lt::load_texture_file(name, path, texture, error)) {
        MessageBoxW(g_hwnd, widen(error).c_str(), L"Environment load failed", MB_OK | MB_ICONERROR);
        return;
    }
    auto shared = std::make_shared<lt::Texture>(std::move(texture));
    g_editor.scene.textures.push_back(shared);
    g_editor.scene.environment.texture = shared;
    g_editor.scene.environment.constant = false;
    reset_accumulation(lt::RenderDirty::Texture | lt::RenderDirty::Environment);
}

void save_scene() {
    if (lowercase_extension(g_editor.scene_path) != ".lt") {
        MessageBoxW(g_hwnd, L"Imported glTF/GLB scenes can be edited, but saving back to that format is not supported yet. Save is limited to .lt scene files.", L"Save skipped", MB_OK | MB_ICONINFORMATION);
        return;
    }
    std::string error;
    if (!lt::save_scene(g_editor.scene, g_editor.scene_path, error)) {
        MessageBoxW(g_hwnd, widen(error).c_str(), L"Save failed", MB_OK | MB_ICONERROR);
    }
}

void add_mesh() {
    int material = lt::find_material(g_editor.scene, "white");
    if (material < 0) material = 0;
    const lt::Vec3 forward = lt::normalize(g_editor.scene.camera.target - g_editor.scene.camera.position);
    g_editor.scene.meshes.push_back(lt::make_cube_mesh("Mesh", material, g_editor.scene.camera.position + forward * 1.5f, 0.35f));
    select_mesh(static_cast<int>(g_editor.scene.meshes.size()) - 1);
    reset_accumulation(lt::RenderDirty::Geometry);
}

void duplicate_selected() {
    if (!has_selection()) return;
    lt::Mesh copy = g_editor.scene.meshes[static_cast<size_t>(g_editor.selected_mesh)];
    copy.name += "_copy";
    copy.translation.x += std::max({copy.scale.x, copy.scale.y, copy.scale.z}) * 1.2f;
    g_editor.scene.meshes.push_back(copy);
    select_mesh(static_cast<int>(g_editor.scene.meshes.size()) - 1);
    reset_accumulation(lt::RenderDirty::Geometry);
}

void delete_selected() {
    if (!has_selection()) return;
    g_editor.scene.meshes.erase(g_editor.scene.meshes.begin() + g_editor.selected_mesh);
    select_mesh(g_editor.scene.meshes.empty() ? -1 : std::min(g_editor.selected_mesh, static_cast<int>(g_editor.scene.meshes.size()) - 1));
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
    if (result.generation != g_editor.render_generation) return;
    g_editor.framebuffer = std::move(result.framebuffer);
    g_editor.frame_index = result.completed_frame + 1u;
    g_editor.last_sample_ms = result.elapsed_ms;
    upload_preview_texture();
}

void launch_render_task() {
    if (g_editor.paused || g_render_future.valid()) return;
    lt::Scene scene = g_editor.scene;
    lt::RenderSettings settings = g_editor.settings;
    lt::Framebuffer framebuffer = g_editor.framebuffer;
    const uint64_t generation = g_editor.render_generation;
    lt::IRenderer* renderer = g_editor.renderer;
    settings.frame_index = g_editor.frame_index;
    settings.dirty = g_editor.dirty;
    g_editor.dirty = lt::RenderDirty::None;
    g_render_future = std::async(std::launch::async, [scene = std::move(scene), settings, framebuffer = std::move(framebuffer), generation, renderer]() mutable {
        const auto begin = std::chrono::steady_clock::now();
        renderer->render(scene, settings, framebuffer);
        const auto end = std::chrono::steady_clock::now();
        const double elapsed_ms = std::chrono::duration<double, std::milli>(end - begin).count();
        return RenderResult{generation, settings.frame_index, elapsed_ms, std::move(framebuffer)};
    });
}

void render_preview_if_needed() {
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
    launch_render_task();
}

ViewTransform make_view_transform() {
    const float aspect = g_editor.viewport_size.x / std::max(1.0f, g_editor.viewport_size.y);
    const float half_height = std::tan(g_editor.scene.camera.fov_degrees * lt::kPi / 360.0f);
    const float half_width = aspect * half_height;
    const lt::Vec3 forward = lt::normalize(g_editor.scene.camera.target - g_editor.scene.camera.position);
    const lt::Vec3 right = lt::normalize(lt::cross(forward, g_editor.scene.camera.up));
    const lt::Vec3 up = lt::cross(right, forward);
    return {forward, right, up, half_width, half_height};
}

lt::Ray make_view_ray_from_screen(ImVec2 mouse) {
    const ViewTransform view = make_view_transform();
    const ImVec2 min = g_editor.viewport_image_min;
    const ImVec2 size = {std::max(1.0f, g_editor.viewport_image_max.x - min.x), std::max(1.0f, g_editor.viewport_image_max.y - min.y)};
    const float u = ((mouse.x - min.x) / size.x * 2.0f - 1.0f) * view.half_width;
    const float v = (1.0f - (mouse.y - min.y) / size.y * 2.0f) * view.half_height;
    return {g_editor.scene.camera.position, lt::normalize(view.forward + view.right * u + view.up * v)};
}

bool project_point(lt::Vec3 point, ImVec2& screen, float* depth = nullptr) {
    const ViewTransform view = make_view_transform();
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

bool pick_mesh(ImVec2 mouse, int& mesh_index) {
    const lt::Ray ray = make_view_ray_from_screen(mouse);
    const lt::RenderScene render_scene = lt::build_render_scene(g_editor.scene);
    float best_t = lt::kInfinity;
    mesh_index = -1;
    for (const lt::Triangle& tri : render_scene.triangles) {
        float t = 0.0f;
        if (intersect_triangle(tri, ray, t) && t < best_t) {
            best_t = t;
            mesh_index = tri.mesh;
        }
    }
    return mesh_index >= 0;
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

void move_camera(float right_delta, float up_delta, float forward_delta) {
    lt::Camera& camera = g_editor.scene.camera;
    const ViewTransform view = make_view_transform();
    const lt::Vec3 delta = view.right * right_delta + view.up * up_delta + view.forward * forward_delta;
    camera.position += delta;
    camera.target += delta;
    reset_accumulation(lt::RenderDirty::Camera);
}

void rotate_camera(float yaw, float pitch) {
    lt::Camera& camera = g_editor.scene.camera;
    lt::Vec3 dir = lt::normalize(camera.target - camera.position);
    const float radius = lt::length(camera.target - camera.position);
    float current_yaw = std::atan2(dir.x, dir.z);
    float current_pitch = std::asin(std::clamp(dir.y, -0.99f, 0.99f));
    current_yaw += yaw;
    current_pitch = std::clamp(current_pitch + pitch, -1.45f, 1.45f);
    camera.target = camera.position + lt::Vec3{std::sin(current_yaw) * std::cos(current_pitch), std::sin(current_pitch), std::cos(current_yaw) * std::cos(current_pitch)} * radius;
    reset_accumulation(lt::RenderDirty::Camera);
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

GizmoHandle pick_move_gizmo(const lt::Mesh& mesh, lt::Vec3 center3, ImVec2 mouse, bool allow_uniform) {
    ImVec2 c{}, x{}, y{}, z{}, xy{}, yz{}, zx{};
    if (!project_point(center3, c)) return GizmoHandle::None;
    if (allow_uniform && distance(mouse, c) <= 10.0f) {
        return GizmoHandle::Uniform;
    }
    const lt::Vec3 ax = transform_axis(mesh, {1.0f, 0.0f, 0.0f});
    const lt::Vec3 ay = transform_axis(mesh, {0.0f, 1.0f, 0.0f});
    const lt::Vec3 az = transform_axis(mesh, {0.0f, 0.0f, 1.0f});
    project_point(center3 + ax * 0.55f, x);
    project_point(center3 + ay * 0.55f, y);
    project_point(center3 + az * 0.55f, z);
    project_point(center3 + (ax + ay) * 0.28f, xy);
    project_point(center3 + (ay + az) * 0.28f, yz);
    project_point(center3 + (az + ax) * 0.28f, zx);
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

GizmoHandle pick_rotate_gizmo(const lt::Mesh& mesh, lt::Vec3 center3, ImVec2 mouse) {
    ImVec2 c{}, px{}, py{}, pz{};
    if (!project_point(center3, c)) return GizmoHandle::None;
    project_point(center3 + local_axis(mesh, {1.0f, 0.0f, 0.0f}) * 0.7f, px);
    project_point(center3 + local_axis(mesh, {0.0f, 1.0f, 0.0f}) * 0.7f, py);
    project_point(center3 + local_axis(mesh, {0.0f, 0.0f, 1.0f}) * 0.7f, pz);
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

void handle_gizmo_drag() {
    if (!has_selection()) {
        g_editor.hovered_gizmo = GizmoHandle::None;
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
    const ViewTransform view = make_view_transform();
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
                reset_accumulation(lt::RenderDirty::Geometry);
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
    reset_accumulation(lt::RenderDirty::Geometry);
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
    if (!has_selection()) return;
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

void draw_mesh_bounds_outline(ImDrawList* draw_list, const lt::Mesh& mesh, ImU32 color, float thickness) {
    if (mesh.vertices.empty()) return;
    lt::Vec3 local_min{lt::kInfinity, lt::kInfinity, lt::kInfinity};
    lt::Vec3 local_max{-lt::kInfinity, -lt::kInfinity, -lt::kInfinity};
    for (lt::Vec3 v : mesh.vertices) {
        local_min = lt::min(local_min, v);
        local_max = lt::max(local_max, v);
    }
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

void draw_scene_reference_outlines(ImDrawList* draw_list) {
    const ImU32 mesh_color = IM_COL32(150, 170, 190, 130);
    const ImU32 selected_color = IM_COL32(255, 150, 35, 230);
    for (int i = 0; i < static_cast<int>(g_editor.scene.meshes.size()); ++i) {
        const bool selected = i == g_editor.selected_mesh;
        draw_mesh_bounds_outline(draw_list, g_editor.scene.meshes[static_cast<size_t>(i)], selected ? selected_color : mesh_color, selected ? 2.0f : 1.0f);
    }
}

void draw_gizmo_overlay() {
    if (!has_selection()) return;
    lt::Vec3 center3{};
    float radius3 = 1.0f;
    mesh_center_radius(g_editor.scene.meshes[static_cast<size_t>(g_editor.selected_mesh)], center3, radius3);
    ImVec2 center{};
    if (!project_point(center3, center)) return;
    const float radius = std::max(18.0f, radius3 * 90.0f);
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    if (g_editor.tool_mode == ToolMode::Select) {
        draw_mesh_bounds_outline(draw_list, g_editor.scene.meshes[static_cast<size_t>(g_editor.selected_mesh)], IM_COL32(255, 150, 35, 255), 2.0f);
    }
    if (g_editor.tool_mode == ToolMode::Move || g_editor.tool_mode == ToolMode::Scale) {
        ImVec2 px{}, py{}, pz{}, pxy{}, pyz{}, pzx{};
        const lt::Mesh& mesh = g_editor.scene.meshes[static_cast<size_t>(g_editor.selected_mesh)];
        const lt::Vec3 ax = transform_axis(mesh, {1.0f, 0.0f, 0.0f});
        const lt::Vec3 ay = transform_axis(mesh, {0.0f, 1.0f, 0.0f});
        const lt::Vec3 az = transform_axis(mesh, {0.0f, 0.0f, 1.0f});
        const bool okx = project_point(center3 + ax * 0.55f, px);
        const bool oky = project_point(center3 + ay * 0.55f, py);
        const bool okz = project_point(center3 + az * 0.55f, pz);
        const bool okxy = project_point(center3 + (ax + ay) * 0.28f, pxy);
        const bool okyz = project_point(center3 + (ay + az) * 0.28f, pyz);
        const bool okzx = project_point(center3 + (az + ax) * 0.28f, pzx);
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
        if (g_editor.tool_mode == ToolMode::Scale) {
            draw_gizmo_center(draw_list, center);
        }
    } else if (g_editor.tool_mode == ToolMode::Rotate) {
        const lt::Mesh& mesh = g_editor.scene.meshes[static_cast<size_t>(g_editor.selected_mesh)];
        ImVec2 px{}, py{}, pz{};
        project_point(center3 + local_axis(mesh, {1.0f, 0.0f, 0.0f}) * 0.7f, px);
        project_point(center3 + local_axis(mesh, {0.0f, 1.0f, 0.0f}) * 0.7f, py);
        project_point(center3 + local_axis(mesh, {0.0f, 0.0f, 1.0f}) * 0.7f, pz);
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
        reset_accumulation(lt::RenderDirty::Geometry);
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
        reset_accumulation(lt::RenderDirty::Geometry);
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

bool toolbar_button(ToolMode mode) {
    const bool active = g_editor.tool_mode == mode;
    if (active) {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.95f, 0.50f, 0.16f, 0.58f));
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.93f, 0.82f, 1.0f));
    }
    const bool clicked = ImGui::Button(tool_mode_icon(mode), ImVec2(36.0f, 34.0f));
    if (active) {
        ImGui::PopStyleColor(2);
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("%s", tool_mode_name(mode));
    }
    if (clicked) {
        g_editor.tool_mode = mode;
    }
    return clicked;
}

void draw_top_bar() {
    if (!ImGui::BeginMainMenuBar()) return;
    if (ImGui::BeginMenu("File")) {
        if (ImGui::MenuItem("Open", "O")) open_scene_dialog();
        if (ImGui::MenuItem("Save", "Ctrl+S")) save_scene();
        if (ImGui::MenuItem("Exit", "Esc")) PostQuitMessage(0);
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Object")) {
        if (ImGui::MenuItem("Add Mesh", "Shift+A")) add_mesh();
        if (ImGui::MenuItem("Duplicate", "Ctrl+D", false, has_selection())) duplicate_selected();
        if (ImGui::MenuItem("Delete", "Del", false, has_selection())) delete_selected();
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Render")) {
        if (ImGui::MenuItem(g_editor.paused ? "Resume" : "Pause", "Space")) g_editor.paused = !g_editor.paused;
        if (ImGui::MenuItem("Reset Accumulation", "Ctrl+R")) reset_accumulation();
        ImGui::EndMenu();
    }
    ImGui::Separator();
    ImGui::Text("%s | %s | samples %u | %.2f ms | %s",
                tool_mode_name(g_editor.tool_mode),
                g_editor.renderer->name(),
                g_editor.frame_index,
                g_editor.last_sample_ms,
                g_editor.scene_path.c_str());
    ImGui::EndMainMenuBar();
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
    if (ImGui::Button(world ? "World" : "Local", ImVec2(56.0f, 28.0f))) {
        g_editor.move_space = world ? TransformSpace::Local : TransformSpace::World;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Move gizmo space");
    }
    ImGui::End();
}

void draw_outliner() {
    ImGui::Begin("Scene Collection", nullptr, ImGuiWindowFlags_NoCollapse);
    if (ImGui::TreeNodeEx("Meshes", ImGuiTreeNodeFlags_DefaultOpen)) {
        for (int i = 0; i < static_cast<int>(g_editor.scene.meshes.size()); ++i) {
            if (ImGui::Selectable(g_editor.scene.meshes[static_cast<size_t>(i)].name.c_str(), g_editor.selected_mesh == i)) select_mesh(i);
        }
        ImGui::TreePop();
    }
    ImGui::End();
}

void draw_properties() {
    ImGui::Begin("Properties", nullptr, ImGuiWindowFlags_NoCollapse);
    if (ImGui::BeginTabBar("PropertiesTabs")) {
        if (ImGui::BeginTabItem("Object")) {
            if (has_selection()) {
                lt::Mesh& mesh = g_editor.scene.meshes[static_cast<size_t>(g_editor.selected_mesh)];
                char name[128] = {};
                std::strncpy(name, mesh.name.c_str(), sizeof(name) - 1);
                if (ImGui::InputText("Name", name, sizeof(name))) mesh.name = name;
                edit_location_world("Location", mesh);
                edit_rotation_degrees("Rotation", mesh.rotation);
                if (edit_vec3("Scale", mesh.scale, 0.01f, lt::RenderDirty::Geometry)) {
                    mesh.scale.x = std::max(0.01f, mesh.scale.x);
                    mesh.scale.y = std::max(0.01f, mesh.scale.y);
                    mesh.scale.z = std::max(0.01f, mesh.scale.z);
                }
                if (ImGui::Combo("Material", &mesh.material, [](void* data, int idx, const char** out) {
                    const auto* scene = static_cast<const lt::Scene*>(data);
                    *out = scene->materials[static_cast<size_t>(idx)]->name.c_str();
                    return true;
                }, &g_editor.scene, static_cast<int>(g_editor.scene.materials.size()))) reset_accumulation(lt::RenderDirty::Geometry);
                if (ImGui::Checkbox("Light", &mesh.light.enabled)) reset_accumulation(lt::RenderDirty::Geometry);
                if (mesh.light.enabled) {
                    color_edit("Light Color", mesh.light.color, lt::RenderDirty::Geometry);
                    if (ImGui::DragFloat("Intensity", &mesh.light.intensity, 0.1f, 0.0f, 1000.0f, "%.2f")) {
                        mesh.light.intensity = std::max(0.0f, mesh.light.intensity);
                        reset_accumulation(lt::RenderDirty::Geometry);
                    }
                }
                ImGui::Text("Vertices: %d", static_cast<int>(mesh.vertices.size()));
                ImGui::Text("Triangles: %d", static_cast<int>(mesh.indices.size() / 3));
            } else {
                ImGui::TextDisabled("No mesh selected");
            }
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Material")) {
            for (std::shared_ptr<lt::Material>& material : g_editor.scene.materials) {
                if (!material) {
                    continue;
                }
                if (ImGui::TreeNodeEx(material->name.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
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
                        reset_accumulation(lt::RenderDirty::Material);
                    }
                    if (ImGui::Button("Load Texture")) {
                        load_texture_dialog(*material);
                    }
                    const char* models[] = {"Lambertian", "Principled", "Mirror", "Dielectric", "Conductor"};
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
                        }
                        material = lt::make_material(name, albedo, static_cast<lt::BrdfModel>(model), roughness, metallic);
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
                    }
                    if (auto* dielectric = dynamic_cast<lt::DielectricMaterial*>(material.get())) {
                        if (ImGui::DragFloat("IOR", &dielectric->ior, 0.01f, 1.0f, 3.0f, "%.2f")) {
                            dielectric->ior = std::clamp(dielectric->ior, 1.0f, 3.0f);
                            reset_accumulation(lt::RenderDirty::Material);
                        }
                    }
                    ImGui::TreePop();
                }
            }
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Camera")) {
            edit_vec3("Position", g_editor.scene.camera.position, 0.02f, lt::RenderDirty::Camera);
            edit_vec3("Target", g_editor.scene.camera.target, 0.02f, lt::RenderDirty::Camera);
            if (ImGui::DragFloat("FOV", &g_editor.scene.camera.fov_degrees, 0.2f, 10.0f, 120.0f, "%.1f")) {
                g_editor.scene.camera.fov_degrees = std::clamp(g_editor.scene.camera.fov_degrees, 10.0f, 120.0f);
                reset_accumulation(lt::RenderDirty::Camera);
            }
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Render")) {
            const bool using_cuda = g_editor.renderer == &g_editor.cuda;
            const char* renderer_name = using_cuda ? "CUDA Path Tracer" : "CPU Path Tracer";
            if (ImGui::BeginCombo("Renderer", renderer_name)) {
                if (ImGui::Selectable("CPU Path Tracer", !using_cuda)) {
                    set_renderer(false);
                }
                const bool cuda_available = g_editor.cuda.available();
                ImGui::BeginDisabled(!cuda_available);
                if (ImGui::Selectable("CUDA Path Tracer", using_cuda)) {
                    set_renderer(true);
                }
                ImGui::EndDisabled();
                ImGui::EndCombo();
            }
            if (ImGui::Checkbox("Multiple importance sampling", &g_editor.settings.use_mis)) {
                reset_accumulation();
            }
            ImGui::BeginDisabled(!g_editor.settings.use_mis);
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
            const char* accel_modes[] = {"Auto (Flat)", "Flat BVH", "Two-level BVH"};
            int accel_mode = static_cast<int>(g_editor.settings.acceleration_structure);
            if (ImGui::Combo("Acceleration", &accel_mode, accel_modes, IM_ARRAYSIZE(accel_modes))) {
                g_editor.settings.acceleration_structure = static_cast<lt::AccelerationStructure>(accel_mode);
                reset_accumulation(lt::RenderDirty::Geometry);
            }
            if (ImGui::Checkbox("Primary hit cache", &g_editor.settings.use_primary_hit_cache)) {
                reset_accumulation(lt::RenderDirty::Camera);
            }
            ImGui::SeparatorText("Environment");
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
            if (ImGui::Button("Load HDRI")) {
                load_environment_texture_dialog();
            }
            ImGui::SameLine();
            if (ImGui::Button("Clear HDRI")) {
                g_editor.scene.environment.texture = nullptr;
                g_editor.scene.environment.constant = false;
                reset_accumulation(lt::RenderDirty::Environment);
            }
            ImGui::Checkbox("Pause", &g_editor.paused);
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    ImGui::End();
}

void handle_viewport_input() {
    ImGuiIO& io = ImGui::GetIO();
    if (!g_editor.viewport_hovered && !g_editor.viewport_focused) return;
    if (g_editor.gizmo_dragging) return;
    const float step = io.KeyShift ? 0.18f : 0.06f;
    if (ImGui::IsKeyDown(ImGuiKey_W)) move_camera(0.0f, 0.0f, step);
    if (ImGui::IsKeyDown(ImGuiKey_S)) move_camera(0.0f, 0.0f, -step);
    if (ImGui::IsKeyDown(ImGuiKey_A)) move_camera(-step, 0.0f, 0.0f);
    if (ImGui::IsKeyDown(ImGuiKey_D)) move_camera(step, 0.0f, 0.0f);
    if (ImGui::IsKeyDown(ImGuiKey_Q)) move_camera(0.0f, -step, 0.0f);
    if (ImGui::IsKeyDown(ImGuiKey_E)) move_camera(0.0f, step, 0.0f);
    if (ImGui::IsMouseDragging(ImGuiMouseButton_Right)) {
        const ImVec2 delta = io.MouseDelta;
        rotate_camera(delta.x * -0.004f, delta.y * -0.004f);
    }
}

void draw_viewport() {
    ImGui::Begin("Viewport", nullptr, ImGuiWindowFlags_NoCollapse);
    ImVec2 available = ImGui::GetContentRegionAvail();
    available.x = std::max(64.0f, available.x);
    available.y = std::max(64.0f, available.y - 26.0f);
    g_editor.viewport_size = available;
    render_preview_if_needed();
    if (g_preview.srv) {
        g_editor.viewport_image_min = ImGui::GetCursorScreenPos();
        ImGui::Image(reinterpret_cast<ImTextureID>(g_preview.srv), available);
        g_editor.viewport_image_max = {g_editor.viewport_image_min.x + available.x, g_editor.viewport_image_min.y + available.y};
    }
    g_editor.viewport_hovered = ImGui::IsItemHovered();
    g_editor.viewport_focused = ImGui::IsWindowFocused();
    if (g_editor.viewport_hovered && g_editor.tool_mode == ToolMode::Select && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        int mesh = -1;
        select_mesh(pick_mesh(ImGui::GetIO().MousePos, mesh) ? mesh : -1);
    }
    handle_gizmo_drag();
    if (g_editor.frame_index == 0) {
        draw_scene_reference_outlines(ImGui::GetWindowDrawList());
    }
    draw_gizmo_overlay();
    if (g_editor.viewport_hovered) {
        if (g_editor.tool_mode == ToolMode::Move) ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeAll);
        if (g_editor.tool_mode == ToolMode::Rotate) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        if (g_editor.tool_mode == ToolMode::Scale) ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNWSE);
    }
    handle_viewport_input();
    ImGui::Text("%s tool | LMB select/drag | RMB look | WASD/QE fly | G/R/S tools", tool_mode_name(g_editor.tool_mode));
    ImGui::End();
}

void draw_status_bar() {
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos({viewport->WorkPos.x, viewport->WorkPos.y + viewport->WorkSize.y - 26.0f});
    ImGui::SetNextWindowSize({viewport->WorkSize.x, 26.0f});
    ImGui::Begin("Status", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings);
    size_t vertex_count = 0;
    size_t triangle_count = 0;
    for (const lt::Mesh& mesh : g_editor.scene.meshes) {
        vertex_count += mesh.vertices.size();
        triangle_count += mesh.indices.size() / 3;
    }
    ImGui::Text("Meshes: %d | Vertices: %zu | Triangles: %zu | %dx%d",
                static_cast<int>(g_editor.scene.meshes.size()),
                vertex_count,
                triangle_count,
                g_editor.settings.width,
                g_editor.settings.height);
    ImGui::End();
}

void draw_ui() {
    draw_top_bar();
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    const float top = ImGui::GetFrameHeight();
    const float bottom = 26.0f;
    const float left = 74.0f;
    const float right = 340.0f;
    const ImVec2 work_pos = viewport->WorkPos;
    const ImVec2 work_size = viewport->WorkSize;
    const float content_height = std::max(64.0f, work_size.y - top - bottom);
    const float viewport_width = std::max(64.0f, work_size.x - left - right);
    ImGui::SetNextWindowPos({work_pos.x, work_pos.y + top}, ImGuiCond_Always);
    ImGui::SetNextWindowSize({left, content_height}, ImGuiCond_Always);
    draw_toolbar();
    ImGui::SetNextWindowPos({work_pos.x + left, work_pos.y + top}, ImGuiCond_Always);
    ImGui::SetNextWindowSize({viewport_width, content_height}, ImGuiCond_Always);
    draw_viewport();
    ImGui::SetNextWindowPos({work_pos.x + left + viewport_width, work_pos.y + top}, ImGuiCond_Always);
    ImGui::SetNextWindowSize({right, content_height * 0.38f}, ImGuiCond_Always);
    draw_outliner();
    ImGui::SetNextWindowPos({work_pos.x + left + viewport_width, work_pos.y + top + content_height * 0.38f}, ImGuiCond_Always);
    ImGui::SetNextWindowSize({right, content_height * 0.62f}, ImGuiCond_Always);
    draw_properties();
    draw_status_bar();
}

void handle_global_shortcuts() {
    ImGuiIO& io = ImGui::GetIO();
    if (ImGui::IsKeyPressed(ImGuiKey_Escape)) PostQuitMessage(0);
    if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_S)) save_scene();
    if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_R)) reset_accumulation();
    if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_D)) duplicate_selected();
    if (ImGui::IsKeyPressed(ImGuiKey_Delete)) delete_selected();
    if (ImGui::IsKeyPressed(ImGuiKey_O)) open_scene_dialog();
    if (ImGui::IsKeyPressed(ImGuiKey_Space)) g_editor.paused = !g_editor.paused;
    if (!io.KeyCtrl) {
        if (io.KeyShift && ImGui::IsKeyPressed(ImGuiKey_A)) add_mesh();
    }
}

LRESULT CALLBACK window_proc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam) {
    if (ImGui_ImplWin32_WndProcHandler(hwnd, msg, wparam, lparam)) return true;
    switch (msg) {
    case WM_SIZE:
        if (g_device && wparam != SIZE_MINIMIZED) {
            cleanup_render_target();
            g_swap_chain->ResizeBuffers(0, LOWORD(lparam), HIWORD(lparam), DXGI_FORMAT_UNKNOWN, 0);
            create_render_target();
        }
        return 0;
    case WM_SYSCOMMAND:
        if ((wparam & 0xfff0) == SC_KEYMENU) return 0;
        break;
    case WM_DESTROY:
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
    g_editor.scene_path = command_line_scene();
    lt::SceneLoadResult loaded = lt::load_scene(g_editor.scene_path);
    if (loaded.error.empty()) g_editor.scene = loaded.scene;
    g_editor.settings.samples_per_pixel = 1;
    g_editor.settings.max_bounces = 5;
    select_mesh(g_editor.scene.meshes.empty() ? -1 : 0);
    if (g_editor.cuda.available()) g_editor.renderer = &g_editor.cuda;

    WNDCLASSW wc{};
    wc.lpfnWndProc = window_proc;
    wc.hInstance = instance;
    wc.lpszClassName = L"LightTransportEditor";
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    RegisterClassW(&wc);
    g_hwnd = CreateWindowExW(0, wc.lpszClassName, L"LightTransport Editor", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, 1440, 900, nullptr, nullptr, instance, nullptr);
    if (!create_device(g_hwnd)) {
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
    ImGui_ImplWin32_Init(g_hwnd);
    ImGui_ImplDX11_Init(g_device, g_context);

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
        handle_global_shortcuts();
        draw_ui();
        ImGui::Render();
        const float clear_color[4] = {0.10f, 0.10f, 0.10f, 1.0f};
        g_context->OMSetRenderTargets(1, &g_main_rtv, nullptr);
        g_context->ClearRenderTargetView(g_main_rtv, clear_color);
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
        g_swap_chain->Present(1, 0);
    }

    if (g_render_future.valid()) {
        g_render_future.wait();
    }

    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();
    cleanup_device();
    DestroyWindow(g_hwnd);
    UnregisterClassW(wc.lpszClassName, instance);
    return 0;
}
