#pragma once

#include "lt/log.h"
#include "lt/renderer.h"

#include <windows.h>
#include <d3d11.h>

#include "imgui.h"

#include <chrono>
#include <future>
#include <mutex>
#include <string>
#include <vector>

namespace lt::editor {

enum class ToolMode { Select, Move, Rotate, Scale };
enum class TransformSpace { Local, World };
enum class GizmoHandle { None, AxisX, AxisY, AxisZ, PlaneXY, PlaneYZ, PlaneZX, Uniform };
enum class SelectionKind { None, Mesh, Sphere };
enum class ViewportPreviewMode { Rendered, MaterialPreview, Solid, Wireframe };

struct GpuPreview {
    ID3D11Texture2D* texture = nullptr;
    ID3D11ShaderResourceView* srv = nullptr;
    int width = 0;
    int height = 0;
};

struct EditorState {
    Scene scene = make_default_scene();
    Framebuffer framebuffer;
    RenderSettings settings;
    CpuPathTracer cpu;
    CudaPathTracer cuda;
    IRenderer* renderer = &cpu;
    RenderDirty dirty = RenderDirty::All;
    std::string scene_path = "scenes/cornell.lt";
    std::shared_ptr<IrradianceVolumeBakeProgress> irradiance_volume_bake_progress = std::make_shared<IrradianceVolumeBakeProgress>();
    std::shared_ptr<LightmapBakeProgress> lightmap_bake_progress = std::make_shared<LightmapBakeProgress>();
    uint32_t frame_index = 0;
    uint64_t render_generation = 1;
    uint64_t content_generation = 1;
    uint64_t geometry_generation = 1;
    SelectionKind selection_kind = SelectionKind::Mesh;
    int selected_mesh = 0;
    int selected_sphere = -1;
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
    Mesh drag_start_mesh = {};
    int drag_start_mesh_index = -1;
    Sphere drag_start_sphere = {};
    int drag_start_sphere_index = -1;
    GizmoHandle hovered_gizmo = GizmoHandle::None;
    GizmoHandle active_gizmo = GizmoHandle::None;
    ImVec2 drag_start_center_screen = {};
    ImVec2 drag_start_axis_a_screen = {};
    ImVec2 drag_start_axis_b_screen = {};
    float drag_start_depth = 1.0f;
    float drag_start_angle = 0.0f;
    double last_sample_ms = 0.0;
    float toolbar_width = 74.0f;
    float properties_width = 340.0f;
    float outliner_fraction = 0.38f;
    ViewportPreviewMode viewport_preview_mode = ViewportPreviewMode::Rendered;
    bool viewport_fullscreen = false;
    bool hide_dirty_wireframes = false;
    bool show_log_panel = false;
    bool show_statistics_panel = false;
    bool log_auto_scroll = true;
    LogLevel log_filter_level = LogLevel::Info;
    std::mutex log_mutex;
    std::vector<LogRecord> pending_logs;
    std::vector<LogRecord> logs;
};

struct RenderResult {
    uint64_t generation = 0;
    uint64_t content_generation = 0;
    uint32_t completed_frame = 0;
    double elapsed_ms = 0.0;
    Framebuffer framebuffer;
};

struct SceneLoadTask {
    uint64_t generation = 0;
    std::string path;
    std::chrono::steady_clock::time_point started;
    std::future<SceneLoadResult> future;
};

struct MeshBoundsCache {
    uint64_t scene_generation = 0;
    std::vector<Vec3> local_min;
    std::vector<Vec3> local_max;
};

struct PickSceneCache {
    uint64_t scene_generation = 0;
    RenderScene render_scene;
};

struct ViewTransform {
    Vec3 forward;
    Vec3 right;
    Vec3 up;
    float half_width = 1.0f;
    float half_height = 1.0f;
};

extern EditorState g_editor;
extern GpuPreview g_preview;
extern HWND g_hwnd;
extern ID3D11Device* g_device;
extern ID3D11DeviceContext* g_context;
extern IDXGISwapChain* g_swap_chain;
extern ID3D11RenderTargetView* g_main_rtv;
extern std::future<RenderResult> g_render_future;
extern SceneLoadTask g_load_task;
extern MeshBoundsCache g_bounds_cache;
extern PickSceneCache g_pick_cache;
extern UINT g_swap_chain_width;
extern UINT g_swap_chain_height;
extern bool g_window_minimized;
extern bool g_shutting_down;

} // namespace lt::editor
