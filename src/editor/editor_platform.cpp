#include "editor_platform.h"

#include "editor_state.h"
#include "solid_preview.h"

#include <algorithm>
#include <future>

namespace lt::editor {

std::wstring widen(const std::string& text) {
    const int count = MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, nullptr, 0);
    std::wstring output(static_cast<size_t>(std::max(1, count)), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, output.data(), count);
    if (!output.empty() && output.back() == L'\0') {
        output.pop_back();
    }
    return output;
}

std::string narrow(const wchar_t* text) {
    const int count = WideCharToMultiByte(CP_UTF8, 0, text, -1, nullptr, 0, nullptr, nullptr);
    std::string output(static_cast<size_t>(std::max(1, count)), '\0');
    WideCharToMultiByte(CP_UTF8, 0, text, -1, output.data(), count, nullptr, nullptr);
    if (!output.empty() && output.back() == '\0') {
        output.pop_back();
    }
    return output;
}

void release_preview_texture() {
    if (g_preview.srv) {
        g_preview.srv->Release();
    }
    if (g_preview.texture) {
        g_preview.texture->Release();
    }
    g_preview = {};
}

void create_render_target() {
    ID3D11Texture2D* back_buffer = nullptr;
    g_swap_chain->GetBuffer(0, IID_PPV_ARGS(&back_buffer));
    g_device->CreateRenderTargetView(back_buffer, nullptr, &g_main_rtv);
    back_buffer->Release();
    RECT client{};
    if (g_hwnd && GetClientRect(g_hwnd, &client)) {
        g_swap_chain_width = static_cast<UINT>(std::max<LONG>(0, client.right - client.left));
        g_swap_chain_height = static_cast<UINT>(std::max<LONG>(0, client.bottom - client.top));
    }
}

void cleanup_render_target() {
    if (g_main_rtv) {
        g_main_rtv->Release();
        g_main_rtv = nullptr;
    }
}

bool create_device(HWND hwnd) {
    DXGI_SWAP_CHAIN_DESC description{};
    description.BufferCount = 2;
    description.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    description.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    description.OutputWindow = hwnd;
    description.SampleDesc.Count = 1;
    description.Windowed = TRUE;
    description.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
    D3D_FEATURE_LEVEL feature_level{};
    const D3D_FEATURE_LEVEL requested[] = {D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_0};
    if (D3D11CreateDeviceAndSwapChain(
            nullptr,
            D3D_DRIVER_TYPE_HARDWARE,
            nullptr,
            0,
            requested,
            2,
            D3D11_SDK_VERSION,
            &description,
            &g_swap_chain,
            &g_device,
            &feature_level,
            &g_context) != S_OK) {
        return false;
    }
    create_render_target();
    return true;
}

void cleanup_device() {
    release_preview_texture();
    release_solid_preview(g_solid_preview);
    cleanup_render_target();
    if (g_swap_chain) {
        g_swap_chain->Release();
        g_swap_chain = nullptr;
    }
    if (g_context) {
        g_context->ClearState();
        g_context->Flush();
        g_context->Release();
        g_context = nullptr;
    }
    if (g_device) {
        g_device->Release();
        g_device = nullptr;
    }
}

void release_editor_memory() {
    if (g_render_future.valid()) {
        g_render_future.wait();
        (void)g_render_future.get();
    }
    if (g_load_task.future.valid()) {
        g_load_task.future.wait();
        (void)g_load_task.future.get();
    }
    g_editor.cuda.reset();
    g_editor.cpu.reset();
    g_editor.renderer = &g_editor.cpu;
    g_editor.scene = {};
    g_editor.framebuffer = {};
    g_editor.drag_start_mesh = {};
    g_bounds_cache = {};
    g_pick_cache = {};
    g_load_task = {};
    g_solid_preview.scene_generation = 0;
    g_solid_preview.geometry_generation = 0;
}

void reset_accumulation(lt::RenderDirty dirty) {
    g_editor.dirty = g_editor.dirty | dirty | lt::RenderDirty::Render;
    ++g_editor.render_generation;
    if (lt::has_dirty(dirty, lt::RenderDirty::Transform) ||
        lt::has_dirty(dirty, lt::RenderDirty::Geometry)) {
        ++g_editor.geometry_generation;
    }
    if (lt::has_dirty(dirty, lt::RenderDirty::Material) ||
        lt::has_dirty(dirty, lt::RenderDirty::Texture) ||
        lt::has_dirty(dirty, lt::RenderDirty::Geometry) ||
        lt::has_dirty(dirty, lt::RenderDirty::Environment) ||
        lt::has_dirty(dirty, lt::RenderDirty::IrradianceVolume) ||
        lt::has_dirty(dirty, lt::RenderDirty::Lightmap)) {
        ++g_editor.content_generation;
    }
    const uint32_t dirty_bits = static_cast<uint32_t>(dirty);
    const uint32_t camera_render_bits =
        static_cast<uint32_t>(lt::RenderDirty::Camera) |
        static_cast<uint32_t>(lt::RenderDirty::Render);
    const bool camera_only_dirty =
        (dirty_bits & static_cast<uint32_t>(lt::RenderDirty::Camera)) != 0u &&
        (dirty_bits & ~camera_render_bits) == 0u;
    const bool preserve_svgf_camera_history =
        lt::svgf_denoising_enabled(g_editor.settings) && camera_only_dirty;
    if (!preserve_svgf_camera_history) {
        g_editor.frame_index = 0;
    }
    g_editor.framebuffer.clear_accumulation();
}

void release_rendered_preview_resources(bool release_renderer_cache, bool release_framebuffer) {
    if (g_render_future.valid()) {
        g_render_future.wait();
        (void)g_render_future.get();
    }
    release_preview_texture();
    if (release_renderer_cache) {
        g_editor.cuda.reset();
    }
    if (release_framebuffer) {
        g_editor.framebuffer = {};
    }
}

void release_realtime_preview_resources(bool release_shaders) {
    (void)release_shaders;
    release_solid_preview(g_solid_preview);
}

void set_viewport_preview_mode(ViewportPreviewMode mode) {
    if (g_editor.viewport_preview_mode == mode) return;

    const ViewportPreviewMode old_mode = g_editor.viewport_preview_mode;
    g_editor.viewport_preview_mode = mode;

    if (old_mode == ViewportPreviewMode::Rendered) {
        release_rendered_preview_resources(/*release_renderer_cache=*/true,
                                           /*release_framebuffer=*/false);
    } else {
        if (mode == ViewportPreviewMode::Rendered) {
            release_realtime_preview_resources(/*release_shaders=*/true);
            reset_accumulation();
        }
    }
}

} // namespace lt::editor
