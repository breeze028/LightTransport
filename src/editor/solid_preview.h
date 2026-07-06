#pragma once

#include "editor_state.h"

#include <d3d11.h>
#include <cstdint>
#include <vector>

namespace lt::editor {

struct SolidPreviewVertex {
    float px, py, pz;
    float nx, ny, nz;
    float u, v;
    uint32_t object_id = 0;
};

struct SolidSelectionRef {
    SelectionKind kind = SelectionKind::None;
    int object_index = -1;
};

struct SolidDrawRange {
    uint32_t object_id = 0;
    int material_index = -1;
    UINT start_index = 0;
    UINT index_count = 0;
};

struct MaterialPreviewData {
    float base_color[4] = {0.8f, 0.8f, 0.8f, 1.0f};
    float emission[4] = {};
    float params[4] = {0.8f, 0.0f, 0.0f, 0.0f}; // roughness, metallic, alpha, flags
};

struct MaterialPreviewTexture {
    const lt::Texture* source = nullptr;
    ID3D11Texture2D* texture = nullptr;
    ID3D11ShaderResourceView* srv = nullptr;
};

enum class GpuPickResult {
    Unavailable,
    Hit,
    Miss,
};

struct SolidPreview {
    // Resolved render target sampled by ImGui.
    ID3D11Texture2D* rt_texture = nullptr;
    ID3D11RenderTargetView* rtv = nullptr;
    ID3D11ShaderResourceView* srv = nullptr;
    ID3D11Texture2D* depth_texture = nullptr;
    ID3D11DepthStencilView* dsv = nullptr;

    // Optional MSAA render targets resolved into rt_texture.
    ID3D11Texture2D* msaa_texture = nullptr;
    ID3D11RenderTargetView* msaa_rtv = nullptr;
    ID3D11Texture2D* msaa_depth_texture = nullptr;
    ID3D11DepthStencilView* msaa_dsv = nullptr;
    UINT sample_count = 1;
    UINT sample_quality = 0;
    int width = 0;
    int height = 0;

    // Geometry buffers
    ID3D11Buffer* vb = nullptr;
    ID3D11Buffer* ib = nullptr;
    UINT vertex_count = 0;
    UINT index_count = 0;
    std::vector<SolidSelectionRef> selection_refs;
    std::vector<SolidDrawRange> draw_ranges;

    // Constant buffer
    ID3D11Buffer* cb = nullptr;
    ID3D11Buffer* material_cb = nullptr;

    // Shaders
    ID3D11VertexShader* vs = nullptr;
    ID3D11PixelShader* ps = nullptr;
    ID3D11PixelShader* material_ps = nullptr;
    ID3D11InputLayout* input_layout = nullptr;

    ID3D11SamplerState* material_sampler = nullptr;
    std::vector<MaterialPreviewData> material_data;
    std::vector<MaterialPreviewTexture> material_textures;

    // Wire pass
    ID3D11VertexShader* wire_vs = nullptr;
    ID3D11PixelShader* wire_ps = nullptr;
    ID3D11RasterizerState* wire_rasterizer = nullptr;

    // Solid pass rasterizer.
    ID3D11RasterizerState* solid_rasterizer = nullptr;

    // Depth/stencil state for solid pass
    ID3D11DepthStencilState* solid_depth_state = nullptr;
    // Depth state for wire pass (depth test on, depth write off, slight bias)
    ID3D11DepthStencilState* wire_depth_state = nullptr;

    // GPU object picking and selected-object outline pass.
    ID3D11VertexShader* id_vs = nullptr;
    ID3D11PixelShader* id_ps = nullptr;
    ID3D11PixelShader* selected_id_ps = nullptr;
    ID3D11VertexShader* fullscreen_vs = nullptr;
    ID3D11PixelShader* outline_ps = nullptr;

    ID3D11Texture2D* object_id_texture = nullptr;
    ID3D11RenderTargetView* object_id_rtv = nullptr;
    ID3D11Texture2D* pick_readback_texture = nullptr;

    ID3D11Texture2D* scene_depth_texture = nullptr;
    ID3D11DepthStencilView* scene_depth_dsv = nullptr;
    ID3D11ShaderResourceView* scene_depth_srv = nullptr;

    ID3D11Texture2D* selected_id_texture = nullptr;
    ID3D11RenderTargetView* selected_id_rtv = nullptr;
    ID3D11ShaderResourceView* selected_id_srv = nullptr;

    ID3D11Texture2D* selected_depth_texture = nullptr;
    ID3D11DepthStencilView* selected_depth_dsv = nullptr;
    ID3D11ShaderResourceView* selected_depth_srv = nullptr;

    ID3D11Texture2D* outline_texture = nullptr;
    ID3D11RenderTargetView* outline_rtv = nullptr;
    ID3D11ShaderResourceView* outline_srv = nullptr;

    bool outline_valid = false;
    uint64_t outline_generation = 0;
    uint64_t outline_content_generation = 0;
    uint32_t outline_object_id = 0;

    // Dirty tracking
    uint64_t scene_generation = 0;
    uint64_t geometry_generation = 0;
    uint64_t material_generation = 0;
};

struct SolidConstantBuffer {
    float view_proj[16];
    float camera_pos[4];
    float light_dir[4];
    float ambient_color[4];
    float clay_color[4];
    float selection_params[4];
};

void init_solid_preview(SolidPreview& sp);
void release_solid_preview(SolidPreview& sp);
void resize_solid_preview(SolidPreview& sp, int width, int height);
void update_solid_preview_buffers(SolidPreview& sp, const Scene& scene);
void render_solid_preview(SolidPreview& sp, const Scene& scene, ImVec2 viewport_size, ViewportPreviewMode mode);
GpuPickResult pick_solid_preview_object(SolidPreview& sp,
                                        const Scene& scene,
                                        ImVec2 viewport_size,
                                        ImVec2 image_min,
                                        ImVec2 image_max,
                                        ImVec2 mouse,
                                        SelectionKind& kind,
                                        int& object_index);
ID3D11ShaderResourceView* render_selection_outline(SolidPreview& sp,
                                                   const Scene& scene,
                                                   ImVec2 viewport_size,
                                                   SelectionKind kind,
                                                   int object_index);

extern SolidPreview g_solid_preview;

} // namespace lt::editor
