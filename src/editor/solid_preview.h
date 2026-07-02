#pragma once

#include "editor_state.h"

#include <d3d11.h>
#include <vector>

namespace lt::editor {

struct SolidPreviewVertex {
    float px, py, pz;
    float nx, ny, nz;
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

    // Constant buffer
    ID3D11Buffer* cb = nullptr;

    // Shaders
    ID3D11VertexShader* vs = nullptr;
    ID3D11PixelShader* ps = nullptr;
    ID3D11InputLayout* input_layout = nullptr;

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

    // Dirty tracking
    uint64_t scene_generation = 0;
    uint64_t content_generation = 0;
};

struct SolidConstantBuffer {
    float view_proj[16];
    float camera_pos[4];
    float light_dir[4];
    float ambient_color[4];
    float clay_color[4];
};

void init_solid_preview(SolidPreview& sp);
void release_solid_preview(SolidPreview& sp);
void resize_solid_preview(SolidPreview& sp, int width, int height);
void update_solid_preview_buffers(SolidPreview& sp, const Scene& scene);
void render_solid_preview(SolidPreview& sp, const Scene& scene, ImVec2 viewport_size, ViewportPreviewMode mode);

extern SolidPreview g_solid_preview;

} // namespace lt::editor
