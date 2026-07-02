#include "solid_preview.h"

#include "lt/scene.h"

#include <d3dcompiler.h>

#include <algorithm>
#include <cmath>
#include <cstring>

#pragma comment(lib, "d3dcompiler.lib")

namespace lt::editor {

// ---------------------------------------------------------------------------
// Embedded HLSL shader source
// ---------------------------------------------------------------------------

static const char* kSolidShaderSource = R"(
cbuffer ConstantBuffer : register(b0) {
    row_major float4x4 view_proj;
    float4 camera_pos;
    float4 light_dir;
    float4 ambient_color;
    float4 clay_color;
};

struct VSInput {
    float3 position : POSITION;
    float3 normal   : NORMAL;
};

struct PSInput {
    float4 position : SV_POSITION;
    float3 normal   : NORMAL;
    float3 world_pos : TEXCOORD0;
};

PSInput VSMain(VSInput input) {
    PSInput output;
    output.position = mul(float4(input.position, 1.0f), view_proj);
    output.normal   = normalize(input.normal);
    output.world_pos = input.position;
    return output;
}

float4 PSMain(PSInput input) : SV_TARGET {
    float3 N = normalize(input.normal);
    float3 L = normalize(light_dir.xyz);

    // Viewport clay shading: stable, scene-light independent, and double-sided.
    float NdotL = abs(dot(N, L));
    float tone = 0.30f
               + smoothstep(0.02f, 0.32f, NdotL) * 0.35f
               + smoothstep(0.35f, 0.88f, NdotL) * 0.30f;

    float3 color = clay_color.rgb * (ambient_color.rgb + tone);

    // Rim light
    float3 V = normalize(camera_pos.xyz - input.world_pos);
    float rim = 1.0f - abs(dot(N, V));
    rim = pow(saturate(rim), 3.0f);
    color += rim * 0.12f;

    return float4(color, 1.0f);
}
)";

static const char* kWireShaderSource = R"(
cbuffer ConstantBuffer : register(b0) {
    row_major float4x4 view_proj;
    float4 camera_pos;
    float4 light_dir;
    float4 ambient_color;
    float4 clay_color;
};

struct VSInput {
    float3 position : POSITION;
    float3 normal   : NORMAL;
};

struct PSInput {
    float4 position : SV_POSITION;
};

PSInput VSMain(VSInput input) {
    PSInput output;
    output.position = mul(float4(input.position, 1.0f), view_proj);
    return output;
}

float4 PSMain(PSInput input) : SV_TARGET {
    return float4(0.08f, 0.09f, 0.10f, 1.0f);
}
)";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

bool compile_shader(const char* source, size_t source_len, const char* entry,
                    const char* target, ID3DBlob** blob) {
    ID3DBlob* error_blob = nullptr;
    UINT flags = D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_WARNINGS_ARE_ERRORS;
#if defined(_DEBUG)
    flags |= D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#else
    flags |= D3DCOMPILE_OPTIMIZATION_LEVEL3;
#endif
    HRESULT hr = D3DCompile(
        source, source_len, nullptr, nullptr, nullptr,
        entry, target, flags, 0, blob, &error_blob);
    if (FAILED(hr) && error_blob) {
        LT_LOG_ERROR("Solid preview shader compile error: {}",
            static_cast<const char*>(error_blob->GetBufferPointer()));
        error_blob->Release();
        return false;
    }
    if (error_blob) error_blob->Release();
    return SUCCEEDED(hr);
}

int create_solid_shaders(SolidPreview& sp) {
    ID3DBlob* vs_blob = nullptr;
    ID3DBlob* ps_blob = nullptr;
    ID3DBlob* wire_vs_blob = nullptr;
    ID3DBlob* wire_ps_blob = nullptr;
    int ok = 0;

    size_t solid_len = std::strlen(kSolidShaderSource);
    size_t wire_len = std::strlen(kWireShaderSource);

    if (!compile_shader(kSolidShaderSource, solid_len, "VSMain", "vs_5_0", &vs_blob)) return ok;
    if (!compile_shader(kSolidShaderSource, solid_len, "PSMain", "ps_5_0", &ps_blob)) { vs_blob->Release(); return ok; }
    if (!compile_shader(kWireShaderSource, wire_len, "VSMain", "vs_5_0", &wire_vs_blob)) { vs_blob->Release(); ps_blob->Release(); return ok; }
    if (!compile_shader(kWireShaderSource, wire_len, "PSMain", "ps_5_0", &wire_ps_blob)) {
        vs_blob->Release(); ps_blob->Release(); wire_vs_blob->Release(); return ok;
    }

    do {
        if (g_device->CreateVertexShader(vs_blob->GetBufferPointer(), vs_blob->GetBufferSize(), nullptr, &sp.vs) != S_OK) break;
        if (g_device->CreatePixelShader(ps_blob->GetBufferPointer(), ps_blob->GetBufferSize(), nullptr, &sp.ps) != S_OK) break;
        if (g_device->CreateVertexShader(wire_vs_blob->GetBufferPointer(), wire_vs_blob->GetBufferSize(), nullptr, &sp.wire_vs) != S_OK) break;
        if (g_device->CreatePixelShader(wire_ps_blob->GetBufferPointer(), wire_ps_blob->GetBufferSize(), nullptr, &sp.wire_ps) != S_OK) break;

        D3D11_INPUT_ELEMENT_DESC layout[] = {
            {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"NORMAL",   0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0},
        };
        if (g_device->CreateInputLayout(layout, 2, vs_blob->GetBufferPointer(), vs_blob->GetBufferSize(), &sp.input_layout) != S_OK) break;

        ok = 1;
    } while (false);

    vs_blob->Release();
    ps_blob->Release();
    wire_vs_blob->Release();
    wire_ps_blob->Release();
    return ok;
}

void create_solid_states(SolidPreview& sp) {
    // Solid preview should show imported meshes even when winding is mixed or
    // transforms mirror an axis, so keep it double-sided.
    D3D11_RASTERIZER_DESC rs_desc{};
    rs_desc.FillMode = D3D11_FILL_SOLID;
    rs_desc.CullMode = D3D11_CULL_NONE;
    rs_desc.FrontCounterClockwise = false;
    rs_desc.DepthClipEnable = true;
    g_device->CreateRasterizerState(&rs_desc, &sp.solid_rasterizer);

    // Wire rasterizer: wireframe, no cull
    rs_desc.FillMode = D3D11_FILL_WIREFRAME;
    rs_desc.CullMode = D3D11_CULL_NONE;
    rs_desc.DepthBias = -1;
    rs_desc.DepthBiasClamp = 0.0f;
    rs_desc.SlopeScaledDepthBias = -1.5f;
    g_device->CreateRasterizerState(&rs_desc, &sp.wire_rasterizer);

    // Solid depth/stencil: write enabled
    D3D11_DEPTH_STENCIL_DESC ds_desc{};
    ds_desc.DepthEnable = true;
    ds_desc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
    ds_desc.DepthFunc = D3D11_COMPARISON_LESS_EQUAL;
    g_device->CreateDepthStencilState(&ds_desc, &sp.solid_depth_state);

    // Wire depth/stencil: test on, write off
    ds_desc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
    g_device->CreateDepthStencilState(&ds_desc, &sp.wire_depth_state);
}

void release_render_targets(SolidPreview& sp) {
    if (sp.srv) { sp.srv->Release(); sp.srv = nullptr; }
    if (sp.rtv) { sp.rtv->Release(); sp.rtv = nullptr; }
    if (sp.rt_texture) { sp.rt_texture->Release(); sp.rt_texture = nullptr; }
    if (sp.dsv) { sp.dsv->Release(); sp.dsv = nullptr; }
    if (sp.depth_texture) { sp.depth_texture->Release(); sp.depth_texture = nullptr; }
    if (sp.msaa_rtv) { sp.msaa_rtv->Release(); sp.msaa_rtv = nullptr; }
    if (sp.msaa_texture) { sp.msaa_texture->Release(); sp.msaa_texture = nullptr; }
    if (sp.msaa_dsv) { sp.msaa_dsv->Release(); sp.msaa_dsv = nullptr; }
    if (sp.msaa_depth_texture) { sp.msaa_depth_texture->Release(); sp.msaa_depth_texture = nullptr; }
    sp.sample_count = 1;
    sp.sample_quality = 0;
}

void choose_msaa(SolidPreview& sp) {
    sp.sample_count = 1;
    sp.sample_quality = 0;
    UINT color_quality = 0;
    UINT depth_quality = 0;
    const HRESULT color_ok = g_device->CheckMultisampleQualityLevels(
        DXGI_FORMAT_B8G8R8A8_UNORM, 4, &color_quality);
    const HRESULT depth_ok = g_device->CheckMultisampleQualityLevels(
        DXGI_FORMAT_D32_FLOAT, 4, &depth_quality);
    if (SUCCEEDED(color_ok) && SUCCEEDED(depth_ok) && color_quality > 0 && depth_quality > 0) {
        sp.sample_count = 4;
        sp.sample_quality = std::min(color_quality, depth_quality) - 1u;
    }
}

// Generate a low-poly UV sphere mesh for analytic sphere preview.
// Returns {vertex_count, index_count}, appends data into vectors.
struct GenCount { size_t verts; size_t indices; };
GenCount generate_sphere_geometry(std::vector<SolidPreviewVertex>& verts,
                                  std::vector<uint32_t>& indices,
                                  const lt::Sphere& sphere,
                                  int segments = 16, int rings = 8) {
    const size_t vert_start = verts.size();
    const float r = sphere.radius;
    const lt::Vec3 c = sphere.center;

    for (int y = 0; y <= rings; ++y) {
        const float v = static_cast<float>(y) / static_cast<float>(rings);
        const float theta = v * lt::kPi;
        const float sin_theta = std::sin(theta);
        const float cos_theta = std::cos(theta);
        for (int x = 0; x <= segments; ++x) {
            const float u = static_cast<float>(x) / static_cast<float>(segments);
            const float phi = u * 2.0f * lt::kPi;
            const lt::Vec3 pos = {
                c.x + r * sin_theta * std::cos(phi),
                c.y + r * cos_theta,
                c.z + r * sin_theta * std::sin(phi),
            };
            const lt::Vec3 n = lt::normalize(pos - c);
            verts.push_back({pos.x, pos.y, pos.z, n.x, n.y, n.z});
        }
    }

    const size_t idx_start = indices.size();
    for (int y = 0; y < rings; ++y) {
        for (int x = 0; x < segments; ++x) {
            const uint32_t a = static_cast<uint32_t>(vert_start + y * (segments + 1) + x);
            const uint32_t b = static_cast<uint32_t>(vert_start + y * (segments + 1) + x + 1);
            const uint32_t c2 = static_cast<uint32_t>(vert_start + (y + 1) * (segments + 1) + x + 1);
            const uint32_t d = static_cast<uint32_t>(vert_start + (y + 1) * (segments + 1) + x);
            indices.push_back(a); indices.push_back(d); indices.push_back(b);
            indices.push_back(b); indices.push_back(d); indices.push_back(c2);
        }
    }
    return {verts.size() - vert_start, indices.size() - idx_start};
}

void build_buffer_from_scene(SolidPreview& sp, const lt::Scene& scene) {
    // Release old buffers
    if (sp.vb) { sp.vb->Release(); sp.vb = nullptr; }
    if (sp.ib) { sp.ib->Release(); sp.ib = nullptr; }
    sp.vertex_count = 0;
    sp.index_count = 0;

    std::vector<SolidPreviewVertex> verts;
    std::vector<uint32_t> indices;

    const lt::RenderScene render_scene = lt::build_render_scene(scene);

    // Use the same world-space geometry as picking and the path tracer.
    for (const lt::Triangle& triangle : render_scene.triangles) {
        const uint32_t base = static_cast<uint32_t>(verts.size());
        verts.push_back({triangle.v0.x, triangle.v0.y, triangle.v0.z, triangle.n0.x, triangle.n0.y, triangle.n0.z});
        verts.push_back({triangle.v1.x, triangle.v1.y, triangle.v1.z, triangle.n1.x, triangle.n1.y, triangle.n1.z});
        verts.push_back({triangle.v2.x, triangle.v2.y, triangle.v2.z, triangle.n2.x, triangle.n2.y, triangle.n2.z});
        indices.push_back(base);
        indices.push_back(base + 1);
        indices.push_back(base + 2);
    }

    for (const lt::RenderSphere& sphere : render_scene.spheres) {
        lt::Sphere preview_sphere;
        preview_sphere.center = sphere.center;
        preview_sphere.radius = sphere.radius;
        preview_sphere.material = sphere.material;
        generate_sphere_geometry(verts, indices, preview_sphere);
    }

    if (verts.empty() || indices.empty()) return;

    // Create vertex buffer
    D3D11_BUFFER_DESC vb_desc{};
    vb_desc.ByteWidth = static_cast<UINT>(verts.size() * sizeof(SolidPreviewVertex));
    vb_desc.Usage = D3D11_USAGE_DEFAULT;
    vb_desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    D3D11_SUBRESOURCE_DATA vb_data{};
    vb_data.pSysMem = verts.data();
    if (g_device->CreateBuffer(&vb_desc, &vb_data, &sp.vb) != S_OK) return;

    // Create index buffer
    D3D11_BUFFER_DESC ib_desc{};
    ib_desc.ByteWidth = static_cast<UINT>(indices.size() * sizeof(uint32_t));
    ib_desc.Usage = D3D11_USAGE_DEFAULT;
    ib_desc.BindFlags = D3D11_BIND_INDEX_BUFFER;
    D3D11_SUBRESOURCE_DATA ib_data{};
    ib_data.pSysMem = indices.data();
    if (g_device->CreateBuffer(&ib_desc, &ib_data, &sp.ib) != S_OK) {
        sp.vb->Release();
        sp.vb = nullptr;
        return;
    }

    sp.vertex_count = static_cast<UINT>(verts.size());
    sp.index_count = static_cast<UINT>(indices.size());
}

void build_constant_buffer(SolidPreview& sp) {
    if (sp.cb) return;
    D3D11_BUFFER_DESC desc{};
    desc.ByteWidth = sizeof(SolidConstantBuffer);
    desc.Usage = D3D11_USAGE_DYNAMIC;
    desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    g_device->CreateBuffer(&desc, nullptr, &sp.cb);
}

} // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

SolidPreview g_solid_preview;

void init_solid_preview(SolidPreview& sp) {
    if (sp.vs) return; // Already initialized
    if (!create_solid_shaders(sp)) {
        LT_LOG_ERROR("Solid preview: failed to compile shaders");
        return;
    }
    create_solid_states(sp);
    build_constant_buffer(sp);
    LT_LOG_INFO("Solid preview initialized");
}

void release_solid_preview(SolidPreview& sp) {
    release_render_targets(sp);
    if (sp.vb) { sp.vb->Release(); sp.vb = nullptr; }
    if (sp.ib) { sp.ib->Release(); sp.ib = nullptr; }
    if (sp.cb) { sp.cb->Release(); sp.cb = nullptr; }
    if (sp.vs) { sp.vs->Release(); sp.vs = nullptr; }
    if (sp.ps) { sp.ps->Release(); sp.ps = nullptr; }
    if (sp.input_layout) { sp.input_layout->Release(); sp.input_layout = nullptr; }
    if (sp.wire_vs) { sp.wire_vs->Release(); sp.wire_vs = nullptr; }
    if (sp.wire_ps) { sp.wire_ps->Release(); sp.wire_ps = nullptr; }
    if (sp.wire_rasterizer) { sp.wire_rasterizer->Release(); sp.wire_rasterizer = nullptr; }
    if (sp.solid_rasterizer) { sp.solid_rasterizer->Release(); sp.solid_rasterizer = nullptr; }
    if (sp.solid_depth_state) { sp.solid_depth_state->Release(); sp.solid_depth_state = nullptr; }
    if (sp.wire_depth_state) { sp.wire_depth_state->Release(); sp.wire_depth_state = nullptr; }
    sp = {};
}

void resize_solid_preview(SolidPreview& sp, int width, int height) {
    if (sp.width == width && sp.height == height && sp.rt_texture) return;

    release_render_targets(sp);
    choose_msaa(sp);

    // Resolved render target texture used by ImGui.
    D3D11_TEXTURE2D_DESC tex_desc{};
    tex_desc.Width = static_cast<UINT>(width);
    tex_desc.Height = static_cast<UINT>(height);
    tex_desc.MipLevels = 1;
    tex_desc.ArraySize = 1;
    tex_desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    tex_desc.SampleDesc.Count = 1;
    tex_desc.SampleDesc.Quality = 0;
    tex_desc.Usage = D3D11_USAGE_DEFAULT;
    tex_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    if (sp.sample_count == 1) {
        tex_desc.BindFlags |= D3D11_BIND_RENDER_TARGET;
    }
    if (g_device->CreateTexture2D(&tex_desc, nullptr, &sp.rt_texture) != S_OK) return;
    g_device->CreateShaderResourceView(sp.rt_texture, nullptr, &sp.srv);

    if (sp.sample_count == 1) {
        g_device->CreateRenderTargetView(sp.rt_texture, nullptr, &sp.rtv);
    } else {
        D3D11_TEXTURE2D_DESC msaa_desc = tex_desc;
        msaa_desc.SampleDesc.Count = sp.sample_count;
        msaa_desc.SampleDesc.Quality = sp.sample_quality;
        msaa_desc.BindFlags = D3D11_BIND_RENDER_TARGET;
        if (g_device->CreateTexture2D(&msaa_desc, nullptr, &sp.msaa_texture) != S_OK) return;
        g_device->CreateRenderTargetView(sp.msaa_texture, nullptr, &sp.msaa_rtv);
    }

    // Depth stencil texture using normal D3D depth: clear to 1, depth test <=.
    D3D11_TEXTURE2D_DESC depth_desc{};
    depth_desc.Width = static_cast<UINT>(width);
    depth_desc.Height = static_cast<UINT>(height);
    depth_desc.MipLevels = 1;
    depth_desc.ArraySize = 1;
    depth_desc.Format = DXGI_FORMAT_D32_FLOAT;
    depth_desc.SampleDesc.Count = sp.sample_count;
    depth_desc.SampleDesc.Quality = sp.sample_count > 1 ? sp.sample_quality : 0;
    depth_desc.Usage = D3D11_USAGE_DEFAULT;
    depth_desc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
    if (sp.sample_count == 1) {
        if (g_device->CreateTexture2D(&depth_desc, nullptr, &sp.depth_texture) != S_OK) return;
        g_device->CreateDepthStencilView(sp.depth_texture, nullptr, &sp.dsv);
    } else {
        if (g_device->CreateTexture2D(&depth_desc, nullptr, &sp.msaa_depth_texture) != S_OK) return;
        g_device->CreateDepthStencilView(sp.msaa_depth_texture, nullptr, &sp.msaa_dsv);
    }

    sp.width = width;
    sp.height = height;
}

void update_solid_preview_buffers(SolidPreview& sp, const Scene& scene) {
    // Rebuild geometry only when scene content changes. Camera-only changes update
    // constants in render_solid_preview and must not touch large vertex buffers.
    if (!sp.vb || !sp.ib || sp.content_generation != g_editor.content_generation) {
        build_buffer_from_scene(sp, scene);
        sp.scene_generation = g_editor.render_generation;
        sp.content_generation = g_editor.content_generation;
    }
}

void render_solid_preview(SolidPreview& sp, const Scene& scene, ImVec2 viewport_size, ViewportPreviewMode mode) {
    ID3D11RenderTargetView* target_rtv = sp.sample_count > 1 ? sp.msaa_rtv : sp.rtv;
    ID3D11DepthStencilView* target_dsv = sp.sample_count > 1 ? sp.msaa_dsv : sp.dsv;
    if (!target_rtv || !target_dsv || !sp.srv || !sp.vb || !sp.ib || !sp.vs || !sp.ps) return;

    // Build view-projection matrix matching the editor projection helpers:
    // camera forward maps to +Z in view space.
    const float aspect = viewport_size.x / std::max(1.0f, viewport_size.y);
    const float fov_rad = scene.camera.fov_degrees * lt::kPi / 180.0f;
    const lt::Vec3 forward = lt::normalize(scene.camera.target - scene.camera.position);
    const float right_sign = scene.camera.right_sign < 0.0f ? -1.0f : 1.0f;
    const lt::Vec3 right = lt::normalize(lt::cross(forward, scene.camera.up)) * right_sign;
    const lt::Vec3 up = lt::cross(right, forward) * right_sign;
    const lt::Vec3& pos = scene.camera.position;

    // Camera looks along +Z in view space.
    // Row-major matrix for HLSL mul(float4(position), view_proj).
    const float view[16] = {
        right.x,         up.x,        forward.x,        0.0f,
        right.y,         up.y,        forward.y,        0.0f,
        right.z,         up.z,        forward.z,        0.0f,
        -lt::dot(right, pos), -lt::dot(up, pos), -lt::dot(forward, pos), 1.0f,
    };

    // D3D perspective projection, row-vector layout, depth 0..1.
    const float n = 0.05f;
    const float f = 1000.0f;
    const float h = 1.0f / std::tan(fov_rad * 0.5f);
    const float w = h / aspect;
    float proj[16] = {0};
    proj[0]  = w;
    proj[5]  = h;
    proj[10] = f / (f - n);
    proj[11] = 1.0f;
    proj[14] = -n * f / (f - n);

    // view * proj
    float view_proj[16] = {0};
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            for (int k = 0; k < 4; ++k) {
                view_proj[row * 4 + col] += view[row * 4 + k] * proj[k * 4 + col];
            }
        }
    }

    // Blender-style solid preview is independent from scene lights. Attach a
    // soft studio direction to the view so shape readability stays stable while
    // orbiting.
    lt::Vec3 light_dir = lt::normalize(right * -0.35f + up * 0.45f + forward * 0.82f);

    // Upload constant buffer
    SolidConstantBuffer cb_data{};
    std::memcpy(cb_data.view_proj, view_proj, sizeof(view_proj));
    cb_data.camera_pos[0] = pos.x;
    cb_data.camera_pos[1] = pos.y;
    cb_data.camera_pos[2] = pos.z;
    cb_data.camera_pos[3] = 1.0f;
    cb_data.light_dir[0] = light_dir.x;
    cb_data.light_dir[1] = light_dir.y;
    cb_data.light_dir[2] = light_dir.z;
    cb_data.light_dir[3] = 0.0f;
    cb_data.ambient_color[0] = 0.18f;
    cb_data.ambient_color[1] = 0.19f;
    cb_data.ambient_color[2] = 0.22f;
    cb_data.ambient_color[3] = 1.0f;
    cb_data.clay_color[0] = 0.62f;
    cb_data.clay_color[1] = 0.62f;
    cb_data.clay_color[2] = 0.60f;
    cb_data.clay_color[3] = 1.0f;

    D3D11_MAPPED_SUBRESOURCE mapped{};
    if (sp.cb && g_context->Map(sp.cb, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped) == S_OK) {
        std::memcpy(mapped.pData, &cb_data, sizeof(cb_data));
        g_context->Unmap(sp.cb, 0);
    }

    // Save existing render target
    ID3D11RenderTargetView* prev_rtv = nullptr;
    ID3D11DepthStencilView* prev_dsv = nullptr;
    g_context->OMGetRenderTargets(1, &prev_rtv, &prev_dsv);

    // Set viewport
    D3D11_VIEWPORT vp{};
    vp.Width = static_cast<float>(sp.width);
    vp.Height = static_cast<float>(sp.height);
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;
    vp.TopLeftX = 0.0f;
    vp.TopLeftY = 0.0f;
    g_context->RSSetViewports(1, &vp);

    // Clear
    const float clear_color[4] = {0.14f, 0.15f, 0.16f, 1.0f};
    g_context->ClearRenderTargetView(target_rtv, clear_color);
    g_context->ClearDepthStencilView(target_dsv, D3D11_CLEAR_DEPTH, 1.0f, 0);

    // Set render target
    g_context->OMSetRenderTargets(1, &target_rtv, target_dsv);
    g_context->OMSetDepthStencilState(sp.solid_depth_state, 0);

    // Input assembly
    const UINT stride = sizeof(SolidPreviewVertex);
    const UINT offset = 0;
    g_context->IASetInputLayout(sp.input_layout);
    g_context->IASetVertexBuffers(0, 1, &sp.vb, &stride, &offset);
    g_context->IASetIndexBuffer(sp.ib, DXGI_FORMAT_R32_UINT, 0);
    g_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

    // Constant buffer
    g_context->VSSetConstantBuffers(0, 1, &sp.cb);
    g_context->PSSetConstantBuffers(0, 1, &sp.cb);

    if (mode == ViewportPreviewMode::Wireframe) {
        g_context->OMSetDepthStencilState(sp.wire_depth_state, 0);
        g_context->RSSetState(sp.wire_rasterizer);
        g_context->VSSetShader(sp.wire_vs, nullptr, 0);
        g_context->PSSetShader(sp.wire_ps, nullptr, 0);
        g_context->DrawIndexed(sp.index_count, 0, 0);
    } else {
        g_context->RSSetState(sp.solid_rasterizer);
        g_context->VSSetShader(sp.vs, nullptr, 0);
        g_context->PSSetShader(sp.ps, nullptr, 0);
        g_context->DrawIndexed(sp.index_count, 0, 0);
    }

    if (sp.sample_count > 1 && sp.msaa_texture && sp.rt_texture) {
        g_context->ResolveSubresource(
            sp.rt_texture, 0, sp.msaa_texture, 0, DXGI_FORMAT_B8G8R8A8_UNORM);
    }

    // Restore previous render target
    g_context->OMSetRenderTargets(1, &prev_rtv, prev_dsv);
    if (prev_rtv) prev_rtv->Release();
    if (prev_dsv) prev_dsv->Release();
}

} // namespace lt::editor
