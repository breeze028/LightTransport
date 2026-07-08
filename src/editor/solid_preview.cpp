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
    float4 selection_params;
};

struct VSInput {
    float3 position : POSITION;
    float3 normal   : NORMAL;
    float2 uv       : TEXCOORD0;
    uint object_id   : TEXCOORD1;
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
    float4 selection_params;
};

struct VSInput {
    float3 position : POSITION;
    float3 normal   : NORMAL;
    float2 uv       : TEXCOORD0;
    uint object_id   : TEXCOORD1;
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
    return float4(1.0f, 0.55f, 0.12f, 1.0f);
}
)";

static const char* kObjectIdShaderSource = R"(
cbuffer ConstantBuffer : register(b0) {
    row_major float4x4 view_proj;
    float4 camera_pos;
    float4 light_dir;
    float4 ambient_color;
    float4 clay_color;
    float4 selection_params;
};

struct VSInput {
    float3 position : POSITION;
    float3 normal   : NORMAL;
    float2 uv       : TEXCOORD0;
    uint object_id   : TEXCOORD1;
};

struct PSInput {
    float4 position : SV_POSITION;
    nointerpolation uint object_id : TEXCOORD1;
};

PSInput VSMain(VSInput input) {
    PSInput output;
    output.position = mul(float4(input.position, 1.0f), view_proj);
    output.object_id = input.object_id;
    return output;
}

uint PSMain(PSInput input) : SV_Target {
    return input.object_id;
}

uint SelectedPSMain(PSInput input) : SV_Target {
    uint selected_id = (uint)(selection_params.x + 0.5f);
    if (input.object_id != selected_id) {
        discard;
    }
    return input.object_id;
}
)";

static const char* kMaterialPreviewShaderSource = R"(
cbuffer ConstantBuffer : register(b0) {
    row_major float4x4 view_proj;
    float4 camera_pos;
    float4 light_dir;
    float4 ambient_color;
    float4 clay_color;
    float4 selection_params;
};

cbuffer MaterialBuffer : register(b1) {
    float4 base_color;
    float4 emission;
    float4 material_params; // roughness, metallic, alpha, flags
};

Texture2D base_color_tx : register(t0);
SamplerState base_color_sampler : register(s0);

struct VSInput {
    float3 position : POSITION;
    float3 normal   : NORMAL;
    float2 uv       : TEXCOORD0;
    uint object_id   : TEXCOORD1;
};

struct PSInput {
    float4 position : SV_POSITION;
    float3 normal   : NORMAL;
    float3 world_pos : TEXCOORD0;
    float2 uv        : TEXCOORD2;
};

PSInput VSMain(VSInput input) {
    PSInput output;
    output.position = mul(float4(input.position, 1.0f), view_proj);
    output.normal = normalize(input.normal);
    output.world_pos = input.position;
    output.uv = input.uv;
    return output;
}

float distribution_ggx(float ndoth, float roughness) {
    float a = max(roughness * roughness, 0.02f);
    float a2 = a * a;
    float d = ndoth * ndoth * (a2 - 1.0f) + 1.0f;
    return a2 / max(3.14159265f * d * d, 0.0001f);
}

float geometry_schlick_ggx(float ndotv, float roughness) {
    float r = roughness + 1.0f;
    float k = (r * r) * 0.125f;
    return ndotv / max(ndotv * (1.0f - k) + k, 0.0001f);
}

float3 fresnel_schlick(float cos_theta, float3 f0) {
    return f0 + (1.0f - f0) * pow(saturate(1.0f - cos_theta), 5.0f);
}

float3 shade_light(float3 base, float3 N, float3 V, float3 L, float3 light_color, float roughness, float metallic) {
    float3 H = normalize(V + L);
    float ndotl = saturate(dot(N, L));
    float ndotv = saturate(dot(N, V));
    float ndoth = saturate(dot(N, H));
    float hdotv = saturate(dot(H, V));
    float3 f0 = lerp(float3(0.04f, 0.04f, 0.04f), base, metallic);
    float3 F = fresnel_schlick(hdotv, f0);
    float D = distribution_ggx(ndoth, roughness);
    float G = geometry_schlick_ggx(ndotv, roughness) * geometry_schlick_ggx(ndotl, roughness);
    float3 spec = (D * G * F) / max(4.0f * ndotv * ndotl, 0.0001f);
    float3 diffuse = (1.0f - F) * base * (1.0f - metallic) * 0.31830989f;
    return (diffuse + spec) * light_color * ndotl;
}

float4 PSMain(PSInput input) : SV_TARGET {
    float texture_enabled = material_params.w;
    float4 texel = lerp(float4(1.0f, 1.0f, 1.0f, 1.0f), base_color_tx.Sample(base_color_sampler, input.uv), texture_enabled);
    float3 base = saturate(base_color.rgb * texel.rgb);
    float roughness = saturate(material_params.x);
    float metallic = saturate(material_params.y);
    float alpha = saturate(material_params.z * texel.a);

    float3 N = normalize(input.normal);
    float3 V = normalize(camera_pos.xyz - input.world_pos);
    if (dot(N, V) < 0.0f) {
        N = -N;
    }

    float3 forward = normalize(light_dir.xyz);
    float3 key = normalize(forward);
    float3 fill = normalize(float3(-forward.x, abs(forward.y) + 0.35f, -forward.z * 0.35f));
    float3 rim = normalize(-V + float3(0.0f, 0.65f, 0.0f));

    float3 color = base * ambient_color.rgb * 0.75f;
    color += shade_light(base, N, V, key, float3(1.28f, 1.20f, 1.05f), roughness, metallic);
    color += shade_light(base, N, V, fill, float3(0.34f, 0.42f, 0.58f), roughness, metallic) * 0.55f;

    float rim_amount = pow(saturate(dot(N, rim)), 2.0f) * 0.16f;
    color += rim_amount * float3(0.90f, 0.96f, 1.0f);

    float3 f0 = lerp(float3(0.04f, 0.04f, 0.04f), base, metallic);
    color += f0 * lerp(0.05f, 0.20f, 1.0f - roughness);
    color += emission.rgb;
    color = color / (color + 1.0f);
    color = pow(max(color, 0.0f), 1.0f / 2.2f);
    return float4(color, max(alpha, 0.20f));
}
)";

static const char* kOutlineShaderSource = R"(
cbuffer ConstantBuffer : register(b0) {
    row_major float4x4 view_proj;
    float4 camera_pos;
    float4 light_dir;
    float4 ambient_color;
    float4 clay_color;
    float4 selection_params;
};

Texture2D<uint> selected_id_tx : register(t0);
Texture2D<float> scene_depth_tx : register(t1);
Texture2D<float> selected_depth_tx : register(t2);

struct VSOutput {
    float4 position : SV_POSITION;
};

VSOutput VSMain(uint vertex_id : SV_VertexID) {
    float2 positions[3] = {
        float2(-1.0f, -1.0f),
        float2(-1.0f,  3.0f),
        float2( 3.0f, -1.0f)
    };
    VSOutput output;
    output.position = float4(positions[vertex_id], 0.0f, 1.0f);
    return output;
}

uint load_id(int2 p, int2 dims) {
    p = min(max(p, int2(0, 0)), dims - int2(1, 1));
    return selected_id_tx.Load(int3(p, 0));
}

float load_selected_depth(int2 p, int2 dims) {
    p = min(max(p, int2(0, 0)), dims - int2(1, 1));
    return selected_depth_tx.Load(int3(p, 0));
}

float load_scene_depth(int2 p, int2 dims) {
    p = min(max(p, int2(0, 0)), dims - int2(1, 1));
    return scene_depth_tx.Load(int3(p, 0));
}

float4 PSMain(VSOutput input) : SV_Target {
    int2 dims = max(int2(selection_params.yz + 0.5f), int2(1, 1));
    int2 p = min(max(int2(input.position.xy), int2(0, 0)), dims - int2(1, 1));
    uint center_id = load_id(p, dims);
    bool center_selected = center_id != 0;

    int2 left_p = p + int2(-1, 0);
    int2 right_p = p + int2(1, 0);
    int2 up_p = p + int2(0, -1);
    int2 down_p = p + int2(0, 1);
    int2 up_left_p = p + int2(-1, -1);
    int2 up_right_p = p + int2(1, -1);
    int2 down_left_p = p + int2(-1, 1);
    int2 down_right_p = p + int2(1, 1);
    int2 left2_p = p + int2(-2, 0);
    int2 right2_p = p + int2(2, 0);
    int2 up2_p = p + int2(0, -2);
    int2 down2_p = p + int2(0, 2);

    bool left_selected = load_id(left_p, dims) != 0;
    bool right_selected = load_id(right_p, dims) != 0;
    bool up_selected = load_id(up_p, dims) != 0;
    bool down_selected = load_id(down_p, dims) != 0;
    bool up_left_selected = load_id(up_left_p, dims) != 0;
    bool up_right_selected = load_id(up_right_p, dims) != 0;
    bool down_left_selected = load_id(down_left_p, dims) != 0;
    bool down_right_selected = load_id(down_right_p, dims) != 0;
    bool left2_selected = load_id(left2_p, dims) != 0;
    bool right2_selected = load_id(right2_p, dims) != 0;
    bool up2_selected = load_id(up2_p, dims) != 0;
    bool down2_selected = load_id(down2_p, dims) != 0;

    float cardinal_selected =
        (left_selected ? 1.0f : 0.0f) +
        (right_selected ? 1.0f : 0.0f) +
        (up_selected ? 1.0f : 0.0f) +
        (down_selected ? 1.0f : 0.0f);
    float diagonal_selected =
        (up_left_selected ? 1.0f : 0.0f) +
        (up_right_selected ? 1.0f : 0.0f) +
        (down_left_selected ? 1.0f : 0.0f) +
        (down_right_selected ? 1.0f : 0.0f);
    float outer_selected =
        (left2_selected ? 1.0f : 0.0f) +
        (right2_selected ? 1.0f : 0.0f) +
        (up2_selected ? 1.0f : 0.0f) +
        (down2_selected ? 1.0f : 0.0f);

    float cardinal_outside = 4.0f - cardinal_selected;
    float diagonal_outside = 4.0f - diagonal_selected;
    float coverage = 0.0f;
    float selected_depth = center_selected ? load_selected_depth(p, dims) : 1.0f;

    if (!center_selected) {
        if (left_selected) selected_depth = min(selected_depth, load_selected_depth(left_p, dims));
        if (right_selected) selected_depth = min(selected_depth, load_selected_depth(right_p, dims));
        if (up_selected) selected_depth = min(selected_depth, load_selected_depth(up_p, dims));
        if (down_selected) selected_depth = min(selected_depth, load_selected_depth(down_p, dims));
        if (up_left_selected) selected_depth = min(selected_depth, load_selected_depth(up_left_p, dims));
        if (up_right_selected) selected_depth = min(selected_depth, load_selected_depth(up_right_p, dims));
        if (down_left_selected) selected_depth = min(selected_depth, load_selected_depth(down_left_p, dims));
        if (down_right_selected) selected_depth = min(selected_depth, load_selected_depth(down_right_p, dims));
        if (left2_selected) selected_depth = min(selected_depth, load_selected_depth(left2_p, dims));
        if (right2_selected) selected_depth = min(selected_depth, load_selected_depth(right2_p, dims));
        if (up2_selected) selected_depth = min(selected_depth, load_selected_depth(up2_p, dims));
        if (down2_selected) selected_depth = min(selected_depth, load_selected_depth(down2_p, dims));
        coverage = cardinal_selected > 0.0f ? saturate(0.86f + cardinal_selected * 0.05f)
                                            : (diagonal_selected > 0.0f ? 0.64f : (outer_selected > 0.0f ? 0.40f : 0.0f));
    } else {
        coverage = cardinal_outside > 0.0f ? 0.32f
                                           : (diagonal_outside > 0.0f ? 0.16f : 0.0f);
    }

    if (coverage <= 0.0f) {
        discard;
    }

    float scene_depth = load_scene_depth(p, dims);
    float alpha = (selected_depth > scene_depth + 0.0005f ? 0.34f : 0.96f) * coverage;
    return float4(1.0f, 0.55f, 0.12f, alpha);
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
    ID3DBlob* material_ps_blob = nullptr;
    ID3DBlob* wire_vs_blob = nullptr;
    ID3DBlob* wire_ps_blob = nullptr;
    ID3DBlob* id_vs_blob = nullptr;
    ID3DBlob* id_ps_blob = nullptr;
    ID3DBlob* selected_id_ps_blob = nullptr;
    ID3DBlob* outline_vs_blob = nullptr;
    ID3DBlob* outline_ps_blob = nullptr;
    int ok = 0;

    size_t solid_len = std::strlen(kSolidShaderSource);
    size_t material_len = std::strlen(kMaterialPreviewShaderSource);
    size_t wire_len = std::strlen(kWireShaderSource);
    size_t object_id_len = std::strlen(kObjectIdShaderSource);
    size_t outline_len = std::strlen(kOutlineShaderSource);

    if (!compile_shader(kSolidShaderSource, solid_len, "VSMain", "vs_5_0", &vs_blob)) return ok;
    if (!compile_shader(kSolidShaderSource, solid_len, "PSMain", "ps_5_0", &ps_blob)) { vs_blob->Release(); return ok; }
    if (!compile_shader(kMaterialPreviewShaderSource, material_len, "PSMain", "ps_5_0", &material_ps_blob)) {
        vs_blob->Release(); ps_blob->Release(); return ok;
    }
    if (!compile_shader(kWireShaderSource, wire_len, "VSMain", "vs_5_0", &wire_vs_blob)) {
        vs_blob->Release(); ps_blob->Release(); material_ps_blob->Release(); return ok;
    }
    if (!compile_shader(kWireShaderSource, wire_len, "PSMain", "ps_5_0", &wire_ps_blob)) {
        vs_blob->Release(); ps_blob->Release(); material_ps_blob->Release(); wire_vs_blob->Release(); return ok;
    }
    if (!compile_shader(kObjectIdShaderSource, object_id_len, "VSMain", "vs_5_0", &id_vs_blob)) {
        vs_blob->Release(); ps_blob->Release(); material_ps_blob->Release(); wire_vs_blob->Release(); wire_ps_blob->Release(); return ok;
    }
    if (!compile_shader(kObjectIdShaderSource, object_id_len, "PSMain", "ps_5_0", &id_ps_blob)) {
        vs_blob->Release(); ps_blob->Release(); material_ps_blob->Release(); wire_vs_blob->Release(); wire_ps_blob->Release(); id_vs_blob->Release(); return ok;
    }
    if (!compile_shader(kObjectIdShaderSource, object_id_len, "SelectedPSMain", "ps_5_0", &selected_id_ps_blob)) {
        vs_blob->Release(); ps_blob->Release(); material_ps_blob->Release(); wire_vs_blob->Release(); wire_ps_blob->Release(); id_vs_blob->Release(); id_ps_blob->Release(); return ok;
    }
    if (!compile_shader(kOutlineShaderSource, outline_len, "VSMain", "vs_5_0", &outline_vs_blob)) {
        vs_blob->Release(); ps_blob->Release(); material_ps_blob->Release(); wire_vs_blob->Release(); wire_ps_blob->Release();
        id_vs_blob->Release(); id_ps_blob->Release(); selected_id_ps_blob->Release(); return ok;
    }
    if (!compile_shader(kOutlineShaderSource, outline_len, "PSMain", "ps_5_0", &outline_ps_blob)) {
        vs_blob->Release(); ps_blob->Release(); material_ps_blob->Release(); wire_vs_blob->Release(); wire_ps_blob->Release();
        id_vs_blob->Release(); id_ps_blob->Release(); selected_id_ps_blob->Release(); outline_vs_blob->Release(); return ok;
    }

    do {
        if (g_device->CreateVertexShader(vs_blob->GetBufferPointer(), vs_blob->GetBufferSize(), nullptr, &sp.vs) != S_OK) break;
        if (g_device->CreatePixelShader(ps_blob->GetBufferPointer(), ps_blob->GetBufferSize(), nullptr, &sp.ps) != S_OK) break;
        if (g_device->CreatePixelShader(material_ps_blob->GetBufferPointer(), material_ps_blob->GetBufferSize(), nullptr, &sp.material_ps) != S_OK) break;
        if (g_device->CreateVertexShader(wire_vs_blob->GetBufferPointer(), wire_vs_blob->GetBufferSize(), nullptr, &sp.wire_vs) != S_OK) break;
        if (g_device->CreatePixelShader(wire_ps_blob->GetBufferPointer(), wire_ps_blob->GetBufferSize(), nullptr, &sp.wire_ps) != S_OK) break;
        if (g_device->CreateVertexShader(id_vs_blob->GetBufferPointer(), id_vs_blob->GetBufferSize(), nullptr, &sp.id_vs) != S_OK) break;
        if (g_device->CreatePixelShader(id_ps_blob->GetBufferPointer(), id_ps_blob->GetBufferSize(), nullptr, &sp.id_ps) != S_OK) break;
        if (g_device->CreatePixelShader(selected_id_ps_blob->GetBufferPointer(), selected_id_ps_blob->GetBufferSize(), nullptr, &sp.selected_id_ps) != S_OK) break;
        if (g_device->CreateVertexShader(outline_vs_blob->GetBufferPointer(), outline_vs_blob->GetBufferSize(), nullptr, &sp.fullscreen_vs) != S_OK) break;
        if (g_device->CreatePixelShader(outline_ps_blob->GetBufferPointer(), outline_ps_blob->GetBufferSize(), nullptr, &sp.outline_ps) != S_OK) break;

        D3D11_INPUT_ELEMENT_DESC layout[] = {
            {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"NORMAL",   0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,   0, 24, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 1, DXGI_FORMAT_R32_UINT,       0, 32, D3D11_INPUT_PER_VERTEX_DATA, 0},
        };
        if (g_device->CreateInputLayout(layout, 4, id_vs_blob->GetBufferPointer(), id_vs_blob->GetBufferSize(), &sp.input_layout) != S_OK) break;

        ok = 1;
    } while (false);

    vs_blob->Release();
    ps_blob->Release();
    material_ps_blob->Release();
    wire_vs_blob->Release();
    wire_ps_blob->Release();
    id_vs_blob->Release();
    id_ps_blob->Release();
    selected_id_ps_blob->Release();
    outline_vs_blob->Release();
    outline_ps_blob->Release();
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

    D3D11_SAMPLER_DESC sampler_desc{};
    sampler_desc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    sampler_desc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
    sampler_desc.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
    sampler_desc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
    sampler_desc.MaxAnisotropy = 1;
    sampler_desc.ComparisonFunc = D3D11_COMPARISON_NEVER;
    sampler_desc.MinLOD = 0.0f;
    sampler_desc.MaxLOD = D3D11_FLOAT32_MAX;
    g_device->CreateSamplerState(&sampler_desc, &sp.material_sampler);
}

void release_material_preview_textures(SolidPreview& sp) {
    for (MaterialPreviewTexture& texture : sp.material_textures) {
        if (texture.srv) { texture.srv->Release(); texture.srv = nullptr; }
        if (texture.texture) { texture.texture->Release(); texture.texture = nullptr; }
        texture.source = nullptr;
    }
    sp.material_textures.clear();
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
    if (sp.object_id_rtv) { sp.object_id_rtv->Release(); sp.object_id_rtv = nullptr; }
    if (sp.object_id_texture) { sp.object_id_texture->Release(); sp.object_id_texture = nullptr; }
    if (sp.pick_readback_texture) { sp.pick_readback_texture->Release(); sp.pick_readback_texture = nullptr; }
    if (sp.scene_depth_srv) { sp.scene_depth_srv->Release(); sp.scene_depth_srv = nullptr; }
    if (sp.scene_depth_dsv) { sp.scene_depth_dsv->Release(); sp.scene_depth_dsv = nullptr; }
    if (sp.scene_depth_texture) { sp.scene_depth_texture->Release(); sp.scene_depth_texture = nullptr; }
    if (sp.selected_id_srv) { sp.selected_id_srv->Release(); sp.selected_id_srv = nullptr; }
    if (sp.selected_id_rtv) { sp.selected_id_rtv->Release(); sp.selected_id_rtv = nullptr; }
    if (sp.selected_id_texture) { sp.selected_id_texture->Release(); sp.selected_id_texture = nullptr; }
    if (sp.selected_depth_srv) { sp.selected_depth_srv->Release(); sp.selected_depth_srv = nullptr; }
    if (sp.selected_depth_dsv) { sp.selected_depth_dsv->Release(); sp.selected_depth_dsv = nullptr; }
    if (sp.selected_depth_texture) { sp.selected_depth_texture->Release(); sp.selected_depth_texture = nullptr; }
    if (sp.outline_srv) { sp.outline_srv->Release(); sp.outline_srv = nullptr; }
    if (sp.outline_rtv) { sp.outline_rtv->Release(); sp.outline_rtv = nullptr; }
    if (sp.outline_texture) { sp.outline_texture->Release(); sp.outline_texture = nullptr; }
    sp.sample_count = 1;
    sp.sample_quality = 0;
    sp.outline_valid = false;
}

bool create_depth_target(UINT width,
                         UINT height,
                         ID3D11Texture2D** texture,
                         ID3D11DepthStencilView** dsv,
                         ID3D11ShaderResourceView** srv) {
    D3D11_TEXTURE2D_DESC desc{};
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R32_TYPELESS;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE;
    if (g_device->CreateTexture2D(&desc, nullptr, texture) != S_OK) return false;

    D3D11_DEPTH_STENCIL_VIEW_DESC dsv_desc{};
    dsv_desc.Format = DXGI_FORMAT_D32_FLOAT;
    dsv_desc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
    if (g_device->CreateDepthStencilView(*texture, &dsv_desc, dsv) != S_OK) return false;

    D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc{};
    srv_desc.Format = DXGI_FORMAT_R32_FLOAT;
    srv_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    srv_desc.Texture2D.MipLevels = 1;
    if (g_device->CreateShaderResourceView(*texture, &srv_desc, srv) != S_OK) return false;
    return true;
}

bool create_selection_targets(SolidPreview& sp, int width, int height) {
    const UINT w = static_cast<UINT>(width);
    const UINT h = static_cast<UINT>(height);

    D3D11_TEXTURE2D_DESC id_desc{};
    id_desc.Width = w;
    id_desc.Height = h;
    id_desc.MipLevels = 1;
    id_desc.ArraySize = 1;
    id_desc.Format = DXGI_FORMAT_R32_UINT;
    id_desc.SampleDesc.Count = 1;
    id_desc.Usage = D3D11_USAGE_DEFAULT;
    id_desc.BindFlags = D3D11_BIND_RENDER_TARGET;
    if (g_device->CreateTexture2D(&id_desc, nullptr, &sp.object_id_texture) != S_OK) return false;
    if (g_device->CreateRenderTargetView(sp.object_id_texture, nullptr, &sp.object_id_rtv) != S_OK) return false;

    D3D11_TEXTURE2D_DESC readback_desc = id_desc;
    readback_desc.Width = 15;
    readback_desc.Height = 15;
    readback_desc.Usage = D3D11_USAGE_STAGING;
    readback_desc.BindFlags = 0;
    readback_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    if (g_device->CreateTexture2D(&readback_desc, nullptr, &sp.pick_readback_texture) != S_OK) return false;

    if (!create_depth_target(w, h, &sp.scene_depth_texture, &sp.scene_depth_dsv, &sp.scene_depth_srv)) return false;

    D3D11_TEXTURE2D_DESC selected_id_desc = id_desc;
    selected_id_desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    if (g_device->CreateTexture2D(&selected_id_desc, nullptr, &sp.selected_id_texture) != S_OK) return false;
    if (g_device->CreateRenderTargetView(sp.selected_id_texture, nullptr, &sp.selected_id_rtv) != S_OK) return false;
    if (g_device->CreateShaderResourceView(sp.selected_id_texture, nullptr, &sp.selected_id_srv) != S_OK) return false;

    if (!create_depth_target(w, h, &sp.selected_depth_texture, &sp.selected_depth_dsv, &sp.selected_depth_srv)) return false;

    D3D11_TEXTURE2D_DESC outline_desc{};
    outline_desc.Width = w;
    outline_desc.Height = h;
    outline_desc.MipLevels = 1;
    outline_desc.ArraySize = 1;
    outline_desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    outline_desc.SampleDesc.Count = 1;
    outline_desc.Usage = D3D11_USAGE_DEFAULT;
    outline_desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    if (g_device->CreateTexture2D(&outline_desc, nullptr, &sp.outline_texture) != S_OK) return false;
    if (g_device->CreateRenderTargetView(sp.outline_texture, nullptr, &sp.outline_rtv) != S_OK) return false;
    if (g_device->CreateShaderResourceView(sp.outline_texture, nullptr, &sp.outline_srv) != S_OK) return false;

    return true;
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
                                  uint32_t object_id,
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
            verts.push_back({pos.x, pos.y, pos.z, n.x, n.y, n.z, u, v, object_id});
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

uint32_t ensure_selection_id(SolidPreview& sp,
                             std::vector<uint32_t>& ids,
                             int object_index,
                             SelectionKind kind) {
    if (object_index < 0) return 0;
    if (object_index >= static_cast<int>(ids.size())) return 0;
    uint32_t& id = ids[static_cast<size_t>(object_index)];
    if (id == 0) {
        id = static_cast<uint32_t>(sp.selection_refs.size() + 1);
        sp.selection_refs.push_back({kind, object_index});
    }
    return id;
}

void append_draw_range(SolidPreview& sp, uint32_t object_id, int material_index, UINT start_index, UINT index_count) {
    if (object_id == 0 || index_count == 0) return;
    if (!sp.draw_ranges.empty()) {
        SolidDrawRange& previous = sp.draw_ranges.back();
        if (previous.object_id == object_id &&
            previous.material_index == material_index &&
            previous.start_index + previous.index_count == start_index) {
            previous.index_count += index_count;
            return;
        }
    }
    sp.draw_ranges.push_back({object_id, material_index, start_index, index_count});
}

float clamp01(float v) {
    return std::max(0.0f, std::min(1.0f, v));
}

lt::Vec3 conductor_f0(const lt::Vec3& eta, const lt::Vec3& k) {
    auto channel = [](float e, float kk) {
        const float e2 = e * e;
        const float k2 = kk * kk;
        return ((e - 1.0f) * (e - 1.0f) + k2) / std::max((e + 1.0f) * (e + 1.0f) + k2, 0.0001f);
    };
    return {channel(eta.x, k.x), channel(eta.y, k.y), channel(eta.z, k.z)};
}

MaterialPreviewData pack_material_preview_data(const std::shared_ptr<lt::Material>& material) {
    MaterialPreviewData data{};
    if (!material) return data;

    lt::Vec3 base = material->albedo;
    lt::Vec3 emission = material->emission;
    float roughness = 0.9f;
    float metallic = 0.0f;

    switch (material->model()) {
    case lt::BrdfModel::Principled:
        if (const auto* principled = dynamic_cast<const lt::PrincipledMaterial*>(material.get())) {
            roughness = principled->roughness;
            metallic = principled->metallic;
        }
        break;
    case lt::BrdfModel::Mirror:
        roughness = 0.02f;
        metallic = 1.0f;
        break;
    case lt::BrdfModel::Dielectric:
        roughness = 0.04f;
        metallic = 0.0f;
        base = base * 0.82f + lt::Vec3{0.18f, 0.22f, 0.28f};
        break;
    case lt::BrdfModel::Conductor:
        if (const auto* conductor = dynamic_cast<const lt::ConductorMaterial*>(material.get())) {
            roughness = conductor->roughness;
            base = conductor_f0(conductor->eta, conductor->k) * material->albedo;
        }
        metallic = 1.0f;
        break;
    case lt::BrdfModel::StandardSurface:
        if (const auto* standard = dynamic_cast<const lt::StandardSurfaceMaterial*>(material.get())) {
            roughness = standard->roughness;
            metallic = standard->metalness;
            base = material->albedo;
        }
        break;
    case lt::BrdfModel::DiffuseTransmission:
        roughness = 0.8f;
        metallic = 0.0f;
        if (const auto* diffuse_transmission = dynamic_cast<const lt::DiffuseTransmissionMaterial*>(material.get())) {
            base = material->albedo * 0.6f + diffuse_transmission->transmittance * 0.4f;
        }
        break;
    case lt::BrdfModel::Lambertian:
    default:
        break;
    }

    data.base_color[0] = clamp01(base.x);
    data.base_color[1] = clamp01(base.y);
    data.base_color[2] = clamp01(base.z);
    data.base_color[3] = clamp01(material->alpha);
    data.emission[0] = std::max(0.0f, emission.x);
    data.emission[1] = std::max(0.0f, emission.y);
    data.emission[2] = std::max(0.0f, emission.z);
    data.emission[3] = 0.0f;
    data.params[0] = std::max(0.02f, std::min(1.0f, roughness));
    data.params[1] = clamp01(metallic);
    data.params[2] = clamp01(material->alpha);
    data.params[3] = material->albedo_texture ? 1.0f : 0.0f;
    return data;
}

bool upload_material_texture(const lt::Texture& source, MaterialPreviewTexture& target) {
    if (source.width <= 0 || source.height <= 0 || source.pixels.empty()) return false;

    std::vector<float> pixels(static_cast<size_t>(source.width) * static_cast<size_t>(source.height) * 4u, 1.0f);
    const size_t count = std::min(source.pixels.size(), static_cast<size_t>(source.width) * static_cast<size_t>(source.height));
    for (size_t i = 0; i < count; ++i) {
        pixels[i * 4u + 0u] = std::max(0.0f, source.pixels[i].x);
        pixels[i * 4u + 1u] = std::max(0.0f, source.pixels[i].y);
        pixels[i * 4u + 2u] = std::max(0.0f, source.pixels[i].z);
        pixels[i * 4u + 3u] = i < source.alpha.size() ? clamp01(source.alpha[i]) : 1.0f;
    }

    D3D11_TEXTURE2D_DESC desc{};
    desc.Width = static_cast<UINT>(source.width);
    desc.Height = static_cast<UINT>(source.height);
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

    D3D11_SUBRESOURCE_DATA data{};
    data.pSysMem = pixels.data();
    data.SysMemPitch = static_cast<UINT>(source.width * sizeof(float) * 4u);
    if (g_device->CreateTexture2D(&desc, &data, &target.texture) != S_OK) return false;
    if (g_device->CreateShaderResourceView(target.texture, nullptr, &target.srv) != S_OK) {
        target.texture->Release();
        target.texture = nullptr;
        return false;
    }
    target.source = &source;
    return true;
}

void build_material_preview_resources(SolidPreview& sp, const lt::Scene& scene) {
    release_material_preview_textures(sp);
    sp.material_data.clear();
    sp.material_data.reserve(scene.materials.size());
    sp.material_textures.resize(scene.materials.size());

    for (size_t i = 0; i < scene.materials.size(); ++i) {
        const std::shared_ptr<lt::Material>& material = scene.materials[i];
        sp.material_data.push_back(pack_material_preview_data(material));
        if (material && material->albedo_texture) {
            MaterialPreviewTexture& texture = sp.material_textures[i];
            if (!upload_material_texture(*material->albedo_texture, texture)) {
                sp.material_data.back().params[3] = 0.0f;
            }
        } else {
            sp.material_data.back().params[3] = 0.0f;
        }
    }
}

void build_buffer_from_scene(SolidPreview& sp, const lt::Scene& scene) {
    // Release old buffers
    if (sp.vb) { sp.vb->Release(); sp.vb = nullptr; }
    if (sp.ib) { sp.ib->Release(); sp.ib = nullptr; }
    sp.vertex_count = 0;
    sp.index_count = 0;
    sp.selection_refs.clear();
    sp.draw_ranges.clear();
    sp.outline_valid = false;

    std::vector<SolidPreviewVertex> verts;
    std::vector<uint32_t> indices;
    std::vector<uint32_t> mesh_ids(scene.meshes.size(), 0);
    std::vector<uint32_t> sphere_ids(scene.spheres.size(), 0);

    const lt::RenderScene render_scene = lt::build_render_scene(scene);

    // Use the same world-space geometry as picking and the path tracer.
    for (const lt::Triangle& triangle : render_scene.triangles) {
        const uint32_t base = static_cast<uint32_t>(verts.size());
        const uint32_t object_id = ensure_selection_id(sp, mesh_ids, triangle.mesh, SelectionKind::Mesh);
        const UINT range_start = static_cast<UINT>(indices.size());
        verts.push_back({triangle.v0.x, triangle.v0.y, triangle.v0.z, triangle.n0.x, triangle.n0.y, triangle.n0.z, triangle.uv0.x, triangle.uv0.y, object_id});
        verts.push_back({triangle.v1.x, triangle.v1.y, triangle.v1.z, triangle.n1.x, triangle.n1.y, triangle.n1.z, triangle.uv1.x, triangle.uv1.y, object_id});
        verts.push_back({triangle.v2.x, triangle.v2.y, triangle.v2.z, triangle.n2.x, triangle.n2.y, triangle.n2.z, triangle.uv2.x, triangle.uv2.y, object_id});
        indices.push_back(base);
        indices.push_back(base + 1);
        indices.push_back(base + 2);
        append_draw_range(sp, object_id, triangle.material, range_start, 3);
    }

    for (const lt::RenderSphere& sphere : render_scene.spheres) {
        lt::Sphere preview_sphere;
        preview_sphere.center = sphere.center;
        preview_sphere.radius = sphere.radius;
        preview_sphere.material = sphere.material;
        const uint32_t object_id = ensure_selection_id(sp, sphere_ids, sphere.sphere, SelectionKind::Sphere);
        const UINT range_start = static_cast<UINT>(indices.size());
        const GenCount count = generate_sphere_geometry(verts, indices, preview_sphere, object_id);
        append_draw_range(sp, object_id, sphere.material, range_start, static_cast<UINT>(count.indices));
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
    if (!sp.cb) {
        D3D11_BUFFER_DESC desc{};
        desc.ByteWidth = sizeof(SolidConstantBuffer);
        desc.Usage = D3D11_USAGE_DYNAMIC;
        desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        g_device->CreateBuffer(&desc, nullptr, &sp.cb);
    }
    if (!sp.material_cb) {
        D3D11_BUFFER_DESC desc{};
        desc.ByteWidth = sizeof(MaterialPreviewData);
        desc.Usage = D3D11_USAGE_DYNAMIC;
        desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        g_device->CreateBuffer(&desc, nullptr, &sp.material_cb);
    }
}

bool upload_solid_constants(SolidPreview& sp,
                            const Scene& scene,
                            ImVec2 viewport_size,
                            uint32_t selected_id = 0) {
    if (!sp.cb) return false;

    // Build view-projection matrix matching the editor projection helpers:
    // camera forward maps to +Z in view space.
    const float aspect = viewport_size.x / std::max(1.0f, viewport_size.y);
    const float fov_rad = scene.camera.fov_degrees * lt::kPi / 180.0f;
    const lt::Vec3 forward = lt::normalize(scene.camera.target - scene.camera.position);
    const float right_sign = scene.camera.right_sign < 0.0f ? -1.0f : 1.0f;
    const lt::Vec3 right = lt::normalize(lt::cross(forward, scene.camera.up)) * right_sign;
    const lt::Vec3 up = lt::cross(right, forward) * right_sign;
    const lt::Vec3& pos = scene.camera.position;

    const float view[16] = {
        right.x,         up.x,        forward.x,        0.0f,
        right.y,         up.y,        forward.y,        0.0f,
        right.z,         up.z,        forward.z,        0.0f,
        -lt::dot(right, pos), -lt::dot(up, pos), -lt::dot(forward, pos), 1.0f,
    };

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
    cb_data.selection_params[0] = static_cast<float>(selected_id);
    cb_data.selection_params[1] = static_cast<float>(sp.width);
    cb_data.selection_params[2] = static_cast<float>(sp.height);
    cb_data.selection_params[3] = 0.0f;

    D3D11_MAPPED_SUBRESOURCE mapped{};
    if (g_context->Map(sp.cb, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped) != S_OK) {
        return false;
    }
    std::memcpy(mapped.pData, &cb_data, sizeof(cb_data));
    g_context->Unmap(sp.cb, 0);
    return true;
}

bool upload_material_preview_constants(SolidPreview& sp, int material_index) {
    if (!sp.material_cb) return false;
    MaterialPreviewData data{};
    if (material_index >= 0 && material_index < static_cast<int>(sp.material_data.size())) {
        data = sp.material_data[static_cast<size_t>(material_index)];
    }

    D3D11_MAPPED_SUBRESOURCE mapped{};
    if (g_context->Map(sp.material_cb, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped) != S_OK) {
        return false;
    }
    std::memcpy(mapped.pData, &data, sizeof(data));
    g_context->Unmap(sp.material_cb, 0);
    return true;
}

ID3D11ShaderResourceView* material_preview_srv(SolidPreview& sp, int material_index) {
    if (material_index < 0 || material_index >= static_cast<int>(sp.material_textures.size())) return nullptr;
    return sp.material_textures[static_cast<size_t>(material_index)].srv;
}

uint32_t find_selection_id(const SolidPreview& sp, SelectionKind kind, int object_index) {
    for (size_t i = 0; i < sp.selection_refs.size(); ++i) {
        const SolidSelectionRef& ref = sp.selection_refs[i];
        if (ref.kind == kind && ref.object_index == object_index) {
            return static_cast<uint32_t>(i + 1);
        }
    }
    return 0;
}

bool has_selection_resources(const SolidPreview& sp) {
    return sp.object_id_rtv &&
        sp.pick_readback_texture &&
        sp.scene_depth_dsv &&
        sp.scene_depth_srv &&
        sp.selected_id_rtv &&
        sp.selected_id_srv &&
        sp.selected_depth_dsv &&
        sp.selected_depth_srv &&
        sp.outline_rtv &&
        sp.outline_srv;
}

void bind_preview_geometry(SolidPreview& sp) {
    const UINT stride = sizeof(SolidPreviewVertex);
    const UINT offset = 0;
    g_context->IASetInputLayout(sp.input_layout);
    g_context->IASetVertexBuffers(0, 1, &sp.vb, &stride, &offset);
    g_context->IASetIndexBuffer(sp.ib, DXGI_FORMAT_R32_UINT, 0);
    g_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    g_context->VSSetConstantBuffers(0, 1, &sp.cb);
    g_context->PSSetConstantBuffers(0, 1, &sp.cb);
    g_context->RSSetState(sp.solid_rasterizer);
    g_context->OMSetDepthStencilState(sp.solid_depth_state, 0);
}

void set_preview_viewport(const SolidPreview& sp) {
    D3D11_VIEWPORT vp{};
    vp.Width = static_cast<float>(sp.width);
    vp.Height = static_cast<float>(sp.height);
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;
    vp.TopLeftX = 0.0f;
    vp.TopLeftY = 0.0f;
    g_context->RSSetViewports(1, &vp);
}

void restore_render_target(ID3D11RenderTargetView* previous_rtv, ID3D11DepthStencilView* previous_dsv) {
    g_context->OMSetRenderTargets(1, &previous_rtv, previous_dsv);
    if (previous_rtv) previous_rtv->Release();
    if (previous_dsv) previous_dsv->Release();
}

bool render_scene_id_pass(SolidPreview& sp, const Scene& scene, ImVec2 viewport_size) {
    if (!sp.vb || !sp.ib || !sp.id_vs || !sp.id_ps || !has_selection_resources(sp)) return false;
    if (!upload_solid_constants(sp, scene, viewport_size)) return false;

    ID3D11ShaderResourceView* null_srvs[3] = {};
    g_context->PSSetShaderResources(0, 3, null_srvs);

    ID3D11RenderTargetView* previous_rtv = nullptr;
    ID3D11DepthStencilView* previous_dsv = nullptr;
    g_context->OMGetRenderTargets(1, &previous_rtv, &previous_dsv);

    const float clear_id[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    g_context->ClearRenderTargetView(sp.object_id_rtv, clear_id);
    g_context->ClearDepthStencilView(sp.scene_depth_dsv, D3D11_CLEAR_DEPTH, 1.0f, 0);

    ID3D11RenderTargetView* id_rtv = sp.object_id_rtv;
    g_context->OMSetRenderTargets(1, &id_rtv, sp.scene_depth_dsv);
    set_preview_viewport(sp);
    bind_preview_geometry(sp);
    g_context->VSSetShader(sp.id_vs, nullptr, 0);
    g_context->PSSetShader(sp.id_ps, nullptr, 0);
    g_context->DrawIndexed(sp.index_count, 0, 0);

    restore_render_target(previous_rtv, previous_dsv);
    return true;
}

bool render_selected_id_pass(SolidPreview& sp, const Scene& scene, ImVec2 viewport_size, uint32_t selected_id) {
    if (!sp.vb || !sp.ib || !sp.id_vs || !sp.selected_id_ps || selected_id == 0 || !has_selection_resources(sp)) return false;
    if (!upload_solid_constants(sp, scene, viewport_size, selected_id)) return false;

    ID3D11ShaderResourceView* null_srvs[3] = {};
    g_context->PSSetShaderResources(0, 3, null_srvs);

    ID3D11RenderTargetView* previous_rtv = nullptr;
    ID3D11DepthStencilView* previous_dsv = nullptr;
    g_context->OMGetRenderTargets(1, &previous_rtv, &previous_dsv);

    const float clear_id[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    g_context->ClearRenderTargetView(sp.selected_id_rtv, clear_id);
    g_context->ClearDepthStencilView(sp.selected_depth_dsv, D3D11_CLEAR_DEPTH, 1.0f, 0);

    ID3D11RenderTargetView* selected_rtv = sp.selected_id_rtv;
    g_context->OMSetRenderTargets(1, &selected_rtv, sp.selected_depth_dsv);
    set_preview_viewport(sp);
    bind_preview_geometry(sp);
    g_context->VSSetShader(sp.id_vs, nullptr, 0);
    g_context->PSSetShader(sp.selected_id_ps, nullptr, 0);
    for (const SolidDrawRange& range : sp.draw_ranges) {
        if (range.object_id == selected_id) {
            g_context->DrawIndexed(range.index_count, range.start_index, 0);
        }
    }

    restore_render_target(previous_rtv, previous_dsv);
    return true;
}

bool render_outline_texture(SolidPreview& sp) {
    if (!sp.fullscreen_vs || !sp.outline_ps || !has_selection_resources(sp)) return false;

    ID3D11RenderTargetView* previous_rtv = nullptr;
    ID3D11DepthStencilView* previous_dsv = nullptr;
    g_context->OMGetRenderTargets(1, &previous_rtv, &previous_dsv);

    const float clear_color[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    g_context->ClearRenderTargetView(sp.outline_rtv, clear_color);
    ID3D11RenderTargetView* outline_rtv = sp.outline_rtv;
    g_context->OMSetRenderTargets(1, &outline_rtv, nullptr);
    g_context->OMSetDepthStencilState(nullptr, 0);
    g_context->OMSetBlendState(nullptr, nullptr, 0xffffffffu);
    set_preview_viewport(sp);

    ID3D11Buffer* null_vb = nullptr;
    UINT stride = 0;
    UINT offset = 0;
    g_context->IASetInputLayout(nullptr);
    g_context->IASetVertexBuffers(0, 1, &null_vb, &stride, &offset);
    g_context->IASetIndexBuffer(nullptr, DXGI_FORMAT_R32_UINT, 0);
    g_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    g_context->VSSetShader(sp.fullscreen_vs, nullptr, 0);
    g_context->PSSetShader(sp.outline_ps, nullptr, 0);
    g_context->VSSetConstantBuffers(0, 1, &sp.cb);
    g_context->PSSetConstantBuffers(0, 1, &sp.cb);
    ID3D11ShaderResourceView* srvs[3] = {sp.selected_id_srv, sp.scene_depth_srv, sp.selected_depth_srv};
    g_context->PSSetShaderResources(0, 3, srvs);
    g_context->Draw(3, 0);

    ID3D11ShaderResourceView* null_srvs[3] = {};
    g_context->PSSetShaderResources(0, 3, null_srvs);
    restore_render_target(previous_rtv, previous_dsv);
    return true;
}

bool decode_selection_id(const SolidPreview& sp, uint32_t object_id, SelectionKind& kind, int& object_index) {
    if (object_id == 0 || object_id > sp.selection_refs.size()) return false;
    const SolidSelectionRef& ref = sp.selection_refs[static_cast<size_t>(object_id - 1)];
    if (ref.kind == SelectionKind::None || ref.object_index < 0) return false;
    kind = ref.kind;
    object_index = ref.object_index;
    return true;
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
    release_material_preview_textures(sp);
    if (sp.vb) { sp.vb->Release(); sp.vb = nullptr; }
    if (sp.ib) { sp.ib->Release(); sp.ib = nullptr; }
    if (sp.cb) { sp.cb->Release(); sp.cb = nullptr; }
    if (sp.material_cb) { sp.material_cb->Release(); sp.material_cb = nullptr; }
    if (sp.vs) { sp.vs->Release(); sp.vs = nullptr; }
    if (sp.ps) { sp.ps->Release(); sp.ps = nullptr; }
    if (sp.material_ps) { sp.material_ps->Release(); sp.material_ps = nullptr; }
    if (sp.input_layout) { sp.input_layout->Release(); sp.input_layout = nullptr; }
    if (sp.wire_vs) { sp.wire_vs->Release(); sp.wire_vs = nullptr; }
    if (sp.wire_ps) { sp.wire_ps->Release(); sp.wire_ps = nullptr; }
    if (sp.id_vs) { sp.id_vs->Release(); sp.id_vs = nullptr; }
    if (sp.id_ps) { sp.id_ps->Release(); sp.id_ps = nullptr; }
    if (sp.selected_id_ps) { sp.selected_id_ps->Release(); sp.selected_id_ps = nullptr; }
    if (sp.fullscreen_vs) { sp.fullscreen_vs->Release(); sp.fullscreen_vs = nullptr; }
    if (sp.outline_ps) { sp.outline_ps->Release(); sp.outline_ps = nullptr; }
    if (sp.wire_rasterizer) { sp.wire_rasterizer->Release(); sp.wire_rasterizer = nullptr; }
    if (sp.solid_rasterizer) { sp.solid_rasterizer->Release(); sp.solid_rasterizer = nullptr; }
    if (sp.solid_depth_state) { sp.solid_depth_state->Release(); sp.solid_depth_state = nullptr; }
    if (sp.wire_depth_state) { sp.wire_depth_state->Release(); sp.wire_depth_state = nullptr; }
    if (sp.material_sampler) { sp.material_sampler->Release(); sp.material_sampler = nullptr; }
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

    if (!create_selection_targets(sp, width, height)) {
        release_render_targets(sp);
        return;
    }

    sp.width = width;
    sp.height = height;
}

void update_solid_preview_buffers(SolidPreview& sp, const Scene& scene) {
    // Rebuild geometry only when scene content changes. Camera-only changes update
    // constants in render_solid_preview and must not touch large vertex buffers.
    if (!sp.vb || !sp.ib || sp.geometry_generation != g_editor.geometry_generation) {
        build_buffer_from_scene(sp, scene);
        sp.scene_generation = g_editor.render_generation;
        sp.geometry_generation = g_editor.geometry_generation;
    }
    if (sp.material_generation != g_editor.content_generation || sp.material_data.size() != scene.materials.size()) {
        build_material_preview_resources(sp, scene);
        sp.material_generation = g_editor.content_generation;
    }
}

void render_solid_preview(SolidPreview& sp, const Scene& scene, ImVec2 viewport_size, ViewportPreviewMode mode) {
    ID3D11RenderTargetView* target_rtv = sp.sample_count > 1 ? sp.msaa_rtv : sp.rtv;
    ID3D11DepthStencilView* target_dsv = sp.sample_count > 1 ? sp.msaa_dsv : sp.dsv;
    if (!target_rtv || !target_dsv || !sp.srv || !sp.vb || !sp.ib || !sp.vs || !sp.ps) return;

    if (!upload_solid_constants(sp, scene, viewport_size)) return;

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
    } else if (mode == ViewportPreviewMode::MaterialPreview && sp.material_ps && sp.material_cb) {
        g_context->RSSetState(sp.solid_rasterizer);
        g_context->VSSetShader(sp.vs, nullptr, 0);
        g_context->PSSetShader(sp.material_ps, nullptr, 0);
        g_context->PSSetConstantBuffers(1, 1, &sp.material_cb);
        g_context->PSSetSamplers(0, 1, &sp.material_sampler);
        for (const SolidDrawRange& range : sp.draw_ranges) {
            if (!upload_material_preview_constants(sp, range.material_index)) continue;
            ID3D11ShaderResourceView* srv = material_preview_srv(sp, range.material_index);
            g_context->PSSetShaderResources(0, 1, &srv);
            g_context->DrawIndexed(range.index_count, range.start_index, 0);
        }
        ID3D11ShaderResourceView* null_srv = nullptr;
        g_context->PSSetShaderResources(0, 1, &null_srv);
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

GpuPickResult pick_solid_preview_object(SolidPreview& sp,
                                        const Scene& scene,
                                        ImVec2 viewport_size,
                                        ImVec2 image_min,
                                        ImVec2 image_max,
                                        ImVec2 mouse,
                                        SelectionKind& kind,
                                        int& object_index) {
    kind = SelectionKind::None;
    object_index = -1;

    const int width = std::max(64, static_cast<int>(viewport_size.x));
    const int height = std::max(64, static_cast<int>(viewport_size.y));
    init_solid_preview(sp);
    resize_solid_preview(sp, width, height);
    update_solid_preview_buffers(sp, scene);

    if (sp.selection_refs.empty()) return GpuPickResult::Miss;
    if (!sp.vb || !sp.ib || !has_selection_resources(sp)) return GpuPickResult::Unavailable;
    if (!render_scene_id_pass(sp, scene, viewport_size)) return GpuPickResult::Unavailable;

    const float rect_w = std::max(1.0f, image_max.x - image_min.x);
    const float rect_h = std::max(1.0f, image_max.y - image_min.y);
    const float u = std::clamp((mouse.x - image_min.x) / rect_w, 0.0f, 0.999999f);
    const float v = std::clamp((mouse.y - image_min.y) / rect_h, 0.0f, 0.999999f);
    const int px = std::clamp(static_cast<int>(std::floor(u * static_cast<float>(sp.width))), 0, sp.width - 1);
    const int py = std::clamp(static_cast<int>(std::floor(v * static_cast<float>(sp.height))), 0, sp.height - 1);

    constexpr int kPickRadius = 7;
    const int left = std::max(0, px - kPickRadius);
    const int top = std::max(0, py - kPickRadius);
    const int right = std::min(sp.width, px + kPickRadius + 1);
    const int bottom = std::min(sp.height, py + kPickRadius + 1);
    const int copy_w = std::max(0, right - left);
    const int copy_h = std::max(0, bottom - top);
    if (copy_w <= 0 || copy_h <= 0) return GpuPickResult::Miss;

    D3D11_BOX box{};
    box.left = static_cast<UINT>(left);
    box.top = static_cast<UINT>(top);
    box.front = 0;
    box.right = static_cast<UINT>(right);
    box.bottom = static_cast<UINT>(bottom);
    box.back = 1;
    g_context->CopySubresourceRegion(sp.pick_readback_texture, 0, 0, 0, 0, sp.object_id_texture, 0, &box);

    D3D11_MAPPED_SUBRESOURCE mapped{};
    if (g_context->Map(sp.pick_readback_texture, 0, D3D11_MAP_READ, 0, &mapped) != S_OK) {
        return GpuPickResult::Unavailable;
    }

    uint32_t best_id = 0;
    const int center_x = px - left;
    const int center_y = py - top;
    for (int radius = 0; radius <= kPickRadius && best_id == 0; ++radius) {
        for (int y = 0; y < copy_h && best_id == 0; ++y) {
            for (int x = 0; x < copy_w; ++x) {
                if (std::max(std::abs(x - center_x), std::abs(y - center_y)) != radius) continue;
                const uint8_t* row = static_cast<const uint8_t*>(mapped.pData) + static_cast<size_t>(y) * mapped.RowPitch;
                const uint32_t id = reinterpret_cast<const uint32_t*>(row)[x];
                if (id != 0) {
                    best_id = id;
                    break;
                }
            }
        }
    }

    g_context->Unmap(sp.pick_readback_texture, 0);
    if (best_id == 0) return GpuPickResult::Miss;
    return decode_selection_id(sp, best_id, kind, object_index) ? GpuPickResult::Hit : GpuPickResult::Miss;
}

ID3D11ShaderResourceView* render_selection_outline(SolidPreview& sp,
                                                   const Scene& scene,
                                                   ImVec2 viewport_size,
                                                   SelectionKind kind,
                                                   int object_index) {
    const int width = std::max(64, static_cast<int>(viewport_size.x));
    const int height = std::max(64, static_cast<int>(viewport_size.y));
    init_solid_preview(sp);
    resize_solid_preview(sp, width, height);
    update_solid_preview_buffers(sp, scene);

    const uint32_t selected_id = find_selection_id(sp, kind, object_index);
    if (selected_id == 0 || !sp.vb || !sp.ib || !has_selection_resources(sp)) {
        sp.outline_valid = false;
        return nullptr;
    }

    if (sp.outline_valid &&
        sp.outline_generation == g_editor.render_generation &&
        sp.outline_content_generation == g_editor.content_generation &&
        sp.outline_object_id == selected_id &&
        sp.width == width &&
        sp.height == height &&
        sp.outline_srv) {
        return sp.outline_srv;
    }

    if (!render_scene_id_pass(sp, scene, viewport_size)) {
        sp.outline_valid = false;
        return nullptr;
    }
    if (!render_selected_id_pass(sp, scene, viewport_size, selected_id)) {
        sp.outline_valid = false;
        return nullptr;
    }
    if (!render_outline_texture(sp)) {
        sp.outline_valid = false;
        return nullptr;
    }

    sp.outline_valid = true;
    sp.outline_generation = g_editor.render_generation;
    sp.outline_content_generation = g_editor.content_generation;
    sp.outline_object_id = selected_id;
    return sp.outline_srv;
}

} // namespace lt::editor
