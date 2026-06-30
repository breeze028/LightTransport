# 公开 API

公开 API 位于 [`include/lt/`](../include/lt/)，命名空间统一为 `lt`。目前没有版本化 ABI 或安装包约定，最稳妥的使用方式是在同一 CMake 工程中链接 `lt_core`。

## 日志 API：`lt/log.h`

应用层可以在启动时调用 `initialize_logging()` 配置控制台、旋转文件和 observer。核心库不会主动创建日志文件；CLI 使用 `logs/lt_render.log`，编辑器使用 `logs/lt_editor.log`。

常用宏是 `LT_LOG_INFO()`、`LT_LOG_WARN()`、`LT_LOG_ERROR()` 等。宏支持顺序替换 `{}`，并自动记录源文件、行号和函数名。详见[日志系统文档](09-logging.md)。

## 最小渲染示例

```cpp
#include "lt/renderer.h"

lt::SceneLoadResult loaded = lt::load_scene("scenes/cornell.lt");
if (!loaded.error.empty()) {
    // loaded.scene 通常仍包含默认场景，可决定继续或中止。
}

lt::RenderSettings settings;
settings.width = 640;
settings.height = 360;
settings.samples_per_pixel = 4;
settings.max_bounces = 6;
settings.frame_index = 0;
settings.dirty = lt::RenderDirty::All;

lt::Framebuffer framebuffer;
framebuffer.resize(settings.width, settings.height);

lt::CpuPathTracer renderer;
renderer.render(loaded.scene, settings, framebuffer);
// framebuffer.rgba: 0xAARRGGBB
// framebuffer.accumulation: 线性 HDR 累积值
```

继续累积下一帧时，将 `frame_index` 设为 1，`dirty` 设为 `None`，保留同一个 Framebuffer。

## 数学 API：`lt/math.h`

### `Vec2` 与 `Vec3`

`Vec3` 支持逐分量加、减、乘，标量乘除和部分复合赋值；`Vec2` 用于 UV。辅助函数包括：

- `dot()`、`cross()`、`length()`、`normalize()`
- `clamp()`、`min()`、`max()`
- `face_forward()`：让法线朝向射线入射侧
- `cosine_sample_hemisphere()`、`to_world()`：余弦半球采样
- `to_rgba8()`：线性 RGB 经固定 gamma 2.2 编码到 `0xAARRGGBB`

注意：零向量 `normalize()` 返回零向量。调用方若要求单位向量，应自行验证长度。

### `Ray`

```cpp
struct Ray {
    Vec3 origin;
    Vec3 direction;
};
```

求交代码默认 `direction` 已归一化。非单位方向会让 `t`、偏移量和距离判断失去一致含义。

### `Rng`

`Rng` 是轻量 PCG 风格随机数状态：

- `next_u32()`：32 位随机整数。
- `next_float()`：`[0, 1)` 浮点数。
- `make_pixel_seed(x, y, frame)`：按像素与帧生成确定性种子。

新增采样逻辑应使用传入的 RNG，不要在材质内部创建固定种子，否则会产生相关噪声。

## 纹理 API：`lt/texture.h`

### `Texture`

主要字段：

- `name`：场景内名称。
- `path`：资源路径；保存 `.lt` 时原样输出。
- `role`：纹理用途（`Unknown`、`Color`、`Data`、`Normal`、`Emission`、`Environment`），导入时由 `apply_texture_role()` 设置。
- `color_space`：导入色彩空间指定（`Auto`、`SceneLinear`、`SRGB`、`Raw`）。
- `width`、`height`
- `pixels`：线性 RGB，大小应为 `width * height`。
- `alpha`：可选 alpha，缺失时采样返回 1。
- `encoded_bytes`、`encoded_extension`：原始编码数据，用于后续重新编码（如 PNG/HDR 导出）。
- `mip_widths`、`mip_heights`、`mip_pixels`：CPU mip 链。

主要方法：

- `sample(uv)`：重复寻址、双线性 RGB 采样。
- `sample_lod(uv, lod)`：三线性 mip 采样。
- `sample_lod(uv, lod, wrap)`：指定二维寻址规则。
- `sample_alpha(uv)`：alpha 采样。
- `build_mips()`：从 `pixels` 重建 mip 链。

`TextureWrap2D`：

- `Repeat`：U/V 重复。
- `RepeatClampY`：U 重复，V 夹紧。
- `OctahedralSphere`：PBRT equal-area 环境图使用的八面体球面包裹。

加载函数：

- `load_ppm_texture()`
- `load_hdr_texture()`
- `load_exr_texture()`
- `load_texture_file()`：按扩展名分派。
- `load_texture_memory()`：嵌入图像，当前依赖 Windows WIC。

角色管理：

- `apply_texture_role(texture, role, color_space)`：根据导入上下文设置纹理的色彩空间和角色标记。例如基础色纹理标记 `Color` + `SRGB`，法线纹理标记 `Normal` + `Raw`。

导出函数：

- `write_texture_png(texture, path, error)`：将纹理编码为 8-bit PNG。
- `write_texture_hdr(texture, path, error)`：将纹理编码为 Radiance HDR RGBE。

直接程序构造 Texture 时，填完像素后应调用 `build_mips()`，否则环境 LOD 只能使用基础层。

### MaterialX 适配器 API：`lt/materialx_adapter.h`

`material_system_status()` 返回 `MaterialSystemStatus` 结构体，指示编译时可用的外部库：

- `materialx_available`：是否链接了 MaterialX。
- `openimageio_available`：是否链接了 OpenImageIO（提供更广泛的纹理格式解码）。
- `opencolorio_available`：是否链接了 OpenColorIO（提供基于角色的色彩空间转换）。
- `surface_model`：当前使用的表面模型标识（`"OpenPBR-compatible standard_surface"`）。
- `texture_pipeline`：当前纹理管线能力描述。

这些标志由 CMake 编译定义控制：`LT_HAS_MATERIALX`、`LT_HAS_OPENIMAGEIO`、`LT_HAS_OPENCOLORIO`。

## 材质 API：`lt/material.h`

### 枚举

- `BrdfModel`：`Lambertian`、`Principled`、`Mirror`、`Dielectric`、`Conductor`、`StandardSurface`
- `AlphaMode`：`Opaque`、`Mask`、`Blend`
- `NprStyle`：`None`、`ColorMap`、`XToon`、`CrossHatching`
- `XToonDetailMode`：常量、深度、轮廓、高光属性
- `MaterialInputChannel`：`RGB`、`R`、`G`、`B`、`A`——从纹理采样指定通道
- `TextureRole`：`Unknown`、`Color`、`Data`、`Normal`、`Emission`、`Environment`
- `TextureColorSpace`：`Auto`、`SceneLinear`、`SRGB`、`Raw`

枚举值会直接转换为 GPU 整数。插入或重排枚举时必须同步 CUDA 代码，并注意旧场景/缓存兼容性。

### `Material`

公共字段覆盖名称、基础色、透明、双面、基础色纹理、法线纹理、发光纹理和 NPR 设置。

非虚拟辅助：

- `base_color(uv)`：`albedo * albedo_texture`。
- `opacity(uv)`：`alpha * texture alpha`，夹紧到 `[0,1]`。
- `emitted(uv)`：`emission * emission_texture`。
- `transmission(uv)`：返回传输系数（默认 0），供 Dielectric/StandardSurface 复写。

派生类必须实现：

```cpp
virtual BrdfModel model() const = 0;
virtual const char* model_name() const = 0;
virtual Vec3 evaluate(Vec3 n, Vec3 wo, Vec3 wi, Vec2 uv) const = 0;
virtual float pdf(Vec3 n, Vec3 wo, Vec3 wi, Vec2 uv) const = 0;
virtual MaterialSample sample(
    Vec3 n, Vec3 wo, Vec2 uv, bool front_face, Rng& rng) const = 0;
virtual std::shared_ptr<Material> clone() const = 0;
```

方向约定：

- `n`：已经朝向当前表面的 shading normal。
- `wo`：从交点指向上一顶点/相机。
- `wi`：从交点指向下一方向/光源。
- 非 delta 材质的 `evaluate()` 返回 BRDF 值。
- `sample().weight` 应包含 `f * abs(n·wi) / pdf`。
- 理想镜面/折射将 `delta = true`、`pdf = 1`，权重直接表示吞吐量倍率。

`front_face` 目前主要供 Dielectric 判断折射率方向。

### `StandardSurfaceMaterial`

`BrdfModel::StandardSurface` 是 OpenPBR 兼容的材质模型，比 Principled 提供更多参数和基于 MaterialInput 的纹理管线：

主要参数：`roughness`、`metalness`、`specular_weight`、`specular_ior`、`transmission_weight`、`transmission_color`、`coat_weight`、`coat_roughness`、`sheen_color`/`sheen_weight`/`sheen_roughness`、`subsurface_weight`/`subsurface_color`、`volume_density`/`volume_color`、`thin_walled`。

每个纹理通道（base_color、roughness、metalness、specular_weight、transmission、opacity、emission、coat、coat_roughness、sheen_color、sheen_roughness）使用 `MaterialInput` 而非裸 Texture shared_ptr。

### `MaterialInput` 与 `TextureTransform`

```cpp
struct MaterialInput {
    std::shared_ptr<Texture> texture;
    Vec3 color_factor = {1.0f, 1.0f, 1.0f};
    float scalar_factor = 1.0f;
    MaterialInputChannel channel = MaterialInputChannel::RGB;
    TextureColorSpace color_space = TextureColorSpace::Auto;
    TextureRole role = TextureRole::Unknown;
    TextureTransform transform;
};
```

`MaterialInput` 封装纹理、颜色/标量因子、采样通道、色彩空间和 UV 变换，由基于角色的纹理管线使用。导入时（如 FBX）通过 `set_material_input()` 填充。`TextureTransform` 包含 `offset`、`scale`、`rotation`（弧度）和 `uv_set` 索引。

工厂和字符串转换：

- `make_material()`：根据 `BrdfModel` 构造内置材质。
- `parse_brdf_model()`：未知字符串回退 Lambertian。
- `parse_npr_style()`、`npr_style_name()`
- `parse_xtoon_detail_mode()`、`xtoon_detail_mode_name()`

## 场景 API：`lt/scene.h`

### 可编辑场景类型

`LightComponent`：

- `enabled`
- `double_sided`
- `color`
- `intensity`

实际面光源辐射为 `color * intensity`。

`DirectionalLight`：

- `direction`：世界空间光源方向。
- `color` 和 `intensity`：颜色和强度倍率。

方向光在 CPU shader 和 CUDA 渲染中作为额外的直接光源，参与 `estimate_direct_lighting()`。它们不附着在 Mesh 上，而是存储在 `Scene::directional_lights` 向量中。

`Mesh`：

- 顶点、法线、UV 和三角形索引
- `material` 是 `Scene::materials` 下标
- `translation`、`rotation`、`scale`
- 可选 `LightComponent`
- `exclude_from_irradiance_volume_bake`：烘焙时跳过该 Mesh

`rotation` 使用弧度，按 X、Y、Z 顺序应用。索引必须每三个组成一个三角形。

`Sphere` 是解析球，包含名称、材质下标、中心和半径。它走独立球求交路径，不会在 `build_render_scene()` 中三角化。同样支持 `exclude_from_irradiance_volume_bake`。

`Camera` 使用 position/target/up/FOV。`right_sign` 用于处理 PBRT 相机手性，普通调用保持默认 1。`up` 向量在非默认相机朝向时需要正确设置。

`Environment` 支持：

- 常量颜色与强度。
- 纹理环境。
- `Equirectangular` 与 `EqualArea` 映射。
- 三个 `light_from_world_*` 基向量，用于旋转环境贴图。

`SceneRenderSettings`：可嵌入 Scene 的渲染设置，包含 NPR 参数和辐照度体积参数。当 Scene 通过编辑器或导入器携带这些设置时，`Scene::has_render_settings` 为 true。它独立于每次渲染的 `RenderSettings`，是场景数据的一部分。

`Scene` 保存所有上述对象。复制 Scene 会克隆材质，但共享纹理，详见[架构文档](01-architecture.md#核心对象的所有权)。Scene 中新增的 `directional_lights` 向量按值存储，在 `save_scene()` 和 `load_scene()` 中读写。

### 渲染场景类型

`Triangle` 是变换后的世界空间三角形，包含：

- 几何法线和三个顶点法线。
- tangent/bitangent。
- UV、centroid、材质和源 mesh 下标。

`RenderSphere` 是验证后的解析球。

`Aabb`、`BvhNode`、`RenderScene` 是公开可见的加速数据结构，主要供渲染器和编辑器拾取使用。应用层通常只调用 `build_render_scene()`。

### 场景函数

- `make_default_scene()`：程序生成 Cornell 风格测试场景。
- `load_scene(path)`：按 URL/扩展名分派，支持 `.lt`、`.gltf`/`.glb`、`.pbrt`、`.fbx`、`.pyscene`；其他扩展名按 `.lt` 解析。
- `load_gltf_scene(path)`
- `load_pbrt_scene(path)`
- `load_fbx_scene(path)`：通过 ufbx 导入 FBX（含嵌入纹理、相机、灯光）；同时查找 `.pyscene` 侧车文件应用材质和环境微调。
- `load_pyscene_scene(path)`：读取 `.pyscene` 文本文件，解析 `importScene()` 指向的 FBX，并应用环境图、emissive 倍率和材质调整。
- `save_scene(scene, path, error)`：写原生 `.lt`。
- `find_material(scene, name)`：精确名称查找，失败返回 -1。
- `build_render_scene(scene)`：世界空间展开并构建加速结构。
- `make_cube_mesh()`、`make_uv_sphere_mesh()`、`make_quad_mesh()`：基础 mesh 生成。`make_quad_mesh` 接受 4 个 Vec3 角点位置。

`SceneLoadResult::error` 非空并不意味着 `scene` 为空。当前许多失败路径会返回默认场景，调用方必须根据产品需求决定是回退还是中止。

## 渲染 API：`lt/renderer.h`

### `RenderSettings`

| 字段 | 含义 |
| --- | --- |
| `width`、`height` | 输出尺寸 |
| `samples_per_pixel` | 当前帧每像素样本数 |
| `max_bounces` | 最大路径顶点数 |
| `use_mis`、`mis_heuristic` | 直接光与 BSDF 的 MIS |
| `acceleration_structure` | Auto、Flat 或 TwoLevel |
| `stylized_samples` | 每个 NPR 顶点的内部估计样本数 |
| `stylized_max_depth` | 一条路径最多风格化多少个表面 |
| `use_irradiance_volume` | 是否在路径追踪中查询辐照度体积间接光 |
| `irradiance_volume_grid_resolution` | 顶层八叉树分辨率 |
| `irradiance_volume_subgrid_resolution` | 子网格分辨率 |
| `irradiance_volume_direction_resolution` | 每探针半球采样方向数 |
| `irradiance_volume_bake_samples` | 烘焙时每方向样本数 |
| `irradiance_volume_bake_bounces` | 烘焙间接光最大弹射次数 |
| `irradiance_volume_bounds_inset` | 自动边界收缩量 |
| `irradiance_volume_principled_gi` | 烘焙时用 Principled 而非 Lambert 求值间接光 |
| `irradiance_volume_debug_probes` | 在渲染中可视化探针球 |
| `irradiance_volume_debug_probe_radius_scale` | 调试探针球半径缩放 |
| `irradiance_volume_cache_enabled` | 启用缓存读写 |
| `irradiance_volume_auto_update` | 编辑器自动重新烘焙 |
| `irradiance_volume_force_rebake` | 强制跳过缓存重新烘焙 |
| `irradiance_volume_cache_path` | 缓存文件路径（默认 `<scene>.ivol`） |
| `irradiance_volume_cache_key` | 缓存 key（默认 scene path） |
| `irradiance_volume_bake_progress` | 指向烘焙进度的原子指针 |
| `irradiance_volume_manual_bounds` | 使用手动烘焙边界 |
| `irradiance_volume_bounds_min` / `_max` | 手动烘焙边界 |
| `frame_index` | 当前累积帧，从 0 开始 |
| `dirty` | 数据和累积失效标记 |

`scene_has_npr_styles()`、`stylized_rendering_enabled()` 和 `irradiance_volume_rendering_enabled()` 是后端选择可直接复用的辅助函数。

### `RenderDirty`

这是位标志枚举，可以用 `operator|` 组合，并用 `has_dirty()` 查询。语义见[渲染管线文档](03-rendering-pipeline.md#脏标记与缓存)。

### `Framebuffer`

- `resize(w,h)`：尺寸变化时重新分配并清空；相同尺寸不清空。
- `clear()`：清空 HDR 累积和 RGBA。
- `clear_accumulation()`：仅清空 HDR 累积，保留 RGBA（用于需要清累积但不丢显示结果的场景）。
- `accumulation`：线性 HDR 累积和，不是平均值。
- `rgba`：显示用 gamma 编码 `0xAARRGGBB`。

CPU 后端会维护宿主侧 `accumulation`；CUDA 后端当前只把 `rgba` 拷回宿主，HDR 累积保留在设备内存中。因此不能直接从 CUDA 渲染后的 `Framebuffer::accumulation` 导出正确 HDR。

### `IRenderer` 与后端

接口：

```cpp
class IRenderer {
public:
    virtual const char* name() const = 0;
    virtual bool available() const = 0;
    virtual void reset() = 0;
    virtual void render(
        const Scene&, const RenderSettings&, Framebuffer&) = 0;
};
```

内置实现：

- `CpuPathTracer`：始终可用，缓存 `RenderScene`。
- `CudaPathTracer`：运行时检查设备；失败时多数路径会自动调用 CPU fallback。

调用 `reset()` 会丢弃后端缓存。CUDA reset 还会释放显存和 texture object。

## 程序构造场景示例

```cpp
lt::Scene scene;
scene.camera.position = {0.0f, 1.0f, -3.0f};
scene.camera.target = {0.0f, 0.5f, 0.0f};

scene.materials.push_back(lt::make_material(
    "blue", {0.2f, 0.35f, 0.9f},
    lt::BrdfModel::Principled, 0.3f, 0.0f));

scene.meshes.push_back(
    lt::make_cube_mesh("Cube", 0, {0.0f, 0.5f, 0.0f}, 1.0f));

scene.environment.constant = true;
scene.environment.color = {0.1f, 0.12f, 0.15f};
scene.environment.strength = 1.0f;
```

在加入 Mesh 前先确保材质下标有效。`build_render_scene()` 会跳过无效球，但 Mesh 的材质合法性主要在后续 shading/GPU 打包时检查，最好在应用层提前验证。
