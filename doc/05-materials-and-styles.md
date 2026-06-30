# 材质与风格系统

## 内置材质

| 模型 | CPU 行为 | 主要参数 |
| --- | --- | --- |
| Lambertian | 余弦采样漫反射 | albedo、albedo texture |
| Principled | Lambert + GGX specular + sheen + clearcoat | roughness、metallic 和相关纹理 |
| Mirror | 无色理想镜面 delta | 当前忽略 albedo |
| Dielectric | Schlick 反射/折射 delta | IOR、transmission tint |
| Conductor | 带 base color tint 的理想镜面 delta | albedo |
| StandardSurface | OpenPBR 兼容模型，含漫反射、GGX specular、transmission、coat、sheen、subsurface 和 volume | roughness、metalness、transmission、coat、sheen 和 MaterialInput 纹理管线 |

`Mirror` 和 `Conductor` 当前不是粗糙微表面模型。若需要粗糙金属，应扩展 Principled 或新增模型，而不是只给 Conductor 增加一个未使用的 roughness 字段。

## StandardSurface 材质

`StandardSurfaceMaterial`（`BrdfModel::StandardSurface`）是 OpenPBR 兼容的综合材质模型，设计用于 FBX 导入等工业格式。相比 Principled，它增加了：

- **分层的 GGX specular**：通过 `specular_weight`、`specular_ior` 控制镜面反射强度和色散。
- **传输（transmission）**：`transmission_weight` 和 `transmission_color` 控制折射。与 Dielectric 不同，StandardSurface 的传输是作为 BSDF lobe 而非 delta 事件参与采样。
- **coat 层**：`coat_weight` 和 `coat_roughness` 提供独立的 coating 反射层。
- **sheen**：`sheen_color`、`sheen_weight`、`sheen_roughness` 提供边缘光织物效果。
- **subsurface**：`subsurface_weight` 和 `subsurface_color` 标记（当前 `unsupported_subsurface` 标志位标记为未实现）。
- **volume**：`volume_density` 和 `volume_color` 标记（当前 `unsupported_volume` 标志位标记为未实现）。
- **thin_walled**：标记薄壁传输模式。

### 基于 MaterialInput 的纹理管线

StandardSurface 不直接使用 `albedo_texture` 等 `shared_ptr<Texture>` 字段，而是通过 `MaterialInput` 结构体封装每个纹理通道：

```
base_color_input    → 替代 Material::albedo_texture
roughness_input     → 替代 PrincipledMaterial::roughness + metallic_roughness_texture
metalness_input     → 替代 PrincipledMaterial::metallic
specular_weight_input
transmission_input
opacity_input       → 替代 Material::alpha 纹理信息
emission_input
coat_input
coat_roughness_input
sheen_color_input
sheen_roughness_input
```

每个 `MaterialInput` 包含纹理指针、颜色/标量因子、采样通道（RGB/R/G/B/A）、色彩空间预设和 UV 变换。访问函数（如 `roughness_at(uv)`、`metalness_at(uv)`）从 MaterialInput 中提取单通道值。

CPU 的 `sample()` 按权重混合 diffuse、specular、transmission 和 coat lobe；`evaluate()` 累加各 lobe 的 BRDF 值。`transmission()` 返回传输权重。GPU 端在 `src/gpu/shading.cuh` 中有对应实现。

### StandardSurface 与 Principled 的关系

- glTF 导入仍使用 Principled。
- FBX 导入使用 StandardSurface（PBR 材质 + MaterialInput 管线）。
- 编辑器 BRDF 下拉菜单同时列出两种模型。
- StandardSurface 在切换 BRDF 时可以被降级转换为 Principled（丢失 transmission/coat/specular 等高级字段）。

## BRDF 接口约束

非 delta 材质的三个函数必须匹配：

- `evaluate(n, wo, wi, uv)` 返回 BRDF。
- `pdf(n, wo, wi, uv)` 返回采样该 `wi` 的概率密度。
- `sample(...)` 生成方向，并返回 `weight = evaluate * abs(n·wi) / pdf`。

如果 sampling 分布与 `pdf()` 不一致，MIS 和吞吐量都会产生偏差。新增 lobe 时必须同时更新 evaluate、pdf 和 sample。

Delta 材质：

- `evaluate()` 和 `pdf()` 对普通方向可返回 0。
- `sample()` 返回唯一离散方向。
- `delta = true`。
- 当前实现约定 `pdf = 1`，`weight` 直接作为路径 throughput 倍率。

## 材质公共字段的渲染语义

### 基础色

`base_color(uv) = albedo * albedo_texture.sample(uv)`。

### Alpha

`material_visible()` 的行为：

- `Opaque`：总是可见。
- `Mask`：opacity 小于 cutoff 时跳过表面。
- `Blend`：以 opacity 为概率随机保留表面。

跳过后沿原方向继续追踪，并且普通路径不消耗 bounce。阴影射线最多连续跳过 8 个透明/介质交点。

### 双面

双面材质会在 evaluate/pdf 和直接光余弦项中使用绝对值。几何命中仍会记录 front face，并将 shading normal 朝向射线。

### 法线贴图

RGB 从 `[0,1]` 映射到 `[-1,1]`，XY 乘 `normal_scale`，再通过 Triangle tangent basis 转到世界空间。

### 发光

材质 emission 可让三角形成为灯。Mesh `LightComponent` 与材质 emission 都存在时，发射计算会按当前函数规则组合/选择；修改这部分前应同时检查：

- CPU `light_emission()`、`material_emission()`、`emitted_radiance()`
- GPU `GpuTriangle::emission` 和设备端对应函数
- `build_render_scene()` 的灯列表判定

## Principled 字段

`PrincipledMaterial` 提供：

- roughness，纹理使用 G 通道。
- metallic，纹理使用 B 通道。
- sheen color/roughness 及纹理。
- clearcoat/clearcoat roughness 及纹理。

当前 sample 的主分布只混合 diffuse 与 GGX 主 specular；sheen 和 clearcoat 被加入 evaluate，但没有各自完整的独立采样分布。clearcoat 只提高 specular 选择概率。这是近似实现，极端参数下方差和能量一致性可能不理想。

## 材质字段的完整传播链

给材质增加字段时逐项检查：

1. `include/lt/material.h`
   - 字段默认值。
   - 对应 `_at(uv)` 访问函数（如有纹理）。
   - 若使用 MaterialInput 管线，定义 `MaterialInput` 成员而非裸 `shared_ptr<Texture>`。
   - clone 是否自动复制。
2. `src/material.cpp`
   - CPU evaluate/pdf/sample。
3. `src/gltf_loader.cpp`、`src/fbx_loader.cpp` 和/或 `src/pbrt/pbrt_loader.cpp`
   - 外部格式映射。
4. `src/scene/scene_io.cpp`
   - `.lt` 读取与保存。
5. `src/gpu/types.cuh::GpuMaterial`
   - 扁平字段或纹理下标。
6. `src/gpu/scene_upload.cuh::pack_scene()`
   - 从多态 Material 提取并打包。MaterialInput 管线需要从 `Scene::textures` 查找 shared_ptr 获得纹理下标。
7. `src/gpu/shading.cuh`
   - GPU evaluate/pdf/sample。
8. `src/editor_win32.cpp`
   - 编辑控件。
   - 切换 BRDF 时保存旧字段并迁移到新对象的代码。
9. `src/cli/render_options.*`
   - 若需要命令行覆盖。
10. 文档和场景样例。

这是项目中最容易”CPU 正常、CUDA 默认值、保存后丢失”的改动类型。

## 新增一种 BRDF

假设新增 `RoughConductor`：

1. 在 `BrdfModel` 追加枚举值，尽量不要插入中间。
2. 在 `material.h` 增加派生类：
   - 参数字段。
   - `model()`、`model_name()`。
   - evaluate/pdf/sample/clone。
3. 在 `material.cpp` 实现 CPU 数学。
4. 更新 `make_material()` 和 `parse_brdf_model()`。
5. 更新 `.lt` 保存所需参数；当前固定 material 行可能需要扩展语法。
6. 更新编辑器 BRDF 列表和切换模型的数据迁移。
7. 若支持 glTF/PBRT/FBX，增加映射。
8. CUDA：
   - `GpuMaterial` 增加参数。
   - `pack_scene()` 提取参数。
   - `evaluate_brdf_gpu()`、`material_pdf_gpu()`、`sample_material_gpu()` 增加分支。
9. 测试：
   - 白炉/能量测试。
   - 不同 roughness。
   - CPU/CUDA 同种子视觉对比。
   - MIS 开关。
   - `.lt` 保存再加载。

若暂时只做 CPU，应在 CLI 和编辑器选择该材质时阻止 CUDA，做法可参考 NPR 的 fallback。

StandardSurface 是目前最完整的 BRDF 新增示例：涵盖了新的 MaterialInput 纹理管线、FBX 导入集成、CPU 多 lobe 采样和 CUDA 设备端实现。

## NPR 数据模型

每个 Material 都有一个 `NprSettings`：

- Color Map：把估计辐射亮度映射到固定渐变。
- X-Toon：基于主光方向、量化 tone 和 detail attribute 生成 toon 色带。
- Cross Hatching：用视图相关周期平面在世界交点上生成线条。

全局 `RenderSettings` 控制风格估计成本和最大风格化深度；具体视觉参数保存在材质中。

### Color Map

输入是 `estimate` 三通道平均值，经 `value_min/value_max` 归一化后进入固定五段渐变。

### X-Toon

- tone 来自法线和“第一个三角灯”方向。
- `xtoon_steps` 量化 tone。
- detail 可取 Constant、Depth、NearSilhouette 或 Highlight。
- shadow/mid/lit 会与 base color 相乘；accent 是直接颜色。

没有三角灯时使用固定方向 `{0.35, 0.85, 0.35}`。

### Cross Hatching

- 由估计亮度决定启用几组线。
- 线条是相对于 Camera basis 的世界空间周期平面。
- `shadow_only` 只参考第一个三角灯，并做一次遮挡测试。
- `passthrough` 控制非线条区域返回原估计还是 paper 色。

## 新增一种 NPR 风格

例如新增 `Outline`：

1. `include/lt/material.h`
   - 向 `NprStyle` 追加值。
   - 在 `NprSettings` 增加参数。
2. `src/material.cpp`
   - 更新 `parse_npr_style()`、`npr_style_name()`。
3. `src/cpu/shading.inl`
   - 实现 `apply_outline_style()`。
   - 在 `apply_npr_style()` 分派。
4. `src/scene/scene_io.cpp`
   - `parse_npr_settings()` 读取。
   - `save_scene()` 写出。
5. `src/cli/render_options.*`
   - 全局参数和 material override。
6. `src/editor_win32.cpp::draw_npr_controls()`
   - Combo 名称和参数控件。
   - 修改后调用 `mark_npr_dirty()`。
7. 添加最小 `.lt` 示例。
8. 若仍仅 CPU，不需要改 fallback；若实现 CUDA，需新增 GPU NPR 路径并重新设计后端能力判断。

屏幕空间轮廓通常更适合后处理，而不是当前按交点返回颜色的 NPR API。若算法需要邻域像素、法线 buffer 或深度 buffer，应新增 render pass/AOV，而不是强塞进 `apply_npr_style()`。

## 纹理角色管线

`TextureRole` 和 `TextureColorSpace` 枚举为纹理提供元数据标记，由导入器设置、渲染器和编辑器消费：

- `apply_texture_role(texture, role, color_space)` 是统一入口，设置在 `include/lt/texture.h`。
- FBX 导入器是主要消费者，通过 `set_material_input()` 为每个 MaterialInput 通道分配正确的角色和色彩空间。
- 角色标记影响：
  - 编辑器纹理列表中的图标/标签。
  - 纹理导出时的色彩空间处理。
  - 未来可能的自动 sRGB↔linear 转换。
- 新增导入器时，应在加载纹理后立即调用 `apply_texture_role()`，建立一致的角色标记。
