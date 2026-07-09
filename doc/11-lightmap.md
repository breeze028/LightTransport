# Lightmap

Lightmap 是把静态间接光烘焙到 mesh 的第二套 UV 上，在运行时按 `uv_lightmap` 查询的一种 GI 缓存。相比 Irradiance Volume，它和表面绑定得更紧，细节可以更贴合几何；代价是需要 UV unwrap、贴图分辨率和 cache 管理。

主要代码：

- `include/lt/renderer.h`：lightmap 设置、进度和 `RenderDirty::Lightmap`。
- `src/cpu/lightmap.inl`：运行时数据结构和查询。
- `src/cpu/lightmap_bake.inl`：unwrap、bake、cache。
- `src/gpu/shading.cuh`、`src/gpu/cuda_path_tracer.cu`：CUDA 数据上传和查询。
- `src/editor_win32.cpp`：Render 面板、overlay、后台 bake。

## 渲染模型

对漫反射表面，可以把光照拆成：

```text
L_o(p, wo) ~= L_direct(p, wo) + rho(p) / pi * E_lm(uv_lm)
```

其中 `E_lm` 是 lightmap 存储的低频 indirect irradiance。运行时只需要根据命中三角形插值得到 `uv_lm`，再做 texture lookup。

lightmap 与完整路径追踪的差异：

- 它适合静态 diffuse GI。
- 不保存视角相关的 specular。
- 对动态物体、动态灯光或材质变化需要重新 bake。
- 分辨率不足时会出现漏光、模糊或 texel 边界问题。

## 数据模型

设置字段在 `RenderSettings`：

```text
use_lightmap
lightmap_resolution
lightmap_padding
lightmap_dilation
lightmap_bake_samples
lightmap_bake_bounces
lightmap_principled_gi
lightmap_bake_backend
lightmap_cache_enabled
lightmap_auto_update
lightmap_force_rebake
lightmap_cache_path
lightmap_cache_key
```

运行时结构在 `src/cpu/lightmap.inl`：

```text
Lightmap
  width / height
  texels
  triangles

LightmapTriangle
  mesh_index
  triangle_index
  uv0 / uv1 / uv2
```

`Lightmap::texels` 存储已经烘焙好的 irradiance。三角形记录保存 lightmap UV 与场景三角形的映射关系，用于 runtime hit 到 texel 的查询。

`.lt` 保存通过 `SceneRenderSettings` 保留 lightmap 参数，但不把大贴图直接内嵌到 scene 文本里；实际 bake 结果走 `.lmap` cache。

## UV Unwrap

lightmap 需要非重叠 UV。当前使用 xatlas：

```text
scene mesh triangles
  -> xatlas chart generation
  -> atlas packing
  -> per-triangle lightmap UV
```

相关依赖由 `CMakeLists.txt` 查找或 FetchContent 拉取：

```text
XATLAS_INCLUDE_DIR
XATLAS_LIBRARY
LT_XATLAS_SOURCES
```

padding 由 `lightmap_padding` 控制，目的是降低 bilinear lookup 和 mip/邻域操作时的 bleeding。`lightmap_dilation` 会把有效 texel 往空洞区域扩张，填补 chart 边缘。

## 烘焙流程

入口在 `src/cpu/lightmap_bake.inl`。

流程：

```text
load .lmap cache, if valid
  -> unwrap scene meshes with xatlas
  -> allocate lightmap texels
  -> for each covered texel
       -> 找到对应 triangle 和 surface point
       -> 按法线半球发射 bake rays
       -> 积分 indirect irradiance
  -> dilate texels
  -> save .lmap cache
```

texel 对应的 surface point 来自 lightmap UV 对三角形的反查。概念公式：

```text
p = b0 * p0 + b1 * p1 + b2 * p2
n = normalize(b0 * n0 + b1 * n1 + b2 * n2)
E_texel = 1 / N * sum_i trace_indirect(p, sample_hemisphere(n, i))
```

其中 `b0,b1,b2` 是 barycentric 坐标，`N` 是 `lightmap_bake_samples`。

`lightmap_bake_bounces` 控制烘焙时的间接光弹射深度。`lightmap_principled_gi` 决定 bake 时使用 Principled 近似还是 Lambert 近似。

## Cache 格式

`.lmap` cache 由 `src/cpu/lightmap_bake.inl` 读写。当前格式包含：

```text
magic: LTLMAP1
version: 1
settings fingerprint
dimensions
texels
triangle mappings
```

缓存 fingerprint 会综合场景和 lightmap 参数。改变几何、UV、材质、灯光或 bake 参数后，旧 cache 应该失效。

默认路径：

```text
<scene>.lmap
```

编辑器中的 URL 或 untitled scene 会放到 `cache/lightmap/` 下。

## 运行时查询

CPU 查询：

```text
query_lightmap()
apply_lightmap_to_render_scene()
```

运行时命中三角形后：

```text
hit triangle id
  -> 找到 LightmapTriangle
  -> 用 hit barycentric 插值 lightmap UV
  -> bilinear sample texels
  -> 返回 indirect irradiance
```

再把查询结果乘以材质 diffuse albedo：

```text
L_indirect = rho / pi * E_lightmap
```

GPU 查询：

```text
GpuLightmap
query_lightmap_gpu()
shade_material_from_lightmap_gpu()
material_uses_lightmap_gi_gpu()
```

GPU 不读取宿主侧 `Lightmap` 容器，而是读取打包后的 texel buffer 和 triangle mapping。

## CPU/CUDA 生命周期

CPU：

- `RenderScene` 构建时关联 lightmap 数据。
- 材质 shading 命中时查询 lightmap。
- cache 加载/烘焙完成后需要重置 accumulation。

CUDA：

- `CudaPathTracer::render()` 根据 dirty flag 判断是否重新上传。
- `GpuLightmap` 持有 device buffer 指针、尺寸和 mapping。
- lightmap 变化后必须释放旧 device buffer 并上传新数据。

这类资源和普通材质参数不同，不适合只做 partial material upload；它影响三角形级别的 shading payload。

## 编辑器入口

Render 面板包含：

- 启用/禁用 Lightmap。
- resolution、padding、dilation。
- bake samples、bounces。
- bake backend。
- cache path、auto update、force bake。
- `Update Lightmap` / bake 按钮。

编辑器还可以显示 lightmap overlay，用于检查 atlas 覆盖、chart 边界和是否存在空 texel。

后台 bake 与 Irradiance Volume 一样，不直接操作 ImGui；主线程轮询 `LightmapBakeProgress`。

## Dirty Flag

相关 flag：

```text
RenderDirty::Lightmap
RenderDirty::Geometry
RenderDirty::Material
RenderDirty::Transform
```

经验规则：

- 改 lightmap 参数：`Lightmap`。
- 改 mesh 顶点/索引/transform：通常要重新 unwrap 和 bake。
- 改 diffuse albedo：运行时乘法变化可能只需要 `Material`，但如果 bake 使用材质属性参与间接光，仍应重新 bake。
- 改 emission 或 mesh light：需要重新 bake。

## 限制

- 只适合静态或低频变化的 GI。
- 目前以 mesh 为主；analytic sphere 如果没有稳定 unwrap，就不适合直接烘焙到 lightmap。
- 分辨率固定为一张 atlas；复杂大场景可能需要多 atlas 或 tiled lightmap。
- 不保存 HDR 输出格式；cache 是内部格式，不是可直接查看的 PNG。

## 验证

命令：

```powershell
.\build\Release\lt_render.exe .\scenes\cornell.lt .\build\lightmap_cpu.ppm --cpu --frames 4 --lightmap --lightmap-force-bake
.\build\Release\lt_render.exe .\scenes\cornell.lt .\build\lightmap_cuda.ppm --cuda --frames 4 --lightmap
```

编辑器检查：

- 开启 Lightmap 后 bake 一次，确认进度走完。
- 保存并重新打开场景，确认 cache 命中。
- 调大/调小 resolution，观察 GI 细节与 bake 时间变化。
- CPU/CUDA 对比，确认没有明显红蓝通道或亮度差异。

## 常见问题

- 画面全黑：没有有效 unwrap、cache 加载失败、texel 没有覆盖到三角形。
- chart 边缘漏光：padding/dilation 太小，或 bilinear lookup 采到了空 texel。
- 修改场景后结果不变：cache key 没失效，或 auto update 关闭。
- CUDA 与 CPU 不一致：triangle mapping 或 texel buffer 上传不同步。
- bake 很慢：resolution、samples、bounces 同时过高；先用低分辨率验证。
