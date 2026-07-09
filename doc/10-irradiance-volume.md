# Irradiance Volume

Irradiance Volume 是 LightTransport 里的低频间接光缓存。它把场景中的漫反射入射光预先烘焙到一组 probe 上，运行时按照 shading point 的位置、法线和方向权重查询，代替完整的多次弹射积分。

这个功能的目标是让编辑器和低样本渲染能快速得到稳定的 diffuse GI。它不是完整路径追踪的替代品，也不会保存镜面、多层透明或高频阴影细节。

主要代码：

- `include/lt/renderer.h`：设置项、烘焙进度和 dirty flag。
- `src/cpu/irradiance_volume.inl`：数据结构、方向基、运行时查询。
- `src/cpu/irradiance_volume_bake.inl`：CPU 烘焙流程。
- `src/gpu/shading.cuh`、`src/gpu/cuda_path_tracer.cu`：CUDA 上传和查询。
- `src/editor_win32.cpp`：Render 面板控件、后台 bake、缓存路径。

## 渲染模型

对于一个漫反射点，间接光可以写成：

```text
L_indirect(p, wo) = rho(p) / pi * E(p, n)
```

其中：

- `rho(p)` 是 diffuse albedo。
- `E(p, n)` 是沿法线半球的 irradiance。
- `p` 是 shading point，`n` 是 shading normal。

真实 irradiance 积分是：

```text
E(p, n) = integral_H L_i(p, wi) max(0, dot(n, wi)) dwi
```

Irradiance Volume 做的是把 `E` 离散为：

```text
E(p, n) ~= query_volume(p, n)
```

也就是先在 probe 位置上按多个方向采样入射光，再在运行时按空间位置插值，并按法线方向选取或混合方向 sample。

当前实现偏向工程实用：

- probe 存的是低频 diffuse irradiance，不存高频 visibility。
- 运行时只给 diffuse/Principled GI 路径使用。
- 直接光、镜面反射、发光面命中仍走原来的路径追踪逻辑。

## 数据模型

公共设置集中在 `RenderSettings`：

```text
use_irradiance_volume
irradiance_volume_grid_resolution
irradiance_volume_subgrid_resolution
irradiance_volume_direction_resolution
irradiance_volume_bake_samples
irradiance_volume_bake_bounces
irradiance_volume_bounds_inset
irradiance_volume_bake_backend
irradiance_volume_principled_gi
irradiance_volume_debug_probes
irradiance_volume_debug_probe_radius_scale
irradiance_volume_cache_enabled
irradiance_volume_auto_update
irradiance_volume_force_rebake
irradiance_volume_cache_path
irradiance_volume_cache_key
irradiance_volume_manual_bounds
irradiance_volume_bounds_min
irradiance_volume_bounds_max
```

`.lt` 保存使用 `SceneRenderSettings` 镜像这些字段。编辑器打开场景后通过 `apply_scene_render_settings()` 把它们复制回 `RenderSettings`。

运行时数据在 `src/cpu/irradiance_volume.inl`：

```text
IrradianceVolume
  bounds_min / bounds_max
  grid_resolution
  subgrid_resolution
  direction_count
  directions
  direction_weights
  grids

IrradianceVolumeGrid
  cell bounds
  samples

IrradianceVolumeSample
  position
  irradiance[direction_count]
```

顶层 grid 用粗分辨率覆盖场景包围盒，子 grid 提供局部 probe 排布。这样比单一大 3D texture 更容易适配当前 CPU 数据结构，也方便按场景范围自动生成。

Mesh 和 analytic sphere 都有：

```text
exclude_from_irradiance_volume_bake
```

它用于把调试物体、代理物体或不希望参与 GI bake 的对象排除掉。注意它只影响烘焙阶段；运行时几何可见性仍由正常渲染路径决定。

## 方向基

方向离散由：

```text
make_irradiance_volume_directions()
make_irradiance_volume_weights()
nearest_irradiance_direction()
```

负责。`direction_resolution` 越高，probe 对法线方向的响应越细，但 bake 时间和缓存体积都会增长。

运行时查询会根据 shading normal 在方向集合里选择最合适的 irradiance sample，或者用方向权重做低频混合。方向基的作用不是做镜面 lobe，而是避免所有法线都读同一个 scalar irradiance，导致墙角、地面和竖直面过度串光。

## 烘焙流程

入口在 `src/cpu/irradiance_volume_bake.inl::bake_irradiance_volume()`。

高层流程：

```text
scene_irradiance_bounds()
  -> 计算自动 bounds，或使用手动 bounds
  -> 按 grid/subgrid 参数布置 probe
  -> 为每个 probe 和方向发射 bake rays
  -> 积分间接 irradiance
  -> 写入 IrradianceVolume
  -> 可选写 .ivol cache
```

自动 bounds 来自场景几何包围盒，并应用 `irradiance_volume_bounds_inset`。手动 bounds 用于建筑室内、薄场景或自动包围盒包含太多空白的情况。

烘焙积分可以概念化为：

```text
E_d = 1 / N * sum_i trace_indirect(probe_position, sample_direction(d, i))
```

其中 `d` 是离散方向，`N` 是 `irradiance_volume_bake_samples`。每条 ray 的最大弹射次数由 `irradiance_volume_bake_bounces` 控制。

`irradiance_volume_principled_gi` 决定 bake 时使用更接近 Principled 材质的 GI 近似，还是更便宜的 Lambert GI。两者都服务于低频 indirect diffuse，不应期待和完整路径追踪逐像素一致。

## 进度与后台任务

进度结构是 `IrradianceVolumeBakeProgress`：

```text
phase
total_samples / completed_samples
total_rays / traced_rays
direction_count
elapsed_ms
```

编辑器中 bake 在后台线程运行。主线程只轮询进度并更新 UI；后台线程不能直接操作 ImGui。完成后会触发 framebuffer 清空和 render refresh。

阶段枚举：

```text
Idle
LoadingCache
Baking
SavingCache
Complete
Failed
```

## 磁盘缓存

`.ivol` 缓存由 bake 代码读写。缓存是否有效由路径、cache key、场景设置和 bake 参数共同决定。

关键设置：

```text
irradiance_volume_cache_enabled
irradiance_volume_cache_path
irradiance_volume_cache_key
irradiance_volume_force_rebake
```

编辑器默认 cache path 是：

```text
<scene>.ivol
```

URL 或无路径场景会落到 `cache/irradiance_volume/` 下的安全文件名。

修改下面内容时通常需要重新 bake：

- 几何、transform、灯光、材质 emission。
- bake bounds、grid/subgrid/direction resolution。
- bake samples/bounces。
- `principled_gi` 或排除 bake 的对象集合。

## 运行时查询

核心查询函数：

```text
query_grid_for_position()
query_irradiance_grid()
query_irradiance_volume()
```

流程：

```text
shading point p, normal n
  -> 找到 p 所在或最近的 grid
  -> 在子 grid probe 中做空间插值
  -> 根据 n 选择/混合方向 irradiance
  -> 返回低频 indirect diffuse
```

空间插值近似是 trilinear lookup。点落在 volume 外时，通常会退化为最近 grid/probe 或返回零，具体行为要以 `query_grid_for_position()` 的当前实现为准。

调试 probe 使用：

```text
make_irradiance_volume_debug_probes()
```

它会生成可视化小球，帮助检查 volume 范围、probe 密度和缓存是否加载成功。

## 路径追踪集成

CPU 路径：

```text
CpuPathTracer::render()
  -> build_render_scene()
  -> trace_path()
  -> shading 命中 diffuse/Principled 表面
  -> query_irradiance_volume()
  -> 把间接 diffuse 加到 radiance
```

CUDA 路径：

```text
CudaPathTracer::render()
  -> pack/upload irradiance volume
  -> render_kernel()
  -> device shading
  -> query irradiance volume buffers
```

GPU 端不会读取宿主侧 `IrradianceVolume` 对象本身，而是读取打包后的 device buffer。改数据布局时必须同步：

- CPU 查询结构。
- GPU types。
- upload/释放路径。
- device shading 函数。

## Dirty Flag 与生命周期

相关 dirty flag：

```text
RenderDirty::IrradianceVolume
RenderDirty::Geometry
RenderDirty::Material
RenderDirty::Transform
RenderDirty::Environment
```

常见规则：

- 只改 debug probe 显示，可以只刷新显示。
- 改 volume 参数或 cache path，要标记 `IrradianceVolume`。
- 改几何、灯光、emission，通常也要让 bake 失效。
- CUDA 后端需要在 volume 变化后重新上传 device buffer。

如果 `irradiance_volume_auto_update` 开启，编辑器会尽量自动重新 bake；如果关闭，用户需要手动按 `Bake Now` 或 `Update Volume`。

## 编辑器入口

Render 面板提供：

- 启用/禁用 Irradiance Volume。
- grid、subgrid、direction resolution。
- bake samples、bounces。
- bake backend：CPU/GPU。
- manual bounds。
- cache path、cache key。
- auto update、force rebake。
- debug probes 和 probe radius。
- Bake/Update 按钮。

状态栏显示当前 bake phase 和进度。日志里会记录 cache hit/miss、bake 开始/完成和失败原因。

## 验证

推荐命令：

```powershell
.\build\Release\lt_render.exe .\scenes\cornell.lt .\build\ivol_cpu.ppm --cpu --frames 4 --irradiance-volume --ivol-force-bake
.\build\Release\lt_render.exe .\scenes\cornell.lt .\build\ivol_cuda.ppm --cuda --frames 4 --irradiance-volume
```

编辑器检查：

- 开启 debug probes，确认 probe 覆盖场景。
- 修改 bounds 后重新 bake，确认缓存路径和 key 更新。
- CPU/CUDA 各跑一次，确认没有明显亮度差异。
- 禁用 Irradiance Volume，确认画面回到路径追踪原有逻辑。

## 常见问题

- 间接光太暗：probe 范围没有覆盖场景、bake samples 太低、物体被排除 bake、cache 过期但没有 force rebake。
- 串光明显：probe 密度太低、场景墙体太薄、方向数太少。
- CPU/CUDA 不一致：host volume 修改后没有重新上传 GPU buffer，或 device 查询逻辑没有同步。
- 编辑器卡顿：auto update 在频繁 transform/material 改动时反复触发 bake。
- Debug probes 看不到：`irradiance_volume_debug_probes` 未开启，或 probe radius scale 太小。
