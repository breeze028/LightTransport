# SVGF

SVGF（Spatiotemporal Variance-Guided Filtering）是 LightTransport 中用于实时路径追踪预览的降噪器。它的核心思路是：用低样本 path tracing 得到 radiance 和 primary-hit AOV，再通过时域重投影、方差估计和 edge-aware A-trous filter 重建稳定图像。

当前实现同时覆盖 CPU 和 CUDA：

- CPU：`src/denoise/svgf.cpp`、`src/cpu/renderer.inl`
- CUDA：`src/gpu/kernel.cuh`、`src/gpu/cuda_path_tracer.cu`
- Raster G-buffer：`src/editor/solid_preview.cpp`、`src/editor_win32.cpp`

整体流程：

```text
path traced radiance + primary-hit AOVs
  -> temporal reprojection
  -> moments / variance
  -> A-trous filtering
  -> final color reconstruction
  -> StablePostAA 或 TAA resolve
```

## 设置项

`RenderSettings` 中相关字段：

```text
denoiser_mode
svgf_iterations
svgf_alpha
svgf_moments_alpha
svgf_phi_color
svgf_phi_normal
svgf_phi_depth
svgf_rasterized_gbuffer
svgf_debug_view
antialiasing_mode
camera_jitter_x
camera_jitter_y
frame_index
dirty
```

模式：

```text
DenoiserMode::Off
DenoiserMode::Svgf
```

Debug view：

```text
Final
Raw
Albedo
Normal
Depth
Variance
HistoryLength
```

AA mode：

```text
Off
StablePostAA
TAA
```

`StablePostAA` 是确定性的空间边缘 AA，不使用 jitter 和历史。`TAA` 使用 final color history；jitter 只在画面稳定、`dirty == RenderDirty::None` 且 frame index 合适时启用。

## AOV

SVGF 需要的 primary-hit AOV 位于 `Framebuffer::AovBuffers`：

```text
radiance
albedo
emission
normal
world_position
linear_depth
object_id
rasterized
```

这些 buffer 的用途：

- `radiance`：当前 frame 的低样本路径追踪结果。
- `albedo`：用于 demodulation/remodulation。
- `emission`：避免发光面被 diffuse filter 错误模糊。
- `normal`、`linear_depth`：A-trous edge stopping。
- `world_position`、`object_id`：history validation 和 reprojection。
- `rasterized`：标记 AOV 是否来自 D3D11 raster G-buffer。

`object_id == 0` 表示背景。

## AOV 生成

### Path-Traced Primary Hit

CPU 中 primary hit 写入在：

```text
src/cpu/renderer.inl::write_primary_aov()
```

它会发射与当前像素一致的 primary ray，记录第一个表面命中的 material/geometry 信息。没有命中时写背景值。

CUDA 中相同信息由 kernel 写入 device AOV buffer，再在需要时用于 SVGF kernel。

### Rasterized G-Buffer

编辑器可以用 D3D11 raster pipeline 生成 G-buffer：

```text
render_svgf_gbuffer()
render_svgf_gbuffer_interop()
read_gbuffer_to_framebuffer()
```

优点：

- primary-hit 数据稳定且便宜。
- Solid/Material Preview 已经有 D3D11 几何缓存，可以复用。
- 对实时编辑更接近成熟引擎做法。

风险：

- raster camera 与 path tracing ray 必须严格对齐。
- 边缘处 path traced sample 与 raster sample 可能命中不同 primitive。
- jitter、TAA、SVGF history 必须使用同一套 projection 约定。

## Radiance Demodulation

SVGF 通常不直接滤波最终颜色，而是先把材质颜色分离：

```text
illumination = max(radiance - emission, 0) / max(albedo, epsilon)
```

滤波后再重构：

```text
final_color = filtered_illumination * albedo + emission
```

这样可以减少贴图高频细节被 A-trous 误模糊。对背景或无效 surface，则直接保留 raw radiance。

## Temporal Reprojection

时域重投影把上一帧历史映射到当前像素：

```text
current world_position
  -> project to previous camera
  -> previous pixel coordinates
  -> sample previous history
  -> validate normal/depth/object_id
```

验证条件包括：

- `object_id` 一致或允许的边缘情况。
- 法线夹角不过大。
- 深度差不过大。
- previous pixel 在屏幕范围内。

通过后做指数混合：

```text
history = lerp(current, previous, 1 - alpha)
```

其中 `svgf_alpha` 控制 illumination history 的响应速度。alpha 越小，静止画面越稳，但运动时越容易拖影。

## Moments 与 Variance

SVGF 使用 luminance 的一阶、二阶矩估计方差：

```text
m1 = E[x]
m2 = E[x^2]
variance = max(m2 - m1 * m1, 0)
```

时域更新：

```text
m1_t = lerp(luminance(current), m1_prev, 1 - moments_alpha)
m2_t = lerp(luminance(current)^2, m2_prev, 1 - moments_alpha)
```

`svgf_moments_alpha` 控制方差历史的稳定程度。历史长度不足时，方差估计更不可靠，filter 应更保守。

## A-trous Filter

A-trous 是带洞卷积。每次迭代扩大采样步长：

```text
step = 1, 2, 4, 8, ...
```

每个邻域样本的权重由三部分组成：

```text
w = w_kernel * w_color * w_normal * w_depth
```

典型形式：

```text
w_color  = exp(-abs(L_i - L_c) / (phi_color * sqrt(variance) + eps))
w_normal = pow(max(dot(n_i, n_c), 0), phi_normal)
w_depth  = exp(-abs(z_i - z_c) / phi_depth)
```

参数：

- `svgf_iterations`：A-trous pass 数量。
- `svgf_phi_color`：颜色/illumination 差异容忍度。
- `svgf_phi_normal`：法线边缘保护强度。
- `svgf_phi_depth`：深度边缘保护强度。

pass 太多会糊，太少会保留噪声。phi 太松会跨边缘漏光，太紧会降噪不足。

## Debug Views

CPU 的 `debug_color()` 和 CUDA 的 `svgf_debug_color_gpu()` 输出 debug view：

- Raw：未降噪 radiance。
- Albedo：primary albedo。
- Normal：法线可视化。
- Depth：线性深度。
- Variance：方差热力。
- HistoryLength：历史长度。

调试顺序建议：

1. 先看 Albedo/Normal/Depth 是否和画面边缘对齐。
2. 再看 HistoryLength 是否在静止时增长。
3. 最后看 Variance 是否在噪声区域高、平坦区域低。

## Final TAA Resolve

SVGF 后还有一层 final color history，和 illumination history 分开：

```text
Framebuffer::TaaHistory
  color
  normal
  world_position
  linear_depth
  object_id
  history_length
  camera
  jitter
```

TAA resolve 的目的：

- 相机移动或物体移动时仍能重用部分 final history。
- 对边缘 jitter 采样做时间累积。
- 在停止移动后逐步稳定。

当前 resolve 使用：

- fractional reprojection。
- bilinear history sampling。
- YCoCg/variance clipping 或邻域 clamp 类似的历史限制。
- 根据 dirty/motion/history length 调整 current-frame weight。

关键约束：

- Geometry/Material/Texture 等大变化会清空历史。
- Camera/Transform 变化不应完全禁用 TAA resolve，但会提高当前帧权重，减少拖影。
- jitter 只在稳定条件下启用；移动中主要依靠 reprojection，而不是继续加大 jitter。

## Edge Coverage

TAA 对几何边缘最敏感。当前实现会检查 object-id 边缘：

```text
taa_object_edge()
```

边缘处的历史更容易无效，因为当前 frame 和上一帧可能命中不同对象。过度拒绝历史会导致“每帧锯齿位置不同”的抖动；过度接受历史会产生拖影。这里需要在稳定性和抗锯齿之间折中。

## Stable Post AA

`StablePostAA` 是默认更稳的后处理边缘 AA：

- 不使用 camera jitter。
- 不依赖 history。
- 根据 luma edge 做局部平滑。
- 作用在 SVGF final 后。

它不会产生 temporal supersampling 的极限效果，但静止画面稳定，适合编辑器默认预览。

## CPU Pipeline

CPU 路径：

```text
CpuPathTracer::render()
  -> framebuffer.resize()
  -> 生成 radiance 和 primary AOV
  -> apply_svgf_denoiser()
       -> demodulate
       -> temporal reprojection
       -> moments/variance
       -> A-trous passes
       -> remodulate
       -> debug view 或 final
       -> apply_final_taa()
  -> framebuffer.rgba
```

主要函数：

```text
apply_svgf_denoiser()
reproject_history()
atrous_pass()
apply_final_taa()
apply_luma_edge_post_aa()
```

CPU 使用 `std::vector` 存储中间 buffer，逻辑更易读，是修改算法时的参考实现。

## CUDA Pipeline

CUDA 路径：

```text
CudaPathTracer::render()
  -> 确认 device buffers
  -> render_kernel 生成 radiance/AOV
  -> SVGF kernels
       -> temporal reprojection
       -> moments/variance
       -> A-trous ping-pong
       -> final resolve
  -> copy rgba to host
```

相关资源：

```text
device_aov_*
device_svgf_history_*
device_taa_history_*
device_svgf_ping/pong
device_variance_ping/pong
```

CUDA 实现应和 CPU 保持同一语义：

- 相同的 history invalidation。
- 相同的 debug view。
- 相同的 AA mode 行为。
- 相同的 rasterized G-buffer 使用条件。

## Dirty Flag 与 History

SVGF history 对 dirty 很敏感。大致规则：

```text
RenderDirty::Geometry      -> 清 SVGF history，通常也清 TAA history
RenderDirty::Material      -> 清 history
RenderDirty::Texture       -> 清 history
RenderDirty::Environment   -> 清 history
RenderDirty::Camera        -> 允许 TAA reproject，SVGF 视具体数据而定
RenderDirty::Transform     -> illumination history 更保守，TAA resolve 仍可工作
```

编辑器连续移动时不能简单把 samples 显示为 0 后完全停掉 SVGF/TAA。目标是：

- path tracing 每帧给出新 radiance。
- G-buffer/AOV 跟当前 camera/scene 对齐。
- history validation 决定可重用多少。
- 移动中提高当前帧权重，停止后逐步稳定。

## CLI 与编辑器

CLI：

```text
--denoiser svgf|off
--aa off|stable|taa
--svgf-iterations N
--svgf-alpha F
--svgf-moments-alpha F
--svgf-phi-color F
--svgf-phi-normal F
--svgf-phi-depth F
--svgf-debug final|raw|albedo|normal|depth|variance|history
```

编辑器 Render 面板：

- Denoiser mode。
- Rasterized G-Buffer。
- Antialiasing。
- Debug View。
- Iterations。
- Alpha / Moments Alpha。
- Phi Color / Normal / Depth。

当前编辑器中启用 SVGF 后，实时预览默认把 `max_bounces` 调到 2，后续可以做成可调控件。

## 验证

CPU：

```powershell
.\build\Release\lt_render.exe .\scenes\cornell.lt .\build\svgf_cpu.ppm --cpu --size 64 48 --frames 16 --denoiser svgf --aa stable --spp 1
.\build\Release\lt_render.exe .\scenes\cornell.lt .\build\svgf_cpu_taa.ppm --cpu --size 64 48 --frames 16 --denoiser svgf --aa taa --spp 1
```

CUDA：

```powershell
.\build\Release\lt_render.exe .\scenes\cornell.lt .\build\svgf_cuda.ppm --cuda --size 64 48 --frames 16 --denoiser svgf --aa stable --spp 1
.\build\Release\lt_render.exe .\scenes\cornell.lt .\build\svgf_cuda_taa.ppm --cuda --size 64 48 --frames 16 --denoiser svgf --aa taa --spp 1
```

编辑器：

- 切换 Raw/Normal/Depth/Variance/History debug view。
- 开关 Rasterized G-Buffer。
- 移动相机，确认 history 不被永久清空。
- 移动物体，确认没有严重拖影。
- 静止 10 秒，确认画面不出现全屏抖动。
- 对比 Off、StablePostAA、TAA 的几何边缘。

## 常见问题

- TAA 只有抖动没有抗锯齿：history 被边缘 validation 持续拒绝，或 jitter 与 G-buffer/path ray 不一致。
- 静止画面仍抖：jitter 在不该启用时仍启用，或 renderer dirty flag 每帧非空。
- 拖影明显：dirty/motion 时 current-frame weight 太低，或 object_id/depth validation 太松。
- 边缘漏色：A-trous 的 normal/depth phi 太松，跨物体混合。
- CPU/CUDA 不一致：AOV 生成、history invalidation、AA mode 或 alpha 参数没有同步。
- Raster G-buffer 锯齿明显：raster sample 和 path traced primary sample 在边缘不一致，需要稳定的 AA resolve 或更完整的 subpixel coverage。
