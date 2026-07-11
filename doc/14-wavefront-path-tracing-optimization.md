# Wavefront Path Tracing 优化笔记

这份文档记录 CUDA wavefront path tracer 这轮迭代里有用的优化、判断依据和 profiling 方法。它不是单纯 changelog，而是把“为什么改”“怎么从 Nsight 数据看出来”“后续遇到类似 profile 应该先看哪里”串起来。

主要相关代码：

- `src/gpu/cuda_path_tracer.cu`：CUDA path tracer 的 host 调度、资源生命周期、kernel launch 顺序。
- `src/gpu/kernel.cuh`：megakernel、wavefront 队列、SVGF 输入、denoise kernel。
- `src/gpu/intersection.cuh`：软件 BVH traversal、compact hit、hit 重建。
- `src/gpu/shading.cuh`：材质、贴图采样、直接光、BSDF sampling。
- `src/gpu/types.cuh`：GPU scene、material、BVH、wavefront path state 布局。
- `src/gpu/scene_upload.cuh`：scene packing、texture object/mipmap 上传、device 资源缓存。

## 1. Megakernel 和 Wavefront 的区别

Megakernel path tracer 的结构通常是：

```text
一次 kernel launch
  -> 生成 camera ray
  -> 在 kernel 内循环 bounce
     -> intersect
     -> shade
     -> direct lighting / shadow
     -> sample BSDF
     -> 更新 ray
  -> 写出 sample
```

优点是调度简单，没有很多中间队列，也没有频繁 kernel launch。缺点是一个大 kernel 同时包含所有路径状态：miss、emissive hit、透明、direct light、shadow、不同 BSDF、Russian roulette 等等。一个 warp 里的线程很容易走不同分支，而且 kernel 需要保留很多分支可能用到的变量，register pressure 会很高。

Wavefront path tracer 把路径追踪拆成一组 work queue：

```text
initialize paths
for each path step:
  active ray queue
    -> intersect kernel
    -> direct light / surface queue
    -> optional GI shortcut queue
    -> shadow ray queue
    -> BSDF sampling queue
    -> next active ray queue
resolve sample
```

Wavefront 的目标不是简单地把 megakernel 拆成 `intersect + shade` 两个 kernel，而是让同一种工作集中执行。这样每个 kernel 的分支更少、寄存器压力更低、profile 也更容易定位。

Blender Cycles 这种成熟 wavefront renderer 的关键启发也是这个：渲染器围绕 work queue 组织，intersection、shading setup、shadow ray、material evaluation、path continuation 是不同阶段。队列计数和路径状态尽量留在 GPU 上，不让 CPU 每个 bounce 都参与调度。

## 2. 第一版 Wavefront 为什么没有变快

早期 profile 暴露了几个问题：

- `wavefront_intersect_kernel` 占主要 GPU kernel 时间。
- `wavefront_shade_kernel` 仍然很大，实际上只是把 megakernel 的 shading 大分支搬到了另一个 kernel。
- 每个 bounce 都把 4 字节 counter 从 device 拷回 host，用来决定下一个 kernel 要不要 launch。

4 字节 D2H copy 本身并不贵，真正贵的是它强制 CPU 等 GPU 前面的工作全部完成。Nsight Systems 里常见现象是：device memcpy 时间很小，但 CUDA runtime 的 `cudaMemcpy` 时间很长，因为它变成了同步点。

第一条原则：

```text
不要用 CPU readback 做 path tracing 内层调度。
```

## 3. Nsight Profile 应该怎么看

不要只看一帧总耗时。至少要分四类看：

1. GPU kernel 时间花在哪里。
2. CPU 是否在 kernel 之间强制同步。
3. 是否有大量 H2D/D2H 拷贝。
4. 当前 profile 是否真的覆盖了想测的路径。

看 SQLite export 时，表名会随 Nsight 版本略有变化，但思路一致：

- 按 kernel 名字聚合 duration。
- 按 CUDA API 名字聚合 runtime duration。
- 按 memcpy 方向和大小统计 copy。
- 看 timeline 上 kernel 和 copy 是否交替串行。

几个高价值问题：

```text
是否有大量 4-byte D2H copy 出现在 bounce loop 里？
cudaMemcpy/cudaMemcpy2DToArray 是否每帧都在发生？
某个 kernel 慢，是因为它做太多工作，还是 launch 次数太多？
editor 路径是否额外跑了 raster AOV 或 CPU upload？
```

要区分：

```text
device memcpy duration：copy engine/GPU 上真实传输时间
CUDA runtime memcpy duration：CPU 调用耗时，包含等待和同步
```

如果 runtime 时间很长但 device copy 时间很短，copy 大概率是在充当同步屏障。

## 4. 已经验证有价值的优化

### 4.1 移除 per-bounce readback

问题结构：

```text
intersect
cudaMemcpy(&shade_count, device counter, 4 bytes, D2H)
if shade_count > 0:
  shade
cudaMemcpy(&active_count, device counter, 4 bytes, D2H)
if active_count == 0:
  break
```

这让 CPU 参与每个 bounce 的控制流。

更合理的结构：

```text
for step in fixed max_path_steps:
  清空本 step 的 device counters
  launch intersect
  无条件 launch direct light / shade stages
  按设置 launch shadow / BSDF stages
  在 GPU 上 promote next queue -> active queue
finalize remaining active paths
```

每个 kernel 自己读取 device-side count，如果 queue 为空就立即 return。空 kernel launch 有成本，但通常比每个 bounce 做 CPU/GPU 同步更便宜。

这和渲染领域 GPU-driven / draw indirect 的思路类似：count 在 GPU 上，后续阶段消费 GPU 上的 count，不 round-trip 回 CPU。这里没有直接用 draw indirect，因为 CUDA path tracer 需要 compute queue 调度，不是 graphics draw call；但 device counter + 固定 staged launch 是同类思想的实用版本。

### 4.2 把 monolithic shade 拆成真实 work queues

Wavefront 只有在 kernel 确实按工作类型变窄时才有价值。

有用的队列：

- Active rays：需要 intersection 的 ray。
- Direct light / surface queue：surface setup、emission、直接光决策。
- GI shortcut queue：lightmap 或 irradiance volume early-out。
- Shadow queue：next-event estimation 的 visibility ray。
- BSDF queue：材质采样和 next ray 生成。
- Next-ray queue：下一 step 的 active ray。

这样 profile 也更清楚：shadow 慢就看 shadow kernel，BVH traversal 慢就看 intersect，material 分支慢就看 BSDF kernel。

### 4.3 降低 queue counter 的 atomic contention

朴素写法是每条 path append 时都对全局 counter 做一次 `atomicAdd`。早期 bounce 全屏 path 都活跃，这会形成 counter contention。

更好的做法是 warp/block aggregation：

```text
warp 内先统计有多少 lane 要 append
leader 用一次 atomicAdd reserve N 个 slot
每个 lane 根据 rank 写入 base + rank
```

这能减少全局 atomic 压力，尤其对 primary bounce 和早期高活跃度 queue 有帮助。

### 4.4 Path state 改成更适合 wavefront 的布局

Wavefront 用更少分支换来更多全局内存读写。每个 stage 都会读写 path state，所以 path state 不能太胖。

有用方向：

- 用 SoA 存热字段：ray、throughput、radiance、RNG、packed state、pixel index。
- intersect 阶段只读 intersect 需要的字段，不读 shading-only 字段。
- 把小状态打包：bounce、transparent steps、transmission count、previous delta。
- 保持 intersect path state 小于完整 shading path state。

核心原则：

```text
不要让 intersect kernel 为 shading 字段付全局内存带宽。
```

### 4.5 调度问题清掉后，重点转向 intersect

当 readback 和调度开销被压下去后，profile 仍然显示 intersect 占主要时间。这时继续拆 shade 不会解决主瓶颈，应该转向软件 BVH traversal。

有价值的方向：

- 用 traversal 专用 compact BVH node layout。
- node 只保留 bounds、child/leaf metadata、primitive range 等 traversal 必需字段。
- traversal 时不要读取 triangle normal、UV、tangent、emission、完整 material。
- 先用 compact triangle 做 candidate intersection，只有 closest hit 确定后再重建完整 `GpuHit`。
- 尽量让 traversal 内层 load 连续、字段少、分支少。

这和 Aila 风格 GPU BVH traversal、Cycles 非 OptiX 分支的思路一致：traversal data 和 shading data 不应该是同一种布局。

### 4.6 SVGF 不应该额外跑一次 primary traversal

把 SVGF 加到 wavefront 后，第一版性能非常差，原因之一是 AOV 通过额外 primary pass 生成，相当于为了 albedo/normal/depth/object-id 又做了一次 primary intersection。

更好的结构：

```text
wavefront first visible primary hit
  -> 写 SVGF AOV
```

first hit 已经有 SVGF 需要的数据：

- albedo
- emission
- normal
- world position
- linear depth
- object id

因此 wavefront + SVGF 应该 inline 写 AOV，而不是再跑 `render_svgf_gbuffer_kernel`。

### 4.7 SVGF 性能爆炸时先看 H2D 和资源生命周期

有一次 profile 显示 97% 帧时间在 host-to-device memory copy，CPU 也长期 100%。这不是 path tracing 算法慢，而是资源生命周期错了。

典型症状：

- `cudaMemcpy2DToArray` 或 `cudaMallocMipmappedArray` 每帧出现。
- texture object / mipmapped array 被重复重建。
- editor 每帧把 CPU/rasterized AOV 上传到 CUDA。
- resolution resize 触发了 full renderer reset，顺带把 scene/texture cache 全丢了。

修正方向：

- 尺寸变化只释放 pixel-sized buffers：accumulation、SVGF buffer、wavefront queue。
- scene/texture cache 在 scene/texture dirty 之前保留。
- CUDA wavefront + SVGF 时，由 wavefront first hit 在 device 写 AOV，跳过 CPU raster G-buffer upload。
- rasterized G-buffer 仍可用于 editor preview 或非 wavefront 路径，但不能强行插入 wavefront steady-state render loop。

原则：

```text
render buffer resize 不应该导致 scene texture re-upload。
```

### 4.8 SVGF albedo 摩尔纹：给 guide albedo 显式 LOD

Path tracing kernel 没有 pixel shader 那种 screen-space derivative。SVGF albedo AOV 如果直接用普通 `tex2D`，高频贴图容易采到过低 mip，debug albedo 会出现摩尔纹。

修正方式是对 SVGF albedo guide 使用显式 LOD：

```text
world_pixel_footprint = 2 * tan(fov / 2) * depth / image_height
uv_density = sqrt(uv_area / world_area)
texture_footprint = world_pixel_footprint * uv_density * texture_size
lod = log2(max(texture_footprint, 1)) + bias
```

这个修正优先只作用在 SVGF guide albedo，不直接改所有 BSDF texture sampling。原因是 guide buffer 更需要稳定和抗 alias；而全局改变材质贴图采样可能改变最终画面语义和噪声特征。

## 5. 不同 Profile 现象对应的判断

### GPU kernel 时间集中在 intersect

优先看：

- BVH node layout 是否 compact。
- triangle traversal data 是否只含 position/edge/material。
- 是否 traversal 阶段提前读取了 shading attributes。
- active queue 是否太大但 rays 完全不 coherent。
- alpha visibility 是否导致 primary traversal 内反复跳过透明面。

### GPU kernel 时间集中在 shade/BSDF

优先看：

- shade kernel 是否仍然包含过多材质分支。
- direct lighting、GI shortcut、BSDF sampling 是否可以继续拆。
- register pressure 是否过高。
- texture fetch 是否集中在某些材质路径。

### CUDA runtime API 时间很高

优先看：

- per-bounce D2H counter readback。
- 同步 `cudaMemcpy`。
- 每帧 `cudaMalloc` / `cudaFree`。
- texture object 或 mipmapped array 是否每帧重建。

### H2D copy 占帧时间大头

优先看：

- scene/texture dirty flag 是否错误。
- resize 是否触发 full reset。
- editor 是否上传 rasterized AOV。
- CPU G-buffer 是否在 CUDA wavefront path 中仍然启用。

### CPU 100% 但 GPU kernel 不算重

优先看：

- CUDA runtime 是否在 busy wait。
- 是否有大量同步 API。
- 是否有 per-frame resource upload。
- editor 主线程是否做了额外 CPU preprocessing。

`cudaDeviceScheduleBlockingSync` 可以降低 CPU spin，但它不是 path tracing 性能优化，只是减少 CPU 等待时的占用。根因仍然要从同步和上传里找。

## 6. 当前实现应该遵守的设计原则

### 用户开关保持稳定

继续保留：

```text
RenderSettings::cuda_wavefront
--cuda-wavefront
--cuda-megakernel
```

优化应尽量留在 CUDA backend 内部，不把临时 profiling 开关暴露给用户。

### CPU 不参与 inner loop

允许：

- 每帧一次 debug stats readback。
- 最终显示/输出 copy。
- scene/texture dirty 后的必要上传。

避免：

- 每 bounce queue counter readback。
- 每 stage 由 CPU 读取 device count 后决定 launch。
- steady-state render 中重复重建 texture object。

### Traversal data 和 Shading data 分开

Traversal 需要：

- bounds
- child/leaf index
- compact triangle position/edge
- alpha visibility 所需的最小 material 信息

Shading 需要：

- normal
- tangent/bitangent
- UV/lightmap UV
- emission
- full material parameters

不要让 traversal 内层为完整 shading 数据付带宽。

### SVGF AOV 是 primary hit 的副产物

Wavefront 路径中：

```text
first visible primary hit writes AOV
not a second traversal
not CPU raster upload
```

这条规则能避免 SVGF 让帧时间成倍膨胀。

## 7. 从 Cycles 非 OptiX 分支可以借鉴什么

不使用 OptiX 时，Cycles 仍然有很多结构值得借鉴：

- queue-oriented path state machine。
- kernel 按工作类型拆分。
- traversal data compact，和 shading data 分离。
- shadow ray、material sampling、path continuation 分阶段。
- 调度计数尽量留在 device。
- profile 每个 stage，而不是只看整帧。

最重要的反面经验：

```text
wavefront 不是 intersect kernel + 一个巨大 shade kernel。
```

如果只是这样拆，往往会比 megakernel 更慢，因为额外 kernel launch 和全局内存队列成本已经付了，但 divergence/register pressure 没有充分下降。

## 8. 下次 Profile 前的检查表

采集前：

1. Release build，确认 `lt_render` 和 `lt_editor` 都更新。
2. 确认当前是 `--cuda-wavefront` 还是 `--cuda-megakernel`。
3. 确认是否启用 SVGF，以及 debug view 是什么。
4. 固定 scene、camera、resolution、frames、samples per pixel。
5. steady-state 性能 profile 前不要改 scene/texture 设置。
6. 捕获足够帧数，区分首帧上传和稳定帧。

分析时：

1. 按 kernel name 聚合 GPU duration。
2. 按 CUDA API name 聚合 runtime duration。
3. 按 memcpy direction 和 size 聚合 copy。
4. 检查 texture upload 是否只在首帧出现。
5. 检查 4-byte D2H copy 是否还在 path step loop 里。
6. 比较 wavefront intersect 与其他 wavefront stage。
7. 比较 SVGF time 与 raw path tracing time。

## 9. 快速诊断表

| Profile 现象 | 优先怀疑 | 优先优化 |
| --- | --- | --- |
| wavefront 比 megakernel 慢，且很多小 D2H | CPU/GPU 同步 | 移除 readback，device counter 调度 |
| `wavefront_intersect_kernel` 占大头 | 软件 BVH traversal | compact node/triangle layout，减少 traversal load |
| shade/BSDF kernel 占大头 | 材质分支和 register pressure | 拆 direct light、GI、shadow、BSDF stage |
| SVGF 后帧时间暴涨 | 额外 traversal 或 AOV upload | first-hit inline AOV，跳过 CPU raster upload |
| H2D copy 占 90%+ | texture/scene 重复上传 | 修 resource cache 和 resize 生命周期 |
| CPU 100% | runtime sync/busy wait | 查同步 API，必要时 blocking sync |
| albedo debug 摩尔纹 | guide albedo 采样过锐 | SVGF albedo 显式 LOD |

## 10. 总结

这轮迭代里真正有用的顺序是：

1. 先用 Nsight 数据定位，不凭体感猜。
2. 移除 per-bounce CPU readback。
3. 把 wavefront 改成真实 work queue，而不是二分 megakernel。
4. 降低 queue append 的 atomic contention。
5. path state 向 SoA/packed state 靠拢，intersect 只读必要字段。
6. 当 intersect 成为主瓶颈后，转向 BVH traversal layout。
7. SVGF AOV 写入合并到 wavefront first hit，避免第二次 primary traversal。
8. resize 时保留 scene/texture cache，避免重复 H2D texture upload。
9. CUDA wavefront + SVGF 跳过 CPU rasterized AOV upload。
10. SVGF albedo guide 使用显式 mip LOD，减少摩尔纹。

更大的原则是：wavefront 只有在减少 divergence、register pressure 或 traversal/shading 内存流量的收益超过额外 launch 和 queue memory traffic 时才会赢。每次改动都应该回到 profile 里验证：kernel 时间、runtime 同步、memcpy、资源生命周期分别对应完全不同的优化方向。

## 11. Sponza `report2` 后续优化记录

固定 workload 为 `scenes/Sponza/Sponza.lt`、RTX 3060 Laptop GPU、`1010 x 789` viewport、1 spp、2 bounces、MIS/NEE、SVGF 5 次 A-Trous和 TAA。`report2`覆盖44个稳定帧，GPU kernel平均约58.43 ms/帧，其中 intersect约40.13 ms、shadow/BSDF约7.10 ms、A-Trous约5.49 ms、surface/direct约3.73 ms。

### 11.1 移除 alpha visibility 导致的空外层 step

旧代码在场景存在 alpha material时使用 `max_bounces * 9 + 12`。Sponza在2 bounces下因此每帧执行30个step，但透明面跳过本来已经在单次 intersect kernel内最多循环8次。profile显示只有step 0和step 1做有效 traversal，后面的launch基本都在读取空queue后返回。

修改后的预算为：

```text
max_path_steps = max_bounces + (has_transmission ? 12 : 0)
```

这保留最多12次delta transmission，不再让alpha material放大外层循环。Sponza的每帧CUDA kernel launch从162次降到22次，intersect从30次降到2次，没有重新引入host counter readback。

### 11.2 将 direct visibility 与 BSDF sampling 分离

旧 `wavefront_shadow_kernel` 同时执行direct-light visibility和BSDF sampling，Nsight Systems报告190 registers/thread。Sponza没有mesh、directional或point light，只有environment，因此这个kernel的主要有效工作其实是BSDF sampling。

修改后surface setup同时生成shadow queue和BSDF queue；direct visibility单独执行并更新RNG/radiance；BSDF kernel在其后读取更新后的RNG以保持随机数顺序；场景没有直接光源时完全跳过direct-visibility launch。Sponza单帧检查中独立BSDF kernel约3.6 ms，通用直接光场景仍保持visibility到BSDF的严格顺序。

### 11.3 Alpha traversal fast path

`GpuTraversalTriangle` 将alpha标志打包进material整数。opaque候选三角形不再加载完整 `GpuMaterial`，也不再为了visibility计算UV；只有标记为alpha的三角形以及解析球体进入 `material_visible_gpu`。closest hit写入shading queue前会恢复普通material index，因此shading数据布局不变。

### 11.4 SVGF temporal直接作为第一轮输入

删除temporal illumination/variance到ping buffer的两次D2D。第一次A-Trous直接读取temporal结果，后续迭代在ping/pong之间交替；0次迭代时resolve直接读取temporal结果。`report2` 中对应的复制合计约12.16 MiB/帧。

### 11.5 Secondary traversal 的 Nsight Compute结论

一次受控的detailed采集得到：79 registers/thread，理论occupancy 50%，实际约37.3%；没有register/local spill；L1/TEX hit约94.8%，L2 hit约83.0%；约35%的global sectors属于非合并访问；long scoreboard和wait是主要warp stall。

因此下一阶段应解决secondary-ray访问一致性，例如有收益门槛的ray reorder或更适合随机traversal的BVH布局；不应仅凭寄存器数量强制 `maxrregcount`。

Nsight Compute的kernel replay会长时间占满GPU。后续自动验证默认只使用正常运行的Nsight Systems或应用内计时；Compute采集必须手工触发、限制到单个kernel，并在机器可承受时进行。

### 11.6 被否决的 CUDA-D3D11输出 interop实验

直接写editor preview texture的实验没有通过性能门槛：`cudaGraphicsMapResources`约9.2 ms/帧，Unmap约10.3 ms/帧，退出时unregister还产生约492 ms阻塞；原始最终RGBA D2H仅约0.5 ms/帧。

因此该实验已回退，继续使用D2H加动态D3D11 texture upload。除非未来改为长期映射的专用共享资源并证明整帧至少提升5%，否则不重新引入。
