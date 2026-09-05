# Wavefront/ReSTIR 性能迭代记录：Sponza 到 Bistro

这份文档记录从 Sponza 约 42 ms/frame 目标优化，到 Bistro 场景 Wavefront Path Tracer 与 ReSTIR DI profiling 的主要过程。重点是保留判断依据：哪些改动已经提交，哪些实验失败并退回，以及当前还卡在哪里。

相关代码：

- `src/gpu/cuda_path_tracer.cu`：wavefront host 调度、kernel launch、ReSTIR 阶段编排。
- `src/gpu/kernel.cuh`：wavefront queue、direct light、direct visibility、BSDF continuation。
- `src/gpu/intersection.cuh`：compact hit、Binary/BVH8/CWBVH traversal、occlusion traversal。
- `src/gpu/shading.cuh`：直接光估计、shadow ray、材质/贴图采样。
- `src/gpu/restir_di.cuh`：ReSTIR DI reservoir、candidate、temporal/spatial reuse、visibility。
- `src/gpu/restir_gi.cuh`：复用 DI 的 triangle light sampling helper。
- `src/gpu/scene_upload.cuh`：GPU scene packing、traversal material flags、BVH8/CWBVH 选择。
- `src/gpu/types.cuh`：GPU-side material、triangle、traversal triangle、reservoir 数据布局。

## 1. 测试条件

主要测试命令形态：

```powershell
.\build\Release\lt_render.exe .\scenes\Sponza\Sponza.lt .\tmp\sponza.png `
  --cuda --cuda-wavefront --sampling nee --denoiser off --aa off `
  --max-bounces 2 --spp 1 --size 1010 789 `
  --frames 10 --benchmark --benchmark-warmup 3 --log-level warn --quiet

.\build\Release\lt_render.exe .\scenes\Bistro_v5_2\BistroInterior_Wine.fbx .\tmp\bistro.png `
  --cuda --cuda-wavefront --sampling nee --denoiser off --aa off `
  --max-bounces 2 --spp 1 --size 1010 789 `
  --frames 10 --benchmark --benchmark-warmup 3 --log-level warn --quiet

.\build\Release\lt_render.exe .\scenes\Bistro_v5_2\BistroInterior_Wine.fbx .\tmp\bistro_restir.png `
  --cuda --cuda-wavefront --sampling nee --denoiser off --aa off `
  --max-bounces 2 --spp 1 --size 1010 789 `
  --frames 10 --benchmark --benchmark-warmup 3 --restir-di --log-level warn --quiet
```

辅助验证：

- `scenes/cornell.lt`
- `scenes/soft_shadow_test.lt`
- Nsight Systems CUDA trace + SQLite export，用 kernel 名称聚合 GPU duration。

注意：Bistro 上同一版本的 CLI `render_ms` 有明显档位波动，尤其长时间连续编译/跑测后更明显。因此单次低点不作为唯一依据，最终判断同时参考 kernel profile 是否切到预期 traversal layout。

## 2. 起点和目标

Sponza 起点截图约为：

- Renderer：CUDA Wavefront Path Tracer
- `max_bounces=2`
- 约 `42 ms/frame`

优化目标是先让 Sponza 接近 `20-25 ms/frame`，同时不能靠只服务 Sponza 的特化逻辑。

Bistro 起点截图约为：

- Renderer：CUDA Wavefront Path Tracer
- `max_bounces=2`
- 约 `1241 ms/frame`

后续 Bistro 重点同时看：

- 普通 wavefront path tracer 的二跳成本。
- ReSTIR DI 的 initial candidates、final visibility、temporal/spatial reuse 成本。
- 是否存在大场景 traversal layout 回退。

## 3. 已提交优化

### 3.1 `08d72b5 perf: bypass ReSTIR DI for environment-only scenes`

问题：Sponza 这类只有环境光、没有非环境 direct lights 的场景，开启 ReSTIR DI 时会走一套没有实际收益的 reservoir 流程。

改动：没有非环境 direct lights 时，ReSTIR DI 自动 disabled，回到 wavefront direct lighting。

结果：

- Sponza ReSTIR DI fallback 正常，最终稳定在约 `23 ms/frame`。
- 日志会输出 warning，说明 ReSTIR DI 被 disabled。

结论：保留。这个不是 Sponza 特化，而是 ReSTIR DI 适用域判断。

### 3.2 `f54809d perf: trim wavefront transmission traversal`

问题：wavefront direct shadow/transmission 允许过多透明步数，大量阴影路径在复杂场景里消耗 traversal。

改动：

- wavefront direct light transparent shadow steps 收紧到较小预算。
- 额外 transmission bounce 预算保留，但避免普通 opaque 路径无意义延长。

结论：保留。对 alpha/transmission 场景要继续关注质量边界，但这类预算是通用路径成本控制。

### 3.3 `658f1ea perf: use compact traversal for wavefront direct shadows`

问题：direct shadow 只需要知道可见性，旧路径会走较重的 full hit。

改动：wavefront direct shadow 改走 compact hit/compact traversal。

结论：保留。这是 wavefront 阴影路径的基础优化。

### 3.4 `b26ce1a perf: avoid texture checks for opaque shadow blockers`

问题：普通 opaque blocker 不需要读取材质纹理/alpha 来判断遮挡。

改动：shadow blocker 可快速判定时，避免进入 texture visibility 检查。

结论：保留。减少阴影 traversal 中最常见 blocker 的材质读取。

### 3.5 `60ef183 perf: skip wavefront direct shadows for delta materials`

问题：Mirror 和足够 smooth 的 Conductor 属于 delta-like surface，显式 direct shadow NEE 对它们通常没有贡献。

改动：wavefront direct light 阶段跳过 delta 材质的 direct shadow enqueue。

结论：保留。减少无贡献阴影队列，同时不影响 delta BSDF continuation。

### 3.6 `bbd7c77 perf: skip ReSTIR initial visibility for basic bias`

问题：Basic bias correction 下，initial visibility trace 成本高，但不是该 bias 模式必须做的工作。

改动：只有 RayTraced bias correction 才执行 ReSTIR initial visibility；Basic 模式跳过。

结果：Bistro ReSTIR DI 曾测到约 `1038 ms/frame`，比更早的 `1241 ms/frame` 降低。

结论：保留。符合 bias mode 语义。

### 3.7 `c2503b8 perf: tag opaque shadow blockers for traversal`

问题：compact shadow traversal 命中三角形后仍要解 material index，再读 material 来判断是否 opaque blocker。

改动：

- 新增 traversal material flag：opaque shadow blocker。
- CWBVH/BVH compact hit 可直接从 packed material flags 判断普通 opaque blocker。
- ReSTIR visibility trace 也复用该 flag。

结果：Bistro 上收益较小且有噪声，但语义清楚、风险低。

结论：保留。

### 3.8 `eb2687a perf: increase wavefront intersect block size`

问题：Bistro traversal-heavy，原 `intersect_block_size=64` 对大规模 wavefront intersect 不够理想。

实验：

- 128 block：Bistro wavefront 约 `524 ms/frame`。
- 256 block：Bistro wavefront 约 `519 ms/frame`，ReSTIR DI 约 `959 ms/frame`。
- Sponza 仍在约 `23 ms/frame`。

改动：`intersect_block_size` 提到 `256`。

结论：保留。对 traversal-heavy 大场景有明显收益，小场景没有灾难性退化。

### 3.9 `c0fa9d3 perf: reuse sampled ReSTIR triangle lights`

问题：ReSTIR triangle proposal 已经采样并读出 `GpuTriangle`，后续 evaluation/pdf 又重复加载同一个 triangle。

改动：

- `restir_sample_triangle_gpu()` 增加输出 `GpuTriangle& light` 的 overload。
- DI initial candidates 与 GI secondary direct candidates 复用 sampled light/material。
- 保留旧 API 给 ReSTIR PT/其它调用点。

结果：

- Bistro ReSTIR DI 约 `949 ms/frame`。
- Bistro wavefront 约 `517 ms/frame`。
- Cornell ReSTIR GI smoke 正常。
- 同帧对比中图像可 bit-identical。

结论：保留。减少重复 global memory 读取。

### 3.10 `cdf90af perf: separate mask alpha traversal flags`

问题：之前 `has_alpha` 同时覆盖 Mask 和 Blend。BVH traversal 内 alpha test 只应对 Mask 做确定性过滤，Blend 的随机透明不适合在 closest-hit traversal 内处理。

改动：

- 新增 `kTraversalMaterialMaskAlphaBit`。
- CWBVH/BVH compact traversal 的 in-traversal alpha test 只看 Mask flag。
- 保留 general alpha flag 给 hit 后 visibility 逻辑。

结果：

- Bistro wavefront 约 `510 ms/frame`。
- Bistro ReSTIR DI 约 `948 ms/frame`。
- Sponza 约 `23 ms/frame`。

结论：保留。减少不必要材质读取，也让 Mask/Blend 语义更清晰。

### 3.11 `83e2462 perf: isolate ReSTIR visibility RNG`

问题：ReSTIR final/environment visibility trace 在 alpha/transparent visibility 路径中使用 `path.rng`，会推进主 path RNG。这样 ReSTIR DI 打开后，后续 BSDF continuation 的随机序列会改变，容易影响和普通 wavefront 的收敛对比。

改动：

- ReSTIR visibility trace 使用从 `path.rng`、pixel、sample type、sample index 派生的局部 visibility RNG。
- 不再把 visibility alpha test 的随机消费写回主 path RNG。

结果：

- 对性能没有稳定大收益，Bistro 上有过 `646 ms/frame` 低点，但重跑会回到更高档位。
- 价值主要在采样维度隔离：ReSTIR visibility 不再扰动主 BSDF stream。

结论：保留。它更像正确性/收敛质量修正，而不是稳定性能优化。

### 3.12 `3230e8a perf: allow wide traversal for larger scenes`

问题：Bistro 的 CWBVH/BVH8 估算内存超过原 `64 MB` budget，导致 wavefront 回退到 Binary traversal。Nsight kernel 名显示 `GpuTraversalLayout)0`，说明实际跑的是 Binary，不是 CWBVH。

改动：

- BVH8/CWBVH wavefront estimated upload budget 从 `64 MB` 提到 `256 MB`。
- 更新注释，说明大场景应该在保守预算内优先使用 CWBVH。

结果：

- 实验中 Bistro wavefront 从约 `775 ms/frame` 档位降到约 `419 ms/frame`。
- Bistro ReSTIR DI 从约 `900-940 ms/frame` 档位降到约 `566 ms/frame`。
- Nsight profile 确认 kernel 切到 `GpuTraversalLayout)2`，即 CWBVH。
- CWBVH 后 `restir_trace_visibility` 从约 `157 ms/call` 降到约 `70-89 ms/call`。
- 非 primary `wavefront_intersect` 从约 `114 ms/call` 降到约 `50-66 ms/call`。

结论：保留。这是 Bistro 阶段最大的一次通用收益。它不是按场景名特化，而是让大生产场景也能使用已经实现的 compact wide traversal。

## 4. 失败或未提交实验

### 4.1 降低 `kRestirInitialCandidates` 到 4

实验结果：

- Bistro ReSTIR DI default 曾到约 `991 ms/frame`。
- 配合 final visibility reuse 曾到约 `954 ms/frame`。

问题：这是直接减少候选数量，会改变 ReSTIR DI 的质量/收敛预算。用户已经反馈收敛速度不如原 wavefront，因此不能用这种方式换时间。

结论：退回，不提交。

### 4.2 把 transmission extra bounce 硬降到 1

结果：对 Bistro 有潜在收益。

问题：会影响玻璃/透明材质的 enter-exit correctness，换场景容易破坏。

结论：退回，不提交。

### 4.3 缓存/复用 `material_emission_gpu()`

实验：在 direct lighting 和 ReSTIR evaluation 中尝试缓存 light/material emission。

结果：Bistro wavefront 变慢，约回到 `838-856 ms/frame` 档位。

推测原因：少算了一些 texture/emission evaluation，但增加了局部变量和寄存器压力，重 kernel 反而更慢。

结论：退回，不提交。

### 4.4 融合 ReSTIR final visibility generate/trace/resolve

实验：写过 fused final visibility kernel，尝试减少 generate/trace/resolve 三段 dispatch 和中间 buffer。

结果：Bistro ReSTIR DI 约 `1029 ms/frame`，比当前路径更差。

推测原因：fused kernel 增加控制流和寄存器压力，丢失了当前 trace kernel 较窄的工作形态。

结论：退回，不提交。

### 4.5 direct visibility block size 从 256 改 128

动机：`wavefront_direct_visibility_kernel` 使用约 176 registers/thread，尝试更小 block 提高 occupancy。

结果：Bistro wavefront 从约 `516 ms/frame` 变到约 `530 ms/frame`。

结论：退回，不提交。

### 4.6 NEE/MIS runtime branch 跳过 BSDF PDF

动机：`--sampling nee` 下 mesh-light direct estimate 不需要 MIS weight，因此理论上不必计算 `material_pdf_gpu()`。

结果：普通 wavefront 略慢，ReSTIR DI 也略慢。

推测原因：runtime branch 没有让编译器真正瘦身，反而增加控制流/寄存器压力。

结论：退回，不提交。

### 4.7 NEE/MIS 编译期专门化 direct visibility

动机：把 `UseMis` 变成模板参数，让 NEE 版本完全不编译 `material_pdf_gpu()` 路径。

结果：

- `wavefront_direct_visibility_kernel` registers 只从 `176` 降到 `174`。
- Bistro wavefront 仍在高档位，未见收益。

结论：退回，不提交。

### 4.8 triangle contribution + proposal PDF 合并 helper

动机：ReSTIR initial candidate 对 triangle light 先 evaluation，再 proposal PDF，重复计算 direction/distance/ldot/area。

结果：Bistro ReSTIR DI 从约 `566 ms/frame` 变到约 `574 ms/frame`。

推测原因：helper 多返回 proposal PDF 并缓存 emission，导致寄存器压力上升，抵消了少量 arithmetic 节省。

结论：退回，不提交。

### 4.9 ReSTIR initial candidates block size 从 256 改 128

动机：`restir_initial_candidates_kernel` 约 100 registers/thread，尝试更小 block。

结果：Bistro ReSTIR DI 约 `658 ms/frame`，明显慢于 256 block 的约 `566 ms/frame`。

结论：退回，不提交。

### 4.10 CWBVH/BVH8 budget 512 MB

动机：确认 Bistro 回退 Binary 是否来自 64 MB 门限。

结果：

- 512 MB 时 Bistro 成功切到 CWBVH。
- wavefront 约 `421 ms/frame`，ReSTIR DI 约 `569 ms/frame`。

后续：256 MB 仍能覆盖 Bistro，且更保守。

结论：最终提交 256 MB，不提交 512 MB。

## 5. 当前性能状态

截至 `3230e8a` 后，较可信的当前状态：

- Sponza：约 `23 ms/frame`，满足 20-25 ms 目标区间。
- Bistro wavefront：最好测到约 `419 ms/frame`；干净重建后也测到过约 `506 ms/frame`。
- Bistro ReSTIR DI：CWBVH 生效时约 `566 ms/frame`；也出现过约 `900 ms/frame` 的 CLI 档位波动，但 Nsight profile 确认 traversal layout 已是 CWBVH。
- Bistro ReSTIR DI + final visibility reuse：约 `549 ms/frame`，只再省约 15-20 ms。

Profile 中最重要的确认点：

```text
Binary 回退时：
  wavefront_intersect non-primary: ~114 ms/call
  restir_trace_visibility:         ~157 ms/call

CWBVH 生效时：
  wavefront_intersect non-primary: ~50-66 ms/call
  restir_trace_visibility:         ~70-89 ms/call
```

这说明 Bistro 当前最大的已知改善来自 traversal backend，而不是 ReSTIR reservoir arithmetic。

## 6. 当前瓶颈

### 6.1 大量二次路径相交

`max_bounces=2` 下，Bistro 的一跳和两跳差距很大：

- Bistro wavefront `max_bounces=1`：约 `159 ms/frame`。
- Bistro wavefront `max_bounces=2`：约 `500 ms/frame` 档位。
- Bistro ReSTIR DI `max_bounces=1`：约 `270 ms/frame`。
- Bistro ReSTIR DI `max_bounces=2`：约 `560-900 ms/frame` 档位，取决于运行状态和 traversal 是否确认为 CWBVH。

这说明第二 bounce 的 intersect + direct visibility 是主要成本。

### 6.2 direct visibility kernel 仍然很重

CWBVH 生效后，`wavefront_direct_visibility_kernel` 仍然是最重 kernel 之一：

- registers/thread 约 `191`。
- 每次调用约 `55-63 ms`，复杂运行中更高。

它包含直接光采样、BSDF evaluation、shadow traversal、透明/alpha blocker 处理等，寄存器压力很高。

### 6.3 ReSTIR initial candidates 固定成本高

`restir_initial_candidates_kernel` 约 `100 regs/thread`，Bistro 上约 `104-105 ms/frame`。

当前实现大致是：

- 有 mesh lights 时，初始 local light 候选补足到 `kRestirInitialCandidates=8`。
- Bistro 的主要成本来自每个 primary surface 评估多个 mesh-light candidates。

不能简单减少 candidate 数量，因为这会伤害收敛质量。

### 6.4 ReSTIR final visibility 仍需追踪大量 shadow rays

CWBVH 后 `restir_trace_visibility_kernel` 仍约 `70-89 ms/frame`。final visibility reuse 只能小幅降低总帧时，说明大部分成本仍来自必要的最终可见性。

### 6.5 仍是软件 traversal，不是硬件 RT

和 Falcor 的实时差距，很可能不只是 ReSTIR 算法层面，而是 traversal backend 和整体渲染架构差距：

- 当前是 CUDA software BVH traversal。
- Falcor 常见路径基于 DXR/RTX hardware RT cores。
- Bistro 这种高复杂度遮挡场景里，shadow ray traversal 占比极高，硬件 RT 的优势会被放大。

## 7. 后续可能优化方向

### 7.1 更彻底的 shadow ray occlusion traversal

当前 CWBVH occlusion traversal 已经比 Binary 快很多，但仍可继续看：

- 对 finite light shadow ray 使用剩余距离裁剪 CWBVH child test。
- 保证 CWBVH occlusion traversal 对 target triangle、alpha/mask、transparent step 的语义和 full path 一致。
- 尽量避免为了透明场景让全场 visibility 都走重模板。

风险：shadow ray 最容易引入漏光/错误遮挡，需要用 alpha、glass、mesh light、point light 场景做图像回归。

### 7.2 ReSTIR initial candidates 降寄存器，而不是降候选数

不要默认降低 `kRestirInitialCandidates`。更合理方向：

- 把 triangle light candidate evaluation 拆成更窄的 helper，但必须检查 register count。
- 预计算 mesh light area、emission luminance、sidedness 等 light sampling 数据，避免 per-candidate 重算。
- 考虑额外 GPU light record，专门服务 light sampling/evaluation，不直接读完整 `GpuTriangle`。

风险：增加 light record 会增加上传和显存，但可能显著降低 initial candidate kernel 的 arithmetic 和 memory footprint。

### 7.3 拆分 direct visibility 的材质/光源路径

`wavefront_direct_visibility_kernel` 过重。可以考虑：

- 为 mesh-light、directional、point light 分离更窄的路径。
- 对没有 point/directional light 的场景编译期专门化，减少循环和变量。
- 对 opaque-only 或 mask-only visibility 进一步选择更窄模板。

风险：模板组合会膨胀编译时间和二进制体积。需要用 Nsight register count 和 benchmark 共同决定。

### 7.4 更准确的 active queue launch

当前很多 kernel 仍按全分辨率 grid launch，再在 device counter 上早退。后续路径活跃数变少时，这会浪费 launch 内线程。

可能方向：

- CUDA cooperative groups 或 indirect-like 调度。
- 使用 persistent kernel/work queue，让 GPU 内部持续消费队列。
- 只在高成本阶段做 device-side compaction，避免 CPU readback。

风险：调度架构改动大，容易影响整个 wavefront pipeline。

### 7.5 考虑硬件 RT 后端

如果目标是向 Falcor Bistro 实时靠近，长期需要评估 DXR/OptiX/RTX traversal 后端，而不是只靠 CUDA software BVH 微调。

建议路线：

- 保留当前 CUDA software traversal 作为可移植 baseline。
- 新增可选 OptiX 或 DXR backend，仅替换 intersect/occlusion traversal。
- ReSTIR DI/GI/PT reservoir pipeline 继续复用现有 CUDA/compute 逻辑，逐步替换最重 shadow/intersect 部分。

### 7.6 建立固定 benchmark protocol

Bistro 数据波动较大，后续需要更稳定的 benchmark 规则：

- 固定分辨率、相机、`max_bounces`、`spp`、AA、denoiser。
- 每次改动至少跑普通 wavefront、ReSTIR DI、Sponza、一个小场景。
- Nsight profile 必须记录 traversal layout，避免误把 Binary/CWBVH 混在一起比较。
- 记录 GPU 温度/功耗/后台负载，避免把时钟档位波动当作代码收益。

## 8. 当前建议优先级

1. 先优化 shadow/intersect traversal，而不是继续拆 kernel 表面结构。
2. 保持 ReSTIR candidate 数量，优先降低每个 candidate 的数据读取和寄存器压力。
3. 给 direct visibility 做更细的场景能力专门化，但每一步都要看 register count。
4. 如果 Bistro 目标是实时级，开始评估硬件 RT 后端；纯 CUDA software BVH 很可能无法靠微优化追到 Falcor。

## 9. 2026-09-02 继续迭代记录

### 9.1 CWBVH budget 和 layout 可观测性

新增 host-side traversal 预算/估算 helper：

- `estimate_wavefront_bvh8_bytes()`
- `estimate_wavefront_cwbvh_bytes()`
- `wavefront_wide_traversal_budget_bytes()`
- `bytes_to_mib()`

预算不再是单一硬编码 `256 MiB`：默认仍为 `256 MiB`，但会按首次查询时 CUDA free memory 的 `1/4` 放宽，最高 `512 MiB`。这样可以避免大场景只因为略微超过固定阈值就从 CWBVH 退回 Binary。

full scene upload 后新增一次性 info log，记录：

- 实际 traversal layout。
- binary BVH 节点数。
- BVH8/CWBVH 节点数。
- CWBVH triangle payload 数量。
- BVH8/CWBVH 估算 MiB、CWBVH 实际 MiB、当前 budget MiB。

Bistro 当前观测：

```text
layout=CWBVH
bvh_nodes=835415
bvh8_nodes=126468
cwbvh_nodes=199443
cwbvh_tri_float4=3960963
bvh8_estimated_mib=106.76
cwbvh_estimated_mib=92.3081
cwbvh_actual_mib=75.6558
budget_mib=512
```

此外 render launch 侧不再只按估算决定 `use_wide_bvh/use_cwbvh`，还会检查 upload 后的 cached node/payload count，避免 CWBVH 构建失败但仍选择 CWBVH 模板路径。

### 9.2 Profiling 更新

Nsight Compute 在当前机器上无法采集硬件计数器：

```text
Profiling failed because a driver resource was unavailable.
Failed to create counter availability image (error = 20)
Failed to get counter availability image (error = ResourceUnavailable)
```

改用 Nsight Systems 做 kernel summary。

Bistro ReSTIR DI，`1010x789`、`spp=1`、`max_bounces=2`、denoiser/AA off、4 frames：

```text
wavefront_direct_visibility_kernel    29.1%
wavefront_intersect_kernel            28.5%
restir_initial_candidates_kernel      18.3%
restir_trace_visibility_kernel        11.8%
```

Bistro 普通 wavefront，同参数：

```text
wavefront_direct_visibility_kernel    48.8%
wavefront_intersect_kernel            47.4%
wavefront_direct_light_kernel          3.2%
```

结论没有改变：Bistro 的核心瓶颈仍是 software traversal 下的 intersect + shadow visibility。ReSTIR DI 的额外成本主要来自 initial candidates 和 final visibility trace，但它们叠在同一个 traversal 天花板上。

### 9.3 本轮失败尝试：CWBVH fast occlusion shadow path

尝试方向：

- 给 `GpuScene` 增加全场 direct-shadow opaque-only 标志。
- 给 CWBVH occlusion 增加 `max_t` 和 `target_triangle`，用于 finite light shadow ray。
- 当场景不需要 alpha/transmission shadow pass 时，让 `direct_shadow_blocked_gpu()` 走距离限制的 occlusion traversal，而不是完整 closest-hit + material-aware traversal。

结果：

- Bistro 的 `opaque_shadow_only=0`，说明该场景存在 alpha/transparent shadow 可能性，fast path 被正确禁用。
- 即使 runtime 未启用，该分支仍被编进 CWBVH direct-shadow 模板，导致 kernel 体积/寄存器压力恶化。
- Bistro 普通 wavefront 从约 `445 ms` 波动到约 `2030 ms`。

处理：

- 已回退该 fast occlusion 实验。
- 保留教训：这个方向不能直接塞进已有 direct visibility 模板。若继续做，应该拆成独立 kernel/template，或只为经确认的 opaque-only 场景编译/launch 更窄路径，避免污染 Bistro 这种混合材质场景。

### 9.4 本轮有效 benchmark

候选版本保留动态 budget、layout log、实际 cached traversal launch guard，回退 fast occlusion 后：

```text
Bistro wavefront:
frames=10 warmup=4 measured=6
mean=444.750 ms
median=442.758 ms
p95=452.370 ms

Bistro ReSTIR DI:
frames=8 warmup=3 measured=5
mean=822.250 ms
median=820.279 ms
p95=846.285 ms
```

ReSTIR DI 本轮数值比早前 `~575 ms` 档更慢，但同一代码路径下普通 wavefront 稳定回到 `~445 ms`，说明需要后续继续用 Nsight Systems/固定 benchmark protocol 排除 GPU 时钟、后台负载和历史状态波动。
