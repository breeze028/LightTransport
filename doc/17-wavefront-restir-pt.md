# CUDA Wavefront ReSTIR PT：GRIS、RTXDI 与 Enhanced 实现对照

ReSTIR PT 的理论来源是 Lin 等人的 GRIS 论文 *Generalized Resampled Importance Sampling: Foundations of ReSTIR*。论文不只提出一个 path tracer，而是给出在不同积分域、相关样本和未知边缘 PDF 下进行 reservoir resampling 的统一框架，并以全路径重用作为主要应用。RTXDI 3.0 在此基础上提供面向实时渲染的 ReSTIR PT pipeline。

本文按 GRIS 原文章节从头推导，再说明 RTXDI 的 hybrid shift 与本项目 CUDA wavefront 实现。内容为详细转述，不是逐字翻译。

## 1. 范围

当前实现：

- 只作用于 CUDA wavefront path tracer；默认关闭。
- ReSTIR PT 接管主表面后的非 delta 间接路径，主表面直接光仍由现有 NEE/MIS 或 ReSTIR DI 负责。
- GI 与 PT 互斥，外部同时开启时 PT 优先；DI 可与 PT 同时开启。
- 支持 `None`、`Temporal`、`Temporal + Spatial`。
- path depth 为 2–8，默认 3；reconnection 可选 `FixedThreshold` 或 `Footprint`。
- CPU、CUDA megakernel、`UNI`、lightmap 和 irradiance volume 保持原行为。

主要源码：

- `src/gpu/restir_pt.cuh`
- `src/gpu/restir_di.cuh`
- `src/gpu/cuda_path_tracer.cu`
- `src/gpu/types.cuh`
- `include/lt/renderer.h`

参考资料：

- Lin et al., *Generalized Resampled Importance Sampling: Foundations of ReSTIR*, SIGGRAPH 2022。
- Lin et al., *ReSTIR PT Enhanced: Algorithmic Advances for Faster and More Robust ReSTIR Path Tracing*, I3D 2026。
- NVIDIA RTXDI: [ReSTIR PT](https://github.com/NVIDIA-RTX/RTXDI/blob/main/Doc/RestirPT.md) 与 [Integration Guide](https://github.com/NVIDIA-RTX/RTXDI/blob/main/Doc/Integration.md)。

## 2. Sections 1–2：为什么需要 GRIS

普通 ReSTIR DI 假设候选最终能在同一个 light sample domain 中比较。全路径复用更难：

- 不同像素的路径从不同 camera/receiver 出发，天然属于不同路径子空间。
- 路径可能由 BSDF sampling、light sampling、Russian roulette、alpha 和 lobe 选择共同生成。
- 样本经过 temporal/spatial reuse 后具有相关性，边缘 PDF 很难显式写出。
- 把一条路径搬到另一像素需要修改路径顶点，概率密度随映射改变。

GRIS 的目标不是消除这些复杂性，而是说明：只要每个输入 sample 携带一个合法的 contribution weight，并且 shift mapping、Jacobian 与 MIS support 正确，就仍可构造一致甚至无偏的重采样估计。

论文 Section 2 回顾了 reservoir sampling、RIS、MIS 和路径空间。这里最关键的概念不是“sample 的 PDF 一定已知”，而是 contribution weight 的期望满足正确积分。

## 3. Section 3：RIS 回顾与局限

### 3.1 Section 3.1：同分布样本

设目标积分为

$$
I=\int_{\Omega}f(x)\,\mathrm dx.
$$

从 proposal $p(x)$ 生成 $M$ 个候选，RIS target 为 $\widehat p(x)$：

$$
w_i=\frac{1}{M}\frac{\widehat p(X_i)}{p(X_i)}.
$$

按 $w_i/\sum_jw_j$ 选择 $Y$ 后，sample contribution weight 为

$$
W_Y=\frac{\sum_iw_i}{\widehat p(Y)},
$$

估计量为

$$
\widehat I=f(Y)W_Y.
$$

这里的 $1/M$ 可以放在每个输入权重里，也可以在 finalize 时统一除；实现必须只做一次。

### 3.2 Section 3.2：不同分布样本

若 $X_i\sim p_i$，引入 partition/MIS weight $m_i(x)$，满足

$$
\sum_i m_i(x)=1
$$

（在有效 support 上），则

$$
w_i=m_i(X_i)\frac{\widehat p(X_i)}{p_i(X_i)}.
$$

这解释了 ReSTIR bias correction 的本质：不同 source reservoir 不能只凭 target 大小合并，还要处理它们的 proposal/support 差异。

### 3.3 Section 3.3：为什么还要推广

全路径样本往往没有一个廉价、显式的 $1/p_i(X_i)$。例如历史 reservoir 已经过多轮相关重采样；random replay 还会改变路径，但无法方便地求最终边缘 PDF。GRIS 因而把“已知 reciprocal PDF”推广为“携带一个无偏 contribution weight”。

## 4. Section 4：Generalized RIS

### 4.1 Sections 4.1–4.2：无偏 contribution weight

若随机样本 $X$ 携带权重 $W_X$，并满足对任意可积函数 $f$：

$$
\mathbb E[f(X)W_X]
=
\int_{\operatorname{supp}(X)}f(x)\,\mathrm dx,
$$

则 $W_X$ 是一个合法的 contribution weight。普通 importance sampling 的 $W_X=1/p(X)$ 只是特例；一个已 finalize 的 reservoir weight 也可以是这种权重。

GRIS 对输入 $(X_i,W_i)$ 的重采样权重写成

$$
w_i=m_i(X_i)\widehat p(X_i)W_i.
$$

选择 $Y$ 后仍使用

$$
W_Y=\frac{\sum_iw_i}{\widehat p(Y)}.
$$

只要 $m_i$ 构成合法 partition，并满足 support 条件，$f(Y)W_Y$ 就保持正确期望。这是“reservoir 可以再次作为 weighted sample 输入下一轮 reservoir”的理论基础。

### 4.2 Section 4.3：Shift Mapping

不同像素的输入可能位于不同域 $\Omega_i$。定义可逆 shift

$$
T_i:\Omega_i\rightarrow\Omega,
\qquad
Y_i=T_i(X_i),
$$

把候选搬到 canonical domain。变量替换要求乘 Jacobian determinant：

$$
J_i(X_i)
=
\left|
\det\frac{\partial T_i(X_i)}{\partial X_i}
\right|.
$$

对应 GRIS weight 为

$$
w_i
=
m_i(Y_i)
\widehat p(Y_i)
W_i
J_i(X_i).
$$

如果实现保存的是 inverse mapping 或 reciprocal contribution weight，Jacobian 可能表现为除法；必须从变量变换方向推导，不能靠经验交换分子分母。

shift 不只是几何地移动一个点。它必须定义：哪些随机决策重放、哪些顶点重连、BSDF lobe 是否一致、visibility 是否成立、路径长度是否改变，以及映射是否在该 sample 上可逆。

### 4.3 Section 4.4：渐近理想重要性采样

论文证明，在合理条件下，候选数增长时 GRIS 输出分布趋向 target $\widehat p$。这解释了 reservoir 重用为何能逐步集中到高贡献路径。但渐近结论不保证有限样本下低方差，也不允许无限保留旧历史：若当前帧的新路径几乎没有机会替换巨大历史 reservoir，算法会停止探索变化后的路径空间。

## 5. Section 5：收敛、方差与奇异性

Section 5 是理论章节，主要回答“相关、shifted、weighted samples 何时仍可信”。工程上需要记住以下结论。

### 5.1 Reasonable distributions

proposal 必须覆盖被积函数的有效 support。若 canonical 当前帧路径能够贡献，但所有 reused proposal 都无法映射到它，MIS partition 必须给 canonical sample 非零权重，否则无法保证收敛。

### 5.2 Asymptotic 与 finite-case variance

target 越接近 $|f|$，渐近方差越低；但有限候选下，极端 contribution weight、近奇异 Jacobian 和弱 support overlap 会产生高方差。一个很亮的 reservoir 样本不一定是“随机数坏”，也可能是 shift density 比值过大。

### 5.3 Avoiding singularities

几何重连在以下位置容易出现奇异或近奇异 Jacobian：

- receiver 与 reconnection vertex 距离接近零；
- cosine 接近零的掠射连接；
- delta/specular 顶点没有普通面积/立体角密度；
- shift 后 lobe 或 visibility support 消失。

这些情况应拒绝 shift，而不是把一个巨大有限数硬塞进 reservoir。clamp 可以稳定画面，但它改变 estimator，必须作为显式偏差记录。

### 5.4 Canonical samples

每轮都保留当前像素新生成的 canonical path，使当前积分域始终有 proposal coverage。它是历史失效、disocclusion 和动态场景下恢复正确结果的锚点。若 zero-contribution canonical path 被从 $M$ 中删除，历史会系统性占优并造成亮偏差。

### 5.5 Robust MIS weights

理想 MIS 要评价所有 source 在 selected shifted path 上的密度，成本过高。论文讨论了更稳健的权重设计，尤其避免只依赖 source 自己的高置信权重。RTXDI/本项目采用 pairwise 形式，让 canonical distribution 与每个 reused source 成对比较，用 `Pi/PiSum` 近似多分布 normalization。

### 5.6 Guaranteeing convergence

收敛依赖持续注入新 canonical samples、正确 support、有限 contribution weight，以及不过度相关的历史。`M` cap、age cap 与 history reset 不是纯性能参数，而是维持路径空间探索的组成部分。

## 6. Section 6：在图像上摊销为 ReSTIR

### 6.1 Section 6.1：每像素积分

每个像素 $q$ 有自己的路径积分

$$
I_q=\int_{\Omega_q}f_q(\bar x)\,\mathrm d\mu(\bar x),
$$

其中 $\bar x=(x_0,x_1,\ldots,x_k)$ 是路径。temporal/spatial reuse 从其他像素/帧取得 weighted path sample，通过 shift $T_{r\rightarrow q}$ 搬到 $\Omega_q$。

### 6.2 Section 6.2：Weighted GRIS reservoir

reservoir 不只保存一条路径，还要保存足以重建 shift 与 contribution weight 的状态：replay RNG seed/dimension、path length、reconnection vertex、部分 Jacobian、sample identity、target、$M$ 和 age。

### 6.3 Section 6.3：Chained GRIS

initial、temporal、spatial 每一阶段都把上阶段 reservoir 当 weighted input，再输出新的 reservoir。每一环都必须 finalize 成合法 contribution weight；不能把尚未 normalization 的 `weightSum` 当作下一阶段 $W_i$。

### 6.4 Section 6.4：M-capping 与探索

历史 reservoir 的 $M$ 若无限增长，当前帧 $M=1$ canonical path 被选中的概率会趋近零。论文把这种现象视为 path-space exploration 问题。cap 可将历史有效 $M$ 限制为当前预算的若干倍：

$$
M_{\mathrm{history}}
\leftarrow
\min(M_{\mathrm{history}},M_{\max}).
$$

缩放 $M$ 时必须同步缩放代表累计候选质量的权重，保持单候选平均权重不变。

### 6.5 Section 6.5：离线渲染

GRIS 并不局限于实时渲染。更多候选、更严格 shift、双向路径技术和多轮迭代也可用于离线场景。RTXDI 的固定小预算只是该理论的一种实时配置。

## 7. Section 7：Shift Mapping 设计

### 7.1 Section 7.1：shift 的责任

对 source path $\bar x_r$ 和 canonical receiver $q$，shift 要生成 $\bar x_q=T_{r\rightarrow q}(\bar x_r)$，并返回：

- shifted path contribution/target；
- forward Jacobian；
- inverse shift 是否存在；
- source/target visibility；
- lobe、path length 与 sample identity 是否兼容。

### 7.2 Section 7.2：常见构件

路径 shift 常由以下操作组成：重放相同随机数、重连顶点、复制 suffix、重新采样离散 lobe、验证 deterministic delta chain，以及转换 area/solid-angle measure。

### 7.3 Section 7.3：Full Shift

完整 shift 会对整条路径构造可逆映射并累计所有 Jacobian，理论最干净，但每个 reused candidate 几乎要重做一次 path tracing，实时成本很高。

### 7.4 Section 7.4：实时优化 shift

三种主要策略：

**Random replay**：在新 receiver 使用 source 保存的随机数重新执行 BSDF、lobe、RR 等决策。它能穿过非 connectable 前缀，但新旧路径可能迅速分叉，成本接近重新追踪。

**Reconnection**：直接把新 receiver 或重放前缀末端连接到 source path 的某个顶点，再复用后缀。它便宜，但 reconnection vertex 必须可连接，且需要几何、BSDF/PDF 和 visibility Jacobian。

**Hybrid shift**：先 random replay 一段前缀，再 reconnect 到缓存 suffix。RTXDI ReSTIR PT 采用这一路线，在鲁棒性与成本之间折中。

### 7.5 Section 7.5：Connectability

delta 顶点、理想介质、不可见连接、背面 BSDF、退化距离和不匹配介质边界通常不可直接 reconnect。粗糙度阈值/footprint 只是在选择候选 reconnect vertex；最终仍需检查实际 BSDF support 和 visibility。

### 7.6 Section 7.6：Lobe indices

混合 BSDF 的离散 lobe 选择是路径 sample 的一部分。random replay 必须重放相同随机维度，shift 必须记录并验证 lobe identity。若 reservoir selection 与 replay 共用 RNG 流，加入一个邻居就可能改变后续 BSDF 路径，破坏可重放性。本项目因此隔离 replay、resampling 与 NEE RNG。

## 8. Section 8：实现所需的数学状态

### 8.1 Section 8.1：Jacobian determinants

hybrid shift 的总 Jacobian 是各连续映射因子的乘积：

$$
J_{\mathrm{total}}
=
\prod_{k}J_k.
$$

可能包含：

- BSDF random-number-to-direction 映射的 PDF 比值；
- reconnection 的 solid-angle/area 几何项；
- Russian roulette continuation probability；
- light endpoint proposal 的 PDF 变换；
- 离散 lobe partition（若映射允许）。

对于 surface reconnection，典型几何比值包含

$$
J_{r\rightarrow q}^{\mathrm{geom}}
=
\frac{|n_c\cdot\omega_{cq}|}{|n_c\cdot\omega_{cr}|}
\frac{\|x_c-x_r\|^2}{\|x_c-x_q\|^2},
$$

其中 $x_c$ 是 reconnect vertex。实际总 Jacobian 还必须乘 replay prefix 和 suffix proposal 的密度比，不能只用这一项。

### 8.2 Section 8.2：Reservoir storage

实时实现不会保存完整 path vertex 数组，而是保存可重建路径的最小状态。当前持久 reservoir 约 64 bytes，包含 translated reconnection position/normal、target/radiance 或 NEE light payload、finalized weight、$M$、age、replay seed/dimension、path length、reconnection length、partial Jacobian、reconnection BSDF PDF 和路径标志。

压缩会影响精度；特别是 packed normal、LogLuv/radiance 和量化 path metadata 必须在测试中检查极端值。

### 8.3 Section 8.3：参数

当前采用一个 initial path、throughput cutoff 0.05、RR continuation 0.5、额外 delta budget 4、boiling 0.2。较积极的初始路径 roulette 用于抵消确定性 replay 增加的存活路径成本；temporal history/age 有上限。spatial 每帧从 3 张 reciprocal pairing map 中选择一张，每个像素恰好有一个互惠邻居，配对距离按 $\sigma=16$ px 的正态分布近似生成。

这些值不是 GRIS 公式常数。改变它们会影响方差、成本和探索速度；改变 clamp/boiling/shift rejection 还可能改变偏差。

## 9. RTXDI ReSTIR PT 工程管线

RTXDI 把 GRIS 组织成：initial path generation、temporal shift/resampling、boiling、spatial shift/resampling 和 final shading。应用通过 bridge 提供材质、BSDF、ray tracing、surface data 和 light sampling，RTXDI 负责 reservoir/shift 的公共数学框架。

RTXDI 的 realtime hybrid shift 通常还配合 sample ID、duplication map、checkerboard、Primary Surface Replacement 和降噪器数据。当前项目实现了基于 replay seed 的 17×17 duplication map 和自适应 temporal history cap，但没有完整复制其余辅助系统。

## 10. CUDA wavefront 管线

```mermaid
flowchart LR
    A["Primary surface"] --> B["Initial path setup"]
    B --> C["Trace / resolve loop"]
    C --> D["NEE proposal"]
    D --> E["NEE visibility"]
    E --> F["Initial PT reservoir"]
    F --> G["Temporal forward shift"]
    G --> H["Temporal selection"]
    H --> I["Temporal inverse shift"]
    I --> J["Pairwise normalize"]
    J --> K["Boiling filter"]
    K --> L["Spatial forward shift"]
    L --> M["Spatial selection"]
    M --> N["Reciprocal pair normalize"]
    N --> P["Final visibility / shading"]
    P --> Q["History"]
```

### 10.1 Initial path

`restir_pt_setup_initial_kernel` 从主表面分流。非 delta path 进入独立 replay RNG；delta chain 继续现有 wavefront 逻辑。每个 bounce 拆成 trace、resolve、NEE generate、NEE visibility 和 continuation，避免把 traversal 与复杂材质状态塞进一个高寄存器 kernel。

Initial sampling 在次级表面完成 NEE 后，对更深的 continuation 应用 Russian roulette；随机数由 path seed 和 depth 哈希得到，不占用 BSDF replay 维度。Random replay 不重新执行随机生死判断，而是在已知 source path 存活的条件下继续应用 $1/p_{rr}$ 补偿；这样既保留 source proposal PDF，又不会让合法 shift 因第二次 roulette 随机失败。固定存活率取 0.5，使 initial path 更短；代价是 initial sample 方差增加，但时空复用会显著抑制这部分噪声。

emissive hit、environment miss 和可见 NEE contribution 在同一 initial path stream 中竞争。每条已生成 initial path 最终都贡献 $M=1$，即使没有找到光或贡献为零。

### 10.2 直接光边界与 NEE/MIS

主表面直接光不属于 PT reservoir。PT 从主表面后的间接路径开始；secondary 及更深顶点的 NEE、emissive hit 和 environment miss 才形成 path contribution。

在 `NEE` 模式，有限灯显式采样，环境主要由 BSDF miss 表示。在 `MIS` 模式，环境/有限灯的 light proposal 与 BSDF proposal 都可参与，并应用现有 heuristic。若 NEE 模式比 MIS 系统性偏暗或偏色，应检查：

- 环境是否在 NEE 下被错误抑制；
- emissive/environment BSDF hit 是否被当作“直接光已处理”删除；
- light endpoint 的 path length 与 reconnection length 是否混淆；
- MIS weight 是否被存入 path contribution 后又在 shift/finalize 重复使用。

### 10.3 Temporal reuse

temporal pass 重投影 current receiver，寻找历史 reservoir，然后：

1. 用 source replay seed 在 current receiver 重放到 reconnection 前缀。
2. forward reconnect 到缓存 suffix，追踪 reconnect visibility。
3. 用 shifted target/contribution weight 参与 reservoir selection。
4. 对最终选中的 source 执行 inverse shift，确认可逆 support。
5. 用 pairwise `Pi/PiSum` normalization finalize。

只有 forward 成功不够。inverse shift 失败意味着该 source/target 映射不满足 pairwise support，应拒绝或在 normalization 中置零。

### 10.4 Spatial reuse

spatial pass 使用 host 端生成并上传的 reciprocal pairing map；该缓冲只在 `temporal-spatial` 模式首次使用时懒分配，`none` 和 `temporal` 不承担生成、上传或显存成本。若 $A$ 与 $B$ 配对，则严格满足 `pair[A] == B` 和 `pair[B] == A`；奇数像素总数下会有一个 self-pair，并在设备端跳过。两边各做一次 forward shift：$B\to A$ 与 $A\to B$；selection 后的 pairwise MIS 直接复用对向已经得到的 target、Jacobian 和 visibility，不再为 selected reservoir 启动第二轮 inverse replay。

配对表只负责摊销 shift，不绕过合法性检查。跨像素 surface compatibility 仍是第一层过滤，路径还必须通过 lobe、footprint reconnection、Jacobian 与 visibility 检查。若一侧 shift 失败，其 reciprocal density 在 normalization 中为零。3 张独立配对表逐帧轮换，减少固定配对形成的时空结构。

### 10.5 Final shading

最终 contribution 抽象为

$$
C=f_q(\bar y)W_{\bar y}.
$$

当 reconnection vertex 较深时，需要按缓存 replay seed 恢复 primary BSDF direction 和 diffuse/specular 分配。final visibility 只验证最终 shifted path；它不能替代 temporal/spatial support correction。

## 11. 原论文逐章节索引

| GRIS 原文章节 | 主题 | 本文位置 | 当前实现对应 |
| --- | --- | --- | --- |
| 1 / 1.1 | 目标与论文路线 | §1–2 | ReSTIR PT 范围 |
| 2 / 2.1 | resampling 背景 | §2–3 | reservoir 基础 |
| 3.1 | 同分布 RIS | §3.1 | initial reservoir |
| 3.2 | 多分布 RIS | §3.2 | source/canonical MIS |
| 3.3 | 推广动机 | §3.3 | weighted history 与 replay |
| 4.1 | GRIS overview | §4 | weighted sample 输入 |
| 4.2 | 无偏 GRIS 积分 | §4.1 | finalized contribution weight |
| 4.3 | Shift Mapping | §4.2 | forward/inverse hybrid shift |
| 4.4 | 渐近理想采样 | §4.3 | target-driven reuse |
| 5.1–5.3 | 分布与方差 | §5.1–5.2 | canonical coverage、离群权重 |
| 5.4 | 避免奇异性 | §5.3 | reconnect rejection/clamp |
| 5.5 | Canonical Samples | §5.4 | 每帧 initial path |
| 5.6 | Robust MIS Weights | §5.5 | pairwise `Pi/PiSum` |
| 5.7 | 保证收敛 | §5.6 | M/age cap、history reset |
| 6.1 | 图像积分表述 | §6.1 | 每像素 path domain |
| 6.2 | Reservoirs/Weighted GRIS | §6.2 | packed PT reservoir |
| 6.3 | Chained GRIS | §6.3 | initial→temporal→spatial |
| 6.4 | M-capping | §6.4 | 路径空间探索 |
| 6.5 | 离线渲染 | §6.5 | 更高预算扩展 |
| 7.1–7.3 | Shift 与 Full Shift | §7.1–7.3 | shift contract |
| 7.4 | 实时 shift | §7.4 | replay + reconnection hybrid |
| 7.5 | Connectability | §7.5 | delta/visibility/support |
| 7.6 | Lobe indices | §7.6 | replay RNG 与 lobe identity |
| 8.1 | Jacobian | §8.1 | partial/total Jacobian |
| 8.2 | Reservoir storage | §8.2 | 约 64-byte packed state |
| 8.3 | Parameters | §8.3 | Medium 风格参数 |
| 9 | Results and Discussion | §12 | 模式分层与高 SPP 验证 |

## 12. 验证方法

1. `resampling=None` 对比相同 max depth 的 baseline，先证明 initial path estimator 正确。
2. 单独测试 NEE 与 MIS，分别覆盖有限面积灯、环境和混合光源；比较线性 HDR 均值和 RGB 比例。
3. 开启 Temporal，检查静止收敛、相机移动、disocclusion、光源背面和阴影位置。
4. 开启 Temporal + Spatial，重点检查跨物体边缘、薄墙、屏幕边缘和高光材质。
5. 分开统计 forward reject、inverse reject、visibility reject、Jacobian reject、$M$ cap 和最大 weight。
6. 使用 `soft_shadow_test.lt` 验证单面面积灯背面必须为黑；该场景比 Cornell 更容易暴露 reconnect 漏光与阴影错位。
7. 检查 delta/介质链、alpha、双面材质、emissive hit、environment miss 和 RR replay 是否确定。

一旦 temporal 与 spatial 同时漏光，应先分别关闭它们定位：`None` 错说明 initial estimator 错；`Temporal` 才错说明 reprojection/replay/history support 错；加 Spatial 才错说明邻居 shift、inverse support 或 pairwise normalization 错。

## 13. 已知差异与未实现项

- 当前是 RTXDI 风格 hybrid shift 的本地重写，不是 SDK HLSL 的逐行移植。
- 没有 previous-frame BVH；temporal visibility 使用当前 BVH，场景变化时清空 history。
- 没有完整 Primary Surface Replacement、checkerboard 或 production denoiser 集成。
- duplication/sample-ID 当前以 replay seed 标识共同 initial candidate，并用 17×17 duplication count 缩短 temporal history；尚未覆盖 mutation、PSR 等更完整策略。
- spatial pass 的实时 pairwise 近似可能有偏；GRIS 理论无偏性依赖更严格的 mapping/support 条件。
- 尚未实现 Enhanced 的 RGB vector-weight marginalization；当前 final visibility 只覆盖最终选中样本，若直接累计未选候选会缺少对应 visibility，不能作为无代价改动加入。
- 尚未实现 dual motion vectors；当前场景数据没有遮挡面/反遮挡面的第二套运动矢量和 previous-frame instance transforms。
- 主表面直接光仍由独立 NEE/MIS 或 ReSTIR DI 处理，尚未实现 Enhanced Section 6.1 的完整 DI/GI unified reservoir。
- footprint/fixed threshold 是 reconnect vertex 启发式，不保证 shift 一定合法。
- Jacobian/throughput cutoff、RR、boiling、M cap 和 shift rejection 都要作为 estimator 设计的一部分验证，不能只按视觉调参。
- path suffix 的 light identity、量化 UV、sampler index 和 environment mapping 必须在 shift 后重新求值；直接复用旧 receiver 的 radiance 会造成漏光和颜色错误。

ReSTIR PT 最难的地方不是 reservoir selection，而是证明“source path 搬到 current receiver 后仍是一条由已知映射产生的合法路径”。forward/inverse shift、Jacobian、support 和 canonical exploration 缺一不可；时空复用只会放大 initial path 或 shift 中已经存在的错误。
