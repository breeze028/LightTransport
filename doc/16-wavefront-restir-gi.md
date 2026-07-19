# CUDA Wavefront ReSTIR GI：从论文到实现

本文逐节转述 Ouyang 等人的 ReSTIR GI 论文，解释它如何把一次 BSDF path sample 变成可时空复用的间接光样本，并对照 RTXDI Full Sample 与本项目 CUDA wavefront 实现。内容不是逐字翻译；理论按论文公式重述，工程行为以当前源码为准。

## 1. 范围与算法边界

当前 ReSTIR GI 是经典的一次间接光版本：

- 只作用于 CUDA wavefront path tracer 的第一个非 delta 主表面。
- 从主表面采一条 BSDF ray，命中非发光 secondary surface 后，估计该点朝主表面的 outgoing radiance。
- secondary sample 经 temporal、boiling、spatial reuse 后，从当前主表面重新连接并追踪 final visibility。
- primary ray 直接命中灯或环境仍属于直接光，不进入 GI reservoir。
- delta reflection/transmission 旁路 GI，继续普通 wavefront path。
- ReSTIR DI 不是 ReSTIR GI 的数学前置条件。当前实现只复用 DI 的 initial-light RIS helper 来估计 secondary direct lighting，即使用户未开启 DI 也能运行 GI。
- ReSTIR GI 与 ReSTIR PT 互斥；PT 开启时优先。

主要源码：

- `src/gpu/restir_gi.cuh`
- `src/gpu/restir_di.cuh`
- `src/gpu/cuda_path_tracer.cu`
- `src/gpu/types.cuh`
- `include/lt/renderer.h`

参考资料：

- Ouyang et al., *ReSTIR GI: Path Resampling for Real-Time Path Tracing*, HPG 2021。
- NVIDIA RTXDI: [ReSTIR GI](https://github.com/NVIDIA-RTX/RTXDI/blob/main/Doc/RestirGI.md) 与 [Shader API](https://github.com/NVIDIA-RTX/RTXDI/blob/main/Doc/ShaderAPI-RestirGI.md)。

## 2. Sections 1–3：从渲染方程到可复用样本

### 2.1 渲染方程

表面点 $x$ 的出射辐射为

$$
L_o(x,\omega_o)
=
L_e(x,\omega_o)
+
\int_{\Omega^+}
L_i(x,\omega_i)
f_r(x,\omega_o,\omega_i)
|n_x\cdot\omega_i|
\,\mathrm d\omega_i.
$$

在无参与介质场景里，入射辐射来自方向 $\omega_i$ 上的最近交点：

$$
L_i(x,\omega_i)
=
L_o(\operatorname{Trace}(x,\omega_i),-\omega_i).
$$

用 $p_x(\omega)$ 从半球采样，普通 Monte Carlo 估计为

$$
\widehat L_o
=
L_e
+
\frac{1}{N}\sum_{j=1}^{N}
\frac{
L_i(x,\omega_j)
f_r(x,\omega_o,\omega_j)
|n_x\cdot\omega_j|
}{p_x(\omega_j)}.
$$

ReSTIR GI 不是直接缓存最终像素颜色。它缓存的是由一条方向射线得到的 secondary sample：位置、法线、朝原 receiver 估计出的 outgoing radiance，以及 reservoir 权重。重用时把这个 secondary sample 搬到新的 receiver 重新评价。

### 2.2 为什么 DI reservoir 不能直接拿来做 GI

DI sample 通常是“灯上的点或无限远方向”，而 GI sample 是“BSDF ray 命中的次级表面及其散射辐射”。两者积分域和几何变换不同：

- DI 在 light sample domain 上重采样。
- GI 在 receiver 半球方向上采样，再通过 ray tracing 映射到次级表面。
- GI 把一个 secondary point 连接到不同 receiver 时，方向域面积发生变化，需要 Jacobian。

因此 DI 和 GI 可以同时启用，但 reservoir、history 和 bias correction 都是独立的。RTXDI Full Sample 常用 ReSTIR DI initial-light sampling 给 secondary surface 估计直接光，这是质量/复用代码的选择，不是 ReSTIR GI 公式要求。

### 2.3 WRS 回顾

对 target $\widehat p(S)$，reservoir 最终权重仍为

$$
W(S)=\frac{w_{\mathrm{sum}}}{M\widehat p(S)},
$$

最终估计是

$$
\widehat I=f(S)W(S).
$$

这里 $S$ 不再只是灯索引，而是 secondary position、normal 和 radiance 的组合。

## 3. Section 4.1：Sample Generation

### 3.1 visible point 与 sample point

论文区分：

- visible point $x_v$：camera ray 首次命中的主表面。
- sample point $x_s$：从 $x_v$ 采样方向并追踪后命中的次级表面。

从 source PDF $p_{x_v}(\omega)$ 采样 $\omega_i$，得到

$$
x_s=\operatorname{Trace}(x_v,\omega_i).
$$

初始 sample 至少要保存 $x_v,n_v,x_s,n_s$ 和 secondary outgoing radiance $\widetilde L_o(x_s,-\omega_i)$。论文还保存生成 secondary radiance 时使用的随机数，供之后 sample validation 重新追踪。

source PDF 可以是均匀半球、余弦或 BSDF PDF。论文发现均匀半球在某些掠射间接光场景反而更稳，因为余弦采样很少探索近切线方向；这不是普遍结论，而是 target、场景和 sample reuse 共同作用的结果。本项目沿用主表面实际 BSDF sampling。

### 3.2 secondary radiance

论文允许在 $x_s$ 继续 path tracing：若 secondary radiance 本身包含 $n$ 次反弹，连接回主表面后得到至多 $n+1$ bounce 的 GI。实时配置通常只在 secondary surface 做一次 NEE，从而得到一次间接光。

本项目在 secondary surface 调用共享 initial-light RIS：发光三角形、点光、方向光和环境都可参加，只对 RIS 最终选中的光追踪一条 visibility ray。该 estimator 不做 secondary DI temporal/spatial reuse，因此 GI reservoir 只重用 secondary sample，不嵌套另一个持久 DI reservoir。

primary BSDF ray 若直接命中发光体或环境，路径只有一个散射事件，属于主表面直接光。它必须交给现有 NEE/MIS/ReSTIR DI 规则结算或抑制，不能再包装成 GI sample，否则会重复计量。

## 4. Section 4.2：Resampling and Shading

### 4.1 receiver-aware target

完整 target 可以包含当前 receiver 的 BSDF 和 cosine：

$$
\widehat p_q(S)
=
Y\!\left(
\widetilde L_o(x_s,-\omega_{qs})
f_r(x_q,\omega_o,\omega_{qs})
|n_q\cdot\omega_{qs}|
\right),
$$

其中 $Y$ 表示标量亮度。这个 target 更接近最终 contribution，但同一个 sample 在不同 receiver 上变化剧烈，空间复用容易出现高权重离群值。

论文还测试了 radiance-only target：

$$
\widehat p(S)=Y\!\left(\widetilde L_o(x_s)\right).
$$

它忽略 receiver BSDF/cosine，重要性匹配较差，却更容易跨像素移植。论文结果认为 radiance-only target 在空间复用时通常更稳定。当前实现仍在 receiver 端计算实际 target，并用 roughness floor 降低高度方向性 secondary radiance 带来的不稳定；这是 RTXDI Full Sample 风格工程取舍，不是原论文唯一方案。

### 4.2 初始 reservoir

若从 primary BSDF solid-angle PDF $p_q(\omega)$ 得到一个 sample，当前实现把初始 contribution weight 写成

$$
w=\frac{\widehat p_q(S)}{p_q(\omega)},
\qquad M=1.
$$

等价地，reservoir 可保存 sample contribution weight $1/p_q(\omega)$，在 stream 时再乘 target。无效、遮挡或零贡献 initial path 仍属于已生成的 Monte Carlo 样本，必须保留 $M=1$ 的统计意义；把它们从 $M$ 删除会造成亮偏差。

### 4.3 temporal reuse

temporal pass 把当前 visible point 重投影到上一帧，选择兼容历史 reservoir。历史 secondary point 在当前 receiver 上形成新方向和距离，所以必须重新计算 target、几何量和 visibility。

当前实现先测试重投影中心，再测试随机旋转五点邻域，并有无 motion vector fallback。兼容条件包括深度、法线、材质和稳定几何标识；history length 上限为 8，reservoir age 上限为 30。

### 4.4 spatial reuse 与 Jacobian

相邻 receiver $x_r$ 的方向样本连接到 secondary point $x_s$ 后，搬到当前 receiver $x_q$ 时发生方向域变换。设

$$
d_r=\|x_s-x_r\|,
\qquad
d_q=\|x_s-x_q\|,
$$

secondary normal 为 $n_s$，从 secondary 指向 receiver 的单位方向分别为 $\omega_{sr}$ 和 $\omega_{sq}$。从旧 receiver 到新 receiver 的 solid-angle Jacobian 为

$$
J_{r\rightarrow q}
=
\frac{|n_s\cdot\omega_{sq}|}{|n_s\cdot\omega_{sr}|}
\frac{d_r^2}{d_q^2}.
$$

若代码把密度从旧域变换到新域，使用乘 $J$ 还是除 $J$ 取决于函数参数定义；必须与 source contribution weight 的方向一致。判断正确性的办法不是看变量名，而是检查变量变换公式

$$
p_q(\omega_q)
=
p_r(\omega_r)
\left|\frac{\partial\omega_r}{\partial\omega_q}\right|.
$$

本项目拒绝 $J\notin[0.1,10]$ 的复用，并把保留值钳制到 $[1/3,3]$。这是稳定性措施，会引入偏差；原论文也强调几何奇异点和极端 Jacobian 是 GI reuse 的主要风险。

空间 pass 默认采两个 32 px 随机磁盘邻居，屏幕边界反射寻址。邻居 surface 必须通过深度、法线和材质检查。

## 5. Section 4.3：Bias

### 5.1 support correction

与 DI 相同，不同 receiver 的 sample support 不一致。对最终选中 sample $S_y$，无偏 normalization 要统计哪些 source reservoir 在该 sample 上有非零 target：

$$
Z
=
\sum_{n\in\mathcal Q}
M_n\,
\mathbf 1\!\left[\widehat p_n(S_y)>0\right],
$$

$$
W
=
\frac{w_{\mathrm{sum}}}{Z\widehat p_q(S_y)}.
$$

若 source receiver 到 selected secondary point 被遮挡，则它不在有效 support 内。`Basic` correction 用几何相似性和非零 target 近似；`Ray Traced` correction 为历史/邻居 source 追踪 conservative visibility。最终 shading 无论哪种模式都要追踪当前 primary 到 selected secondary 的 visibility。

### 5.2 时间偏差与 sample validation

即使表面重投影正确，历史 sample 中的 secondary radiance 也可能因动态光照或遮挡变化而失效。论文提出 sample validation：重放 initial sample 保存的随机数，重新追踪 secondary radiance；实验中约每 6 帧验证一次。当前项目没有完整实现这种周期性随机数验证，而是在场景/材质/光源/环境变化时清空 history，并始终使用当前 BVH 做 reconnect visibility。

这意味着纯相机移动可稳定重用，但动态遮挡或未触发 history reset 的光照变化仍可能短时陈旧。

### 5.3 非方向性 radiance 近似

论文缓存 $x_s$ 朝原 receiver 的 outgoing radiance，却在重连到新 receiver 时把它近似当作方向无关。对 diffuse secondary surface 合理，对 glossy/specular surface 不成立，会造成能量偏差和高亮传播。

当前实现把 secondary roughness 至少钳制到 0.5；Final MIS 使用的辅助 BSDF roughness floor 为 0.3。它们降低方向性误差和离群值，但改变了原始材质，因此属于显式偏差。理想方案需要缓存/重放方向相关路径信息，这正是 ReSTIR PT/GRIS shift 更复杂的原因。

## 6. Section 5：原论文实现细节

论文报告的主要实现选择：

- spatial reuse 只接受法线夹角小于约 25°、归一化深度差小于约 0.05 的邻居。
- 当 reservoir 的 $M$ 很小时可尝试更多邻居，最多约 9 次；稳定后减少到约 3 次。
- 自适应空间半径从图像尺寸的约 10% 开始，找不到兼容邻居时逐步缩小，最小约 3 px。
- multi-bounce 配置只让部分 tile 追踪额外 bounce，以控制成本。
- sample validation 周期性更新历史 radiance。

当前项目没有逐项采用这些启发式：邻居数、半径、history cap 和 Jacobian clamp 固定，参数更接近 RTXDI 示例而非论文实验配置。

## 7. RTXDI GI 与本项目的关系

RTXDI 把 ReSTIR GI 暴露为 reservoir API 和应用侧 bridge：应用提供 G-buffer、材质/BRDF、ray tracing、surface similarity 和 secondary radiance，库负责 temporal/spatial resampling 数学。Full Sample 的典型顺序是：

1. 生成 BRDF ray 并追踪 secondary surface。
2. 在 secondary surface 计算直接光。
3. 形成 initial GI reservoir。
4. temporal resampling。
5. boiling filter。
6. spatial resampling。
7. final shading 与 reconnect visibility。

本项目没有调用 RTXDI runtime，而是在 CUDA 中重写同一责任划分。DI 与 GI 可同时启用：DI 估计主表面直接光，GI 估计主表面经 secondary surface 的间接光。二者不会把同一个持久 reservoir 串联起来。

## 8. CUDA wavefront 管线

```mermaid
flowchart LR
    A["Primary surface"] --> B["Sample primary BSDF"]
    B -->|"delta"| C["Existing continuation"]
    B -->|"non-delta"| D["Generate secondary ray"]
    D --> E["Trace secondary"]
    E --> F["Resolve secondary"]
    F --> G["Secondary direct RIS"]
    G --> H["Trace selected-light visibility"]
    H --> I["Build initial GI reservoir"]
    I --> J["Temporal reuse"]
    J --> K["Boiling filter"]
    K --> L["Spatial reuse"]
    L --> M["Generate reconnect ray"]
    M --> N["Trace final visibility"]
    N --> O["Final MIS / accumulate"]
    O --> P["Store history"]
```

关键 pass：

1. `restir_gi_generate_secondary_kernel` 分流 delta/non-delta，保存 primary BSDF direction 和 solid-angle PDF。
2. `restir_gi_trace_secondary_kernel` 独立执行 traversal。
3. setup/resolve 区分 miss、emissive hit 和普通 secondary surface。
4. secondary direct initial RIS 生成灯候选，只追踪最终选中灯的一条 visibility ray。
5. initial reservoir 保存 secondary position、normal、radiance、weight、$M$ 和 age；持久形式压缩 normal 与 LogLuv radiance。
6. temporal stream/finalize 执行重投影、surface validation、Jacobian 与 correction。
7. boiling filter 在 temporal 后限制 tile 内离群 reservoir。
8. spatial stream/finalize 使用随机磁盘邻居与 canonical receiver normalization。
9. final visibility 从当前 primary 连接 selected secondary；resolve 计算真实 primary BSDF、cosine、radiance 和 reservoir weight。

## 9. Final MIS

仅用 resampled sample 会丢掉当前帧 initial BSDF sample 的某些高频信息。RTXDI Full Sample 用启发式在原始 initial estimator 与 resampled estimator 之间混合。本项目保存原 initial sample，并用 roughness floor 0.3 的辅助 BSDF 构造权重，再用真实 BSDF 计算最终颜色。

抽象地写，若两个无偏估计分别为 $C_i$ 和 $C_r$，混合为

$$
C=w_iC_i+w_rC_r,
\qquad
w_i+w_r=1.
$$

权重必须与各自可采样 support 和 PDF 一致。Final MIS 不能用来修复错误的 reservoir normalization；若关闭 resampling 的 initial estimator 已过亮、漏光或偏色，应先修 initial path 与直接/间接边界。

## 10. 原论文逐章节索引

| 原论文章节 | 主题 | 本文位置 | 当前实现对应 |
| --- | --- | --- | --- |
| 1 | 一次/多次间接光的时空路径重用 | §1–2 | 主表面 GI 接管 |
| 2 | 之前的实时 GI 与 sample reuse | §2.2 | DI/GI 域差异 |
| 3 | 渲染方程、MC、RIS/WRS | §2 | initial reservoir 数学 |
| 4 | ReSTIR GI 总体算法 | §3–5 | secondary sample pipeline |
| 4.1 | Sample Generation | §3 | BSDF ray、secondary radiance |
| 4.2 | Resampling and Shading | §4 | temporal/spatial/Jacobian |
| 4.3 | Bias | §5 | support、validation、方向近似 |
| 5 | Implementation | §6–9 | wavefront、参数、Final MIS |
| 6 | Results | §11 | 与 initial/baseline 的线性均值比较 |
| 6.1 | Limitations | §12 | glossy radiance、动态历史等 |
| 7 | Conclusion and Future Work | §12 | 当前缺口与扩展方向 |

## 11. 正确性验证顺序

1. `resampling=None`：与相同 primary BSDF direction 的一次间接 baseline 比较，确认 secondary direct estimator、PDF 和直接事件抑制正确。
2. `Temporal`：静止相机下比较线性均值；再移动相机检查重投影和 disocclusion。
3. `Temporal + Spatial`：重点测试薄墙、柱体阴影、光源背面和屏幕边缘。
4. 分别测试 `Basic` 与 `Ray Traced`，确认差异只来自 correction visibility，而非最终 visibility。
5. 关闭 Final MIS 比较 resampled estimator；再开启确认 MIS 只改变方差/混合，不系统改变颜色。
6. 对面积灯背面使用零环境场景。灯背面仍亮时，优先检查 secondary light emission sidedness、primary-to-secondary final visibility 和历史 support，而不是先调亮度常数。

## 12. 已知差异、偏差与限制

- 只实现一次间接光，不是完整 path suffix resampling。
- secondary outgoing radiance 近似方向无关，并用 roughness floor 稳定；对高光材质有偏。
- Jacobian reject/clamp 有偏，但避免 grazing/近距离奇异权重。
- boiling filter 有偏，主要抑制历史离群值。
- 没有论文中的周期性 sample validation，也没有 previous-frame BVH。
- temporal correction 使用当前 BVH；场景变化依赖 history reset。
- 当前固定两个空间邻居和 32 px 半径，没有论文的自适应 radius/iteration count。
- packed normal/LogLuv 会引入量化误差；运算阶段解包为全精度。
- `Ray Traced` correction 使用额外可见性 ray，但不能补偿错误的 Jacobian、PDF 或 radiance sidedness。

ReSTIR GI 的正确性链条是：primary PDF 正确、secondary radiance 是合法估计、直接事件不重复、receiver 变化带上 Jacobian、source support 正确、最终连接可见。任何一环出错，时空复用都会把局部错误稳定地传播到历史中。
