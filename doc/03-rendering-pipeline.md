# 渲染管线

## 一帧怎样生成

CPU 后端的调用链：

```text
CpuPathTracer::render()
  -> build_render_scene()（首次或 Geometry dirty）
  -> 多线程逐行
  -> make_camera_ray()
  -> trace_path()
     -> intersect_scene()
     -> 环境 / 发光面 / 直接光
     -> Material::sample()
     -> 下一次 bounce
  -> accumulation += sample
  -> to_rgba8(accumulation / (frame_index + 1))
```

CUDA 后端的调用链：

```text
CudaPathTracer::render()
  -> pack_scene() + 上传（按 dirty 决定）
  -> render_kernel()
  -> camera_ray()
  -> trace_gpu()
  -> 拷回 rgba
```

CUDA 只拷回显示用 `rgba`，设备上的 HDR accumulation 继续保留。CPU Framebuffer 的 `accumulation` 则由宿主维护。

## `Scene` 到 `RenderScene`

[`build_render_scene()`](../src/scene/render_scene.cpp) 完成：

1. 验证并复制解析球到 `RenderSphere`。
2. 对每个 Mesh 的顶点应用 `scale -> rotate XYZ -> translate`。
3. 生成世界空间三角形、几何法线、插值法线、UV 和切线空间。
4. 根据 Mesh light 或材质 emission 建立 `light_triangle_indices`。
5. 为每个 Mesh 建 BLAS。
6. 为 Mesh instance 建 TLAS。
7. 再建一份覆盖所有三角形的扁平 BVH。

当前“实例”并不共享一份局部空间几何；顶点已经展开到世界空间。Two-level BVH 的价值是遍历组织，而不是节省重复顶点内存。

### 切线空间

若有合法 UV，切线和副切线由 UV 导数计算。UV 退化或缺失时，从法线构造一个正交基。法线贴图在 CPU `apply_normal_map()` 和 GPU `apply_normal_map_gpu()` 中使用该基。

## 加速结构

`AccelerationStructure`：

- `Flat`：遍历覆盖所有三角形的单层 BVH。
- `TwoLevel`：先 TLAS，再进入每个 Mesh 的 BLAS。
- `Auto`：当 mesh instance 数量大于 1 时使用 TwoLevel。

CPU 选择发生在 `src/cpu/intersection.inl` 的 `use_two_level()`；GPU 选择发生在 `src/gpu/scene_upload.cuh` 的 `use_two_level_accel()`。修改策略时要保持两处一致。

BVH 构建使用：

- 最长 centroid 轴。
- `std::nth_element` 中位划分。
- BLAS/Flat 叶节点最多 4 个三角形。
- TLAS 叶节点最多 2 个 instance。

若要替换为 SAH、LBVH 或可更新 BVH，主要修改 `src/scene/render_scene.cpp`；若节点布局变化，还需同步 CPU/GPU 遍历和 `GpuBvhNode` 打包。

## 路径追踪

CPU 主循环在 `src/cpu/shading.inl::trace_path()`：

1. 求最近交点；未命中时累加环境。
2. 处理 alpha mask/blend 的随机可见性。
3. 应用法线贴图。
4. 命中发光三角形时，根据路径类型和 MIS 累加 emission。
5. 从三角形灯列表均匀选一个灯，进行 next-event estimation。
6. 从材质采样下一方向。
7. 第 4 个 bounce 起使用 Russian roulette。
8. 更新 throughput，继续追踪。

单样本 radiance 会被夹紧：相机首段上限 64，后续上限 8。这是当前降低 firefly 的偏差策略；做无偏参考渲染时需要显式调整或移除 CPU/GPU 两端的 clamp。

### 直接光和灯列表

当前可直接采样的灯是三角形灯：

- `Mesh::light.enabled && intensity > 0`
- 或三角形材质 `Material::emission` 非零

解析球不在三角灯列表中。当前还有一个后端差异：CPU 发光命中逻辑只处理 Triangle，因此会忽略解析球材质的 emission；GPU 会在命中后叠加材质 emission，但仍不会对该球做 next-event estimation。若要正式支持发光解析球，应先统一两端语义，再为它增加可采样的灯表示和 PDF。

灯选择对所有发光三角形等概率，不按面积或功率加权。若要做重要性采样，需要修改：

- `RenderScene` 灯数据。
- `build_render_scene()`。
- CPU `estimate_direct_lighting()` 与 `light_pdf_solid_angle()`。
- GPU 打包、`GpuScene` 和对应设备函数。

### MIS

启用 `use_mis` 后：

- 直接光样本用灯 PDF 对 BSDF PDF 加权。
- 非 delta BSDF 路径命中灯时，用 BSDF PDF 对灯 PDF 加权。
- `Balance` 使用一次幂，`Power` 使用平方启发式。

Delta 材质命中灯不做常规 MIS，直接累加。

## 环境光

`environment_radiance()` 根据 `Environment` 选择常量或纹理：

- Equirectangular：经纬映射。
- EqualArea：PBRT 环境图的八面体 equal-area 映射。
- `light_from_world_*` 把世界方向变换到贴图方向。

环境 LOD 根据图像尺寸、每帧样本数和累计帧估计。修改环境采样时要同步：

- CPU `src/cpu/shading.inl`
- GPU `src/gpu/shading.cuh`
- GPU `GpuScene` 字段与上传
- PBRT 环境导入（若映射语义变化）

当前环境只在射线 miss 时贡献，不在 next-event estimation 中被主动采样。因此小而亮的 HDRI 光源可能收敛较慢。

## NPR 路径

`trace_path()` 首先检查 `stylized_rendering_enabled()`。启用后进入递归的 `trace_stylized_radiance()`：

- 普通路径估计仍包含直接光、间接光和 MIS。
- 当前表面需要风格化时，重复 `stylized_samples` 次估计。
- 最多风格化 `stylized_max_depth` 个路径表面。
- 最终调用 `apply_npr_style()`。

该路径目前只有 CPU 实现。新增 GPU NPR 前，不应删除 CLI/编辑器的 CPU fallback。

## 脏标记与缓存

| 标记 | 语义 | CPU 行为 | CUDA 行为 |
| --- | --- | --- | --- |
| `Render` | 累积结果无效 | 调用方通常负责清 Framebuffer | 清设备 accumulation |
| `Camera` | 相机变化 | 直接读取新 Camera | 只上传 Camera |
| `Material` | 材质变化 | 路径追踪直接读新材质 | 重新 pack/upload 全场景核心缓冲 |
| `Texture` | 纹理变化 | 直接读共享纹理 | 重建 CUDA texture objects 并完整上传 |
| `Geometry` | 几何、变换、灯列表或 BVH 变化 | 重建 `RenderScene` | 完整 pack/upload |
| `Environment` | 环境参数变化 | 直接读取 | 局部上传环境字段 |

重要细节：

- CPU `CpuPathTracer::render()` 本身不会因为 `RenderDirty::Render` 自动清空 Framebuffer；调用方必须清空并重置 `frame_index`。
- `Framebuffer::resize()` 只有尺寸变化才清空。
- 材质 emission 从零变非零或反向变化会改变 CPU 的三角灯列表。虽然它是材质字段，CPU 还需要 `Geometry` 来重建 `RenderScene`。编辑此类字段时建议标记 `Material | Geometry`。
- Mesh light 的 enabled/color/intensity/double-sided 被 `RenderScene` 灯列表和 shading 使用，统一标记 `Geometry` 最安全。

## 新增渲染设置时的传播链

例如新增 `float exposure`：

1. 在 `include/lt/renderer.h::RenderSettings` 增加字段和默认值。
2. 决定设置在 CPU 的哪个阶段生效。
3. 若 CUDA 支持，确保 `RenderSettings` 可按值传入 kernel，并在设备代码使用。
4. 在 `src/cli/render_options.cpp` 增加参数解析。
5. 在 `src/editor_win32.cpp::draw_properties()` 增加控件。
6. 控件变化后调用 `reset_accumulation()`。
7. 若该字段改变缓存结构而不仅是像素结果，增加或复用适当 `RenderDirty`。
8. 为 CLI 和编辑器各做一次 CPU/CUDA 冒烟测试。

若设置只用于最终显示（例如曝光），更合理的设计可能是增加独立 tone mapping 阶段，而不是污染积分器的 HDR accumulation。

## 新增渲染后端

实现 `IRenderer` 后，还需接入：

- CLI：`src/main.cpp` 的后端选择。
- 编辑器：`EditorState` 持有实例，`set_renderer()` 和 Render 面板列出选项。
- CMake：加入源文件、依赖和编译定义。
- 功能协商：`available()`、NPR/材质支持限制和 fallback。

后端必须遵守 Framebuffer 尺寸、frame index 和累积语义，否则编辑器渐进预览会出现跳帧或亮度错误。
