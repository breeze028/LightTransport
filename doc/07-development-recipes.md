# 功能开发配方

本页按“我要实现什么”列出最短修改路径。每项都区分核心实现、可保存/可编辑支持和验证。

如果要改 Irradiance Volume、Lightmap、Viewport View 或 SVGF，先读对应专题再回到本页查跨模块清单：

- [10-irradiance-volume.md](10-irradiance-volume.md)
- [11-lightmap.md](11-lightmap.md)
- [12-viewport-view.md](12-viewport-view.md)
- [13-svgf.md](13-svgf.md)
- [13.5-cwbvh.md](13.5-cwbvh.md)
- [14-wavefront-path-tracing-optimization.md](14-wavefront-path-tracing-optimization.md)
- [15-wavefront-restir-di.md](15-wavefront-restir-di.md)
- [16-wavefront-restir-gi.md](16-wavefront-restir-gi.md)
- [17-wavefront-restir-pt.md](17-wavefront-restir-pt.md)

## 修改点总表

| 功能类型 | 数据模型 | CPU | CUDA | I/O/导入 | CLI | 编辑器 |
| --- | --- | --- | --- | --- | --- | --- |
| 新材质字段 | `material.h` | `material.cpp`/shading | types + pack + shading | `.lt`/glTF/PBRT/FBX | 可选 | Material tab |
| 新 BRDF | `BrdfModel` + class | evaluate/pdf/sample | 三个设备函数 | parse/save/import | 可选 | BRDF Combo |
| 新 NPR | `NprStyle` + settings | `shading.inl` | 当前可不支持 | npr parse/save | style options | NPR controls |
| 新几何 | `scene.h` | build + intersect | pack + intersect | scene/importer | 可选 | selection/UI |
| 新渲染设置 | `RenderSettings` | 使用点 | 使用点 | 通常无 | parser | Render tab |
| 新灯型 | Scene/RenderScene | sampling/PDF | types/pack/sampling | loader/save | 可选 | object properties |
| 新文件格式 | Scene API | 无 | 无 | loader + dispatch | 无 | Open filter |
| 新输出格式 | 可选 image API | 无 | 无 | writer | main | 可选 |
| 新日志/诊断点 | `lt/log.h` | 边界事件 | host fallback | loader warning/error | log options | Log 面板 |
| 辐照度体积参数 | `RenderSettings` + `SceneRenderSettings` | bake + lookup | types + upload + lookup | `.ivol` 缓存文件 | ivol options | IV panel + bake |
| 纹理角色/色彩空间 | `texture.h` | 无 | 无 | loader 标记 | 无 | 纹理列表显示 |

## 新增一种 BRDF

核心：

- [ ] `include/lt/material.h`：枚举、派生类、字段、clone。
- [ ] `src/material.cpp`：evaluate/pdf/sample。
- [ ] `make_material()`、`parse_brdf_model()`。

GPU：

- [ ] `src/gpu/types.cuh::GpuMaterial`。
- [ ] `src/gpu/scene_upload.cuh::pack_scene()`。
- [ ] `src/gpu/shading.cuh` 的 evaluate/pdf/sample。

外围：

- [ ] `scene_io.cpp` 读写参数。
- [ ] glTF/PBRT 映射。
- [ ] 编辑器模型列表、参数控件、模型切换时字段迁移。
- [ ] 示例 Scene。

验证：

- [ ] 只照环境光。
- [ ] 只照面光源，MIS 开/关。
- [ ] 极端参数值。
- [ ] CPU/CUDA。
- [ ] 保存再加载。

## 给现有材质加纹理槽

例如 roughness detail texture：

- [ ] Material 持有 `shared_ptr<Texture>`。
- [ ] 添加 `roughness_at(uv)` 组合规则。
- [ ] CPU evaluate/pdf/sample 全部使用 accessor，不能有一处仍读常量。
- [ ] glTF/PBRT 导入。
- [ ] `.lt` 如需保存，增加可识别的纹理引用语法。
- [ ] `GpuMaterial` 保存 texture index。
- [ ] `pack_scene()` 从 `Scene::textures` 查找相同 shared_ptr。
- [ ] GPU 采样相同通道和组合规则。
- [ ] 编辑器 Combo/Load 按钮。
- [ ] 修改后 dirty 至少为 `Material`；新纹理资源还要 `Texture`。

纹理必须存在于 `Scene::textures`，仅把一个未登记的 shared_ptr 放进 Material 会导致 GPU `pack_scene()` 找不到下标。

## 新增一种 NPR 风格

- [ ] `NprStyle` 追加值。
- [ ] `NprSettings` 参数与默认值。
- [ ] 字符串 parse/name。
- [ ] CPU `apply_npr_style()` 分支。
- [ ] `.lt` parse/save 对称。
- [ ] CLI 全局参数和 per-material override。
- [ ] 编辑器 Combo 和控件。
- [ ] 保持 CPU fallback，或完成完整 GPU 实现后再放开。

如果风格依赖像素邻域，先设计 G-buffer/AOV 和后处理 pass。

## 新增一种几何体

### 优先方案：生成 Mesh

适合圆柱、圆锥、圆盘、曲面细分结果：

1. 在 `scene_geometry.cpp` 增加 `make_xxx_mesh()`。
2. 在 `scene.h` 声明。
3. 生成 vertices/normals/texcoords/indices。
4. 接入导入器或编辑器 Add 菜单。
5. 使用现有 `build_render_scene()`、BVH 和三角求交。

这是改动最小、CPU/CUDA 自动兼容度最高的方案。

### 解析图元方案

适合必须保持解析表达的图元：

- [ ] `Scene` 新增对象 vector。
- [ ] `RenderScene` 新增验证后的 render 类型。
- [ ] `build_render_scene()` 转换。
- [ ] CPU `Hit` 和 intersection。
- [ ] GPU type、pack、intersection。
- [ ] 材质、UV、切线空间和 normal mapping 语义。
- [ ] `.lt` 读写。
- [ ] 编辑器 selection、outliner、properties、pick、gizmo、outline、duplicate/delete。

解析球展示了完整链路，可作为模板。

## 新增灯光类型

当前直接光采样三角灯和方向光。新增点光/spot 时建议显式建立统一 Light 数据，而不是继续伪装成 Mesh。

需要设计：

- Light 类型与参数放在 `Scene` 还是 `RenderScene`。
- 每种灯的 `sample_li()`、PDF、是否 delta。
- 灯选择 PMF。
- BSDF 路径命中灯时的 MIS 处理。
- CPU/GPU 一致布局。
- `.lt`、glTF/PBRT/FBX 和编辑器表示。

若先做最小实现，可像 PBRT importer 一样把点光转为小型发光球 Mesh，但它不是数学意义上的 delta point light，会受半径和三角化影响。

方向光是现有 delta 方向光实现的参考：`DirectionalLight` 存储在 `Scene` 向量中，CPU 和 CUDA 的 `estimate_direct_lighting()` 独立遍历并累加无障碍贡献，打包为 `GpuDirectionalLight`。

## 新增 FBX 导入功能

给 FBX loader 增加新材质通道或几何特性：

- [ ] 在 `src/fbx_loader.cpp` 中增加 ufbx 材质通道读取（如 pbr.clearcoat）。
- [ ] 若增加纹理通道，使用 `set_material_input()` 并指定正确的 `MaterialInputChannel`、`TextureRole` 和 `TextureColorSpace`。
- [ ] 同时支持 FBX PBR 扩展和传统 FBX 材质回退。
- [ ] 若需要新命名约定纹理，更新 `load_convention_texture()` 的 suffix 逻辑。
- [ ] 添加 `.pyscene` 侧车解析支持（若新参数需要调整）。
- [ ] 测试嵌入纹理和外部纹理路径。
- [ ] 验证 BC5/ATI2 DDS 法线的跳过逻辑仍正确。

## 新增辐照度体积功能

修改辐照度体积烘焙或查找：

- [ ] `src/cpu/irradiance_volume.inl`：八叉树构建和查找逻辑。
- [ ] `src/cpu/irradiance_volume_bake.inl`：烘焙线程和积分。
- [ ] `src/gpu/types.cuh`：`GpuIrradianceVolume` 相关类型。
- [ ] `src/gpu/cuda_path_tracer.cu`：buffer 上传/释放和 device 指针。
- [ ] `src/gpu/shading.cuh`：设备端查找。
- [ ] `include/lt/renderer.h`：`RenderSettings` 和 `RenderDirty` 字段。
- [ ] `include/lt/scene.h`：`SceneRenderSettings` 和 per-object 排除字段。
- [ ] CLI：`src/cli/render_options.cpp` 的 `--ivol-*` 参数。
- [ ] 编辑器：`draw_irradiance_volume_panel()`、`start_irradiance_volume_bake()`。
- [ ] `.ivol` 缓存格式版本控制和向前兼容。

## 修改光源采样策略

例如从“按三角形均匀”改为“按功率”：

- [ ] `RenderScene` 保存每灯权重和 CDF/alias table。
- [ ] `build_render_scene()` 计算面积、辐射和权重。
- [ ] CPU 采样返回选择 PMF。
- [ ] `light_pdf_solid_angle()` 使用同一 PMF。
- [ ] BSDF 命中灯时能查询该三角形 PMF。
- [ ] GPU types、pack 和设备采样同步。
- [ ] 零功率、极小面积和双面灯测试。

只改抽样、不改命中灯 PDF 会破坏 MIS。

## 新增 `.lt` 指令

- [ ] 在 `load_scene()` 主 while 中增加 tag 分支。
- [ ] 校验 token 数量、范围和引用。
- [ ] 错误包含 line number。
- [ ] 在 `save_scene()` 对称写出。
- [ ] 定义声明顺序和向后兼容。
- [ ] 添加一个最小场景。
- [ ] 做 load -> save -> load 值一致性检查。

若指令引用稍后声明的对象，仿照 `pending_lights` 或 texture name 二次解析，不要保存裸指针。

## 新增场景导入格式

- [ ] 新 loader 只输出合法 `Scene`。
- [ ] `scene.h` 声明。
- [ ] `CMakeLists.txt` 加源文件。
- [ ] `load_scene()` 扩展名分派。
- [ ] 坐标系、手性、法线变换和 UV 翻转。
- [ ] 外部资源相对路径。
- [ ] 编辑器 Open filter。
- [ ] 错误回退策略。
- [ ] 大场景性能与索引越界测试。

## 新增 CLI 渲染选项

- [ ] `RenderOptions` 或 `RenderSettings` 字段。
- [ ] `parse_render_options()`。
- [ ] 参数缺失和非法值处理。
- [ ] `main.cpp` 应用。
- [ ] 若是核心 setting，CPU/CUDA 两端。
- [ ] 编辑器 Render tab 提供同等能力。
- [ ] 修改后累积失效。

## 新增编辑器操作

例如“复制材质”：

- [ ] 确定操作函数，避免把逻辑全写在 ImGui 分支里。
- [ ] 使用 `material->clone()`。
- [ ] 生成唯一名称。
- [ ] 更新 Scene vector 和选择状态。
- [ ] reset `Material`。
- [ ] 如果影响 Mesh material index，检查 vector 插入位置；追加比中间插入安全。
- [ ] 加菜单和快捷键。
- [ ] 快捷键避免与文本输入冲突。

## 新增 RenderDirty 类别

只有现有类别无法表达上传成本时才新增：

1. 在枚举追加 bit，并更新 `All`。
2. 更新 CPU cache 失效判断。
3. 更新 CUDA full/partial upload。
4. 更新编辑器所有修改点。
5. 更新文档矩阵。

如果只是像素结果变化，使用 `Render`；不要为每个参数创建一个 bit。

## 新增输出格式

- [ ] 在 `include/lt/image_io.h` / `src/image_io.cpp` 增加 writer。
- [ ] API 接收平均 HDR 或 `Framebuffer::rgba`，明确色彩空间。
- [ ] 在 `lt::write_image()` 中按扩展名选择 writer。
- [ ] CLI 输出错误返回非零。
- [ ] PNG 测 alpha/通道顺序。
- [ ] EXR 测 float 范围，不应用 gamma/clamp。

## 新增日志或诊断点

- [ ] 使用 `LT_LOG_*` 宏，不直接写 `std::cout` / `std::cerr`，除非是 CLI 进度条或最终用户输出。
- [ ] 加载器入口记录 `info`，失败记录 `error`，可恢复资源缺失记录 `warn`。
- [ ] GPU fallback 记录 `warn`，同一原因重复出现时可以降为 `debug`。
- [ ] 编辑器后台任务只写日志，不直接操作 ImGui；Log 面板由主线程消费 `EditorState::pending_logs`。
- [ ] 不在 CPU 像素循环、sample 循环、BRDF 高频求值或 CUDA device code 中写日志。
- [ ] 如果新增 CLI 控制项，同步更新 `doc/09-logging.md` 和 `doc/06-cli-and-editor.md`。

## 修改相机模型

景深、全景或正交相机至少影响：

- `Camera` 数据。
- CPU `make_camera_ray()`。
- GPU `camera_ray()`。
- `.lt` I/O。
- glTF/PBRT 映射。
- 编辑器 Camera tab 和 viewport picking/project。

编辑器的 `make_view_transform()`、`make_view_ray_from_screen()` 和 `project_point()` 假定透视相机。只改渲染射线会导致选择和 gizmo 与画面不一致。

## 提交前的功能完整性检查

- [ ] 默认值不会改变旧场景结果。
- [ ] Scene copy 不丢字段（含 `directional_lights` 和 `render_settings`）。
- [ ] `.lt` 读写对称，或明确标注不支持保存。
- [ ] CPU 有实现。
- [ ] CUDA 有实现或可靠 fallback。
- [ ] 新导入格式（FBX/PyScene）正确处理坐标系和单位。
- [ ] 新材质（StandardSurface）的 MaterialInput 纹理在 `Scene::textures` 中已注册。
- [ ] CLI 和编辑器都能触达功能。
- [ ] dirty 正确，修改后不会混入旧累积帧。
- [ ] 小场景、空场景、无灯场景能运行。
- [ ] 辐照度体积烘焙在有/无缓存、手动/自动边界情况下都能完成。
- [ ] 文档和示例同步。
