# LightTransport 开发文档

这组文档面向准备修改或扩展 LightTransport 的开发者。重点不是重复源码，而是回答两个问题：

1. 项目对外提供了哪些 API，它们的生命周期和约束是什么？
2. 想实现某个功能时，必须修改哪些文件，CPU、CUDA、场景文件、CLI 和编辑器之间需要怎样保持一致？

文档基于当前仓库源码编写。代码符号比行号更稳定，因此文档统一使用“文件路径 + 类型/函数名”定位代码。

## 阅读顺序

| 文档 | 适合解决的问题 |
| --- | --- |
| [01-architecture.md](01-architecture.md) | 项目由哪些模块组成，数据怎样从场景流到图像 |
| [02-public-api.md](02-public-api.md) | `include/lt/` 中每个公开类型和函数如何使用 |
| [03-rendering-pipeline.md](03-rendering-pipeline.md) | CPU/CUDA 路径追踪、BVH、MIS、累积帧和脏标记怎样工作 |
| [04-scene-and-assets.md](04-scene-and-assets.md) | `.lt`、glTF、PBRT、纹理和环境贴图如何加载与保存 |
| [05-materials-and-styles.md](05-materials-and-styles.md) | BRDF、材质纹理、透明、发光和 NPR 风格如何扩展 |
| [06-cli-and-editor.md](06-cli-and-editor.md) | 命令行参数和编辑器 UI 的入口、状态和刷新规则 |
| [07-development-recipes.md](07-development-recipes.md) | 按功能查找要改的代码，包含实施与验证清单 |
| [08-build-and-verification.md](08-build-and-verification.md) | 如何构建、冒烟测试和检查 CPU/CUDA 一致性 |

## 按需求快速查找

| 想实现的功能 | 首先阅读 | 主要代码位置 |
| --- | --- | --- |
| 新增一种 BRDF/材质模型 | [材质与风格](05-materials-and-styles.md#新增一种-brdf) | `include/lt/material.h`、`src/material.cpp`、`src/gpu/types.cuh`、`src/gpu/shading.cuh` |
| 给 Principled 增加参数或纹理 | [材质字段的完整传播链](05-materials-and-styles.md#材质字段的完整传播链) | 材质头文件、导入器、GPU 打包、编辑器 |
| 新增 NPR 风格 | [新增 NPR 风格](05-materials-and-styles.md#新增一种-npr-风格) | `NprSettings`、CPU shading、CLI、`.lt` I/O、编辑器 |
| 新增几何体 | [几何体配方](07-development-recipes.md#新增一种几何体) | `include/lt/scene.h`、`src/scene/scene_geometry.cpp`、CPU/GPU intersection |
| 新增场景语法 | [原生场景格式](04-scene-and-assets.md#扩展-lt-语法) | `src/scene/scene_io.cpp` |
| 支持新的文件格式 | [新增场景导入器](04-scene-and-assets.md#新增场景导入器) | 新 loader、`load_scene()`、`CMakeLists.txt` |
| 支持新的纹理格式 | [纹理 API](04-scene-and-assets.md#纹理加载与采样) | `include/lt/texture.h`、`src/texture.cpp` |
| 新增 CLI 参数 | [CLI 扩展](06-cli-and-editor.md#新增命令行参数) | `src/cli/render_options.*`、`src/main.cpp` |
| 新增编辑器面板或控件 | [编辑器扩展](06-cli-and-editor.md#新增编辑器控件) | `src/editor_win32.cpp`、`src/editor/editor_state.h` |
| 新增渲染选项 | [RenderSettings 传播链](03-rendering-pipeline.md#新增渲染设置时的传播链) | `include/lt/renderer.h`、CPU、CUDA、CLI、编辑器 |
| 修改 BVH/求交 | [加速结构](03-rendering-pipeline.md#加速结构) | `src/scene/render_scene.cpp`、CPU/GPU intersection |
| 新增渲染后端 | [IRenderer](02-public-api.md#irenderer-与后端) | `include/lt/renderer.h`、入口和编辑器后端选择 |
| 改输出格式 | [离线输出](06-cli-and-editor.md#离线输出) | `src/main.cpp` 中的 `write_ppm()` |

## 最容易遗漏的联动点

- CPU 和 CUDA 有两套材质求值、采样、求交和路径追踪实现。修改核心渲染算法时，要明确是只支持 CPU，还是同步实现 GPU。
- GPU 不直接使用多态 `Material`，而是通过 `pack_scene()` 转成扁平的 `GpuMaterial`。新增材质字段必须经过这一步。
- `.lt` 的读取和保存是两段独立代码。只修改读取会导致编辑器保存时丢失新字段。
- 编辑器修改状态后必须调用 `reset_accumulation()`，并传入正确的 `RenderDirty`。
- `RenderScene` 是从 `Scene` 构建的加速数据。修改几何、三角灯列表或会影响 BVH 的数据时必须标记 `Geometry`。
- 当前 NPR 只在 CPU 实现；启用 NPR 时 CLI 和编辑器都会回退到 CPU。
- 当前原生 `.lt` 保存并不能完整保留 glTF/PBRT 导入的所有高级材质字段和 UV 数据。详见 [场景保存限制](04-scene-and-assets.md#保存限制)。

## 代码入口

- 公共 API：[`include/lt/`](../include/lt/)
- 核心库：[`src/`](../src/)
- CPU 路径追踪：[`src/cpu/`](../src/cpu/)
- CUDA 路径追踪：[`src/gpu/`](../src/gpu/)
- 场景转换和 I/O：[`src/scene/`](../src/scene/)
- 命令行：[`src/main.cpp`](../src/main.cpp)、[`src/cli/`](../src/cli/)
- 编辑器：[`src/editor_win32.cpp`](../src/editor_win32.cpp)、[`src/editor/`](../src/editor/)
- 示例场景：[`scenes/`](../scenes/)
