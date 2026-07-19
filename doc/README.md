# LightTransport 开发文档

这组文档面向需要修改或扩展 LightTransport 的开发者。重点不是从 API 生成注释，而是说明数据放在哪里、CPU/CUDA/编辑器路径怎样保持一致，以及哪些文件必须一起改。

代码引用优先使用文件路径与函数/类型名，不绑定具体行号，因为实现还在快速移动。

## 阅读顺序

| 文档 | 适合查什么 |
| --- | --- |
| [01-architecture.md](01-architecture.md) | 模块布局、Scene 如何变成像素、主要线程/资源边界 |
| [02-public-api.md](02-public-api.md) | `include/lt/` 公共类型、Renderer 接口、Scene/Material API |
| [03-rendering-pipeline.md](03-rendering-pipeline.md) | CPU/CUDA 路径追踪、加速结构、灯光、dirty flag、帧累积 |
| [04-scene-and-assets.md](04-scene-and-assets.md) | `.lt`、glTF、PBRT、FBX、PyScene、纹理、色彩空间、场景保存/加载 |
| [05-materials-and-styles.md](05-materials-and-styles.md) | BRDF、MaterialInput、纹理、透明、发光、NPR 风格 |
| [06-cli-and-editor.md](06-cli-and-editor.md) | CLI 参数、编辑器状态、预览任务生命周期、面板与控件 |
| [07-development-recipes.md](07-development-recipes.md) | 跨 CPU/CUDA/I/O/编辑器加功能时的检查表 |
| [08-build-and-verification.md](08-build-and-verification.md) | 构建命令、冒烟测试、CPU/CUDA 验证 |
| [09-logging.md](09-logging.md) | 日志 API、CLI 日志控制、编辑器 Log 面板 |

## 专题文档

下面几个功能已经拆成独立文件。每个专题都从理论公式讲到当前代码实现、编辑器入口、缓存/dirty 规则和验证方式。

| 功能 | 文档 | 主要代码 |
| --- | --- | --- |
| Irradiance Volume | [10-irradiance-volume.md](10-irradiance-volume.md) | `src/cpu/irradiance_volume*.inl`、`src/gpu/shading.cuh`、`src/gpu/cuda_path_tracer.cu`、编辑器 Render 面板 |
| Lightmap | [11-lightmap.md](11-lightmap.md) | `src/cpu/lightmap*.inl`、`src/gpu/shading.cuh`、`src/gpu/cuda_path_tracer.cu`、xatlas 集成 |
| Viewport View | [12-viewport-view.md](12-viewport-view.md) | `src/editor_win32.cpp`、`src/editor/solid_preview.*` |
| SVGF | [13-svgf.md](13-svgf.md) | `src/denoise/svgf.cpp`、`src/gpu/kernel.cuh`、`src/gpu/cuda_path_tracer.cu`、raster G-buffer 代码 |
| CWBVH | [13.5-cwbvh.md](13.5-cwbvh.md) | `src/gpu/scene_upload.cuh`、`src/gpu/intersection.cuh`、`src/gpu/types.cuh` |
| Wavefront Path Tracing 优化 | [14-wavefront-path-tracing-optimization.md](14-wavefront-path-tracing-optimization.md) | `src/gpu/cuda_path_tracer.cu`、`src/gpu/kernel.cuh`、`src/gpu/intersection.cuh` |
| ReSTIR DI | [15-wavefront-restir-di.md](15-wavefront-restir-di.md) | `src/gpu/restir_di.cuh`、`src/gpu/cuda_path_tracer.cu` |
| ReSTIR GI | [16-wavefront-restir-gi.md](16-wavefront-restir-gi.md) | `src/gpu/restir_gi.cuh`、`src/gpu/restir_di.cuh` |
| ReSTIR PT | [17-wavefront-restir-pt.md](17-wavefront-restir-pt.md) | `src/gpu/restir_pt.cuh`、`src/gpu/restir_di.cuh` |

## 快速定位

| 我要改什么 | 先看哪里 | 主要文件 |
| --- | --- | --- |
| 新增 BRDF/材质模型 | [05-materials-and-styles.md](05-materials-and-styles.md) | `include/lt/material.h`、`src/material.cpp`、`src/gpu/shading.cuh` |
| 新增纹理通道 | [05-materials-and-styles.md](05-materials-and-styles.md) | Material accessor、loader、`pack_scene()`、编辑器 Material tab |
| 新增场景对象 | [07-development-recipes.md](07-development-recipes.md) | `include/lt/scene.h`、scene geometry、CPU/GPU intersection、编辑器 selection |
| 新增 CLI 参数 | [06-cli-and-editor.md](06-cli-and-editor.md) | `src/cli/render_options.cpp`、`src/main.cpp` |
| 新增编辑器控件 | [06-cli-and-editor.md](06-cli-and-editor.md) | `src/editor_win32.cpp`、`src/editor/editor_state.h` |
| 修改路径追踪 | [03-rendering-pipeline.md](03-rendering-pipeline.md) | `src/cpu/shading.inl`、`src/gpu/shading.cuh`、`src/gpu/kernel.cuh` |
| 修改 Irradiance Volume | [10-irradiance-volume.md](10-irradiance-volume.md) | CPU bake/lookup、GPU upload/lookup、`.ivol` cache、编辑器 Render 面板 |
| 修改 Lightmap | [11-lightmap.md](11-lightmap.md) | xatlas unwrap、bake、cache、runtime lookup、GPU upload |
| 修改 Viewport 预览 | [12-viewport-view.md](12-viewport-view.md) | D3D11 solid/material/wireframe、GPU picking、outline、raster G-buffer |
| 修改 SVGF/TAA/Post AA | [13-svgf.md](13-svgf.md) | AOV 生成、temporal reprojection、A-trous filter、final resolve |
| 修改 CUDA wavefront traversal | [13.5-cwbvh.md](13.5-cwbvh.md)、[14-wavefront-path-tracing-optimization.md](14-wavefront-path-tracing-optimization.md) | BVH8/CWBVH 打包、compact hit、kernel layout 选择、profile |
| 修改 ReSTIR | [15-wavefront-restir-di.md](15-wavefront-restir-di.md)、[16-wavefront-restir-gi.md](16-wavefront-restir-gi.md)、[17-wavefront-restir-pt.md](17-wavefront-restir-pt.md) | reservoir、visibility、secondary trace、history |

## 跨模块规则

- CPU 和 CUDA 通常是两套实现。一个渲染功能完成前，必须确认两条路径的语义一致，或明确写出 fallback。
- GPU 代码读取的是 `pack_scene()` 打包后的 POD 风格数据，不读取宿主侧多态 `Material` 对象。
- `.lt` 加载和保存是两条路径。loader 支持某个字段，不代表编辑器保存时会保留它。
- 编辑器中影响像素的控件必须用合适的 `RenderDirty` 调用 `reset_accumulation()`。
- `RenderScene` 派生自 `Scene`。几何、三角灯列表和加速结构都依赖正确的 dirty 传播。
- D3D11 编辑器资源和 CUDA 资源生命周期不同。D3D11 texture 传给 CUDA interop 时，注册、map/unmap 和 release 路径必须清楚。
- CPU 高频 sample loop 和 CUDA device code 中不要写日志；日志只放在边界、fallback、缓存决策和用户触发操作处。

## 重要入口

- 公共 API：[`include/lt/`](../include/lt/)
- 核心源码：[`src/`](../src/)
- CPU 渲染器：[`src/cpu/`](../src/cpu/)
- CUDA 渲染器：[`src/gpu/`](../src/gpu/)
- 降噪器：[`src/denoise/`](../src/denoise/)
- 编辑器：[`src/editor_win32.cpp`](../src/editor_win32.cpp)、[`src/editor/`](../src/editor/)
- CLI：[`src/main.cpp`](../src/main.cpp)、[`src/cli/`](../src/cli/)
- Scene I/O：[`src/scene/`](../src/scene/)
- 示例场景：[`scenes/`](../scenes/)
