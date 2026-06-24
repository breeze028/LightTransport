# 构建与验证

## 构建

Visual Studio 2022：

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DLT_ENABLE_CUDA=ON
cmake --build build --config Release
```

Ninja：

```powershell
cmake -S . -B build-ninja -G Ninja -DCMAKE_BUILD_TYPE=Release -DLT_ENABLE_CUDA=ON
cmake --build build-ninja
```

CPU-only：

```powershell
cmake -S . -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DLT_ENABLE_CUDA=OFF
cmake --build build-cpu
```

不要在同一 build 目录混用 Visual Studio 和 Ninja generator。

## 基础冒烟测试

以下命令以 Visual Studio Release 目录为例。

CPU 基础：

```powershell
.\build\Release\lt_render.exe scenes\cornell.lt build\smoke_cpu.ppm --cpu --size 64 64 --spp 1 --frames 1
```

CUDA 基础：

```powershell
.\build\Release\lt_render.exe scenes\cornell.lt build\smoke_cuda.ppm --cuda --size 64 64 --spp 1 --frames 1
```

MIS：

```powershell
.\build\Release\lt_render.exe scenes\cornell.lt build\smoke_mis.ppm --cpu --mis --mis-heuristic power --size 64 64 --spp 2
```

NPR：

```powershell
.\build\Release\lt_render.exe scenes\toon_material_test.lt build\smoke_npr.ppm --cpu --size 64 64 --style-samples 2 --style-depth 1
```

解析球与 Mesh 混合由 `cornell.lt` 和 `toon_material_test.lt` 覆盖。

## 功能对应测试矩阵

| 改动 | 最少测试 |
| --- | --- |
| BRDF | CPU/CUDA、MIS 开关、不同 bounce、极端参数 |
| 纹理 | CPU/CUDA、无 mip/有 mip、alpha、环境 |
| 法线贴图 | 有效 UV 和退化 UV |
| 几何/求交 | Flat/TwoLevel/Auto，单 Mesh 和多 Mesh |
| 灯光 | 单面/双面、材质 emission、Mesh light、无灯 |
| Scene I/O | load、save、reload、非法 token |
| glTF | 外部资源、GLB 嵌入资源、多个 primitive |
| PBRT | Include、PLY、实例、环境 |
| CLI | 默认值、缺失参数、非法值 |
| 编辑器 | 修改后清累积、异步结果不过期覆盖 |
| NPR | 每种 style、style depth、CPU fallback |

## CPU/CUDA 对比

两端使用相同 pixel/frame seed，但实现细节和浮点执行顺序不同，不应要求逐像素 bit-exact。推荐：

1. 固定尺寸、SPP、frame 数和场景。
2. 输出 CPU/CUDA 两张图。
3. 比较平均亮度、最大误差和结构差异。
4. 对 delta 材质、透明和 emission 单独建小场景。

如果 CUDA 失败会在 `CudaPathTracer::render()` 内静默回退 CPU，仅观察“命令成功”不能证明 kernel 被使用。运行时 renderer name 和编辑器状态栏只能证明选择了 CUDA wrapper；严格确认需要增加 fallback 日志、检查 CUDA 错误，或使用 GPU profiler。

## Scene I/O 回归

建议为每个新增字段做：

```text
构造 Scene
  -> save_scene()
  -> load_scene()
  -> 比较字段
  -> 再 save
  -> 比较规范化文本或语义
```

当前没有正式测试框架。若开始增加自动化测试，建议：

- 新建 `tests/`。
- 为 `lt_core` 建独立 test executable。
- 先覆盖纯函数：材质 parse、纹理采样、场景 I/O、BVH 与 brute-force 求交对比。
- 再增加小分辨率 golden image，并为随机噪声使用统计阈值而非逐像素完全相等。

## 文档校验

修改 API 后检查：

- 文档中的文件路径仍存在。
- 新枚举、字段和命令行参数已加入相应表格。
- `.lt` 语法与 `scene_io.cpp` 的读取和保存一致。
- CPU-only 或平台限制已写明。
- 示例命令能在当前构建目录运行。
