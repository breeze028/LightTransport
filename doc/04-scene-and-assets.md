# 场景、导入器与纹理

## 统一加载入口

`lt::load_scene(path)` 位于 `src/scene/scene_io.cpp`，分派规则：

1. `http://` 或 `https://`：Windows 下下载到临时目录，再按扩展名加载。
2. `.glb` / `.gltf`：`load_gltf_scene()`。
3. `.pbrt`：`load_pbrt_scene()`。
4. `.fbx`：`load_fbx_scene()`（通过 ufbx 库）。
5. `.pyscene`：`load_pyscene_scene()`（解析侧车文件中的 FBX 引用并应用材质/环境调整）。
6. 其他扩展名：按原生 `.lt` 文本解析。

失败时许多路径返回 `make_default_scene()` 加错误字符串。严格工具应把 `error` 视为失败，而不是因为 Scene 非空就继续。

## 原生 `.lt` 格式

### 通用规则

- 一行一个指令；`#` 后是注释。
- 名称和路径按空白分词，不支持带空格或引号转义。
- 材质必须在引用它的 `npr`、`sphere` 和 `mesh` 前声明。
- Mesh rotation 是弧度。
- 颜色和强度使用线性值，可大于 1。

### 相机

```text
camera px py pz  tx ty tz  fov_degrees [up_x up_y up_z]
```

`up` 是可选的 3 个浮点数，用于非默认相机朝向。省略时使用 `(0,1,0)`。`right_sign` 不在 `.lt` 中保存。

### 纹理

```text
texture name relative/or/absolute/path.ext
```

相对路径以场景文件目录为基准。纹理名称在场景内应唯一。

### 环境

```text
environment r g b strength [texture_name]
```

- 无 texture：常量环境。
- 有 texture：纹理环境，颜色与强度作为倍率。
- 原生格式不保存 Mapping 和环境旋转基，读取后使用默认 Equirectangular/单位变换。

### 材质

```text
material name r g b brdf roughness_or_ior metallic [albedo_texture]
```

`brdf` 支持 `lambertian`、`principled`、`mirror`、`dielectric`、`conductor` 及解析函数中的别名。

- Principled：字段为 roughness、metallic。
- Dielectric：roughness 位置实际保存 IOR。
- 其他模型仍保留两个占位数值。

旧格式允许在 albedo 后先写三个 emission 数值；加载器会把它转换成使用该材质 Mesh 上的 `LightComponent`。

### NPR

Color Map：

```text
npr material_name color_map value_min value_max
```

X-Toon：

```text
npr material_name x_toon detail_mode steps
    shadow_r shadow_g shadow_b
    mid_r mid_g mid_b
    lit_r lit_g lit_b
    accent_r accent_g accent_b
    detail_strength detail_threshold detail_power depth_near depth_far
```

Cross Hatching：

```text
npr material_name cross_hatching
    sets spacing width angle value_min value_max
    ink_r ink_g ink_b paper_r paper_g paper_b
    passthrough shadow_only
```

这些内容实际写在一行。布尔值用 `0/1`。

### 面光源

```text
light mesh_name color_r color_g color_b intensity [double_sided]
```

光源可以先于 Mesh 声明，加载结束后按名称绑定。若重名，当前会绑定第一个匹配 Mesh。

### 方向光

```text
directional_light direction_x direction_y direction_z r g b intensity
```

方向光是无限远光源，方向为归一化的世界空间方向。加载时不验证归一化。保存时写出所有 `Scene::directional_lights`。

### 解析球

```text
sphere name material_name cx cy cz radius
```

解析球不能直接附加 `light` 指令，因为 `light` 按 Mesh 名称查找。

### Mesh

```text
mesh name material_name tx ty tz rx ry rz sx sy sz vertex_count triangle_count
x y z
...
[normals normal_count]
[nx ny nz]
...
i0 i1 i2
...
```

兼容旧 header：只写一个统一 scale。法线数量若非零必须与顶点数一致。

原生格式没有 UV block。加载时 UV 由局部顶点 `x/z` 生成：

```cpp
uv = {vertex.x + 0.5f, vertex.z + 0.5f};
```

## 保存限制

`save_scene()` 只面向原生 `.lt`，当前会保存：

- 相机 position/target/FOV/up
- 纹理名称和 path
- 基础环境颜色、强度和可选纹理
- 材质基础色、模型、roughness/IOR、metallic、albedo texture
- NPR
- Mesh light
- 方向光
- 解析球
- Mesh 顶点、可选法线和索引

当前不会完整保存：

- Mesh UV
- alpha/alpha mode/cutoff/double-sided
- normal texture 和 normal scale
- material emission/emission texture
- dielectric transmission tint
- Principled metallic-roughness、sheen、clearcoat 纹理和大部分高级字段
- StandardSurface 的全部 MaterialInput、transmission、coat、sheen 等字段
- Environment mapping 和旋转基
- 辐照度体积设置（`SceneRenderSettings`）
- glTF/PBRT/FBX 原始层级、动画或实例关系

因此，把 glTF/PBRT 导入为 Scene 后再写 `.lt` 会丢高级信息。编辑器也只允许原路径为 `.lt` 时执行保存。

## 纹理加载与采样

`load_texture_file()` 支持：

| 格式 | 平台 | 说明 |
| --- | --- | --- |
| PPM P3/P6 | 全平台 | 内置解析 |
| Radiance HDR RGBE | 全平台 | 内置解析 |
| EXR | 全平台 | RGB，half/float，none 或 ZIP 压缩 |
| PNG/JPEG 等 WIC 格式 | Windows | 通过 Windows Imaging Component |

非 Windows 下的错误消息仍写“Only PPM”，但代码实际上会先处理 HDR/EXR；新增平台解码器时应顺便修正这条消息。

所有成功的文件加载都会调用 `build_mips()`。纹理 RGB 被当作线性数据；当前没有显式 sRGB/linear 色彩空间元数据。导入普通 PNG/JPEG 基础色时也不会做 sRGB 解码，这是扩展色彩管理时需要统一处理的地方。

### 新增纹理格式

1. 在 `src/texture.cpp` 实现 `load_xxx_texture()`。
2. 需要公共调用时在 `include/lt/texture.h` 声明。
3. 在 `load_texture_file()` 按扩展名分派。
4. 填充 name/path/width/height/pixels/alpha。
5. 设置适当的 `role` 和 `color_space`（如果 loader 上下文已知）。
6. 调用 `build_mips()`。
7. 若支持内存嵌入，还要扩展 `load_texture_memory()`。
8. 更新编辑器文件对话框过滤器。
9. 测试材质纹理、环境纹理和 CUDA texture object 上传。
10. 若需保留原始编码数据用于导出，填充 `encoded_bytes` 和 `encoded_extension`。

## glTF 2.0 导入

入口：`src/gltf_loader.cpp::load_gltf_scene()`。

当前支持：

- `.glb` 和 `.gltf`
- 外部/二进制 buffer
- triangle primitive（mode 4）
- POSITION、NORMAL、TEXCOORD_0、indices
- node matrix 或 TRS，导入时烘焙到顶点
- Perspective camera
- 基础色、metallic、roughness 及其纹理
- normal texture
- emissive factor/texture 和 `KHR_materials_emissive_strength`
- alpha mode/cutoff、double-sided
- `KHR_materials_transmission`、`KHR_materials_ior`
- `KHR_materials_volume` 的 attenuation color 作为 transmission tint
- `KHR_materials_sheen`
- `KHR_materials_clearcoat`
- Windows 下外部及嵌入 PNG/JPEG 图像

重要限制：

- 只导入静态三角形；不支持动画、skin、morph。
- 不支持保存回 glTF。
- 每个 primitive 变成一个 Mesh。
- node 变换被烘焙，Scene 中 Mesh transform 保持默认。
- 相机加载只使用找到的 camera node 局部变换，不处理其父节点世界变换。
- 传输材质简化为理想 Dielectric，不完整实现 glTF 体积和 transmission texture。

扩展 glTF 材质时，通常修改 `load_materials()`；扩展顶点属性时修改 accessor 读取和 `import_primitive()`。

## PBRT 导入

入口：`src/pbrt/pbrt_loader.cpp::load_pbrt_scene()`。

当前处理的主要指令：

- `Include`
- Attribute/Transform 栈
- `Identity`、`Translate`、`Scale`、`Rotate`、`Transform`、`ConcatTransform`
- `LookAt`、`Camera`
- `Texture` imagemap
- `MakeNamedMaterial`、`NamedMaterial`、`Material`
- `AreaLightSource`
- `LightSource`：infinite，以及简化为小球面光源的 point/spot/goniometric
- `ObjectBegin`、`ObjectEnd`、`ObjectInstance`
- `Shape`：trianglemesh、plymesh、bilinearmesh、disk、sphere

材质映射是近似的：

- diffuse/matte -> Lambertian
- dielectric/thindielectric/glass -> Dielectric
- conductor -> Conductor
- coateddiffuse/coatedconductor -> 带 clearcoat 的 Principled
- 其他 -> Principled

PLY 加载使用异步任务，支持常见顶点位置、法线、UV 和多边形三角化。PBRT 环境采用 EqualArea mapping 并保留旋转基。

PBRT 导入器同时处理 `LightSource "distant"` 方向光，将 `point from` 转为世界空间方向，`L` 转为 `color * intensity`。

## FBX 导入

入口：`src/fbx_loader.cpp::load_fbx_scene()`，通过 ufbx 库解析 FBX 文件。

当前支持：

- 三角面 mesh（自动三角化四边形和 n-gon）。
- 顶点位置、法线、UV。
- 按 material slot 拆分为独立 Mesh，每个 node 可能产生多个 Mesh。
- PBR 材质（base color、roughness、metalness、normal、emission、specular），优先使用 FBX PBR 扩展并回退至传统 FBX 材质。
- 按命名约定查找纹理（`<material_name>_BaseColor`、`_Normal`、`_Roughness` 等后缀，搜 `Textures/` 子目录）。
- 嵌入纹理（通过 ufbx 的嵌入数据）。
- 透视相机导入，支持投影轴和 InterestPosition/UpVector 属性。
- 点光/面光源转换为小型发光 quad Mesh（方向光和体积光被跳过）。
- 自动摄像机：找不到相机时根据场景 AABB 计算默认视角。
- `.pyscene` 侧车文件自动应用（若存在同名的 `.pyscene` 文件）。

重要限制：

- 材质导入为 `StandardSurfaceMaterial`，不使用旧的 Principled 材质。
- BC5/ATI2 DDS 法线贴图当前被跳过（未做 BC5 → tangent-space normal 展开）。
- 不支持动画、blend shape、NURBS。
- node 变换烘焙到顶点世界空间，Mesh transform 保留默认值。
- 不支持保存回 FBX。

### PyScene 侧车文件

`.pyscene` 是文本辅助文件，与同名 FBX 配套使用。`load_pyscene_scene(path)` 还支持 `.pyscene` 作为主入口：它会解析 `sceneBuilder.importScene("path.fbx")` 加载 FBX，再应用以下指令：

- `EnvMap("path.hdr")` + `envMap.intensity = N`：设置 HDR 环境贴图。
- `emissiveFactor *= N`：对全部有 emission 的材质应用倍率。
- `sceneBuilder.getMaterial("name")`：建立材料变量引用。
- `variable.roughness = N` / `.metallic` / `.indexOfRefraction` / `.specularTransmission` / `.doubleSided`：调整单个材质参数（支持 Principled、StandardSurface 和 Dielectric）。

## 纹理角色与色彩空间

导入器通过 `apply_texture_role()` 为纹理设置角色和色彩空间：

| 纹理类型 | `TextureRole` | `TextureColorSpace` |
| --- | --- | --- |
| 基础色 / diffuse | `Color` | `SRGB`（或 `SceneLinear` 当 FBX 已知为线性） |
| roughness / metallic / specular / data | `Data` | `Raw` |
| 法线 | `Normal` | `Raw` |
| emission | `Emission` | `SceneLinear` |
| 环境 | `Environment` | 取决于源格式 |

色彩空间标签目前是元数据标记；实际 sRGB→linear 转换在导入时已由各 loader 完成。标签用于后续处理（如纹理导出和编辑器信息显示）。

## 纹理导出

- `write_texture_png(texture, path, error)`：将 RGB 像素编码为 8-bit RGBA PNG。如果 `encoded_bytes` 有原始编码数据则优先使用，否则从 `pixels` 重新编码。
- `write_texture_hdr(texture, path, error)`：将 RGB 像素编码为 Radiance HDR RGBE 格式。

这些函数用于编辑器 Save Texture 和日志诊断导出，不是渲染管线的一部分。

## 扩展 `.lt` 语法

以给 Camera 增加 aperture 为例：

1. 在 `include/lt/scene.h::Camera` 增加字段及默认值。
2. 在 `load_scene()` 的 `tag == "camera"` 分支读取可选字段，并兼容旧文件。
3. 在 `save_scene()` 对称写出。
4. 在 CPU/CUDA camera ray 中实现语义。
5. 若 CUDA 使用 Camera 按值传输，检查类型仍可直接拷贝。
6. 在编辑器 Camera tab 增加控件并标记 `Camera`。
7. 如需 CLI 覆盖，增加命令行参数。
8. 添加最小 `.lt` 示例测试读取、保存、再读取。

增加新指令（如 `directional_light`）同理：

1. 在 `Scene` 增加数据字段或 vector。
2. 在 `load_scene()` 主 while 增加 tag 分支，解析参数，校验范围。
3. 在 `save_scene()` 对称写出。
4. 在 CPU/CUDA 使用新数据。
5. 若引用其他对象（如材质名称），在加载结束后做二次解析。

原则：读取器应尽量兼容旧语法，保存器只输出一种规范新语法。

## 新增场景导入器

1. 在 `include/lt/scene.h` 声明 `load_xxx_scene()`。
2. 新建 `src/xxx_loader.cpp`，失败时返回清晰 error。
3. 把源文件加入 `lt_core`（含第三方依赖如 ufbx）。
4. 在 `load_scene()` 根据扩展名分派。
5. 将外部数据映射到 Scene，而不是直接生成 GPU 数据。
6. 保证所有 Mesh 材质下标、索引、法线/UV 数量合法。
7. 决定坐标系、手性、UV 原点和单位转换。
8. 更新编辑器 Open 对话框过滤器。
9. 说明是否支持保存回源格式。
10. 若格式支持嵌入纹理或自定义材质管线（如 FBX 的 MaterialInput），导入器应使用基于角色的纹理标记。

FBX/PyScene 导入器提供了完整参考：使用 ufbx 库作为解析前端，支持嵌入和外部纹理、PBR/传统材质回退、自动相机和灯光转换，以及 `.pyscene` 侧车文件的材质/环境调整。
