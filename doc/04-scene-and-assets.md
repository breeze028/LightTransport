# 场景、导入器与纹理

## 统一加载入口

`lt::load_scene(path)` 位于 `src/scene/scene_io.cpp`，分派规则：

1. `http://` 或 `https://`：Windows 下下载到临时目录，再按扩展名加载。
2. `.glb` / `.gltf`：`load_gltf_scene()`。
3. `.pbrt`：`load_pbrt_scene()`。
4. 其他扩展名：按原生 `.lt` 文本解析。

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
camera px py pz  tx ty tz  fov_degrees
```

没有 `up` 和 `right_sign` 字段；加载使用默认值。

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

- 相机 position/target/FOV
- 纹理名称和 path
- 基础环境颜色、强度和可选纹理
- 材质基础色、模型、roughness/IOR、metallic、albedo texture
- NPR
- Mesh light
- 解析球
- Mesh 顶点、可选法线和索引

当前不会完整保存：

- Mesh UV
- alpha/alpha mode/cutoff/double-sided
- normal texture 和 normal scale
- material emission/emission texture
- dielectric transmission tint
- Principled metallic-roughness、sheen、clearcoat 纹理和大部分高级字段
- Environment mapping 和旋转基
- glTF/PBRT 原始层级、动画或实例关系

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
5. 调用 `build_mips()`。
6. 若支持内存嵌入，还要扩展 `load_texture_memory()`。
7. 更新编辑器文件对话框过滤器。
8. 测试材质纹理、环境纹理和 CUDA texture object 上传。

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

原则：读取器应尽量兼容旧语法，保存器只输出一种规范新语法。

## 新增场景导入器

1. 在 `include/lt/scene.h` 声明 `load_xxx_scene()`。
2. 新建 `src/xxx_loader.cpp`，失败时返回清晰 error。
3. 把源文件加入 `lt_core`。
4. 在 `load_scene()` 根据扩展名分派。
5. 将外部数据映射到 Scene，而不是直接生成 GPU 数据。
6. 保证所有 Mesh 材质下标、索引、法线/UV 数量合法。
7. 决定坐标系、手性、UV 原点和单位转换。
8. 更新编辑器 Open 对话框过滤器。
9. 说明是否支持保存回源格式。
