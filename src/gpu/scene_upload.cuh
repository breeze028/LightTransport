bool size_fits_int(size_t size) {
    return size <= static_cast<size_t>(std::numeric_limits<int>::max());
}

bool use_two_level_accel(const RenderScene& render_scene, AccelerationStructure acceleration_structure) {
    if (acceleration_structure == AccelerationStructure::TwoLevel) {
        return true;
    }
    if (acceleration_structure == AccelerationStructure::Flat) {
        return false;
    }
    return render_scene.mesh_instances.size() > 1;
}

bool pack_scene_from_render_scene(const Scene& scene, const RenderSettings& settings, const RenderScene& render_scene, PackedGpuScene& packed) {
    packed = {};
    GpuScene& gpu = packed.scene;
    gpu.camera = scene.camera;
    gpu.environment_color = scene.environment.color;
    gpu.environment_strength = scene.environment.strength;
    gpu.environment_constant = scene.environment.constant;
    gpu.environment_mapping = static_cast<int>(scene.environment.mapping);
    gpu.environment_light_from_world_x = scene.environment.light_from_world_x;
    gpu.environment_light_from_world_y = scene.environment.light_from_world_y;
    gpu.environment_light_from_world_z = scene.environment.light_from_world_z;
    if (!size_fits_int(scene.textures.size()) || !size_fits_int(scene.materials.size())) {
        return false;
    }
    if (!size_fits_int(scene.directional_lights.size())) {
        return false;
    }
    gpu.directional_light_count = static_cast<int>(scene.directional_lights.size());
    packed.directional_lights.resize(scene.directional_lights.size());
    for (int i = 0; i < gpu.directional_light_count; ++i) {
        const DirectionalLight& light = scene.directional_lights[static_cast<size_t>(i)];
        packed.directional_lights[static_cast<size_t>(i)] = {light.direction, light.color, light.intensity};
    }
    gpu.texture_count = static_cast<int>(scene.textures.size());
    packed.textures.resize(scene.textures.size());
    for (int i = 0; i < gpu.texture_count; ++i) {
        const std::shared_ptr<Texture>& texture = scene.textures[static_cast<size_t>(i)];
        if (!texture || texture->width <= 0 || texture->height <= 0) {
            return false;
        }
        const size_t pixels = static_cast<size_t>(texture->width) * static_cast<size_t>(texture->height);
        if (pixels != texture->pixels.size() || !size_fits_int(pixels)) {
            return false;
        }
        const int mip_levels = std::max(1, static_cast<int>(texture->mip_pixels.size()));
        packed.textures[static_cast<size_t>(i)] = {texture->width, texture->height, mip_levels, 0};
    }
    gpu.material_count = static_cast<int>(scene.materials.size());
    packed.materials.resize(scene.materials.size());
    for (int i = 0; i < gpu.material_count; ++i) {
        const std::shared_ptr<Material>& material = scene.materials[static_cast<size_t>(i)];
        if (!material) {
            return false;
        }
        float roughness = 0.5f;
        float metallic = 0.0f;
        Vec3 conductor_eta = {0.200438f, 0.924033f, 1.102212f};
        Vec3 conductor_k = {3.912949f, 2.452848f, 2.142188f};
        int normal_texture_index = -1;
        int emission_texture_index = -1;
        int sheen_color_texture_index = -1;
        int sheen_roughness_texture_index = -1;
        int clearcoat_texture_index = -1;
        int clearcoat_roughness_texture_index = -1;
        int transmission_texture_index = -1;
        Vec3 sheen_color;
        float sheen_roughness = 0.0f;
        float clearcoat = 0.0f;
        float clearcoat_roughness = 0.0f;
        Vec3 transmission_tint = {1.0f, 1.0f, 1.0f};
        float transmission = 0.0f;
        float specular_ior = 1.5f;
        float specular_weight = 1.0f;
        if (const auto* principled = dynamic_cast<const PrincipledMaterial*>(material.get())) {
            roughness = principled->roughness;
            metallic = principled->metallic;
            sheen_color = principled->sheen_color;
            sheen_roughness = principled->sheen_roughness;
            clearcoat = principled->clearcoat;
            clearcoat_roughness = principled->clearcoat_roughness;
        } else if (const auto* dielectric = dynamic_cast<const DielectricMaterial*>(material.get())) {
            roughness = dielectric->ior;
            transmission = 1.0f;
            specular_ior = dielectric->ior;
            transmission_tint = dielectric->transmission_tint;
        } else if (const auto* conductor = dynamic_cast<const ConductorMaterial*>(material.get())) {
            roughness = conductor->roughness;
            conductor_eta = conductor->eta;
            conductor_k = conductor->k;
        } else if (const auto* standard = dynamic_cast<const StandardSurfaceMaterial*>(material.get())) {
            roughness = standard->roughness;
            metallic = standard->metalness;
            sheen_color = standard->sheen_color;
            sheen_roughness = standard->sheen_roughness;
            clearcoat = standard->coat_weight;
            clearcoat_roughness = standard->coat_roughness;
            transmission = standard->transmission_weight;
            transmission_tint = standard->transmission_color;
            specular_ior = standard->specular_ior;
            specular_weight = standard->specular_weight;
        }
        int texture_index = -1;
        int metallic_roughness_texture_index = -1;
        int roughness_texture_index = -1;
        int metallic_texture_index = -1;
        int specular_texture_index = -1;
        TextureTransform base_texture_transform;
        TextureTransform emission_texture_transform;
        const auto find_texture_index = [&](const std::shared_ptr<Texture>& texture) {
            if (!texture) {
                return -1;
            }
            for (int t = 0; t < gpu.texture_count; ++t) {
                if (scene.textures[static_cast<size_t>(t)] == texture) {
                    return t;
                }
            }
            return -1;
        };
        if (material->albedo_texture) {
            for (int t = 0; t < gpu.texture_count; ++t) {
                if (scene.textures[static_cast<size_t>(t)] == material->albedo_texture) {
                    texture_index = t;
                    break;
                }
            }
        }
        if (const auto* principled = dynamic_cast<const PrincipledMaterial*>(material.get())) {
            if (principled->metallic_roughness_texture) {
                for (int t = 0; t < gpu.texture_count; ++t) {
                    if (scene.textures[static_cast<size_t>(t)] == principled->metallic_roughness_texture) {
                        metallic_roughness_texture_index = t;
                        break;
                    }
                }
            }
            sheen_color_texture_index = find_texture_index(principled->sheen_color_texture);
            sheen_roughness_texture_index = find_texture_index(principled->sheen_roughness_texture);
            clearcoat_texture_index = find_texture_index(principled->clearcoat_texture);
            clearcoat_roughness_texture_index = find_texture_index(principled->clearcoat_roughness_texture);
        } else if (const auto* standard = dynamic_cast<const StandardSurfaceMaterial*>(material.get())) {
            base_texture_transform = standard->base_color_input.transform;
            emission_texture_transform = standard->emission_input.transform;
            if (standard->roughness_input.texture && standard->roughness_input.texture == standard->metalness_input.texture) {
                metallic_roughness_texture_index = find_texture_index(standard->roughness_input.texture);
            }
            roughness_texture_index = find_texture_index(standard->roughness_input.texture);
            metallic_texture_index = find_texture_index(standard->metalness_input.texture);
            specular_texture_index = find_texture_index(standard->specular_weight_input.texture);
            sheen_color_texture_index = find_texture_index(standard->sheen_color_input.texture);
            sheen_roughness_texture_index = find_texture_index(standard->sheen_roughness_input.texture);
            clearcoat_texture_index = find_texture_index(standard->coat_input.texture);
            clearcoat_roughness_texture_index = find_texture_index(standard->coat_roughness_input.texture);
            transmission_texture_index = find_texture_index(standard->transmission_input.texture);
        }
        if (material->normal_texture) {
            for (int t = 0; t < gpu.texture_count; ++t) {
                if (scene.textures[static_cast<size_t>(t)] == material->normal_texture) {
                    normal_texture_index = t;
                    break;
                }
            }
        }
        if (material->emission_texture) {
            for (int t = 0; t < gpu.texture_count; ++t) {
                if (scene.textures[static_cast<size_t>(t)] == material->emission_texture) {
                    emission_texture_index = t;
                    break;
                }
            }
        }
        const float packed_roughness = material->model() == BrdfModel::Dielectric
            ? std::clamp(roughness, 1.0f, 3.0f)
            : (material->model() == BrdfModel::Conductor ? std::clamp(roughness, 0.0f, 1.0f) : std::clamp(roughness, 0.02f, 1.0f));
        packed.materials[static_cast<size_t>(i)] = {
            material->albedo,
            static_cast<int>(material->model()),
            packed_roughness,
            std::clamp(metallic, 0.0f, 1.0f),
            conductor_eta,
            conductor_k,
            texture_index,
            metallic_roughness_texture_index,
            roughness_texture_index,
            metallic_texture_index,
            specular_texture_index,
            base_texture_transform.offset,
            base_texture_transform.scale,
            base_texture_transform.rotation,
            sheen_color,
            std::clamp(sheen_roughness, 0.0f, 1.0f),
            sheen_color_texture_index,
            sheen_roughness_texture_index,
            std::clamp(clearcoat, 0.0f, 1.0f),
            std::clamp(clearcoat_roughness, 0.0f, 1.0f),
            clearcoat_texture_index,
            clearcoat_roughness_texture_index,
            normal_texture_index,
            material->normal_scale,
            material->emission,
            emission_texture_index,
            emission_texture_transform.offset,
            emission_texture_transform.scale,
            emission_texture_transform.rotation,
            transmission_tint,
            std::clamp(transmission, 0.0f, 1.0f),
            transmission_texture_index,
            std::clamp(specular_ior, 1.0f, 3.0f),
            std::max(0.0f, specular_weight),
            material->alpha,
            material->alpha_cutoff,
            static_cast<int>(material->alpha_mode),
            material->double_sided ? 1 : 0,
        };
    }
    if (scene.environment.texture) {
        for (int t = 0; t < gpu.texture_count; ++t) {
            if (scene.textures[static_cast<size_t>(t)] == scene.environment.texture) {
                gpu.environment_texture = t;
                break;
            }
        }
        if (gpu.environment_texture < 0) {
            return false;
        }
    }
    if (!size_fits_int(render_scene.triangles.size()) || !size_fits_int(render_scene.spheres.size()) ||
        !size_fits_int(render_scene.triangle_indices.size()) ||
        !size_fits_int(render_scene.flat_triangle_indices.size()) || !size_fits_int(render_scene.flat_bvh_nodes.size()) ||
        !size_fits_int(render_scene.bvh_nodes.size()) || !size_fits_int(render_scene.light_triangle_indices.size()) ||
        !size_fits_int(render_scene.mesh_instances.size()) || !size_fits_int(render_scene.mesh_instance_indices.size()) ||
        !size_fits_int(render_scene.tlas_nodes.size())) {
        return false;
    }
    gpu.use_two_level = use_two_level_accel(render_scene, settings.acceleration_structure) ? 1 : 0;
    gpu.triangle_count = static_cast<int>(render_scene.triangles.size());
    gpu.sphere_count = static_cast<int>(render_scene.spheres.size());
    gpu.bvh_node_count = gpu.use_two_level ? static_cast<int>(render_scene.bvh_nodes.size()) : static_cast<int>(render_scene.flat_bvh_nodes.size());
    gpu.mesh_instance_count = static_cast<int>(render_scene.mesh_instances.size());
    gpu.tlas_node_count = static_cast<int>(render_scene.tlas_nodes.size());
    packed.triangles.resize(render_scene.triangles.size());
    packed.spheres.resize(render_scene.spheres.size());
    packed.triangle_indices = gpu.use_two_level ? render_scene.triangle_indices : render_scene.flat_triangle_indices;
    const std::vector<BvhNode>& source_bvh_nodes = gpu.use_two_level ? render_scene.bvh_nodes : render_scene.flat_bvh_nodes;
    packed.bvh_nodes.resize(source_bvh_nodes.size());
    packed.mesh_instances.resize(render_scene.mesh_instances.size());
    packed.mesh_instance_indices = render_scene.mesh_instance_indices;
    packed.tlas_nodes.resize(render_scene.tlas_nodes.size());
    for (size_t i = 0; i < source_bvh_nodes.size(); ++i) {
        const BvhNode& node = source_bvh_nodes[i];
        packed.bvh_nodes[i] = {node.bounds.min, node.bounds.max, node.left, node.right, node.first, node.count};
    }
    for (size_t i = 0; i < render_scene.mesh_instances.size(); ++i) {
        const RenderScene::MeshInstance& instance = render_scene.mesh_instances[i];
        packed.mesh_instances[i] = {instance.bounds.min, instance.bounds.max, instance.bvh_root, instance.mesh};
    }
    for (size_t i = 0; i < render_scene.tlas_nodes.size(); ++i) {
        const BvhNode& node = render_scene.tlas_nodes[i];
        packed.tlas_nodes[i] = {node.bounds.min, node.bounds.max, node.left, node.right, node.first, node.count};
    }
    packed.light_indices.reserve(render_scene.light_triangle_indices.size());
    for (int i = 0; i < gpu.triangle_count; ++i) {
        const Triangle& tri = render_scene.triangles[static_cast<size_t>(i)];
        if (tri.material < 0 || tri.material >= gpu.material_count) {
            return false;
        }
        Vec3 emission;
        int light_double_sided = 0;
        if (tri.mesh >= 0 && tri.mesh < static_cast<int>(scene.meshes.size())) {
            const LightComponent& light = scene.meshes[static_cast<size_t>(tri.mesh)].light;
            if (light.enabled && light.intensity > 0.0f) {
                emission = light.color * light.intensity;
                light_double_sided = light.double_sided ? 1 : 0;
            }
        }
        packed.triangles[static_cast<size_t>(i)] = {
            tri.v0, tri.v1, tri.v2, tri.normal, tri.n0, tri.n1, tri.n2, tri.tangent, tri.bitangent, tri.centroid,
            emission, tri.uv0, tri.uv1, tri.uv2, tri.lightmap_uv0, tri.lightmap_uv1, tri.lightmap_uv2,
            tri.material, tri.mesh, light_double_sided, tri.has_lightmap ? 1 : 0,
        };
    }
    for (int i = 0; i < gpu.sphere_count; ++i) {
        const RenderSphere& sphere = render_scene.spheres[static_cast<size_t>(i)];
        if (sphere.material < 0 || sphere.material >= gpu.material_count) {
            return false;
        }
        packed.spheres[static_cast<size_t>(i)] = {sphere.center, sphere.radius, sphere.material, sphere.sphere};
    }
    for (int i = 0; i < static_cast<int>(packed.triangle_indices.size()); ++i) {
        const int tri_index = packed.triangle_indices[static_cast<size_t>(i)];
        if (tri_index < 0 || tri_index >= gpu.triangle_count) {
            return false;
        }
        packed.triangle_indices[static_cast<size_t>(i)] = tri_index;
    }
    for (int i = 0; i < static_cast<int>(render_scene.light_triangle_indices.size()); ++i) {
        const int light_index = render_scene.light_triangle_indices[static_cast<size_t>(i)];
        if (light_index < 0 || light_index >= gpu.triangle_count) {
            return false;
        }
        packed.light_indices.push_back(light_index);
    }
    gpu.light_count = static_cast<int>(packed.light_indices.size());
    return true;
}

bool pack_scene(const Scene& scene, const RenderSettings& settings, PackedGpuScene& packed) {
    return pack_scene_from_render_scene(scene, settings, build_render_scene(scene), packed);
}

template <typename T>
bool upload_buffer(void*& device, int& cached_count, const std::vector<T>& host) {
    const int count = static_cast<int>(host.size());
    if (cached_count != count) {
        cudaFree(device);
        device = nullptr;
        cached_count = 0;
    }
    if (count == 0) {
        return true;
    }
    if (!device && cudaMalloc(&device, host.size() * sizeof(T)) != cudaSuccess) {
        return false;
    }
    if (cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess) {
        return false;
    }
    cached_count = count;
    return true;
}

void release_texture_objects(std::vector<void*>& arrays, std::vector<uint64_t>& objects) {
    for (uint64_t object : objects) {
        if (object != 0) {
            cudaDestroyTextureObject(static_cast<cudaTextureObject_t>(object));
        }
    }
    for (void* array : arrays) {
        if (array) {
            cudaFreeMipmappedArray(static_cast<cudaMipmappedArray_t>(array));
        }
    }
    arrays.clear();
    objects.clear();
}

bool upload_texture_objects(const Scene& scene, PackedGpuScene& packed, std::vector<void*>& arrays, std::vector<uint64_t>& objects) {
    release_texture_objects(arrays, objects);
    arrays.resize(scene.textures.size(), nullptr);
    objects.resize(scene.textures.size(), 0);
    for (int i = 0; i < static_cast<int>(scene.textures.size()); ++i) {
        const std::shared_ptr<Texture>& texture = scene.textures[static_cast<size_t>(i)];
        if (!texture || texture->width <= 0 || texture->height <= 0) {
            return false;
        }
        cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();
        cudaMipmappedArray_t array = nullptr;
        const unsigned int mip_levels = static_cast<unsigned int>(std::max(1, static_cast<int>(texture->mip_pixels.size())));
        if (cudaMallocMipmappedArray(&array, &channel_desc, make_cudaExtent(static_cast<size_t>(texture->width), static_cast<size_t>(texture->height), 0), mip_levels) != cudaSuccess) {
            release_texture_objects(arrays, objects);
            return false;
        }
        arrays[static_cast<size_t>(i)] = array;
        for (unsigned int level = 0; level < mip_levels; ++level) {
            const std::vector<Vec3>& level_pixels = level == 0 || texture->mip_pixels.empty() ? texture->pixels : texture->mip_pixels[static_cast<size_t>(level)];
            const int level_width = level == 0 || texture->mip_widths.empty() ? texture->width : texture->mip_widths[static_cast<size_t>(level)];
            const int level_height = level == 0 || texture->mip_heights.empty() ? texture->height : texture->mip_heights[static_cast<size_t>(level)];
            std::vector<float4> pixels(level_pixels.size());
            for (size_t p = 0; p < level_pixels.size(); ++p) {
                const Vec3 color = level_pixels[p];
                const float alpha = level == 0 && p < texture->alpha.size() ? texture->alpha[p] : 1.0f;
                pixels[p] = make_float4(color.x, color.y, color.z, alpha);
            }
            cudaArray_t level_array = nullptr;
            if (cudaGetMipmappedArrayLevel(&level_array, array, level) != cudaSuccess) {
                release_texture_objects(arrays, objects);
                return false;
            }
            const size_t pitch = static_cast<size_t>(level_width) * sizeof(float4);
            if (cudaMemcpy2DToArray(level_array, 0, 0, pixels.data(), pitch, pitch, static_cast<size_t>(level_height), cudaMemcpyHostToDevice) != cudaSuccess) {
                release_texture_objects(arrays, objects);
                return false;
            }
        }

        cudaResourceDesc resource_desc{};
        resource_desc.resType = cudaResourceTypeMipmappedArray;
        resource_desc.res.mipmap.mipmap = array;
        cudaTextureDesc texture_desc{};
        texture_desc.addressMode[0] = cudaAddressModeClamp;
        texture_desc.addressMode[1] = cudaAddressModeClamp;
        texture_desc.filterMode = cudaFilterModeLinear;
        texture_desc.mipmapFilterMode = cudaFilterModeLinear;
        texture_desc.minMipmapLevelClamp = 0.0f;
        texture_desc.maxMipmapLevelClamp = static_cast<float>(mip_levels - 1u);
        texture_desc.readMode = cudaReadModeElementType;
        texture_desc.normalizedCoords = 1;
        cudaTextureObject_t object = 0;
        if (cudaCreateTextureObject(&object, &resource_desc, &texture_desc, nullptr) != cudaSuccess) {
            release_texture_objects(arrays, objects);
            return false;
        }
        objects[static_cast<size_t>(i)] = static_cast<uint64_t>(object);
        packed.textures[static_cast<size_t>(i)].object = object;
    }
    return true;
}

bool apply_cached_texture_objects(PackedGpuScene& packed, const std::vector<uint64_t>& objects) {
    if (objects.size() != packed.textures.size()) {
        return false;
    }
    for (size_t i = 0; i < packed.textures.size(); ++i) {
        packed.textures[i].object = static_cast<cudaTextureObject_t>(objects[i]);
    }
    return true;
}

template <typename T>
bool upload_scene_field(GpuScene* device_scene, size_t offset, const T& value) {
    char* base = reinterpret_cast<char*>(device_scene);
    return cudaMemcpy(base + offset, &value, sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess;
}

bool upload_camera(GpuScene* device_scene, const Camera& camera) {
    return upload_scene_field(device_scene, offsetof(GpuScene, camera), camera);
}

bool upload_environment(GpuScene* device_scene, const GpuScene& scene) {
    return upload_scene_field(device_scene, offsetof(GpuScene, environment_texture), scene.environment_texture) &&
           upload_scene_field(device_scene, offsetof(GpuScene, environment_color), scene.environment_color) &&
           upload_scene_field(device_scene, offsetof(GpuScene, environment_strength), scene.environment_strength) &&
           upload_scene_field(device_scene, offsetof(GpuScene, environment_constant), scene.environment_constant) &&
           upload_scene_field(device_scene, offsetof(GpuScene, environment_mapping), scene.environment_mapping) &&
           upload_scene_field(device_scene, offsetof(GpuScene, environment_light_from_world_x), scene.environment_light_from_world_x) &&
           upload_scene_field(device_scene, offsetof(GpuScene, environment_light_from_world_y), scene.environment_light_from_world_y) &&
           upload_scene_field(device_scene, offsetof(GpuScene, environment_light_from_world_z), scene.environment_light_from_world_z);
}

bool make_environment_gpu(const Scene& scene, GpuScene& gpu) {
    gpu.environment_texture = -1;
    gpu.environment_color = scene.environment.color;
    gpu.environment_strength = scene.environment.strength;
    gpu.environment_constant = scene.environment.constant;
    gpu.environment_mapping = static_cast<int>(scene.environment.mapping);
    gpu.environment_light_from_world_x = scene.environment.light_from_world_x;
    gpu.environment_light_from_world_y = scene.environment.light_from_world_y;
    gpu.environment_light_from_world_z = scene.environment.light_from_world_z;
    if (!scene.environment.texture) {
        return true;
    }
    for (int i = 0; i < static_cast<int>(scene.textures.size()); ++i) {
        if (scene.textures[static_cast<size_t>(i)] == scene.environment.texture) {
            gpu.environment_texture = i;
            return true;
        }
    }
    return false;
}
