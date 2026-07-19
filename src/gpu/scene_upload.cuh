bool size_fits_int(size_t size) {
    return size <= static_cast<size_t>(std::numeric_limits<int>::max());
}

bool use_two_level_accel(const RenderScene& render_scene, AccelerationStructure acceleration_structure) {
    (void)render_scene;
    return acceleration_structure == AccelerationStructure::TwoLevel;
}

bool use_wavefront_bvh8_layout(const RenderScene& render_scene, const RenderSettings& settings) {
    if (!settings.cuda_wavefront || !use_two_level_accel(render_scene, settings.acceleration_structure)) {
        return false;
    }
    constexpr size_t kMaxEstimatedBvh8Bytes = 64ull * 1024ull * 1024ull;
    const size_t estimated_wide_nodes = (render_scene.bvh_nodes.size() + 1u) / 2u;
    const size_t estimated_bytes = estimated_wide_nodes * sizeof(GpuTraversalBvh8Node);
    return estimated_bytes > 0 && estimated_bytes <= kMaxEstimatedBvh8Bytes;
}

bool use_wavefront_cwbvh_layout(const RenderScene& render_scene, const RenderSettings& settings) {
    // CWBVH v1 is kept as an internal experiment, but is not the default
    // wavefront layout yet. Current Sponza profiling shows it reduces register
    // count but increases intersect frame time, so keep the production path on
    // the full-float BVH8 baseline until the remaining CWBVH traversal pieces
    // are implemented and pass the >=5% median / <=3% P95 acceptance gate.
    constexpr bool kEnableExperimentalCwBvh = true;
    if (!kEnableExperimentalCwBvh) {
        (void)render_scene;
        (void)settings;
        return false;
    }
    if (!settings.cuda_wavefront || !use_two_level_accel(render_scene, settings.acceleration_structure)) {
        return false;
    }
    constexpr size_t kMaxEstimatedCwBvhBytes = 64ull * 1024ull * 1024ull;
    const size_t estimated_wide_nodes = (render_scene.bvh_nodes.size() + 1u) / 2u;
    const size_t estimated_bytes = estimated_wide_nodes * sizeof(GpuCwBvhNode) +
        render_scene.triangle_indices.size() * 3u * sizeof(float4);
    return estimated_bytes > 0 && estimated_bytes <= kMaxEstimatedCwBvhBytes;
}

float pack_bvh_surface_area(const Aabb& bounds) {
    const Vec3 extent = bounds.max - bounds.min;
    return std::max(0.0f, 2.0f * (extent.x * extent.y + extent.y * extent.z + extent.z * extent.x));
}

int pack_bvh_child_octant(const Aabb& parent_bounds, const Aabb& child_bounds) {
    const Vec3 parent_center = (parent_bounds.min + parent_bounds.max) * 0.5f;
    const Vec3 child_center = (child_bounds.min + child_bounds.max) * 0.5f;
    int octant = 0;
    if (child_center.x >= parent_center.x) octant |= 1;
    if (child_center.y >= parent_center.y) octant |= 2;
    if (child_center.z >= parent_center.z) octant |= 4;
    return octant;
}

int build_traversal_bvh8_node(
    const std::vector<BvhNode>& nodes,
    int binary_root,
    std::vector<int>& binary_to_wide,
    std::vector<GpuTraversalBvh8Node>& wide_nodes)
{
    if (binary_root < 0 || binary_root >= static_cast<int>(nodes.size())) {
        return -1;
    }
    int& cached = binary_to_wide[static_cast<size_t>(binary_root)];
    if (cached >= 0) {
        return cached;
    }

    const int wide_index = static_cast<int>(wide_nodes.size());
    cached = wide_index;
    wide_nodes.push_back({});

    std::vector<int> children;
    const BvhNode& root = nodes[static_cast<size_t>(binary_root)];
    if (root.count > 0) {
        children.push_back(binary_root);
    } else {
        if (root.left >= 0) children.push_back(root.left);
        if (root.right >= 0) children.push_back(root.right);
    }

    while (children.size() < 8) {
        int expand_slot = -1;
        float expand_area = -1.0f;
        for (int i = 0; i < static_cast<int>(children.size()); ++i) {
            const int child_index = children[static_cast<size_t>(i)];
            if (child_index < 0 || child_index >= static_cast<int>(nodes.size())) {
                continue;
            }
            const BvhNode& child = nodes[static_cast<size_t>(child_index)];
            if (child.count > 0 || child.left < 0 || child.right < 0) {
                continue;
            }
            const float area = pack_bvh_surface_area(child.bounds);
            if (area > expand_area) {
                expand_area = area;
                expand_slot = i;
            }
        }
        if (expand_slot < 0) {
            break;
        }
        const BvhNode& child = nodes[static_cast<size_t>(children[static_cast<size_t>(expand_slot)])];
        children.erase(children.begin() + expand_slot);
        children.insert(children.begin() + expand_slot, child.right);
        children.insert(children.begin() + expand_slot, child.left);
    }

    for (int slot = 0; slot < static_cast<int>(children.size()) && slot < 8; ++slot) {
        const int child_index = children[static_cast<size_t>(slot)];
        if (child_index < 0 || child_index >= static_cast<int>(nodes.size())) {
            continue;
        }
        const BvhNode& child = nodes[static_cast<size_t>(child_index)];
        GpuTraversalBvh8Child gpu_child;
        const int child_octant = pack_bvh_child_octant(root.bounds, child.bounds);
        gpu_child.bounds_min = child.bounds.min;
        gpu_child.bounds_max = child.bounds.max;
        int leaf_mask = 0;
        if (child.count > 0) {
            leaf_mask = 1 << slot;
            gpu_child.index = child.first;
            gpu_child.count = child.count;
        } else {
            gpu_child.index = build_traversal_bvh8_node(nodes, child_index, binary_to_wide, wide_nodes);
            gpu_child.count = 0;
        }

        GpuTraversalBvh8Node& wide = wide_nodes[static_cast<size_t>(wide_index)];
        wide.valid_mask |= 1 << slot;
        wide.leaf_mask |= leaf_mask;
        wide.child_octants |= static_cast<unsigned int>(child_octant & 7) << (slot * 3);
        wide.children[slot] = gpu_child;
    }
    return wide_index;
}

#if 0
struct CwBvhBuildChild {
    Aabb bounds;
    int binary_node = -1;
    int first = 0;
    int count = 0;
    bool leaf = false;
};

int cwbvh_effective_child_slots(const BvhNode& node) {
    if (node.count > 0) {
        return (node.count + 2) / 3;
    }
    return 1;
}

Aabb cwbvh_triangle_span_bounds(
    const std::vector<Triangle>& triangles,
    const std::vector<int>& indices,
    int first,
    int count)
{
    Aabb bounds;
    for (int i = 0; i < count; ++i) {
        const int index_offset = first + i;
        if (index_offset < 0 || index_offset >= static_cast<int>(indices.size())) {
            continue;
        }
        const int tri_index = indices[static_cast<size_t>(index_offset)];
        if (tri_index < 0 || tri_index >= static_cast<int>(triangles.size())) {
            continue;
        }
        const Triangle& tri = triangles[static_cast<size_t>(tri_index)];
        bounds.min = min(bounds.min, min(tri.v0, min(tri.v1, tri.v2)));
        bounds.max = max(bounds.max, max(tri.v0, max(tri.v1, tri.v2)));
    }
    constexpr float kBoundsEpsilon = 1.0e-4f;
    bounds.min = bounds.min - Vec3{kBoundsEpsilon};
    bounds.max = bounds.max + Vec3{kBoundsEpsilon};
    return bounds;
}

void cwbvh_append_child(
    const std::vector<Triangle>& triangles,
    const std::vector<int>& indices,
    const std::vector<BvhNode>& nodes,
    int binary_node,
    std::vector<CwBvhBuildChild>& output)
{
    if (binary_node < 0 || binary_node >= static_cast<int>(nodes.size())) {
        return;
    }
    const BvhNode& node = nodes[static_cast<size_t>(binary_node)];
    if (node.count <= 0) {
        output.push_back({node.bounds, binary_node, 0, 0, false});
        return;
    }
    for (int offset = 0; offset < node.count; offset += 3) {
        const int count = std::min(3, node.count - offset);
        output.push_back({
            cwbvh_triangle_span_bounds(triangles, indices, node.first + offset, count),
            binary_node,
            node.first + offset,
            count,
            true,
        });
    }
}

std::vector<CwBvhBuildChild> cwbvh_collect_children(
    const std::vector<Triangle>& triangles,
    const std::vector<int>& indices,
    const std::vector<BvhNode>& nodes,
    int binary_root)
{
    std::vector<int> frontier;
    if (binary_root < 0 || binary_root >= static_cast<int>(nodes.size())) {
        return {};
    }
    const BvhNode& root = nodes[static_cast<size_t>(binary_root)];
    if (root.count > 0) {
        std::vector<CwBvhBuildChild> leaf_children;
        cwbvh_append_child(triangles, indices, nodes, binary_root, leaf_children);
        return leaf_children;
    }
    if (root.left >= 0) frontier.push_back(root.left);
    if (root.right >= 0) frontier.push_back(root.right);

    auto slot_count = [&]() {
        int count = 0;
        for (int child : frontier) {
            if (child >= 0 && child < static_cast<int>(nodes.size())) {
                count += cwbvh_effective_child_slots(nodes[static_cast<size_t>(child)]);
            }
        }
        return count;
    };

    while (slot_count() < 8) {
        int expand_slot = -1;
        float expand_area = -1.0f;
        const int available_slots = 8 - slot_count();
        for (int i = 0; i < static_cast<int>(frontier.size()); ++i) {
            const int child_index = frontier[static_cast<size_t>(i)];
            if (child_index < 0 || child_index >= static_cast<int>(nodes.size())) {
                continue;
            }
            const BvhNode& child = nodes[static_cast<size_t>(child_index)];
            if (child.count > 0 || child.left < 0 || child.right < 0) {
                continue;
            }
            int replacement_slots = 0;
            if (child.left >= 0) replacement_slots += cwbvh_effective_child_slots(nodes[static_cast<size_t>(child.left)]);
            if (child.right >= 0) replacement_slots += cwbvh_effective_child_slots(nodes[static_cast<size_t>(child.right)]);
            const int delta = replacement_slots - cwbvh_effective_child_slots(child);
            if (delta > available_slots) {
                continue;
            }
            const float area = pack_bvh_surface_area(child.bounds);
            if (area > expand_area) {
                expand_area = area;
                expand_slot = i;
            }
        }
        if (expand_slot < 0) {
            break;
        }
        const BvhNode& child = nodes[static_cast<size_t>(frontier[static_cast<size_t>(expand_slot)])];
        frontier.erase(frontier.begin() + expand_slot);
        if (child.right >= 0) frontier.insert(frontier.begin() + expand_slot, child.right);
        if (child.left >= 0) frontier.insert(frontier.begin() + expand_slot, child.left);
    }

    std::vector<CwBvhBuildChild> children;
    for (int child : frontier) {
        cwbvh_append_child(triangles, indices, nodes, child, children);
    }
    if (children.size() > 8) {
        children.resize(8);
    }
    return children;
}

int cwbvh_slot_assignment(const Aabb& parent_bounds, const Aabb& child_bounds, int slot) {
    const Vec3 parent_centroid = (parent_bounds.min + parent_bounds.max) * 0.5f;
    const Vec3 child_centroid = (child_bounds.min + child_bounds.max) * 0.5f;
    const Vec3 delta = child_centroid - parent_centroid;
    const Vec3 direction = {
        ((slot >> 2) & 1) ? -1.0f : 1.0f,
        ((slot >> 1) & 1) ? -1.0f : 1.0f,
        ((slot >> 0) & 1) ? -1.0f : 1.0f,
    };
    return static_cast<int>((delta.x * direction.x + delta.y * direction.y + delta.z * direction.z) * 1024.0f);
}

std::array<int, 8> cwbvh_assign_slots(const Aabb& parent_bounds, const std::vector<CwBvhBuildChild>& children) {
    std::array<int, 8> slot_to_child;
    slot_to_child.fill(-1);
    std::array<int, 8> child_to_slot;
    child_to_slot.fill(-1);
    std::array<bool, 8> slot_empty;
    slot_empty.fill(true);

    while (true) {
        int best_slot = -1;
        int best_child = -1;
        int best_score = std::numeric_limits<int>::min();
        for (int slot = 0; slot < 8; ++slot) {
            if (!slot_empty[static_cast<size_t>(slot)]) {
                continue;
            }
            for (int child = 0; child < static_cast<int>(children.size()) && child < 8; ++child) {
                if (child_to_slot[static_cast<size_t>(child)] >= 0) {
                    continue;
                }
                const int score = cwbvh_slot_assignment(parent_bounds, children[static_cast<size_t>(child)].bounds, slot);
                if (score > best_score) {
                    best_score = score;
                    best_slot = slot;
                    best_child = child;
                }
            }
        }
        if (best_slot < 0 || best_child < 0) {
            break;
        }
        slot_empty[static_cast<size_t>(best_slot)] = false;
        child_to_slot[static_cast<size_t>(best_child)] = best_slot;
        slot_to_child[static_cast<size_t>(best_slot)] = best_child;
    }
    return slot_to_child;
}

int cwbvh_quant_exponent(float extent) {
    const float safe_extent = std::max(extent, 1.0e-20f);
    const int exponent = static_cast<int>(std::ceil(std::log2(safe_extent / 255.0f)));
    return std::clamp(exponent, -126, 127);
}

unsigned int cwbvh_pack_exponents_and_mask(int ex, int ey, int ez, unsigned int imask) {
    const unsigned int ux = static_cast<unsigned int>(static_cast<unsigned char>(static_cast<signed char>(ex)));
    const unsigned int uy = static_cast<unsigned int>(static_cast<unsigned char>(static_cast<signed char>(ey)));
    const unsigned int uz = static_cast<unsigned int>(static_cast<unsigned char>(static_cast<signed char>(ez)));
    return ux | (uy << 8) | (uz << 16) | ((imask & 0xffu) << 24);
}

unsigned char cwbvh_meta_at(const GpuCwBvhNode& node, int slot) {
    return static_cast<unsigned char>((node.meta4[slot >> 2] >> ((slot & 3) * 8)) & 0xffu);
}

void cwbvh_set_meta(GpuCwBvhNode& node, int slot, unsigned char meta) {
    const int word = slot >> 2;
    const int shift = (slot & 3) * 8;
    node.meta4[word] &= ~(0xffu << shift);
    node.meta4[word] |= static_cast<unsigned int>(meta) << shift;
}

bool build_cwbvh_node_at(
    const std::vector<Triangle>& triangles,
    const std::vector<int>& indices,
    const std::vector<BvhNode>& source_nodes,
    int binary_root,
    int cwbvh_index,
    std::vector<int>& binary_to_cwbvh,
    std::vector<GpuCwBvhNode>& cwbvh_nodes,
    std::vector<int>& cwbvh_triangle_indices)
{
    if (binary_root < 0 || binary_root >= static_cast<int>(source_nodes.size()) ||
        cwbvh_index < 0 || cwbvh_index >= static_cast<int>(cwbvh_nodes.size())) {
        return false;
    }

    binary_to_cwbvh[static_cast<size_t>(binary_root)] = cwbvh_index;
    const BvhNode& source = source_nodes[static_cast<size_t>(binary_root)];
    std::vector<CwBvhBuildChild> children = cwbvh_collect_children(triangles, indices, source_nodes, binary_root);
    if (children.empty() || children.size() > 8) {
        return false;
    }

    GpuCwBvhNode node;
    node.bounds_min = source.bounds.min;
    const std::array<int, 8> slot_to_child = cwbvh_assign_slots(source.bounds, children);

    int ex = cwbvh_quant_exponent(source.bounds.max.x - source.bounds.min.x);
    int ey = cwbvh_quant_exponent(source.bounds.max.y - source.bounds.min.y);
    int ez = cwbvh_quant_exponent(source.bounds.max.z - source.bounds.min.z);
    std::array<int, 8> qlo_x{};
    std::array<int, 8> qlo_y{};
    std::array<int, 8> qlo_z{};
    std::array<int, 8> qhi_x{};
    std::array<int, 8> qhi_y{};
    std::array<int, 8> qhi_z{};
    for (int attempt = 0; attempt < 16; ++attempt) {
        bool fits = true;
        const float sx = std::ldexp(1.0f, ex);
        const float sy = std::ldexp(1.0f, ey);
        const float sz = std::ldexp(1.0f, ez);
        for (int slot = 0; slot < 8; ++slot) {
            const int child_index = slot_to_child[static_cast<size_t>(slot)];
            if (child_index < 0) {
                continue;
            }
            const Aabb& bounds = children[static_cast<size_t>(child_index)].bounds;
            qlo_x[static_cast<size_t>(slot)] = static_cast<int>(std::floor((bounds.min.x - source.bounds.min.x) / sx));
            qlo_y[static_cast<size_t>(slot)] = static_cast<int>(std::floor((bounds.min.y - source.bounds.min.y) / sy));
            qlo_z[static_cast<size_t>(slot)] = static_cast<int>(std::floor((bounds.min.z - source.bounds.min.z) / sz));
            qhi_x[static_cast<size_t>(slot)] = static_cast<int>(std::ceil((bounds.max.x - source.bounds.min.x) / sx));
            qhi_y[static_cast<size_t>(slot)] = static_cast<int>(std::ceil((bounds.max.y - source.bounds.min.y) / sy));
            qhi_z[static_cast<size_t>(slot)] = static_cast<int>(std::ceil((bounds.max.z - source.bounds.min.z) / sz));
            fits = fits &&
                qlo_x[static_cast<size_t>(slot)] >= 0 && qlo_x[static_cast<size_t>(slot)] <= 255 &&
                qlo_y[static_cast<size_t>(slot)] >= 0 && qlo_y[static_cast<size_t>(slot)] <= 255 &&
                qlo_z[static_cast<size_t>(slot)] >= 0 && qlo_z[static_cast<size_t>(slot)] <= 255 &&
                qhi_x[static_cast<size_t>(slot)] >= 0 && qhi_x[static_cast<size_t>(slot)] <= 255 &&
                qhi_y[static_cast<size_t>(slot)] >= 0 && qhi_y[static_cast<size_t>(slot)] <= 255 &&
                qhi_z[static_cast<size_t>(slot)] >= 0 && qhi_z[static_cast<size_t>(slot)] <= 255;
        }
        if (fits) {
            break;
        }
        ex = std::min(ex + 1, 127);
        ey = std::min(ey + 1, 127);
        ez = std::min(ez + 1, 127);
        if (attempt == 15) {
            return false;
        }
    }

    unsigned int imask = 0;
    int internal_count = 0;
    for (int slot = 0; slot < 8; ++slot) {
        const int child_index = slot_to_child[static_cast<size_t>(slot)];
        if (child_index >= 0 && !children[static_cast<size_t>(child_index)].leaf) {
            if (internal_count == 0) {
                node.child_base = static_cast<int>(cwbvh_nodes.size());
            }
            ++internal_count;
            imask |= 1u << slot;
        }
    }
    if (internal_count > 0) {
        cwbvh_nodes.resize(cwbvh_nodes.size() + static_cast<size_t>(internal_count));
    }

    int leaf_triangle_count = 0;
    node.triangle_base = static_cast<int>(cwbvh_triangle_indices.size());
    for (int slot = 0; slot < 8; ++slot) {
        const int child_index = slot_to_child[static_cast<size_t>(slot)];
        if (child_index < 0) {
            continue;
        }
        const CwBvhBuildChild& child = children[static_cast<size_t>(child_index)];
        node.qlo_x[slot] = static_cast<unsigned char>(std::clamp(qlo_x[static_cast<size_t>(slot)], 0, 255));
        node.qlo_y[slot] = static_cast<unsigned char>(std::clamp(qlo_y[static_cast<size_t>(slot)], 0, 255));
        node.qlo_z[slot] = static_cast<unsigned char>(std::clamp(qlo_z[static_cast<size_t>(slot)], 0, 255));
        node.qhi_x[slot] = static_cast<unsigned char>(std::clamp(qhi_x[static_cast<size_t>(slot)], 0, 255));
        node.qhi_y[slot] = static_cast<unsigned char>(std::clamp(qhi_y[static_cast<size_t>(slot)], 0, 255));
        node.qhi_z[slot] = static_cast<unsigned char>(std::clamp(qhi_z[static_cast<size_t>(slot)], 0, 255));
        if (child.leaf) {
            if (child.count <= 0 || child.count > 3 || leaf_triangle_count + child.count > 31) {
                return false;
            }
            const int unary_count = child.count == 1 ? 0b001 : child.count == 2 ? 0b011 : 0b111;
            cwbvh_set_meta(node, slot, static_cast<unsigned char>((unary_count << 5) | leaf_triangle_count));
            for (int tri = 0; tri < child.count; ++tri) {
                const int source_index = child.first + tri;
                if (source_index < 0 || source_index >= static_cast<int>(indices.size())) {
                    return false;
                }
                cwbvh_triangle_indices.push_back(indices[static_cast<size_t>(source_index)]);
            }
            leaf_triangle_count += child.count;
        }
    }

    int relative_internal = 0;
    for (int slot = 0; slot < 8; ++slot) {
        const int child_index = slot_to_child[static_cast<size_t>(slot)];
        if (child_index < 0) {
            continue;
        }
        const CwBvhBuildChild& child = children[static_cast<size_t>(child_index)];
        if (!child.leaf) {
            cwbvh_set_meta(node, slot, static_cast<unsigned char>((1u << 5) | (24u + static_cast<unsigned int>(slot))));
            const int child_cwbvh_index = node.child_base + relative_internal++;
            if (!build_cwbvh_node_at(triangles, indices, source_nodes, child.binary_node, child_cwbvh_index,
                    binary_to_cwbvh, cwbvh_nodes, cwbvh_triangle_indices)) {
                return false;
            }
        }
    }
    node.exyz_imask = cwbvh_pack_exponents_and_mask(ex, ey, ez, imask);
    cwbvh_nodes[static_cast<size_t>(cwbvh_index)] = node;
    return true;
}

int build_traversal_cwbvh_node(
    const std::vector<Triangle>& triangles,
    const std::vector<int>& indices,
    const std::vector<BvhNode>& source_nodes,
    int binary_root,
    std::vector<int>& binary_to_cwbvh,
    std::vector<GpuCwBvhNode>& cwbvh_nodes,
    std::vector<int>& cwbvh_triangle_indices)
{
    if (binary_root < 0 || binary_root >= static_cast<int>(source_nodes.size())) {
        return -1;
    }
    int& cached = binary_to_cwbvh[static_cast<size_t>(binary_root)];
    if (cached >= 0) {
        return cached;
    }
    const int cwbvh_index = static_cast<int>(cwbvh_nodes.size());
    cwbvh_nodes.push_back({});
    if (!build_cwbvh_node_at(triangles, indices, source_nodes, binary_root, cwbvh_index,
            binary_to_cwbvh, cwbvh_nodes, cwbvh_triangle_indices)) {
        return -1;
    }
    return cwbvh_index;
}
#endif

__host__ __forceinline__ uint32_t cwbvh_float_to_uint(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

__host__ __forceinline__ float cwbvh_uint_to_float(uint32_t value) {
    float bits = 0.0f;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float4 make_cwbvh_float4_from_tiny(const tinybvh::bvhvec4& v) {
    return make_float4(v.x, v.y, v.z, v.w);
}

bool build_tinybvh_cwbvh_node(
    const std::vector<Triangle>& triangles,
    const std::vector<int>& indices,
    const std::vector<BvhNode>& source_nodes,
    int binary_root,
    std::vector<int>& binary_to_cwbvh,
    std::vector<GpuCwBvhNode>& cwbvh_nodes,
    std::vector<float4>& cwbvh_triangles)
{
    if (binary_root < 0 || binary_root >= static_cast<int>(source_nodes.size())) {
        return false;
    }
    int& cached = binary_to_cwbvh[static_cast<size_t>(binary_root)];
    if (cached >= 0) {
        return true;
    }

    std::vector<int> subtree_triangle_indices;
    std::vector<int> stack;
    stack.push_back(binary_root);
    while (!stack.empty()) {
        const int node_index = stack.back();
        stack.pop_back();
        if (node_index < 0 || node_index >= static_cast<int>(source_nodes.size())) {
            return false;
        }
        const BvhNode& node = source_nodes[static_cast<size_t>(node_index)];
        if (node.count > 0) {
            if (node.first < 0 || node.first + node.count > static_cast<int>(indices.size())) {
                return false;
            }
            for (int i = 0; i < node.count; ++i) {
                subtree_triangle_indices.push_back(indices[static_cast<size_t>(node.first + i)]);
            }
        } else {
            if (node.right >= 0) stack.push_back(node.right);
            if (node.left >= 0) stack.push_back(node.left);
        }
    }

    std::vector<tinybvh::bvhvec4> vertices;
    vertices.reserve(subtree_triangle_indices.size() * 3u);
    std::vector<int> local_to_global;
    local_to_global.reserve(subtree_triangle_indices.size());
    for (int tri_index : subtree_triangle_indices) {
        if (tri_index < 0 || tri_index >= static_cast<int>(triangles.size())) {
            return false;
        }
        const Triangle& tri = triangles[static_cast<size_t>(tri_index)];
        vertices.emplace_back(tri.v0.x, tri.v0.y, tri.v0.z, 0.0f);
        vertices.emplace_back(tri.v1.x, tri.v1.y, tri.v1.z, 0.0f);
        vertices.emplace_back(tri.v2.x, tri.v2.y, tri.v2.z, 0.0f);
        local_to_global.push_back(tri_index);
    }
    if (local_to_global.empty()) {
        return false;
    }

    tinybvh::BVH8_CWBVH tiny_cwbvh;
    tiny_cwbvh.Build(vertices.data(), static_cast<uint32_t>(local_to_global.size()));
    if (tiny_cwbvh.bvh8Data == nullptr || tiny_cwbvh.bvh8Tris == nullptr || tiny_cwbvh.usedBlocks == 0 ||
        (tiny_cwbvh.usedBlocks % 5u) != 0u) {
        return false;
    }

    const int node_offset = static_cast<int>(cwbvh_nodes.size());
    const uint32_t node_float4_count = tiny_cwbvh.usedBlocks;
    const uint32_t node_count = node_float4_count / 5u;
    if (!size_fits_int(cwbvh_nodes.size() + static_cast<size_t>(node_count))) {
        return false;
    }
    cwbvh_nodes.resize(cwbvh_nodes.size() + static_cast<size_t>(node_count));
    for (uint32_t node = 0; node < node_count; ++node) {
        GpuCwBvhNode& dst = cwbvh_nodes[static_cast<size_t>(node_offset) + node];
        for (int block = 0; block < 5; ++block) {
            dst.block[block] = make_cwbvh_float4_from_tiny(tiny_cwbvh.bvh8Data[node * 5u + static_cast<uint32_t>(block)]);
        }
        uint32_t child_base = cwbvh_float_to_uint(dst.block[1].x);
        if (child_base != 0u) {
            child_base += static_cast<uint32_t>(node_offset);
            dst.block[1].x = cwbvh_uint_to_float(child_base);
        }
    }

    const uint32_t tri_float4_count = tiny_cwbvh.bvh8.idxCount * 3u;
    const uint32_t tri_float4_offset = static_cast<uint32_t>(cwbvh_triangles.size());
    if (!size_fits_int(cwbvh_triangles.size() + static_cast<size_t>(tri_float4_count))) {
        return false;
    }
    for (uint32_t i = 0; i < tri_float4_count; ++i) {
        cwbvh_triangles.push_back(make_cwbvh_float4_from_tiny(tiny_cwbvh.bvh8Tris[i]));
    }
    for (uint32_t tri = 0; tri + 2u < tri_float4_count; tri += 3u) {
        float4& v0 = cwbvh_triangles[static_cast<size_t>(tri_float4_offset + tri + 2u)];
        const uint32_t local_index = cwbvh_float_to_uint(v0.w);
        if (local_index >= local_to_global.size()) {
            return false;
        }
        v0.w = cwbvh_uint_to_float(static_cast<uint32_t>(local_to_global[static_cast<size_t>(local_index)]));
    }
    for (uint32_t node = 0; node < node_count; ++node) {
        GpuCwBvhNode& dst = cwbvh_nodes[static_cast<size_t>(node_offset) + node];
        uint32_t triangle_base = cwbvh_float_to_uint(dst.block[1].y);
        triangle_base += tri_float4_offset;
        dst.block[1].y = cwbvh_uint_to_float(triangle_base);
    }

    cached = node_offset;
    return true;
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
    if (!size_fits_int(scene.point_lights.size())) {
        return false;
    }
    gpu.point_light_count = static_cast<int>(scene.point_lights.size());
    packed.point_lights.resize(scene.point_lights.size());
    for (int i = 0; i < gpu.point_light_count; ++i) {
        const PointLight& light = scene.point_lights[static_cast<size_t>(i)];
        packed.point_lights[static_cast<size_t>(i)] = {light.position, light.color, light.intensity};
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
        } else if (const auto* diffuse_transmission = dynamic_cast<const DiffuseTransmissionMaterial*>(material.get())) {
            transmission_tint = diffuse_transmission->transmittance;
            transmission = std::clamp(std::max(transmission_tint.x, std::max(transmission_tint.y, transmission_tint.z)), 0.0f, 1.0f);
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
        } else if (const auto* diffuse_transmission = dynamic_cast<const DiffuseTransmissionMaterial*>(material.get())) {
            transmission_texture_index = find_texture_index(diffuse_transmission->transmittance_texture);
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
    const bool pack_wide_bvh = use_wavefront_bvh8_layout(render_scene, settings);
    const bool pack_cwbvh = use_wavefront_cwbvh_layout(render_scene, settings);
    gpu.triangle_count = static_cast<int>(render_scene.triangles.size());
    gpu.sphere_count = static_cast<int>(render_scene.spheres.size());
    gpu.bvh_node_count = gpu.use_two_level ? static_cast<int>(render_scene.bvh_nodes.size()) : static_cast<int>(render_scene.flat_bvh_nodes.size());
    gpu.mesh_instance_count = static_cast<int>(render_scene.mesh_instances.size());
    gpu.tlas_node_count = static_cast<int>(render_scene.tlas_nodes.size());
    packed.triangles.resize(render_scene.triangles.size());
    packed.traversal_triangles.resize(render_scene.triangles.size());
    packed.spheres.resize(render_scene.spheres.size());
    packed.triangle_indices = gpu.use_two_level ? render_scene.triangle_indices : render_scene.flat_triangle_indices;
    const std::vector<BvhNode>& source_bvh_nodes = gpu.use_two_level ? render_scene.bvh_nodes : render_scene.flat_bvh_nodes;
    packed.bvh_nodes.resize(source_bvh_nodes.size());
    packed.traversal_bvh_nodes.resize(source_bvh_nodes.size());
    std::vector<int> bvh8_root_map(source_bvh_nodes.size(), -1);
    if (pack_wide_bvh && !source_bvh_nodes.empty()) {
        for (const RenderScene::MeshInstance& instance : render_scene.mesh_instances) {
            if (instance.bvh_root >= 0 && instance.bvh_root < static_cast<int>(source_bvh_nodes.size())) {
                build_traversal_bvh8_node(source_bvh_nodes, instance.bvh_root, bvh8_root_map, packed.traversal_bvh8_nodes);
            }
        }
    }
    std::vector<int> cwbvh_root_map(source_bvh_nodes.size(), -1);
    if (pack_cwbvh && gpu.use_two_level && !source_bvh_nodes.empty()) {
        const size_t saved_node_count = packed.traversal_cwbvh_nodes.size();
        const size_t saved_triangle_count = packed.cwbvh_triangles.size();
        bool cwbvh_ok = true;
        for (const RenderScene::MeshInstance& instance : render_scene.mesh_instances) {
            if (instance.bvh_root >= 0 && instance.bvh_root < static_cast<int>(source_bvh_nodes.size()) &&
                !build_tinybvh_cwbvh_node(render_scene.triangles, packed.triangle_indices, source_bvh_nodes,
                    instance.bvh_root, cwbvh_root_map, packed.traversal_cwbvh_nodes,
                    packed.cwbvh_triangles)) {
                cwbvh_ok = false;
                break;
            }
        }
        if (!cwbvh_ok) {
            packed.traversal_cwbvh_nodes.resize(saved_node_count);
            packed.cwbvh_triangles.resize(saved_triangle_count);
            std::fill(cwbvh_root_map.begin(), cwbvh_root_map.end(), -1);
        }
    }
    gpu.bvh8_node_count = static_cast<int>(packed.traversal_bvh8_nodes.size());
    gpu.cwbvh_node_count = static_cast<int>(packed.traversal_cwbvh_nodes.size());
    gpu.cwbvh_triangle_index_count = static_cast<int>(packed.cwbvh_triangle_indices.size());
    gpu.cwbvh_triangle_count = static_cast<int>(packed.cwbvh_triangles.size());
    packed.mesh_instances.resize(render_scene.mesh_instances.size());
    packed.mesh_instance_indices = render_scene.mesh_instance_indices;
    packed.tlas_nodes.resize(render_scene.tlas_nodes.size());
    packed.traversal_tlas_nodes.resize(render_scene.tlas_nodes.size());
    const auto make_traversal_node = [](const BvhNode& node) {
        GpuTraversalBvhNode gpu_node;
        gpu_node.bounds_min = node.bounds.min;
        gpu_node.bounds_max = node.bounds.max;
        if (node.count > 0) {
            gpu_node.left_or_first = node.first;
            gpu_node.right_or_neg_count = -node.count;
        } else {
            gpu_node.left_or_first = node.left;
            gpu_node.right_or_neg_count = node.right;
        }
        return gpu_node;
    };
    for (size_t i = 0; i < source_bvh_nodes.size(); ++i) {
        const BvhNode& node = source_bvh_nodes[i];
        packed.bvh_nodes[i] = {node.bounds.min, node.bounds.max, node.left, node.right, node.first, node.count};
        packed.traversal_bvh_nodes[i] = make_traversal_node(node);
    }
    for (size_t i = 0; i < render_scene.mesh_instances.size(); ++i) {
        const RenderScene::MeshInstance& instance = render_scene.mesh_instances[i];
        const int bvh8_root = instance.bvh_root >= 0 && instance.bvh_root < static_cast<int>(bvh8_root_map.size())
            ? bvh8_root_map[static_cast<size_t>(instance.bvh_root)]
            : -1;
        const int cwbvh_root = instance.bvh_root >= 0 && instance.bvh_root < static_cast<int>(cwbvh_root_map.size())
            ? cwbvh_root_map[static_cast<size_t>(instance.bvh_root)]
            : -1;
        packed.mesh_instances[i] = {instance.bounds.min, instance.bounds.max, instance.bvh_root, bvh8_root, cwbvh_root, instance.mesh};
    }
    for (size_t i = 0; i < render_scene.tlas_nodes.size(); ++i) {
        const BvhNode& node = render_scene.tlas_nodes[i];
        packed.tlas_nodes[i] = {node.bounds.min, node.bounds.max, node.left, node.right, node.first, node.count};
        packed.traversal_tlas_nodes[i] = make_traversal_node(node);
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
        packed.traversal_triangles[static_cast<size_t>(i)] = {
            tri.v0,
            sub(tri.v1, tri.v0),
            sub(tri.v2, tri.v0),
            tri.material |
                (scene.materials[static_cast<size_t>(tri.material)]->alpha_mode != AlphaMode::Opaque
                    ? kTraversalMaterialAlphaBit
                    : 0),
        };
    }
    for (int tri_addr = 0; tri_addr + 2 < static_cast<int>(packed.cwbvh_triangles.size()); tri_addr += 3) {
        float4& edge2 = packed.cwbvh_triangles[static_cast<size_t>(tri_addr)];
        const float4& v0 = packed.cwbvh_triangles[static_cast<size_t>(tri_addr + 2)];
        const int tri_index = static_cast<int>(cwbvh_float_to_uint(v0.w));
        if (tri_index < 0 || tri_index >= gpu.triangle_count) {
            return false;
        }
        edge2.w = cwbvh_uint_to_float(static_cast<uint32_t>(
            packed.traversal_triangles[static_cast<size_t>(tri_index)].material_and_flags));
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
           upload_scene_field(device_scene, offsetof(GpuScene, environment_light_from_world_z), scene.environment_light_from_world_z) &&
           upload_scene_field(device_scene, offsetof(GpuScene, environment_sampler), scene.environment_sampler);
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
