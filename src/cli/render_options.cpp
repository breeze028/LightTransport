#include "render_options.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <ostream>
#include <utility>

namespace lt::cli {
namespace {

template <size_t N>
void copy_setting_text(char (&target)[N], const std::string& value) {
    std::snprintf(target, N, "%s", value.c_str());
}

void apply_style(const RenderOptions& options, std::shared_ptr<Material>& material, NprStyle style) {
    if (!material) {
        return;
    }
    material->npr.style = style;
    if (style == NprStyle::ColorMap) {
        material->npr.value_min = options.global_color_min;
        material->npr.value_max = options.global_color_max;
    } else if (style == NprStyle::XToon) {
        material->npr.xtoon_detail_mode = options.global_xtoon_mode;
    } else if (style == NprStyle::CrossHatching) {
        material->npr.hatch_sets = options.global_hatch_sets;
        material->npr.hatch_spacing = options.global_hatch_spacing;
        material->npr.hatch_width = std::clamp(options.global_hatch_width, 0.0001f, options.global_hatch_spacing);
        material->npr.hatch_angle = options.global_hatch_angle;
        material->npr.hatch_value_min = options.global_color_min;
        material->npr.hatch_value_max = options.global_color_max;
        material->npr.hatch_ink = options.global_hatch_ink;
        material->npr.hatch_paper = options.global_hatch_paper;
        material->npr.hatch_passthrough = options.global_hatch_passthrough;
        material->npr.hatch_shadow_only = options.global_hatch_shadow_only;
    }
}

} // namespace

RenderOptions parse_render_options(int argc, char** argv) {
    RenderOptions options;
    options.scene_path = argc > 1 ? argv[1] : options.scene_path;
    options.output_path = argc > 2 ? argv[2] : options.output_path;
    options.settings.width = 1280;
    options.settings.height = 720;
    options.settings.samples_per_pixel = 1;
    options.settings.max_bounces = 6;

    for (int i = 3; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--cuda") {
            options.prefer_cuda = true;
        } else if (argument == "--cpu") {
            options.prefer_cuda = false;
        } else if (argument == "--verbose") {
            options.log_console_level = LogLevel::Debug;
            options.log_file_level = LogLevel::Debug;
        } else if (argument == "--quiet") {
            options.quiet = true;
            options.log_console_level = LogLevel::Warn;
        } else if (argument == "--log-level" && i + 1 < argc) {
            const LogLevel level = parse_log_level(argv[++i], options.log_console_level);
            options.log_console_level = level;
            options.log_file_level = level;
        } else if (argument == "--log-file" && i + 1 < argc) {
            options.log_file_enabled = true;
            options.log_file_path = argv[++i];
        } else if (argument == "--no-log-file") {
            options.log_file_enabled = false;
        } else if (argument == "--mis") {
            options.settings.use_mis = true;
        } else if (argument == "--no-mis") {
            options.settings.use_mis = false;
        } else if (argument == "--mis-heuristic" && i + 1 < argc) {
            const std::string heuristic = argv[++i];
            options.settings.mis_heuristic = heuristic == "balance" ? MisHeuristic::Balance : MisHeuristic::Power;
        } else if (argument == "--accel" && i + 1 < argc) {
            const std::string acceleration = argv[++i];
            if (acceleration == "two-level" || acceleration == "twolevel" || acceleration == "tlas") {
                options.settings.acceleration_structure = AccelerationStructure::TwoLevel;
            } else if (acceleration == "flat") {
                options.settings.acceleration_structure = AccelerationStructure::Flat;
            } else {
                options.settings.acceleration_structure = AccelerationStructure::Auto;
            }
        } else if (argument == "--spp" && i + 1 < argc) {
            options.settings.samples_per_pixel = std::max(1, std::atoi(argv[++i]));
        } else if (argument == "--max-bounces" && i + 1 < argc) {
            options.settings.max_bounces = std::clamp(std::atoi(argv[++i]), 1, 32);
        } else if (argument == "--frames" && i + 1 < argc) {
            options.settings.frame_index = static_cast<uint32_t>(std::max(0, std::atoi(argv[++i]) - 1));
        } else if (argument == "--size" && i + 2 < argc) {
            options.settings.width = std::max(1, std::atoi(argv[++i]));
            options.settings.height = std::max(1, std::atoi(argv[++i]));
        } else if (argument == "--irradiance-volume") {
            options.settings.use_irradiance_volume = true;
        } else if (argument == "--no-irradiance-volume") {
            options.settings.use_irradiance_volume = false;
        } else if (argument == "--ivol-grid" && i + 1 < argc) {
            options.settings.irradiance_volume_grid_resolution = std::max(2, std::atoi(argv[++i]));
        } else if (argument == "--ivol-subgrid" && i + 1 < argc) {
            options.settings.irradiance_volume_subgrid_resolution = std::max(2, std::atoi(argv[++i]));
        } else if (argument == "--ivol-dir" && i + 1 < argc) {
            options.settings.irradiance_volume_direction_resolution = std::max(1, std::atoi(argv[++i]));
        } else if (argument == "--ivol-bake-samples" && i + 1 < argc) {
            options.settings.irradiance_volume_bake_samples = std::max(1, std::atoi(argv[++i]));
        } else if (argument == "--ivol-bake-bounces" && i + 1 < argc) {
            options.settings.irradiance_volume_bake_bounces = std::max(1, std::atoi(argv[++i]));
        } else if (argument == "--ivol-bounds-inset" && i + 1 < argc) {
            options.settings.irradiance_volume_bounds_inset = std::clamp(static_cast<float>(std::atof(argv[++i])), 0.0f, 0.45f);
        } else if (argument == "--ivol-principled-gi") {
            options.settings.irradiance_volume_principled_gi = true;
        } else if (argument == "--no-ivol-principled-gi") {
            options.settings.irradiance_volume_principled_gi = false;
        } else if (argument == "--ivol-debug-probes") {
            options.settings.irradiance_volume_debug_probes = true;
        } else if (argument == "--no-ivol-debug-probes") {
            options.settings.irradiance_volume_debug_probes = false;
        } else if (argument == "--ivol-probe-radius-scale" && i + 1 < argc) {
            options.settings.irradiance_volume_debug_probe_radius_scale = std::clamp(static_cast<float>(std::atof(argv[++i])), 0.0f, 2.0f);
        } else if (argument == "--ivol-cache" && i + 1 < argc) {
            options.settings.irradiance_volume_cache_enabled = true;
            copy_setting_text(options.settings.irradiance_volume_cache_path, argv[++i]);
        } else if (argument == "--no-ivol-cache") {
            options.settings.irradiance_volume_cache_enabled = false;
        } else if (argument == "--ivol-auto-update") {
            options.settings.irradiance_volume_auto_update = true;
        } else if (argument == "--no-ivol-auto-update") {
            options.settings.irradiance_volume_auto_update = false;
        } else if (argument == "--ivol-force-bake") {
            options.settings.irradiance_volume_force_rebake = true;
        } else if (argument == "--ivol-bounds" && i + 6 < argc) {
            options.settings.irradiance_volume_manual_bounds = true;
            options.settings.irradiance_volume_bounds_min = {
                static_cast<float>(std::atof(argv[++i])),
                static_cast<float>(std::atof(argv[++i])),
                static_cast<float>(std::atof(argv[++i])),
            };
            options.settings.irradiance_volume_bounds_max = {
                static_cast<float>(std::atof(argv[++i])),
                static_cast<float>(std::atof(argv[++i])),
                static_cast<float>(std::atof(argv[++i])),
            };
        } else if (argument == "--lightmap") {
            options.settings.use_lightmap = true;
        } else if (argument == "--no-lightmap") {
            options.settings.use_lightmap = false;
        } else if (argument == "--lightmap-resolution" && i + 1 < argc) {
            options.settings.lightmap_resolution = std::clamp(std::atoi(argv[++i]), 16, 16384);
        } else if (argument == "--lightmap-padding" && i + 1 < argc) {
            options.settings.lightmap_padding = std::clamp(std::atoi(argv[++i]), 0, 64);
        } else if (argument == "--lightmap-dilation" && i + 1 < argc) {
            options.settings.lightmap_dilation = std::clamp(std::atoi(argv[++i]), 0, 64);
        } else if (argument == "--lightmap-bake-samples" && i + 1 < argc) {
            options.settings.lightmap_bake_samples = std::max(1, std::atoi(argv[++i]));
        } else if (argument == "--lightmap-bake-bounces" && i + 1 < argc) {
            options.settings.lightmap_bake_bounces = std::max(1, std::atoi(argv[++i]));
        } else if (argument == "--lightmap-principled-gi") {
            options.settings.lightmap_principled_gi = true;
        } else if (argument == "--no-lightmap-principled-gi") {
            options.settings.lightmap_principled_gi = false;
        } else if (argument == "--lightmap-cache" && i + 1 < argc) {
            options.settings.lightmap_cache_enabled = true;
            copy_setting_text(options.settings.lightmap_cache_path, argv[++i]);
        } else if (argument == "--no-lightmap-cache") {
            options.settings.lightmap_cache_enabled = false;
        } else if (argument == "--lightmap-auto-update") {
            options.settings.lightmap_auto_update = true;
        } else if (argument == "--no-lightmap-auto-update") {
            options.settings.lightmap_auto_update = false;
        } else if (argument == "--lightmap-force-bake") {
            options.settings.lightmap_force_rebake = true;
        } else if (argument == "--style" && i + 1 < argc) {
            options.global_style = parse_npr_style(argv[++i]);
            options.global_style_set = true;
        } else if (argument == "--material-style" && i + 2 < argc) {
            MaterialStyleOverride override;
            override.material = argv[++i];
            override.style = parse_npr_style(argv[++i]);
            options.material_styles.push_back(std::move(override));
        } else if (argument == "--style-samples" && i + 1 < argc) {
            options.settings.stylized_samples = std::max(1, std::atoi(argv[++i]));
        } else if (argument == "--style-depth" && i + 1 < argc) {
            options.settings.stylized_max_depth = std::max(0, std::atoi(argv[++i]));
        } else if (argument == "--style-range" && i + 2 < argc) {
            options.global_color_min = static_cast<float>(std::atof(argv[++i]));
            options.global_color_max = std::max(static_cast<float>(std::atof(argv[++i])), options.global_color_min + 1.0e-4f);
        } else if (argument == "--xtoon-mode" && i + 1 < argc) {
            options.global_xtoon_mode = parse_xtoon_detail_mode(argv[++i]);
        } else if (argument == "--hatch-sets" && i + 1 < argc) {
            options.global_hatch_sets = std::clamp(std::atoi(argv[++i]), 1, 8);
        } else if (argument == "--hatch-spacing" && i + 1 < argc) {
            options.global_hatch_spacing = std::max(0.001f, static_cast<float>(std::atof(argv[++i])));
            options.global_hatch_width = std::min(options.global_hatch_width, options.global_hatch_spacing);
        } else if (argument == "--hatch-width" && i + 1 < argc) {
            options.global_hatch_width = std::max(0.0001f, static_cast<float>(std::atof(argv[++i])));
        } else if (argument == "--hatch-angle" && i + 1 < argc) {
            options.global_hatch_angle = static_cast<float>(std::atof(argv[++i]));
        } else if (argument == "--hatch-ink" && i + 3 < argc) {
            options.global_hatch_ink = {
                static_cast<float>(std::atof(argv[++i])),
                static_cast<float>(std::atof(argv[++i])),
                static_cast<float>(std::atof(argv[++i])),
            };
        } else if (argument == "--hatch-paper" && i + 3 < argc) {
            options.global_hatch_paper = {
                static_cast<float>(std::atof(argv[++i])),
                static_cast<float>(std::atof(argv[++i])),
                static_cast<float>(std::atof(argv[++i])),
            };
        } else if (argument == "--hatch-passthrough") {
            options.global_hatch_passthrough = true;
        } else if (argument == "--hatch-shadow-only") {
            options.global_hatch_shadow_only = true;
        }
    }
    return options;
}

void apply_material_styles(const RenderOptions& options, Scene& scene, std::ostream& errors) {
    if (options.global_style_set) {
        for (std::shared_ptr<Material>& material : scene.materials) {
            apply_style(options, material, options.global_style);
        }
    }
    for (const MaterialStyleOverride& override : options.material_styles) {
        const int material_index = find_material(scene, override.material);
        if (material_index < 0) {
            LT_LOG_WARN("Material not found for --material-style: {}", override.material);
            errors << "Material not found for --material-style: " << override.material << "\n";
            continue;
        }
        apply_style(options, scene.materials[static_cast<size_t>(material_index)], override.style);
    }
}

} // namespace lt::cli
