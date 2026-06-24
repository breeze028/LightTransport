#include "render_options.h"

#include <algorithm>
#include <cstdlib>
#include <ostream>
#include <utility>

namespace lt::cli {
namespace {

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
        } else if (argument == "--frames" && i + 1 < argc) {
            options.settings.frame_index = static_cast<uint32_t>(std::max(0, std::atoi(argv[++i]) - 1));
        } else if (argument == "--size" && i + 2 < argc) {
            options.settings.width = std::max(1, std::atoi(argv[++i]));
            options.settings.height = std::max(1, std::atoi(argv[++i]));
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
            errors << "Material not found for --material-style: " << override.material << "\n";
            continue;
        }
        apply_style(options, scene.materials[static_cast<size_t>(material_index)], override.style);
    }
}

} // namespace lt::cli
