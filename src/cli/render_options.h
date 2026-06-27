#pragma once

#include "lt/log.h"
#include "lt/renderer.h"

#include <iosfwd>
#include <string>
#include <vector>

namespace lt::cli {

struct MaterialStyleOverride {
    std::string material;
    NprStyle style = NprStyle::None;
};

struct RenderOptions {
    std::string scene_path = "scenes/cornell.lt";
    std::string output_path = "out.ppm";
    RenderSettings settings;
    bool prefer_cuda = false;
    bool quiet = false;
    bool log_file_enabled = true;
    std::string log_file_path = "logs/lt_render.log";
    LogLevel log_console_level = LogLevel::Info;
    LogLevel log_file_level = LogLevel::Debug;
    bool global_style_set = false;
    NprStyle global_style = NprStyle::None;
    float global_color_min = 0.0f;
    float global_color_max = 1.0f;
    XToonDetailMode global_xtoon_mode = XToonDetailMode::Highlight;
    int global_hatch_sets = 3;
    float global_hatch_spacing = 0.08f;
    float global_hatch_width = 0.012f;
    float global_hatch_angle = 0.0f;
    Vec3 global_hatch_ink = {0.05f, 0.045f, 0.04f};
    Vec3 global_hatch_paper = {0.96f, 0.94f, 0.88f};
    bool global_hatch_passthrough = false;
    bool global_hatch_shadow_only = false;
    std::vector<MaterialStyleOverride> material_styles;
};

RenderOptions parse_render_options(int argc, char** argv);
void apply_material_styles(const RenderOptions& options, Scene& scene, std::ostream& errors);

} // namespace lt::cli
