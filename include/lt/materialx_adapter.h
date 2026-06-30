#pragma once

namespace lt {

struct MaterialSystemStatus {
    bool materialx_available = false;
    bool openimageio_available = false;
    bool opencolorio_available = false;
    const char* surface_model = "OpenPBR-compatible standard_surface";
    const char* texture_pipeline = "role-based sRGB/raw fallback";
};

MaterialSystemStatus material_system_status();

} // namespace lt
