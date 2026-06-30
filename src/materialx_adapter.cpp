#include "lt/materialx_adapter.h"

namespace lt {

MaterialSystemStatus material_system_status() {
    MaterialSystemStatus status;
#if LT_HAS_MATERIALX
    status.materialx_available = true;
#endif
#if LT_HAS_OPENIMAGEIO
    status.openimageio_available = true;
    status.texture_pipeline = "OpenImageIO decode with role-based color conversion";
#endif
#if LT_HAS_OPENCOLORIO
    status.opencolorio_available = true;
    status.texture_pipeline = "OpenImageIO/fallback decode with OpenColorIO-capable role conversion";
#endif
    return status;
}

} // namespace lt
