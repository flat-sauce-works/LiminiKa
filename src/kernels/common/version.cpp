// SPDX-License-Identifier: MIT OR Apache-2.0

#include "liminika/gcso_config.h"

namespace liminika {

// Returns the runtime version string of the LiminiKa C++ kernel module.
const char* get_version() GCSO_NOEXCEPT {
    return "0.1.1";
}

} // namespace liminika