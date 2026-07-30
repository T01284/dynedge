#pragma once

#include <stdint.h>

// Steady-clock µs; used for software latency (USB power meter is separate).
uint64_t fer_now_us();
