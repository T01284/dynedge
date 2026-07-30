#pragma once

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Start STA + TCP command server (port FER_WIFI_TCP_PORT). Returns false if disabled/fail. */
bool fer_wifi_start(void);

/** Current STA IPv4 string, or "0.0.0.0" if not connected. */
const char *fer_wifi_ip_str(void);

#ifdef __cplusplus
}
#endif
