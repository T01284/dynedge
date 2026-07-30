#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Line-oriented CMD protocol (serial USB or WiFi TCP). */
typedef void (*fer_reply_fn)(const char *line, void *ctx);

void fer_cmd_boot_banner(fer_reply_fn reply, void *ctx);
void fer_cmd_handle_line(const char *line, fer_reply_fn reply, void *ctx);

#ifdef __cplusplus
}
#endif
