// fer-dynedge ESP-DL CMD console.
// S3 power bench: GPIO UART (TX16/RX15) + USB-TTL; flash still via USB.
// P4 default: USB Serial/JTAG (see sdkconfig.defaults).
#include <cstdio>

#include "esp_log.h"
#include "fer_cmd.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "fer_main";

static void serial_reply(const char *line, void * /*ctx*/) {
    fputs(line, stdout);
    fflush(stdout);
}

extern "C" void app_main(void) {
#if defined(FER_UART_TX_GPIO) && defined(FER_UART_RX_GPIO)
    ESP_LOGI(TAG, "CMD UART TX=GPIO%d RX=GPIO%d @115200 (USB-TTL for power bench)", FER_UART_TX_GPIO,
             FER_UART_RX_GPIO);
#else
    ESP_LOGI(TAG, "CMD via USB Serial/JTAG @115200");
#endif
    fer_cmd_boot_banner(serial_reply, nullptr);

    char line[160];
    size_t len = 0;
    while (true) {
        int c = getchar();
        if (c == EOF) {
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }
        if (c == '\r') continue;
        if (c == '\n') {
            line[len] = 0;
            if (len) fer_cmd_handle_line(line, serial_reply, nullptr);
            len = 0;
        } else if (len + 1 < sizeof(line)) {
            line[len++] = (char)c;
        } else {
            len = 0;
        }
    }
}
