// WiFi STA + TCP line server for power-bench (5V supply, no USB data).
// ESP32-S3: native Wi-Fi. ESP32-P4 (Tab5): ESP-Hosted + esp_wifi_remote via C6.
#include "fer_wifi.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "lwip/sockets.h"
#include "nvs_flash.h"

#include "fer_cmd.h"
#include "wifi_secrets.h"

#if CONFIG_IDF_TARGET_ESP32P4
#include "esp_hosted.h"
#endif

static const char *TAG = "fer_wifi";
static EventGroupHandle_t s_wifi_events;
static constexpr int WIFI_OK = BIT0;
static constexpr int WIFI_FAIL = BIT1;
static char s_ip[16] = "0.0.0.0";
static SemaphoreHandle_t s_cmd_mu;

extern "C" const char *fer_wifi_ip_str(void) { return s_ip; }

static void tcp_reply(const char *line, void *ctx) {
    int fd = (int)(intptr_t)ctx;
    if (fd < 0 || !line) return;
    const char *p = line;
    size_t left = strlen(line);
    while (left) {
        int n = ::send(fd, p, left, 0);
        if (n <= 0) break;
        p += n;
        left -= (size_t)n;
    }
}

static void serial_reply(const char *line, void * /*ctx*/) {
    fputs(line, stdout);
    fflush(stdout);
}

static void on_wifi_event(void * /*arg*/, esp_event_base_t base, int32_t id, void *data) {
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        ESP_LOGW(TAG, "disconnected; retry");
        esp_wifi_connect();
        xEventGroupClearBits(s_wifi_events, WIFI_OK);
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        auto *e = (ip_event_got_ip_t *)data;
        snprintf(s_ip, sizeof(s_ip), IPSTR, IP2STR(&e->ip_info.ip));
        ESP_LOGI(TAG, "got ip %s", s_ip);
        xEventGroupSetBits(s_wifi_events, WIFI_OK);
    }
}

static void handle_client(int fd) {
    char line[160];
    size_t len = 0;
    tcp_reply("READY via=tcp\n", (void *)(intptr_t)fd);
    fer_cmd_boot_banner(tcp_reply, (void *)(intptr_t)fd);

    while (true) {
        char c;
        int n = ::recv(fd, &c, 1, 0);
        if (n <= 0) break;
        if (c == '\r') continue;
        if (c == '\n') {
            line[len] = 0;
            if (len) {
                if (xSemaphoreTake(s_cmd_mu, portMAX_DELAY) == pdTRUE) {
                    fer_cmd_handle_line(line, tcp_reply, (void *)(intptr_t)fd);
                    xSemaphoreGive(s_cmd_mu);
                }
            }
            len = 0;
        } else if (len + 1 < sizeof(line)) {
            line[len++] = c;
        } else {
            len = 0;
        }
    }
    ::close(fd);
}

static void tcp_server_task(void * /*arg*/) {
    int listen_fd = ::socket(AF_INET, SOCK_STREAM, IPPROTO_IP);
    if (listen_fd < 0) {
        ESP_LOGE(TAG, "socket failed");
        vTaskDelete(nullptr);
        return;
    }
    int yes = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(FER_WIFI_TCP_PORT);
    if (bind(listen_fd, (sockaddr *)&addr, sizeof(addr)) != 0) {
        ESP_LOGE(TAG, "bind :%d failed", FER_WIFI_TCP_PORT);
        ::close(listen_fd);
        vTaskDelete(nullptr);
        return;
    }
    listen(listen_fd, 1);
    ESP_LOGI(TAG, "TCP CMD server on %s:%d", s_ip, FER_WIFI_TCP_PORT);

    while (true) {
        sockaddr_in client = {};
        socklen_t clen = sizeof(client);
        int fd = accept(listen_fd, (sockaddr *)&client, &clen);
        if (fd < 0) {
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }
        ESP_LOGI(TAG, "client connected");
        handle_client(fd);
        ESP_LOGI(TAG, "client closed");
    }
}

static void serial_cmd_task(void * /*arg*/) {
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
            if (len) {
                if (xSemaphoreTake(s_cmd_mu, portMAX_DELAY) == pdTRUE) {
                    fer_cmd_handle_line(line, serial_reply, nullptr);
                    xSemaphoreGive(s_cmd_mu);
                }
            }
            len = 0;
        } else if (len + 1 < sizeof(line)) {
            line[len++] = (char)c;
        } else {
            len = 0;
        }
    }
}

extern "C" bool fer_wifi_start(void) {
    if (std::strcmp(FER_WIFI_PASS, "CHANGE_ME") == 0 || FER_WIFI_PASS[0] == '\0') {
        ESP_LOGE(TAG, "WiFi password unset — copy wifi_secrets.h.example -> wifi_secrets.h");
        return false;
    }

    s_cmd_mu = xSemaphoreCreateMutex();
    s_wifi_events = xEventGroupCreate();

#if CONFIG_IDF_TARGET_ESP32P4
    ESP_LOGI(TAG, "P4: init ESP-Hosted (C6 WiFi)");
    esp_err_t herr = esp_hosted_init();
    if (herr != ESP_OK) {
        ESP_LOGE(TAG, "esp_hosted_init failed: %s", esp_err_to_name(herr));
        return false;
    }
    herr = esp_hosted_connect_to_slave();
    if (herr != ESP_OK) {
        ESP_LOGE(TAG, "esp_hosted_connect_to_slave failed: %s", esp_err_to_name(herr));
        return false;
    }
#endif

    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        nvs_flash_erase();
        err = nvs_flash_init();
    }
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "nvs_flash_init: %s", esp_err_to_name(err));
        return false;
    }

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &on_wifi_event, nullptr,
                                                       nullptr));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &on_wifi_event, nullptr,
                                                       nullptr));

    wifi_config_t wcfg = {};
    std::snprintf((char *)wcfg.sta.ssid, sizeof(wcfg.sta.ssid), "%s", FER_WIFI_SSID);
    std::snprintf((char *)wcfg.sta.password, sizeof(wcfg.sta.password), "%s", FER_WIFI_PASS);
    wcfg.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wcfg));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_LOGI(TAG, "connecting SSID=%s ...", FER_WIFI_SSID);

    EventBits_t bits =
        xEventGroupWaitBits(s_wifi_events, WIFI_OK | WIFI_FAIL, pdFALSE, pdFALSE, pdMS_TO_TICKS(30000));
    if (!(bits & WIFI_OK)) {
        ESP_LOGE(TAG, "WiFi connect timeout");
        return false;
    }

    xTaskCreate(tcp_server_task, "fer_tcp", 8192, nullptr, 5, nullptr);
    xTaskCreate(serial_cmd_task, "fer_serial", 8192, nullptr, 5, nullptr);
    return true;
}
