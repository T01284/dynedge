// Shared CMD protocol for USB serial and WiFi TCP.
#include "fer_cmd.h"

#include <array>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "dl_image_preprocessor.hpp"
#include "dl_model_base.hpp"
#include "esp_log.h"
#include "esp_timer.h"
#include "ref_images.h"

static const char *TAG = "fer_cmd";

extern const uint8_t model_espdl[];
extern const size_t model_espdl_len;

#ifndef FER_DEVICE_NAME
#define FER_DEVICE_NAME "esp32"
#endif

static dl::Model *g_model = nullptr;
static dl::image::ImagePreprocessor *g_pre = nullptr;
static const std::array<float, 3> kMean = {0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f};
static const std::array<float, 3> kStd = {0.229f * 255.f, 0.224f * 255.f, 0.225f * 255.f};

static void reply_fmt(fer_reply_fn reply, void *ctx, const char *fmt, ...) {
    char buf[768];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (reply) {
        reply(buf, ctx);
    } else {
        fputs(buf, stdout);
        fflush(stdout);
    }
}

static void print_ready(fer_reply_fn reply, void *ctx) {
    reply_fmt(reply, ctx,
              "READY device=%s fw=0.5.0-espdl-wifi model=embedded "
              "input=64x64x3 ref_n=%d model_bytes=%u\n",
              FER_DEVICE_NAME, REF_IMG_N, (unsigned)model_espdl_len);
}

static bool ensure_model(fer_reply_fn reply, void *ctx) {
    if (g_model) return true;
    g_model = new dl::Model((const char *)model_espdl, fbs::MODEL_LOCATION_IN_FLASH_RODATA);
    if (!g_model) {
        reply_fmt(reply, ctx, "ERR model_new_failed\n");
        return false;
    }
    ESP_LOGI(TAG, "model loaded bytes=%u", (unsigned)model_espdl_len);
    return true;
}

static bool ensure_preprocessor(fer_reply_fn reply, void *ctx) {
    if (!ensure_model(reply, ctx)) return false;
    if (g_pre) return true;
    g_pre = new dl::image::ImagePreprocessor(g_model, kMean, kStd, /*rgb_swap=*/false);
    return true;
}

static void run_logits(int idx, const char *kind, fer_reply_fn reply, void *ctx) {
    if (!ensure_preprocessor(reply, ctx)) return;
    if (idx < 0 || idx >= REF_IMG_N) {
        reply_fmt(reply, ctx, "ERR bad_idx=%d\n", idx);
        return;
    }

    dl::image::img_t img = {};
    img.data = (void *)ref_images[idx];
    img.width = REF_IMG_W;
    img.height = REF_IMG_H;
    img.pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB888;

    g_pre->preprocess(img);

    const int64_t t0 = esp_timer_get_time();
    g_model->run();
    const int64_t t1 = esp_timer_get_time();

    auto outputs = g_model->get_outputs();
    if (outputs.empty()) {
        reply_fmt(reply, ctx, "ERR no_outputs\n");
        return;
    }
    dl::TensorBase *out = outputs.begin()->second;
    const int n = (int)out->get_size();
    const int exp = (int)out->exponent;

    char head[192];
    snprintf(head, sizeof(head),
             "RESULT kind=%s idx=%d latency_us=%lld nlogits=%d out_exp=%d host_pred=%d logits=",
             kind, idx, (long long)(t1 - t0), n, exp, ref_host_pred[idx]);

    std::string line = head;
    if (out->get_dtype() == dl::DATA_TYPE_FLOAT) {
        float *p = (float *)out->data;
        for (int i = 0; i < n; ++i) {
            char num[32];
            snprintf(num, sizeof(num), "%s%.6f", i ? "," : "", p[i]);
            line += num;
        }
    } else if (out->get_dtype() == dl::DATA_TYPE_INT8) {
        int8_t *p = (int8_t *)out->data;
        for (int i = 0; i < n; ++i) {
            char num[16];
            snprintf(num, sizeof(num), "%s%d", i ? "," : "", (int)p[i]);
            line += num;
        }
    } else if (out->get_dtype() == dl::DATA_TYPE_INT16) {
        int16_t *p = (int16_t *)out->data;
        for (int i = 0; i < n; ++i) {
            char num[16];
            snprintf(num, sizeof(num), "%s%d", i ? "," : "", (int)p[i]);
            line += num;
        }
    } else {
        char num[48];
        snprintf(num, sizeof(num), "unsupported_dtype=%d", (int)out->get_dtype());
        line += num;
    }
    line += "\n";
    if (reply) {
        reply(line.c_str(), ctx);
    } else {
        fputs(line.c_str(), stdout);
        fflush(stdout);
    }
}

extern "C" void fer_cmd_boot_banner(fer_reply_fn reply, void *ctx) {
    reply_fmt(reply, ctx, "\n=== fer-dynedge ESP-DL (wifi+serial) ===\n");
    if (!ensure_model(reply, ctx)) return;
    print_ready(reply, ctx);
    reply_fmt(reply, ctx,
              "HINT CMD INFO | CMD LOGITS <i> | CMD CHECK | CMD BENCH | CMD BENCH_SYN | CMD IP\n");
}

extern "C" void fer_cmd_handle_line(const char *line, fer_reply_fn reply, void *ctx) {
    if (!line || !line[0]) return;

    if (strcmp(line, "CMD INFO") == 0 || strcmp(line, "INFO") == 0) {
        print_ready(reply, ctx);
    } else if (strncmp(line, "CMD LOGITS", 10) == 0 || strncmp(line, "LOGITS", 6) == 0) {
        int idx = 0;
        const char *p = strstr(line, "LOGITS");
        if (p) sscanf(p + 6, "%d", &idx);
        run_logits(idx, "logits", reply, ctx);
    } else if (strcmp(line, "CMD CHECK") == 0 || strcmp(line, "CHECK") == 0) {
        for (int i = 0; i < REF_IMG_N; ++i) run_logits(i, "check", reply, ctx);
        reply_fmt(reply, ctx, "CHECK_DONE\n");
    } else if (strcmp(line, "CMD BENCH_SYN") == 0 || strcmp(line, "BENCH_SYN") == 0) {
        if (!ensure_model(reply, ctx)) return;
        auto inputs = g_model->get_inputs();
        if (inputs.empty()) {
            reply_fmt(reply, ctx, "ERR no_inputs\n");
        } else {
            for (auto &kv : inputs) {
                dl::TensorBase *t = kv.second;
                if (t->get_dtype() == dl::DATA_TYPE_FLOAT) {
                    float *p = (float *)t->data;
                    const int n = (int)t->get_size();
                    for (int i = 0; i < n; ++i) p[i] = 0.1f;
                } else {
                    std::memset(t->data, 0, t->get_bytes());
                }
            }
            for (int i = 0; i < 3; ++i) g_model->run();
            for (int i = 0; i < 10; ++i) {
                const int64_t t0 = esp_timer_get_time();
                g_model->run();
                const int64_t t1 = esp_timer_get_time();
                reply_fmt(reply, ctx, "RESULT kind=op_bench idx=%d latency_us=%lld\n", i,
                          (long long)(t1 - t0));
            }
            reply_fmt(reply, ctx, "BENCH_DONE\n");
        }
    } else if (strncmp(line, "CMD BENCH", 9) == 0 || strncmp(line, "BENCH", 5) == 0) {
        for (int i = 0; i < 3; ++i) run_logits(0, "warmup", reply, ctx);
        for (int i = 0; i < 10; ++i) run_logits(0, "bench", reply, ctx);
        reply_fmt(reply, ctx, "BENCH_DONE\n");
    } else if (strcmp(line, "CMD INFER") == 0 || strcmp(line, "INFER") == 0) {
        run_logits(0, "single", reply, ctx);
    } else if (strcmp(line, "CMD IP") == 0 || strcmp(line, "IP") == 0) {
#if defined(FER_UART_TX_GPIO) && defined(FER_UART_RX_GPIO)
        reply_fmt(reply, ctx, "UART tx_gpio=%d rx_gpio=%d baud=115200\n", FER_UART_TX_GPIO,
                  FER_UART_RX_GPIO);
#else
        reply_fmt(reply, ctx, "LINK usb_serial_jtag baud=115200\n");
#endif
    } else {
        reply_fmt(reply, ctx, "ERR unknown_cmd=%s\n", line);
    }
}
