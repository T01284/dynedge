#pragma once

#include <stddef.h>
#include <stdint.h>

#include "fer_config.h"

struct FerInferResult {
  int pred;
  float confidence;
  float logits[FER_NUM_CLASSES];
};

struct FerBenchStats {
  uint32_t warmup;
  uint32_t iters;
  double mean_us;
  double p50_us;
  double p95_us;
  double p99_us;
  FerInferResult last;
};

// Backend-agnostic runner (NCNN CPU today; NPU wrapper later on T527).
class FerRunner {
 public:
  virtual ~FerRunner() = default;
  virtual bool load(const char *param_path, const char *bin_path) = 0;
  virtual bool ready() const = 0;
  virtual const char *backend() const = 0;
  virtual const char *device_name() const = 0;

  // Optional runtime input size (default FER_INPUT_*). Call before infer/bench.
  virtual void set_input_hwc(int h, int w, int c) {
    (void)h;
    (void)w;
    (void)c;
  }
  virtual int input_h() const { return FER_INPUT_H; }
  virtual int input_w() const { return FER_INPUT_W; }
  virtual int input_c() const { return FER_INPUT_C; }

  // input: float NCHW; size = input_c()*input_h()*input_w().
  virtual void infer(const float *input, FerInferResult *out) = 0;
};

FerRunner *fer_create_ncnn_runner(const char *device_name, int num_threads,
                                  bool use_vulkan);

void fer_fill_synthetic_nchw(float *dst, uint32_t seed);
void fer_fill_synthetic_nchw_hwc(float *dst, int h, int w, int c, uint32_t seed);
void fer_print_ready(const FerRunner *r);
void fer_print_mem();
FerBenchStats fer_run_bench(FerRunner *r, const float *input, uint32_t warmup,
                            uint32_t iters);
void fer_print_bench_result(const FerRunner *r, const FerBenchStats &s);
