#include "fer_runner.h"
#include "fer_timing.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#if defined(FER_HAS_NCNN)
#include "net.h"
#endif

namespace {

uint32_t xorshift32(uint32_t &s) {
  s ^= s << 13;
  s ^= s >> 17;
  s ^= s << 5;
  return s;
}

double percentile(std::vector<double> &v, double p) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const double idx = p * static_cast<double>(v.size() - 1);
  const size_t lo = static_cast<size_t>(idx);
  const size_t hi = std::min(lo + 1, v.size() - 1);
  const double frac = idx - static_cast<double>(lo);
  return v[lo] * (1.0 - frac) + v[hi] * frac;
}

#if defined(FER_HAS_NCNN)

class NcnnRunner final : public FerRunner {
 public:
  NcnnRunner(const char *device_name, int num_threads, bool use_vulkan)
      : device_name_(device_name ? device_name : "linux"),
        num_threads_(num_threads),
        use_vulkan_(use_vulkan),
        in_h_(FER_INPUT_H),
        in_w_(FER_INPUT_W),
        in_c_(FER_INPUT_C) {
    net_.opt.num_threads = num_threads_;
    net_.opt.use_vulkan_compute = use_vulkan_;
    // Prefer INT8 when model is quantized; harmless for FP32 param/bin.
    net_.opt.use_packing_layout = true;
    // T527 A55 auto-enables FP16; v4 Hardswish mid-activations overflow → NaN.
    net_.opt.use_fp16_packed = false;
    net_.opt.use_fp16_storage = false;
    net_.opt.use_fp16_arithmetic = false;
    net_.opt.use_bf16_storage = false;
  }

  bool load(const char *param_path, const char *bin_path) override {
    if (net_.load_param(param_path) != 0) {
      std::fprintf(stderr, "ERR load_param %s\n", param_path);
      return false;
    }
    if (net_.load_model(bin_path) != 0) {
      std::fprintf(stderr, "ERR load_model %s\n", bin_path);
      return false;
    }
    ready_ = true;
    return true;
  }

  bool ready() const override { return ready_; }
  const char *backend() const override {
    return use_vulkan_ ? "ncnn_vulkan" : "ncnn_cpu";
  }
  const char *device_name() const override { return device_name_.c_str(); }

  void set_input_hwc(int h, int w, int c) override {
    if (h > 0) in_h_ = h;
    if (w > 0) in_w_ = w;
    if (c > 0) in_c_ = c;
  }
  int input_h() const override { return in_h_; }
  int input_w() const override { return in_w_; }
  int input_c() const override { return in_c_; }

  void infer(const float *input, FerInferResult *out) override {
    ncnn::Mat in(in_w_, in_h_, in_c_);
    // NCHW planar copy
    const size_t plane = static_cast<size_t>(in_h_) * static_cast<size_t>(in_w_);
    for (int c = 0; c < in_c_; ++c) {
      float *dst = in.channel(c);
      std::memcpy(dst, input + c * plane, plane * sizeof(float));
    }

    ncnn::Extractor ex = net_.create_extractor();
    ex.input("input", in);
    ncnn::Mat scores;
    // Default output name; override via FER_NCNN_OUTPUT if needed at build time.
#ifdef FER_NCNN_OUTPUT
    ex.extract(FER_NCNN_OUTPUT, scores);
#else
    ex.extract("output", scores);
#endif

    const int n = std::min(FER_NUM_CLASSES, scores.w * scores.h * scores.c);
    float max_v = -1e30f;
    int best = 0;
    for (int i = 0; i < n; ++i) {
      out->logits[i] = scores[i];
      if (scores[i] > max_v) {
        max_v = scores[i];
        best = i;
      }
    }
    for (int i = n; i < FER_NUM_CLASSES; ++i) out->logits[i] = 0.f;

    float sum = 0.f;
    float probs[FER_NUM_CLASSES];
    for (int i = 0; i < FER_NUM_CLASSES; ++i) {
      probs[i] = std::exp((out->logits[i] - max_v) * 1.f);
      sum += probs[i];
    }
    out->pred = best;
    out->confidence = probs[best] / sum;
  }

 private:
  ncnn::Net net_;
  std::string device_name_;
  int num_threads_;
  bool use_vulkan_;
  int in_h_;
  int in_w_;
  int in_c_;
  bool ready_ = false;
};

#else  // !FER_HAS_NCNN

// Host-side stub so the CLI / CSV path can be developed without a board.
class StubRunner final : public FerRunner {
 public:
  explicit StubRunner(const char *device_name)
      : device_name_(device_name ? device_name : "stub") {}

  bool load(const char *, const char *) override {
    ready_ = true;
    return true;
  }
  bool ready() const override { return ready_; }
  const char *backend() const override { return "stub_cpu"; }
  const char *device_name() const override { return device_name_.c_str(); }

  void infer(const float *input, FerInferResult *out) override {
    // Cheap deterministic hash of a few pixels → fake logits.
    double h = 0.0;
    for (int i = 0; i < 64; ++i) h += input[i] * (i + 1);
    for (int i = 0; i < FER_NUM_CLASSES; ++i) {
      out->logits[i] = static_cast<float>(std::sin(h + i) * 2.0);
    }
    int best = 0;
    float max_v = out->logits[0];
    for (int i = 1; i < FER_NUM_CLASSES; ++i) {
      if (out->logits[i] > max_v) {
        max_v = out->logits[i];
        best = i;
      }
    }
    float sum = 0.f;
    float probs[FER_NUM_CLASSES];
    for (int i = 0; i < FER_NUM_CLASSES; ++i) {
      probs[i] = std::exp(out->logits[i] - max_v);
      sum += probs[i];
    }
    out->pred = best;
    out->confidence = probs[best] / sum;
    // Burn a little CPU so stub timings are non-zero.
    volatile float sink = 0.f;
    for (int i = 0; i < 2000; ++i) sink += input[i % FER_INPUT_PIXELS];
    (void)sink;
  }

 private:
  std::string device_name_;
  bool ready_ = false;
};

#endif

}  // namespace

FerRunner *fer_create_ncnn_runner(const char *device_name, int num_threads,
                                  bool use_vulkan) {
#if defined(FER_HAS_NCNN)
  return new NcnnRunner(device_name, num_threads, use_vulkan);
#else
  (void)num_threads;
  (void)use_vulkan;
  return new StubRunner(device_name);
#endif
}

void fer_fill_synthetic_nchw_hwc(float *dst, int h, int w, int c, uint32_t seed) {
  uint32_t s = seed ? seed : 1u;
  for (int ch = 0; ch < c; ++ch) {
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const int cx = x - w / 2;
        const int cy = y - h / 2;
        const float r2 = static_cast<float>(cx * cx + cy * cy);
        float v = 0.3f + 0.5f * std::exp(-r2 / (0.15f * w * w));
        v += (static_cast<int>(xorshift32(s) % 21) - 10) * 0.002f;
        dst[ch * h * w + y * w + x] = v;
      }
    }
  }
}

void fer_fill_synthetic_nchw(float *dst, uint32_t seed) {
  fer_fill_synthetic_nchw_hwc(dst, FER_INPUT_H, FER_INPUT_W, FER_INPUT_C, seed);
}

void fer_print_ready(const FerRunner *r) {
  std::printf(
      "READY device=%s fw=%s backend=%s input=%dx%dx%d classes=%d "
      "ncnn=%d\n",
      r->device_name(), FER_FW_VERSION, r->backend(), r->input_h(), r->input_w(),
      r->input_c(), FER_NUM_CLASSES,
#if defined(FER_HAS_NCNN)
      1
#else
      0
#endif
  );
}

void fer_print_mem() {
  // Best-effort RSS from /proc (Linux boards).
  std::ifstream f("/proc/self/status");
  std::string line;
  long rss_kb = -1;
  long vms_kb = -1;
  while (std::getline(f, line)) {
    if (line.rfind("VmRSS:", 0) == 0) {
      std::sscanf(line.c_str(), "VmRSS: %ld", &rss_kb);
    } else if (line.rfind("VmSize:", 0) == 0) {
      std::sscanf(line.c_str(), "VmSize: %ld", &vms_kb);
    }
  }
  std::printf("MEM rss_kb=%ld vms_kb=%ld\n", rss_kb, vms_kb);
}

FerBenchStats fer_run_bench(FerRunner *r, const float *input, uint32_t warmup,
                            uint32_t iters) {
  FerBenchStats s{};
  s.warmup = warmup;
  s.iters = iters;
  FerInferResult tmp{};
  for (uint32_t i = 0; i < warmup; ++i) r->infer(input, &tmp);

  std::vector<double> samples;
  samples.reserve(iters);
  for (uint32_t i = 0; i < iters; ++i) {
    const uint64_t t0 = fer_now_us();
    r->infer(input, &s.last);
    const uint64_t t1 = fer_now_us();
    samples.push_back(static_cast<double>(t1 - t0));
  }
  s.mean_us = std::accumulate(samples.begin(), samples.end(), 0.0) /
              static_cast<double>(samples.size());
  s.p50_us = percentile(samples, 0.50);
  s.p95_us = percentile(samples, 0.95);
  s.p99_us = percentile(samples, 0.99);
  return s;
}

void fer_print_bench_result(const FerRunner *r, const FerBenchStats &s) {
  std::printf(
      "RESULT kind=bench device=%s backend=%s warmup=%u iters=%u "
      "input=%dx%dx%d "
      "latency_us_mean=%.1f latency_us_p50=%.1f latency_us_p95=%.1f "
      "latency_us_p99=%.1f pred=%d class=%s conf=%.4f\n",
      r->device_name(), r->backend(), s.warmup, s.iters, r->input_h(),
      r->input_w(), r->input_c(), s.mean_us, s.p50_us, s.p95_us, s.p99_us,
      s.last.pred, kFerClassNames[s.last.pred],
      static_cast<double>(s.last.confidence));
  fer_print_mem();
  std::printf("BENCH_DONE\n");
}
