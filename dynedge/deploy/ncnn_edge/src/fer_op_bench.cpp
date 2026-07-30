// Standalone single-graph latency bench for LUT op mining.
// Unlike fer_ncnn_bench, input H/W/C and blob names are CLI-configurable.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "net.h"

namespace {

double percentile(std::vector<double> &v, double p) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const double idx = p * static_cast<double>(v.size() - 1);
  const size_t lo = static_cast<size_t>(idx);
  const size_t hi = std::min(lo + 1, v.size() - 1);
  const double frac = idx - static_cast<double>(lo);
  return v[lo] * (1.0 - frac) + v[hi] * frac;
}

void usage(const char *argv0) {
  std::fprintf(
      stderr,
      "Usage:\n"
      "  %s --device NAME --param P --bin B --h H --w W --c C\n"
      "     [--input NAME] [--output NAME] [--threads N] [--warmup W] [--iters I]\n",
      argv0);
}

}  // namespace

int main(int argc, char **argv) {
  const char *device = "h618";
  const char *param = nullptr;
  const char *bin = nullptr;
  const char *in_name = "input";
  const char *out_name = "out";
  int h = 0, w = 0, c = 0;
  int threads = 4;
  unsigned warmup = 20;
  unsigned iters = 200;

  for (int i = 1; i < argc; ++i) {
    auto need = [&](const char *flag) -> const char * {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "ERR missing value for %s\n", flag);
        std::exit(2);
      }
      return argv[++i];
    };
    if (!std::strcmp(argv[i], "--device")) device = need("--device");
    else if (!std::strcmp(argv[i], "--param")) param = need("--param");
    else if (!std::strcmp(argv[i], "--bin")) bin = need("--bin");
    else if (!std::strcmp(argv[i], "--input")) in_name = need("--input");
    else if (!std::strcmp(argv[i], "--output")) out_name = need("--output");
    else if (!std::strcmp(argv[i], "--h")) h = std::atoi(need("--h"));
    else if (!std::strcmp(argv[i], "--w")) w = std::atoi(need("--w"));
    else if (!std::strcmp(argv[i], "--c")) c = std::atoi(need("--c"));
    else if (!std::strcmp(argv[i], "--threads")) threads = std::atoi(need("--threads"));
    else if (!std::strcmp(argv[i], "--warmup")) warmup = static_cast<unsigned>(std::atoi(need("--warmup")));
    else if (!std::strcmp(argv[i], "--iters")) iters = static_cast<unsigned>(std::atoi(need("--iters")));
    else if (!std::strcmp(argv[i], "--help") || !std::strcmp(argv[i], "-h")) {
      usage(argv[0]);
      return 0;
    } else {
      std::fprintf(stderr, "ERR unknown arg %s\n", argv[i]);
      usage(argv[0]);
      return 2;
    }
  }

  if (!param || !bin || h <= 0 || w <= 0 || c <= 0) {
    usage(argv[0]);
    return 2;
  }

  ncnn::Net net;
  net.opt.num_threads = threads;
  net.opt.use_packing_layout = true;
  net.opt.use_fp16_packed = false;
  net.opt.use_fp16_storage = false;
  net.opt.use_fp16_arithmetic = false;
  net.opt.use_bf16_storage = false;
  if (net.load_param(param) != 0) {
    std::fprintf(stderr, "ERR load_param %s\n", param);
    return 1;
  }
  if (net.load_model(bin) != 0) {
    std::fprintf(stderr, "ERR load_model %s\n", bin);
    return 1;
  }

  ncnn::Mat in(w, h, c);
  in.fill(0.1f);

  auto once = [&]() {
    ncnn::Extractor ex = net.create_extractor();
    ex.input(in_name, in);
    ncnn::Mat out;
    ex.extract(out_name, out);
  };

  for (unsigned i = 0; i < warmup; ++i) once();

  std::vector<double> samples;
  samples.reserve(iters);
  for (unsigned i = 0; i < iters; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    once();
    const auto t1 = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
  }

  double sum = 0.0;
  for (double x : samples) sum += x;
  const double mean = sum / static_cast<double>(samples.size());
  auto sorted = samples;
  const double p50 = percentile(sorted, 0.50);
  sorted = samples;
  const double p95 = percentile(sorted, 0.95);
  sorted = samples;
  const double p99 = percentile(sorted, 0.99);

  std::printf(
      "RESULT kind=op_bench device=%s backend=ncnn_cpu warmup=%u iters=%u "
      "h=%d w=%d c=%d threads=%d "
      "latency_us_mean=%.1f latency_us_p50=%.1f latency_us_p95=%.1f latency_us_p99=%.1f\n",
      device, warmup, iters, h, w, c, threads, mean, p50, p95, p99);
  return 0;
}
