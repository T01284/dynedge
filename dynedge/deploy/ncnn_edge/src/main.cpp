#include "fer_runner.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#if defined(FER_HAS_NCNN)
#include "cpu.h"
#endif

namespace {

void usage(const char *argv0) {
  std::fprintf(
      stderr,
      "Usage:\n"
      "  %s --device t527|h618|t113 [--param P --bin B] [--threads N]\n"
      "     [--h H --w W --c C] [--warmup W] [--iters I] [--seed S] [--vulkan] [--info]\n"
      "\n"
      "Without NCNN (no -DFER_HAS_NCNN), runs stub backend for protocol bring-up.\n",
      argv0);
}

}  // namespace

int main(int argc, char **argv) {
  const char *device = "linux";
  const char *param = nullptr;
  const char *bin = nullptr;
  int threads = 4;
  int in_h = FER_INPUT_H;
  int in_w = FER_INPUT_W;
  int in_c = FER_INPUT_C;
  uint32_t warmup = FER_BENCH_WARMUP;
  uint32_t iters = FER_BENCH_ITERS;
  uint32_t seed = 42;
  bool vulkan = false;
  bool info_only = false;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
      device = argv[++i];
    } else if (std::strcmp(argv[i], "--param") == 0 && i + 1 < argc) {
      param = argv[++i];
    } else if (std::strcmp(argv[i], "--bin") == 0 && i + 1 < argc) {
      bin = argv[++i];
    } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
      threads = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--h") == 0 && i + 1 < argc) {
      in_h = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--w") == 0 && i + 1 < argc) {
      in_w = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--c") == 0 && i + 1 < argc) {
      in_c = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
      warmup = static_cast<uint32_t>(std::atoi(argv[++i]));
    } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
      iters = static_cast<uint32_t>(std::atoi(argv[++i]));
    } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
      seed = static_cast<uint32_t>(std::atoi(argv[++i]));
    } else if (std::strcmp(argv[i], "--vulkan") == 0) {
      vulkan = true;
    } else if (std::strcmp(argv[i], "--info") == 0) {
      info_only = true;
    } else if (std::strcmp(argv[i], "--help") == 0) {
      usage(argv[0]);
      return 0;
    } else {
      std::fprintf(stderr, "ERR unknown_arg=%s\n", argv[i]);
      usage(argv[0]);
      return 2;
    }
  }

  std::unique_ptr<FerRunner> runner(fer_create_ncnn_runner(device, threads, vulkan));
  runner->set_input_hwc(in_h, in_w, in_c);

#if defined(FER_HAS_NCNN)
  if (!param || !bin) {
    std::fprintf(stderr, "ERR need --param and --bin when built with NCNN\n");
    return 2;
  }
  if (!runner->load(param, bin)) return 1;
#else
  if (!runner->load(param ? param : "", bin ? bin : "")) return 1;
  if (param || bin) {
    std::fprintf(stderr,
                 "WARN stub build ignores --param/--bin (rebuild with "
                 "-DFER_HAS_NCNN=ON)\n");
  }
#endif

  fer_print_ready(runner.get());
  std::printf("INFO threads=%d vulkan=%d\n", threads, vulkan ? 1 : 0);
#if defined(FER_HAS_NCNN)
  std::printf(
      "NCNN_CPU neon=%d asimdhp=%d asimddp=%d asimdfhm=%d i8mm=%d\n",
      ncnn::cpu_support_arm_neon() ? 1 : 0,
      ncnn::cpu_support_arm_asimdhp() ? 1 : 0,
      ncnn::cpu_support_arm_asimddp() ? 1 : 0,
      ncnn::cpu_support_arm_asimdfhm() ? 1 : 0,
      ncnn::cpu_support_arm_i8mm() ? 1 : 0);
#endif
  fer_print_mem();

  if (info_only) return 0;

  const size_t npix = static_cast<size_t>(in_h) * static_cast<size_t>(in_w) *
                      static_cast<size_t>(in_c);
  std::vector<float> input(npix);
  fer_fill_synthetic_nchw_hwc(input.data(), in_h, in_w, in_c, seed);
  std::printf("BENCH_START warmup=%u iters=%u seed=%u input=%dx%dx%d\n",
              warmup, iters, seed, in_h, in_w, in_c);
  const FerBenchStats stats =
      fer_run_bench(runner.get(), input.data(), warmup, iters);
  fer_print_bench_result(runner.get(), stats);
  return 0;
}
