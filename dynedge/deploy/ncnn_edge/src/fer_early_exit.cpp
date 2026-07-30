// Early-exit board runner.
// Modes:
//   1) threshold (T3): --tau 0.85  — max-softmax gate (NOT G5)
//   2) forced exits (G5 host-decided): --exits exits.txt  — one index 0..3 per line
//   3) --force-final — always extract final
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include "net.h"

static const char *need(int &i, int argc, char **argv, const char *flag) {
  if (i + 1 >= argc) {
    std::fprintf(stderr, "ERR missing value for %s\n", flag);
    std::exit(2);
  }
  return argv[++i];
}

static float softmax_max(const ncnn::Mat &out) {
  const int n = out.w * out.h * out.c;
  float m = out[0];
  for (int i = 1; i < n; ++i) m = std::max(m, out[i]);
  double s = 0;
  for (int i = 0; i < n; ++i) s += std::exp(static_cast<double>(out[i] - m));
  double best = 0;
  for (int i = 0; i < n; ++i) {
    best = std::max(best, std::exp(static_cast<double>(out[i] - m)) / s);
  }
  return static_cast<float>(best);
}

static std::vector<int> load_exits(const char *path) {
  std::vector<int> exits;
  std::ifstream f(path);
  if (!f) {
    std::fprintf(stderr, "ERR open exits %s\n", path);
    std::exit(1);
  }
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    int v = std::atoi(line.c_str());
    if (v < 0 || v > 3) {
      std::fprintf(stderr, "ERR bad exit idx %s\n", line.c_str());
      std::exit(1);
    }
    exits.push_back(v);
  }
  return exits;
}

int main(int argc, char **argv) {
  const char *param = nullptr;
  const char *bin = nullptr;
  const char *list_path = nullptr;
  const char *exits_path = nullptr;
  const char *o0 = "out0", *o1 = "out1", *o2 = "out2", *o3 = "out3";
  int h = 112, w = 112, c = 3, threads = 4;
  float tau = 0.85f;
  int limit = 0;
  int force_final = 0;

  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--param")) param = need(i, argc, argv, argv[i]);
    else if (!std::strcmp(argv[i], "--bin")) bin = need(i, argc, argv, argv[i]);
    else if (!std::strcmp(argv[i], "--list")) list_path = need(i, argc, argv, argv[i]);
    else if (!std::strcmp(argv[i], "--exits")) exits_path = need(i, argc, argv, argv[i]);
    else if (!std::strcmp(argv[i], "--h")) h = std::atoi(need(i, argc, argv, argv[i]));
    else if (!std::strcmp(argv[i], "--w")) w = std::atoi(need(i, argc, argv, argv[i]));
    else if (!std::strcmp(argv[i], "--c")) c = std::atoi(need(i, argc, argv, argv[i]));
    else if (!std::strcmp(argv[i], "--tau")) tau = std::atof(need(i, argc, argv, argv[i]));
    else if (!std::strcmp(argv[i], "--threads")) threads = std::atoi(need(i, argc, argv, argv[i]));
    else if (!std::strcmp(argv[i], "--out0")) o0 = need(i, argc, argv, argv[i]);
    else if (!std::strcmp(argv[i], "--out1")) o1 = need(i, argc, argv, argv[i]);
    else if (!std::strcmp(argv[i], "--out2")) o2 = need(i, argc, argv, argv[i]);
    else if (!std::strcmp(argv[i], "--out3")) o3 = need(i, argc, argv, argv[i]);
    else if (!std::strcmp(argv[i], "--limit")) limit = std::atoi(need(i, argc, argv, argv[i]));
    else if (!std::strcmp(argv[i], "--force-final")) force_final = 1;
    else {
      std::fprintf(stderr, "ERR unknown %s\n", argv[i]);
      return 2;
    }
  }
  if (!param || !bin || !list_path) {
    std::fprintf(stderr, "ERR need --param/--bin/--list\n");
    return 2;
  }

  std::vector<int> forced_exits;
  if (exits_path) {
    if (force_final) {
      std::fprintf(stderr, "ERR --exits incompatible with --force-final\n");
      return 2;
    }
    forced_exits = load_exits(exits_path);
  }

  ncnn::Net net;
  net.opt.num_threads = threads;
  net.opt.use_fp16_packed = false;
  net.opt.use_fp16_storage = false;
  net.opt.use_fp16_arithmetic = false;
  net.opt.use_bf16_storage = false;
  if (net.load_param(param) != 0 || net.load_model(bin) != 0) {
    std::fprintf(stderr, "ERR load model\n");
    return 1;
  }

  const char *outs[4] = {o0, o1, o2, o3};
  int hits[4] = {0, 0, 0, 0};
  std::vector<double> ms;
  std::ifstream list(list_path);
  std::string line;
  int n_seen = 0;
  const size_t elems = static_cast<size_t>(h) * w * c;
  std::vector<float> input(elems);

  while (std::getline(list, line)) {
    if (line.empty()) continue;
    if (limit > 0 && n_seen >= limit) break;
    if (!forced_exits.empty() && static_cast<size_t>(n_seen) >= forced_exits.size()) {
      std::fprintf(stderr, "ERR exits shorter than list at n=%d\n", n_seen);
      return 1;
    }
    std::ifstream bf(line, std::ios::binary);
    if (!bf) {
      std::fprintf(stderr, "ERR open %s\n", line.c_str());
      return 1;
    }
    bf.read(reinterpret_cast<char *>(input.data()), static_cast<std::streamsize>(elems * 4));
    if (bf.gcount() != static_cast<std::streamsize>(elems * 4)) {
      std::fprintf(stderr, "ERR short %s\n", line.c_str());
      return 1;
    }

    ncnn::Mat in(w, h, c);
    const size_t plane = static_cast<size_t>(h) * w;
    for (int ch = 0; ch < c; ++ch) {
      std::memcpy(in.channel(ch), input.data() + ch * plane, plane * sizeof(float));
    }

    auto t0 = std::chrono::steady_clock::now();
    ncnn::Extractor ex = net.create_extractor();
    if (ex.input("input", in) != 0) {
      std::fprintf(stderr, "ERR input\n");
      return 1;
    }
    int exited = 3;
    if (force_final) {
      exited = 3;
      ncnn::Mat out;
      if (ex.extract(outs[3], out) != 0) {
        std::fprintf(stderr, "ERR extract final\n");
        return 1;
      }
    } else if (!forced_exits.empty()) {
      // Honest online-gate wall clock: extract outs[0..chosen] in order.
      exited = forced_exits[static_cast<size_t>(n_seen)];
      for (int k = 0; k <= exited; ++k) {
        ncnn::Mat out;
        if (ex.extract(outs[k], out) != 0) {
          std::fprintf(stderr, "ERR extract exit=%d\n", k);
          return 1;
        }
      }
    } else {
      for (int k = 0; k < 3; ++k) {
        ncnn::Mat out;
        if (ex.extract(outs[k], out) != 0) continue;
        if (softmax_max(out) >= tau) {
          exited = k;
          break;
        }
      }
      if (exited == 3) {
        ncnn::Mat out;
        if (ex.extract(outs[3], out) != 0) {
          std::fprintf(stderr, "ERR extract final\n");
          return 1;
        }
      }
    }
    auto t1 = std::chrono::steady_clock::now();
    ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    hits[exited] += 1;
    ++n_seen;
  }

  if (ms.empty()) {
    std::fprintf(stderr, "ERR no samples\n");
    return 1;
  }
  std::sort(ms.begin(), ms.end());
  auto pct = [&](double p) {
    size_t i = static_cast<size_t>(p * (ms.size() - 1));
    return ms[i];
  };
  double mean = std::accumulate(ms.begin(), ms.end(), 0.0) / ms.size();
  const char *mode = exits_path ? "g5_forced_exits" : (force_final ? "force_final" : "threshold_not_G5");
  std::printf(
      "RESULT n=%d tau=%.3f force_final=%d threads=%d e0=%.4f e1=%.4f e2=%.4f e3=%.4f "
      "p50_ms=%.3f p99_ms=%.3f mean_ms=%.3f fp16=off note=%s\n",
      n_seen, tau, force_final, threads,
      hits[0] / static_cast<double>(n_seen), hits[1] / static_cast<double>(n_seen),
      hits[2] / static_cast<double>(n_seen), hits[3] / static_cast<double>(n_seen), pct(0.50),
      pct(0.99), mean, mode);
  return 0;
}
