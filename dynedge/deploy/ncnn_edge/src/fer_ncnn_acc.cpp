// Board-side: FP32 vs INT8 NCNN top-1 match on preprocessed float32 NCHW bins.
// Usage:
//   fer_ncnn_acc --fp32-param P --fp32-bin B --int8-param P --int8-bin B \
//     --list bins.txt --h H --w W --c C --threads 2
// bins.txt: one absolute path to .bin per line (float32 NCHW, size=c*h*w)
#include "net.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>

static const char *need(int &i, int argc, char **argv, const char *flag) {
  if (i + 1 >= argc) {
    std::fprintf(stderr, "ERR missing value for %s\n", flag);
    std::exit(2);
  }
  return argv[++i];
}

static void disable_fp16(ncnn::Option &opt) {
  // T527 A55 enables FP16 by default; v4 Hardswish activations overflow FP16.
  opt.use_fp16_packed = false;
  opt.use_fp16_storage = false;
  opt.use_fp16_arithmetic = false;
  opt.use_bf16_storage = false;
}

static bool load_net(ncnn::Net &net, const char *param, const char *bin, int threads) {
  net.opt.num_threads = threads;
  net.opt.use_packing_layout = true;
  disable_fp16(net.opt);
  if (net.load_param(param) != 0) {
    std::fprintf(stderr, "ERR load_param %s\n", param);
    return false;
  }
  if (net.load_model(bin) != 0) {
    std::fprintf(stderr, "ERR load_model %s\n", bin);
    return false;
  }
  return true;
}

static bool infer(ncnn::Net &net, const float *input, int h, int w, int c,
                  std::vector<float> &logits) {
  ncnn::Mat in(w, h, c);
  const size_t plane = static_cast<size_t>(h) * static_cast<size_t>(w);
  for (int ch = 0; ch < c; ++ch) {
    std::memcpy(in.channel(ch), input + ch * plane, plane * sizeof(float));
  }
  ncnn::Extractor ex = net.create_extractor();
  if (ex.input("input", in) != 0) return false;
  ncnn::Mat out;
  if (ex.extract("output", out) != 0) return false;
  const int n = out.w * out.h * out.c;
  logits.resize(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) logits[static_cast<size_t>(i)] = out[i];
  return true;
}

static int argmax(const std::vector<float> &v) {
  int best = 0;
  float mv = v.empty() ? 0.f : v[0];
  for (int i = 1; i < static_cast<int>(v.size()); ++i) {
    if (v[static_cast<size_t>(i)] > mv) {
      mv = v[static_cast<size_t>(i)];
      best = i;
    }
  }
  return best;
}

static float cosine(const std::vector<float> &a, const std::vector<float> &b) {
  if (a.size() != b.size() || a.empty()) return 0.f;
  double dot = 0, na = 0, nb = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  if (na < 1e-12 || nb < 1e-12) return 0.f;
  return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb)));
}

int main(int argc, char **argv) {
  const char *fp32_param = nullptr;
  const char *fp32_bin = nullptr;
  const char *int8_param = nullptr;
  const char *int8_bin = nullptr;
  const char *list_path = nullptr;
  int h = 112, w = 112, c = 3, threads = 2;
  bool fp32_only = false;
  int limit = 0;

  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--fp32-param")) fp32_param = need(i, argc, argv, argv[i]);
    else if (!std::strcmp(argv[i], "--fp32-bin")) fp32_bin = need(i, argc, argv, argv[i]);
    else if (!std::strcmp(argv[i], "--int8-param")) int8_param = need(i, argc, argv, argv[i]);
    else if (!std::strcmp(argv[i], "--int8-bin")) int8_bin = need(i, argc, argv, argv[i]);
    else if (!std::strcmp(argv[i], "--list")) list_path = need(i, argc, argv, argv[i]);
    else if (!std::strcmp(argv[i], "--h")) h = std::atoi(need(i, argc, argv, argv[i]));
    else if (!std::strcmp(argv[i], "--w")) w = std::atoi(need(i, argc, argv, argv[i]));
    else if (!std::strcmp(argv[i], "--c")) c = std::atoi(need(i, argc, argv, argv[i]));
    else if (!std::strcmp(argv[i], "--threads")) threads = std::atoi(need(i, argc, argv, argv[i]));
    else if (!std::strcmp(argv[i], "--fp32-only")) fp32_only = true;
    else if (!std::strcmp(argv[i], "--limit")) limit = std::atoi(need(i, argc, argv, argv[i]));
    else {
      std::fprintf(stderr, "ERR unknown %s\n", argv[i]);
      return 2;
    }
  }
  if (!fp32_param || !fp32_bin || !list_path) {
    std::fprintf(stderr, "ERR need --fp32-param/--fp32-bin/--list\n");
    return 2;
  }
  if (!fp32_only && (!int8_param || !int8_bin)) {
    std::fprintf(stderr, "ERR need int8 param/bin (or pass --fp32-only)\n");
    return 2;
  }

  ncnn::Net net_fp, net_i8;
  if (!load_net(net_fp, fp32_param, fp32_bin, threads)) return 1;
  if (!fp32_only && !load_net(net_i8, int8_param, int8_bin, threads)) return 1;

  std::ifstream lf(list_path);
  if (!lf) {
    std::fprintf(stderr, "ERR open list %s\n", list_path);
    return 1;
  }

  const size_t elems = static_cast<size_t>(h) * static_cast<size_t>(w) * static_cast<size_t>(c);
  std::vector<float> input(elems);
  std::vector<float> z_fp, z_i8;
  int n = 0, match = 0, nan_fp = 0, nan_i8 = 0, seen = 0;
  double cos_sum = 0.0;
  std::string line;
  while (std::getline(lf, line)) {
    if (line.empty() || line[0] == '#') continue;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (limit > 0 && seen >= limit) break;
    ++seen;

    std::ifstream bf(line, std::ios::binary);
    if (!bf) {
      std::fprintf(stderr, "ERR open bin %s\n", line.c_str());
      return 1;
    }
    bf.read(reinterpret_cast<char *>(input.data()), static_cast<std::streamsize>(elems * sizeof(float)));
    if (bf.gcount() != static_cast<std::streamsize>(elems * sizeof(float))) {
      std::fprintf(stderr, "ERR short bin %s\n", line.c_str());
      return 1;
    }
    if (!infer(net_fp, input.data(), h, w, c, z_fp)) {
      std::fprintf(stderr, "ERR fp32 infer %s\n", line.c_str());
      return 1;
    }

    bool bad_fp = false;
    for (float v : z_fp) if (!std::isfinite(v)) bad_fp = true;

    if (fp32_only) {
      // Dump one LOGITS line for host ONNX cosine check.
      std::printf("LOGITS idx=%d path=%s finite=%d n=%zu vals=", seen - 1, line.c_str(),
                  bad_fp ? 0 : 1, z_fp.size());
      for (size_t i = 0; i < z_fp.size(); ++i) {
        if (i) std::printf(",");
        std::printf("%.8g", z_fp[i]);
      }
      std::printf("\n");
      if (bad_fp) ++nan_fp;
      else ++n;
      continue;
    }

    if (!infer(net_i8, input.data(), h, w, c, z_i8)) {
      std::fprintf(stderr, "ERR int8 infer %s\n", line.c_str());
      return 1;
    }
    bool bad_i8 = false;
    for (float v : z_i8) if (!std::isfinite(v)) bad_i8 = true;
    if (bad_fp) {
      ++nan_fp;
      continue;
    }
    if (bad_i8) {
      ++nan_i8;
      continue;
    }
    const int p_fp = argmax(z_fp);
    const int p_i8 = argmax(z_i8);
    if (p_fp == p_i8) ++match;
    cos_sum += cosine(z_fp, z_i8);
    ++n;
    if ((n % 50) == 0) {
      std::fprintf(stderr, "progress %d match=%.4f\n", n, n ? double(match) / n : 0.0);
    }
  }

  if (fp32_only) {
    std::printf("RESULT mode=fp32_only n_ok=%d nan_fp=%d n_seen=%d hw=%dx%dx%d fp16=off\n",
                n, nan_fp, seen, h, w, c);
    return (nan_fp == 0 && n > 0) ? 0 : 1;
  }

  if (n == 0) {
    std::fprintf(stderr, "ERR no valid samples nan_fp=%d nan_i8=%d\n", nan_fp, nan_i8);
    return 1;
  }
  std::printf(
      "RESULT n_eval=%d fp32_int8_match=%.6f logits_cosine_mean=%.6f "
      "nan_fp=%d nan_i8=%d hw=%dx%dx%d fp16=off\n",
      n, double(match) / n, cos_sum / n, nan_fp, nan_i8, h, w, c);
  return 0;
}
