// Tiny probe: print ncnn ARM CPU feature detection at runtime.
#include <cstdio>
#include "cpu.h"

int main() {
  std::printf("NCNN_CPU_PROBE\n");
  std::printf("neon=%d\n", ncnn::cpu_support_arm_neon() ? 1 : 0);
  std::printf("asimdhp=%d\n", ncnn::cpu_support_arm_asimdhp() ? 1 : 0);
  std::printf("asimddp=%d\n", ncnn::cpu_support_arm_asimddp() ? 1 : 0);
  std::printf("asimdfhm=%d\n", ncnn::cpu_support_arm_asimdfhm() ? 1 : 0);
  std::printf("i8mm=%d\n", ncnn::cpu_support_arm_i8mm() ? 1 : 0);
  std::printf("cpucount=%d\n", ncnn::get_cpu_count());
  return 0;
}
