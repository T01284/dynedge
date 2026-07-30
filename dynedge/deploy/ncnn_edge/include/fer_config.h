#pragma once

// Shared Linux-edge FER probe / bench config (T527 / H618, paper Step 1.2–1.3).
// Probe default: MobileNetV3-Small style @112×112 RGB (Linux tier).
// MCU stays at 64×64; do not mix tiers in one CSV row without noting input=.

#ifndef FER_INPUT_H
#define FER_INPUT_H 112
#endif
#ifndef FER_INPUT_W
#define FER_INPUT_W 112
#endif
#ifndef FER_INPUT_C
#define FER_INPUT_C 3
#endif
#ifndef FER_NUM_CLASSES
#define FER_NUM_CLASSES 7
#endif

#ifndef FER_BENCH_WARMUP
#define FER_BENCH_WARMUP 50
#endif
#ifndef FER_BENCH_ITERS
#define FER_BENCH_ITERS 1000
#endif

#define FER_INPUT_PIXELS (FER_INPUT_H * FER_INPUT_W * FER_INPUT_C)
#define FER_FW_VERSION "0.2.0-ncnn-v3"

#if FER_NUM_CLASSES == 8
// FERPlus official majority order used in training
static const char *const kFerClassNames[FER_NUM_CLASSES] = {
    "neutral", "happiness", "surprise", "sadness",
    "anger", "disgust", "fear", "contempt",
};
#else
static const char *const kFerClassNames[FER_NUM_CLASSES] = {
    "angry", "disgust", "fear", "happy", "sad", "surprise", "neutral",
};
#endif
