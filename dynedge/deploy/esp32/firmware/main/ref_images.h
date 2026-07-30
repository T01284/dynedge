#pragma once
#include <stddef.h>
#include <stdint.h>
#define REF_IMG_N 8
#define REF_IMG_H 64
#define REF_IMG_W 64
#define REF_IMG_C 3
extern const uint8_t ref_images[REF_IMG_N][REF_IMG_H * REF_IMG_W * REF_IMG_C];
extern const int ref_host_pred[REF_IMG_N];
