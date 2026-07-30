# fer-dynedge MCU ESP-DL path (P4 / S3)

正式路径：**蒸馏子网 → ONNX → ESP-PPQ PTQ@64 灰度→3ch → `.espdl` → ESP-DL 上板 → 宿主机/板端 logits**。

## 状态（2026-07-20）

| 步骤 | 状态 |
|---|---|
| v1 d2222 ONNX @64 | ✅ |
| PTQ `esp32p4` | ✅ `models/d2222_ferplus_64_esp32p4.espdl` |
| 宿主机 FP32 vs INT8 | ✅ 31/32，cos≈0.998 |
| P4 烧录 + `CMD CHECK` | ✅ **8/8，cos=1.0，≈27.15 ms** |
| S3（`target=esp32s3`） | 待做 |
| 换 v4 权重 | 重跑下方命令即可 |

## 主机复现

```bat
python tools/export_subnet_onnx.py --ckpt results/from_autodl/subnets/subnet_ferplus_d2222_seed42_v1_best.pt --depths 2,2,2,2 --img-size 64 --relu-for-deploy --num-classes 8 --out deploy/mcu_espdl/models/d2222_ferplus_64.onnx

python tools/espdl_ptq_d2222.py --onnx deploy/mcu_espdl/models/d2222_ferplus_64.onnx --calib-root data/calib/ferplus_mcu64/train --test-root data/calib/ferplus_mcu64/test --target esp32p4 --img-size 64 --gray-to-3ch --out-dir deploy/mcu_espdl/models --name d2222_ferplus_64

python tools/espdl_pack_ref_images.py
```

生成固件内嵌 blob（首次 / 换权重后）：

```bat
cd deploy\mcu_espdl\firmware
python -c "from pathlib import Path; d=Path('main'); b=(d/'d2222_ferplus_64_esp32p4.espdl').read_bytes() if (d/'d2222_ferplus_64_esp32p4.espdl').exists() else Path('../models/d2222_ferplus_64_esp32p4.espdl').read_bytes(); open(d/'model_espdl_data.c','w',newline='\n').write('#include <stdint.h>\n#include <stddef.h>\nconst size_t model_espdl_len=%d;\nconst uint8_t model_espdl[] __attribute__((aligned(16))) = {\n'%len(b)+''.join('  '+', '.join(f'0x{x:02x}' for x in b[i:i+16])+',\n' for i in range(0,len(b),16))+'};\n')"
```

更简单：若 `main/d2222_ferplus_64_esp32p4.espdl` 已在，用仓库内既有 `model_espdl_data.c`；换权重后重新生成该 `.c`。

## Flash / 分区

- 板载 flash：**16MB**（`esptool flash-id`，证据 `results/mcu_p4_partition/`）
- `partitions.csv`：**factory = 12MB**（`0xC00000`）；旧 4MB 会导致大 `.espdl` 嵌入后无法 boot
- 改分区后首次上板建议先 erase：`.\pio.cmd run -e esp32p4_espdl -t erase --upload-port SERIAL_PORT`

## 上板

```bat
cd deploy\mcu_espdl\firmware
.\pio.cmd run -e esp32p4_espdl -t upload --upload-port SERIAL_PORT
python ..\..\..\tools\espdl_board_check.py --port SERIAL_PORT
```

串口：`CMD INFO` / `CMD LOGITS i` / `CMD CHECK` / `CMD BENCH`。

## 功耗测量（S3：GPIO UART + USB-TTL）

S3 当前固件 CMD 走 **板丝印 UART：TX=GPIO43 / RX=GPIO44 @115200**（烧录仍用 USB SERIAL_PORT）。

1. USB-TTL（例 SERIAL_PORT / CH340）接线：
   - 板丝印 **TX → TTL RX**
   - 板丝印 **RX → TTL TX**
   - **GND 共地**
2. 板子 **5V 外供**，P5S1 ±3.3 A 高端串外供。
3. 主机对 TTL 口：`CMD INFO` / `CMD BENCH`。

| 板 | Flash USB | CMD（功耗） |
|---|---|---|
| ESP32-S3 | SERIAL_PORT | TTL（丝印 TX=GPIO43 / RX=GPIO44） |
| ESP32-P4 | SERIAL_PORT | TTL Grove（TX=GPIO54 / RX=GPIO53） |
