# T527 DEMO CAR — 刷 Buildroot Linux（PhoenixSuit）

> 前置：公开 ACUITY（`ubuntu-npu:v1.8.13`）已在本机 WSL Docker 就绪。

## 材料

| 项 | 路径 |
|---|---|
| 镜像 | `E:\t527\myd_lt527_emmc_buildroot\myd_lt527_emmc_buildroot.img`（`IMAGEWTY`） |
| 工具 | `E:\t527\PhoenixSuit\app\PhoenixSuit\PhoenixSuit.exe` |
| USB 驱动 | 同目录 `Drivers\AW_Driver\InstallUSBDrv.exe` |
| 串口 | `SERIAL_PORT` @115200（CH340） |

## 步骤

1. 启动 PhoenixSuit；必要时先跑一遍 `InstallUSBDrv.exe`。
2. USB 线接 PC ↔ 板子 **OTG/Device** 口（不是串口线）。
3. 进 FEL：按住 FEL → 点 RESET → 松 FEL（或断电后按住 FEL 上电）。
4. 设备管理器出现 Allwinner/AW USB（常见 `VID_1F3A`）。
5. PhoenixSuit → 选镜像 → Upgrade / 一键刷机。
6. 完成后断电上电；串口应出现 Buildroot 登录提示。
7. 配网后继续 Phase B：确认 `/dev/vipcore`、重跑 NCNN。

## 风险

- 会擦除当前 Android；过渡 CSV 已在仓库 `results/t527_ncnn_*_probe.csv`。
