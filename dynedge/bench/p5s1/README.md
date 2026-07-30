# P5S1 电压电流表功耗采集

用于论文设备矩阵 **Orin / T527 / H618 / ESP32-S3 / ESP32-P4** 的统一板级能耗测量。仪表型号：**星澜科创 P5S1**（五位半，协议 v2）。

| 项目 | 值 |
|---|---|
| 通信 | USB CDC 或 UART；主机侧按 **921600 8N1** 打开（CDC 实际忽略 Line Coding） |
| 采样率 | 1 / 10 / **100** Hz（论文默认 100 Hz） |
| 电压量程 | 0–48 V |
| **论文统一电流量程** | **±3.3 A**（全设备：Orin / T527 / H618 / S3 / P4） |
| **论文统一电流分辨率** | **0.01 mA（= 10 µA）** |
| 厂商规格另列 | ±33 mA / ±330 mA / ±3.3 A（分流 5 / 0.5 / 0.05 Ω）；本仓库实测 INFO 恒为 3.3 A，串口协议**无切档命令** |
| 上位机工具 | `python -m bench.p5s1.capture`（本目录，跨平台，依赖 `pyserial`） |

厂商协议原文：`电流电压表串口通信格式说明_v2.pdf`；产品说明书：`电压电流表说明书.pdf`。本工具实现与协议 PDF 金样一致（`selftest` 校验 START/STOP/SET_RATE CRC）。

---

## 1. 接线原则（所有设备通用）

1. **仪表自身**用 USB 或串口模块 5 V 供电（与被测电源分离）。
2. **被测板（DUT）**由实验电源供电；仪表串在供电回路中同时测 V/I。
3. 优先 **高端测量**（仪表串在电源正极侧，DUT 与电源共地）。无特殊需求不要低端串入地线。
4. 确认极性；勿超量程。**全论文统一 ±3.3 A / 0.01 mA**，不按 DUT 换档（保证跨设备可比）。
5. 若仪表供电与 DUT 电源无法共地，使用隔离串口模块，并将 V− 与电源负极共地（说明书 5.3）。

### 分设备建议测点

| 设备 | 推荐串入位置 | 电流档 | 备注 |
|---|---|---|---|
| ESP32-S3 / P4 | 开发板 **5 V 外供**（USB 线 **剪断 +5V**，只留数据/地做 CMD） | ±3.3 A | 见 `deploy/mcu_espdl/README.md`；勿 USB+外供双供 |
| H618 (OrangePi Zero3) | 5 V 输入正极高端 | ±3.3 A | 锁 `performance` 后再测 |
| T527 | 底板 5 V / 12 V 主输入高端 | ±3.3 A | 与延迟脚本同条件（governor=performance） |
| Orin Nano | DC 电源输入高端（或 USB-C PD 输入侧，视套件供电方式） | ±3.3 A | 报告所用 `nvpmodel` / `jetson_clocks` 模式；可与板载 INA 交叉校验 |

> Orin 若瞬时电流接近/超过 3.3 A，改测更高电压输入侧（电流更小），或缩短峰值窗口并在论文中声明量程上限。

---

## 2. 采集程序用法

```bash
pip install pyserial

# 列端口
python -m bench.p5s1.capture list

# 协议自检
python -m bench.p5s1.capture selftest

# 读设备 INFO
python -m bench.p5s1.capture info --port SERIAL_PORT

# 定长采集（默认 100 Hz，30 s）
python -m bench.p5s1.capture capture --port SERIAL_PORT --device esp32s3 \
  --rate 100 --duration 30 --live \
  --out results/power/esp32s3_probe.csv

# 交互打标（推理窗口 / 空闲基线）
python -m bench.p5s1.capture capture --port SERIAL_PORT --device orin --rate 100 \
  --interactive --out results/power/orin_run.csv
# stdin 输入: idle_start → idle_stop → start → stop → quit
```

输出：

- `*.csv`：`host_s,t_ms,voltage_v,current_a,power_w,current_ma,power_mw,mark`
- `*.csv.meta.json`：端口、量程、采样率、设备标签

积分汇总（梯形积分 `P=V·I`，可选空闲功率扣减）：

```bash
python -m bench.p5s1.capture summarize \
  --csv results/power/orin_run.csv \
  --n-infer 1000 --device orin \
  --out results/power/orin_run_summary.json
```

论文主表优先报 **`energy_mj_per_infer_net`**（扣 idle）；同时保留 gross 便于复核。

---

## 3. MCU（S3/P4）功耗窗口（USB CMD + 剪断 5V）

1. USB 线剪断 **+5V**，接主机做 Serial/JTAG CMD；板子 **5V 外供** + P5S1 高端。
2. 空闲基线：`CMD INFO`（模型已加载、不 BENCH）。
3. 推理窗：打标 `start` → `CMD BENCH` → `stop`。
4. summarize 时 `--n-infer` 与实际次数一致（默认 BENCH=warmup3+bench10）。

---

## 4. 实验过程标准流程（Phase 3 / 主表）

对每个 `{数据集 × 设备 × 子网 × 位宽}`：

1. **工况锁定**：与延迟测量完全一致（锁频 / MAXN / governor=performance / 同输入分辨率与后端）。
2. **预热**：Linux ≥50 次，MCU 至稳态；此阶段可不计入能量窗。
3. **空闲基线**（≥5 s）：DUT 已上电、模型已加载，但不跑推理；打标 `idle_start` / `idle_stop`。
4. **推理窗**：用测试集真实图像流跑 N 次（Linux 1000 / MCU ≥200）；同步打标 `start` / `stop`。N 写入 summarize 的 `--n-infer`。
5. **重复**：同一配置至少 3 轮，报均值；异常波动检查接线/档位/温漂。
6. **落盘**：CSV + meta + summary JSON；文件名含 `device/subnet/dtype/run#`，保证主表数字可溯源。

能耗定义：

\[
E_{\mathrm{net}} = \int_{T_{\mathrm{act}}} V(t)I(t)\,dt - P_{\mathrm{idle}}\cdot T_{\mathrm{act}},\quad
\mathrm{mJ/inf} = 1000\cdot E_{\mathrm{net}} / N
\]

---

## 5. 与旧方案的关系

| 原计划 | 现方案 |
|---|---|
| Linux：USB 功耗计 | **P5S1** 串供电回路（分辨率与可脚本化更好） |
| MCU：INA219/INA226 | **P5S1 ±3.3 A / 0.01 mA**（与 Linux 同量程、同脚本） |
| GPU 服务器 | 仍用 `nvidia-smi`（`bench/gpu_power.py`），不经过本表 |

协议注意：先发 **START** 锁定 UART/CDC 活动链路，再设采样率；活动链路锁定后另一接口命令会被忽略直至重启仪表。
