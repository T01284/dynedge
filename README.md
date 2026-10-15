# DynEdge

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-green.svg)](requirements.txt)

Code for *When Is FLOPs Enough? An Empirical Study of Latency Proxies, Numeric Format, and Energy for Elastic CNNs on Five Edge Devices*.

Li Zhao (Heilongjiang Jiaotong Polytechnic),
Ming Zhao (Harbin Cambridge University),
Shu Wang (Heilongjiang Tianyouwei Technology Co., Ltd.; corresponding author, shu.wang@hljtyw.com)

The manuscript is being submitted to Neurocomputing. DynEdge is the elastic early-exit supernet used as the test bed. The repository contains the training, search, deployment, and measurement code, plus operator lookup tables and result tables. It does not contain dataset images or trained weights.

## Devices

| Device | Backend in the paper | Role |
|---|---|---|
| Jetson Orin Nano 8GB | TensorRT FP16 / INT8 | Edge GPU |
| Allwinner T527 | NCNN and VIPLite NBG | CPU and NPU |
| H618 (Orange Pi Zero 3) | NCNN, 4 threads | CPU board |
| ESP32-P4 | ESP-DL INT8 | Static microcontroller anchor |
| ESP32-S3 | ESP-DL INT8 | Static microcontroller anchor |

A CUDA server is used only to train models and to report reference accuracy. It is not a deployment target.

Toolchain pins are in [`dynedge/configs/versions.yaml`](dynedge/configs/versions.yaml).

## Setup

```bash
git clone https://github.com/T01284/dynedge.git
cd dynedge
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export PYTHONPATH=$PWD             # Windows: set PYTHONPATH=%CD%
```

FERPlus and ExpW images are not included. Download them yourself and follow [`dynedge/datasets/README.md`](dynedge/datasets/README.md).

```bash
python -m dynedge.datasets.prepare_ferplus --help
python -m dynedge.datasets.prepare_expw --help
```

## What you can run

```bash
python -m dynedge.search.search_subnets_lut --help
python -m dynedge.search.search_subnets_flops_ctrl --help
python -m dynedge.train.train_subnet_distill --help
python -m dynedge.train.train_controller --help
python dynedge/bench/run_matrix.py --help
python -m dynedge.bench.p5s1.capture --help
```

Board addresses stay in a local file:

```bash
cp dynedge/configs/config.example.yaml config.local.yaml
```

Deployment entry points are under `dynedge/deploy/` (`h618`, `t527`, `orin`, `esp32`, and the shared `ncnn_edge` path). ESP32 builds use ESP-IDF 5.5.x and ESP-DL.

## Layout

```
dynedge/                 # package; PYTHONPATH is the repository root
  datasets/              # FERPlus / ExpW preparation (no images)
  models/                # elastic supernet, early-exit heads, controller
  train/                 # supernet, distillation, controller
  lut/                   # operator lookup tables and MCU enumeration
  search/                # evolutionary search (LUT cost and FLOPs control)
  deploy/                # h618, t527, orin, esp32, ncnn_edge
  bench/                 # latency matrix and the P5S1 energy harness
  configs/               # example board config, toolchain pins, label maps
artifacts/
  luts/                  # operator tables and on-device end-to-end tables
  results/               # result CSVs from the measurement campaign
  subnet_configs/        # subnet structure files
docs/
  figures/               # overview figure used above
  MEASUREMENT.md
  REPRODUCE.md
```

The P5S1 harness is `dynedge/bench/p5s1`.

## Analysis code and cited results

`reproduction/scripts/` holds the scripts that compute the statistics in the manuscript (stage A and stage B analyses, and the figure scripts). `reproduction/results/` holds the tables those scripts read and the summaries cited in the paper:

- `stageB/c1_t527_clean/` — iso-accuracy and iso-latency comparisons
- `stageB/c4_orin_int8/summary.json` — calibrated TensorRT INT8 accuracy and latency
- `stageB/orin_sweep/` — TensorRT search latencies, including the repeat pass (`*_r2.csv`)
- `stageB/b5_energy_window/` — energy summaries and the raw windows they were fit on, for Orin, T527, H618, ESP32-P4, and ESP32-S3
- `stageA/` — proxy rank, predictor, and energy-rate tables
- `matrices/` — NCNN matrix summaries for the formal and FLOPs-control bakes
- `mcu_board_acc/`, `npu_full_acc/`, `board_full_acc/` — full-test board scores
- `c3_budget_tracking/`, `c2c3_joint_tracking/`, `controller/` — early-exit budget curves (controller weights are not included)

`reproduction/process/` holds the records those tables were computed from:

- `power/orin_p5s1`, `power/t527_p5s1`, `power/h618_p5s1` — P5S1 captures for the Linux energy rows
- `power/esp32p4_ppk2`, `power/esp32s3_ppk2` — PPK2 5.0 V streams for the microcontroller rows
- `stageB/c1_fix`, `stageB/b6_t527_p99` — comparison and tail-latency run logs
- `stageB/` search logs and the T527 / H618 compile logs
- `training_logs/` and `dataset_splits/` — supernet training logs and the FERPlus / ExpW split lists

Board addresses in metadata are replaced with `redacted`. Machine-specific paths in logs are replaced with placeholders such as `<repo>/`. The multi-hour H618 PPK2 stream is omitted because individual files exceed GitHub's per-file size limit; the cited H618 cross-check is the summary already under `reproduction/results/`.

## Not in this repository

- FERPlus and ExpW images.
- Board IP addresses and passwords. `config.example.yaml` has placeholders only. Capture metadata keeps the bench PC's COM-port labels, which identify nothing outside the lab.
- Checkpoints, NCNN parameter files, NBG files, TensorRT engines, and ESP-DL images. Rebuild them with the scripts under `dynedge/deploy/`.

## Citation

```bibtex
@article{zhao2026dynedge,
  title   = {When Is {FLOPs} Enough? An Empirical Study of Latency Proxies,
             Numeric Format, and Energy for Elastic {CNNs} on Five Edge Devices},
  author  = {Zhao, Li and Zhao, Ming and Wang, Shu},
  journal = {Neurocomputing},
  year    = {2026},
  note    = {Manuscript submitted for publication.
             Code: https://github.com/T01284/dynedge}
}
```

See [`CITATION.cff`](CITATION.cff).

## License

Apache License 2.0. See [LICENSE](LICENSE).
