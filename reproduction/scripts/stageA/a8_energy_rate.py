#!/usr/bin/env python3
"""A8: system-level energy vs inference rate for the cabin-monitoring scenario.

E_sys(f) = E_net + P_idle / f, from the idle-subtracted P5S1 measurements
(results/power/*/energy_matrix.csv; T527 NPU, Orin and the two MCUs from the B5 window
sessions: Orin is the slope of window energy on N, MCUs are the 30 s window means).
Reports the daily energy at f in {1,5,10,30,100} Hz and the crossover rates.

Outputs: results/stageA/a8_*.csv|json, fig_a8_energy_rate.{pdf,png}
"""
from __future__ import annotations

import json

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from common import OUT, ROOT  # noqa: E402
from paper_style import H618, ORIN, P4, S3, T527, TEXT_IN, apply, save  # noqa: E402

SERIES = {
    "Orin TRT INT8": ("Orin", ORIN, "-", "o"),
    "T527 NPU (ReLU NBG)": ("T527 NPU", T527, "-", "s"),
    "T527 NCNN INT8": ("T527 NCNN", T527, "--", "s"),
    "H618 NCNN INT8": ("H618", H618, "-", "D"),
    "ESP32-P4 ESP-DL": ("ESP32-P4", P4, "-", "^"),
    "ESP32-S3 ESP-DL": ("ESP32-S3", S3, "-", "v"),
}

FREQS = [1, 5, 10, 30, 100]
# Table II operating point: Linux mid w0.5_d3333@112; MCU smallest anchor w0.35_d2222@64g (PPK2 30 s).
PICKS = [
    ("t527_p5s1", "ncnn_int8", "w05_d3333_s112c", "T527 NCNN INT8"),
    ("h618_p5s1", "ncnn_int8", "w05_d3333_s112c", "H618 NCNN INT8"),
]
# T527 NPU: B5 30 s windows on the ImageNet-normalised NBGs (native ReLU is the Table II row).
NPU_B5 = [("v4_npu_relu_fix", "T527 NPU (ReLU NBG)", True), ("v4_npu_fix", "T527 NPU (Hardswish NBG)", False)]


def main() -> int:
    devices = []
    for nb, label, plot in NPU_B5:
        p = ROOT / "results/stageB/b5_energy_window/t527" / nb / "summary.json"
        a = json.loads(p.read_text(encoding="utf-8"))["by_anchor"]["w05_d3333_s112c"]
        w = a["windows"]["30"]
        devices.append({"label": label, "source": str(p.relative_to(ROOT)), "artifact": f"w05_d3333_s112c {nb}",
                        "n_runs": int(w["n_reps"]), "e_net_mj": float(w["net_mean"]),
                        "e_net_std_mj": float(w["net_sd"]), "p_idle_w": float(a["idle"]["mean_w"]), "in_fig": plot})
    p = ROOT / "results/stageB/b5_energy_window/orin/summary.json"
    o = json.loads(p.read_text(encoding="utf-8"))["by_config"]["w05_d3333_s112c_int8"]
    devices.insert(0, {"label": "Orin TRT INT8", "source": str(p.relative_to(ROOT)),
                       "artifact": "w05_d3333_s112c trt_int8, slope of E_win on N", "n_runs": o["fit"]["n_windows"],
                       "e_net_mj": o["fit"]["e_mj_per_infer"],
                       "e_net_std_mj": (o["fit"]["e_ci95"][1] - o["fit"]["e_ci95"][0]) / 3.92,
                       "p_idle_w": o["p_idle_w_mean"], "in_fig": True})
    for folder, backend, tag, label in PICKS:
        em = pd.read_csv(ROOT / "results/power" / folder / "energy_matrix.csv")
        r = em[(em.backend == backend) & (em.tag == tag)].iloc[0]
        devices.append({"label": label, "source": f"results/power/{folder}/energy_matrix.csv",
                        "artifact": f"{tag} {backend}", "n_runs": int(r.n_runs),
                        "e_net_mj": float(r.energy_mj_per_infer_net_mean),
                        "e_net_std_mj": float(r.energy_mj_per_infer_net_std),
                        "p_idle_w": float(r.p_idle_w_mean), "in_fig": True})
    for dev, label in (("esp32p4_ppk2", "ESP32-P4 ESP-DL"), ("esp32s3_ppk2", "ESP32-S3 ESP-DL")):
        p = ROOT / "results/stageB/b5_energy_window" / dev / "summary.json"
        a = json.loads(p.read_text(encoding="utf-8"))["by_anchor"]["w035_d2222_s64g"]
        w = a["windows"]["30"]
        devices.append({"label": label, "source": str(p.relative_to(ROOT)),
                        "artifact": "w0.35_d2222_s64g PPK2 30 s windows", "n_runs": int(w["n_reps"]),
                        "e_net_mj": float(w["net_mean"]), "e_net_std_mj": float(w["net_sd"]),
                        "p_idle_w": float(a["idle"]["mean_w"]), "in_fig": True})
    dev = pd.DataFrame(devices)

    rows = []
    for _, d in dev.iterrows():
        for f in FREQS:
            e = d.e_net_mj + 1000.0 * d.p_idle_w / f  # mJ per inference, system level
            rows.append({"device": d.label, "f_hz": f, "e_sys_mj": e,
                         "idle_share": (1000.0 * d.p_idle_w / f) / e,
                         "wh_per_day": e * f * 86400 / 3.6e6})
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "a8_daily_energy.csv", index=False)

    # crossover rates: where device i becomes cheaper per inference than device j
    cross = []
    for i, a in dev.iterrows():
        for j, b in dev.iterrows():
            if i >= j or abs(a.p_idle_w - b.p_idle_w) < 1e-6:
                continue
            f_star = 1000.0 * (b.p_idle_w - a.p_idle_w) / (a.e_net_mj - b.e_net_mj)
            if f_star > 0:
                cheaper_below = a.label if a.p_idle_w < b.p_idle_w else b.label
                cross.append({"a": a.label, "b": b.label, "f_star_hz": f_star,
                              "cheaper_below_fstar": cheaper_below})
    cross = pd.DataFrame(cross).sort_values("f_star_hz")
    cross.to_csv(OUT / "a8_crossovers.csv", index=False)
    dev.to_csv(OUT / "a8_devices.csv", index=False)

    # the paper's stated critical rate, recomputed from the same numbers
    orin_idle = dev[dev.label == "Orin TRT INT8"].p_idle_w.iloc[0]
    h618_enet = dev[dev.label == "H618 NCNN INT8"].e_net_mj.iloc[0]
    stated = {"paper_formula": "R* = P_idle(Orin) / E_net(H618 mid)",
              "orin_idle_w_measured": orin_idle, "h618_mid_enet_mj": h618_enet,
              "r_star_hz": 1000.0 * orin_idle / h618_enet, "paper_claims_hz": 373,
              "paper_used_orin_idle_w": 8.6, "paper_used_h618_enet_mj": 23.35}

    apply()
    fig, ax = plt.subplots(figsize=(TEXT_IN, 2.85))
    fs = np.logspace(0, 3, 300)
    ax.axvspan(1, 30, color="0.93", zorder=0)
    for _, d in dev[dev["in_fig"]].iterrows():
        name, color, ls, marker = SERIES[d.label]
        y = (d.e_net_mj + 1000.0 * d.p_idle_w / fs) * fs * 86400 / 3.6e6
        ax.plot(fs, y, color=color, ls=ls, lw=1.1, zorder=2)
        ax.plot(fs[::40], y[::40], color=color, ls="none", marker=marker, ms=4.5, label=name, zorder=3)
    for hz, lab in ((26, "26"), (62, "62"), (391, "391")):
        ax.axvline(hz, color="0.25", lw=0.55, ls=(0, (1.2, 1.4)), zorder=1)
        ax.text(hz, 0.98, lab, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7, color="0.2")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Inference rate (Hz)")
    ax.set_ylabel("Energy per day (Wh)")
    ax.set_xlim(1, 1000)
    ax.legend(frameon=False, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    save(fig, "fig_a8_energy_rate")
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_a8_energy_rate.{ext}", dpi=200)
    plt.close(fig)

    (OUT / "a8_summary.json").write_text(json.dumps({"stated_critical_rate": stated}, indent=2), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(dev.round(3).to_string(index=False))
    print(tab.pivot(index="device", columns="f_hz", values="wh_per_day").round(2).to_string())
    print(cross.round(2).to_string(index=False))
    print(json.dumps(stated, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())