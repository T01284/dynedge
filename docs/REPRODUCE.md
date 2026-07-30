# Reproduce paper tables & figures (command map — stub)

Map each published table/figure to a concrete command. Expand as Release assets land.

| Paper item | Source artifact | Command / notes |
|---|---|---|
| Fig. 1 iso-FLOPs | `artifacts/results/fig1_iso_flops_cluster.csv` | cluster measured offline; CSV is the source of truth |
| Fig. 2 overview | `docs/figures/fig2_overview.png` | illustration |
| Fig. 4 six-device Pareto | `artifacts/results/fig4_six_device_pareto.csv` | regenerate with local plotting scripts if needed |
| Table main (H618/T527) | `artifacts/results/tab_main_*_ncnn.csv` | board matrix summaries |
| Table C1 iso | `artifacts/results/tab_c1_iso_{acc,lat}.csv` | LUT vs FLOPs frontiers |
| Energy rows | `artifacts/results/energy_*_p5s1.csv` | from `python -m dynedge.bench.p5s1.capture` pipelines |
| Subnet structures | `artifacts/subnet_configs/*.json` | seed-42 formal + FLOPs-control |

**Table II row → exact CLI** listings will be filled once checkpoint Release assets are published.
