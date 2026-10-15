# H618 P5S1 ready

- Models pushed under `/root/fer-dynedge/models/v4_predistill_baked/`
- Bench: `/root/fer-dynedge/deploy/ncnn_edge/build-ncnn8/fer_ncnn_bench`
- **Wait until T527 P5S1 finishes**, move meter to H618 5V high-side,
  then: `python deploy/h618/scripts/run_p5s1_energy_matrix.py --port COMxx`
