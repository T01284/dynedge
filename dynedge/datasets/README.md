# Datasets (FERPlus & ExpW)

This package ships **download and alignment scripts only**. Raw images, aligned caches,
calibration dumps, and preview grids are **not** redistributed (dataset licenses).

## FERPlus

1. Obtain FER2013 pixels (`fer2013.csv`) from the [FER2013 Kaggle challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)
   (you need a Kaggle account; store the token in `~/.kaggle/kaggle.json` **locally**, never in git).
2. Obtain majority-vote labels from [microsoft/FERPlus](https://github.com/microsoft/FERPlus) (`fer2013new.csv`).
3. Build image folders:

```bash
python -m dynedge.datasets.prepare_ferplus \
  --fer2013 /path/to/fer2013.csv \
  --ferplus /path/to/fer2013new.csv \
  --out $DATA_ROOT/ferplus
```

Label protocol matches Barsoum et al. (majority after single-vote outlier zeroing; discard unknown/NF).

## ExpW

1. Obtain the Expression in-the-Wild (ExpW) release from the original authors / dataset page
   (Images + `label.lst` style annotations with bounding boxes).
2. Run bbox crop → optional MTCNN align → 112×112 cache + stratified split:

```bash
python -m dynedge.datasets.prepare_expw \
  --src /path/to/expw \
  --out $DATA_ROOT/expw \
  --mtcnn-input 160
# debug / no GPU: add --no-mtcnn (resize-only)
```

### MTCNN align & 19.7% resize-fallback

- Batch align lives in `prepare_expw.mtcnn_batch_align` (facenet-pytorch `MTCNN`).
- Thin re-export: `dynedge.datasets.align_mtcnn`.
- When MTCNN fails to return a face, the pipeline **falls back to a resized bbox crop**.
  On the formal ExpW cache used in the paper this resize-fallback share is about **19.7%**
  of successfully written aligned samples (`n_resize_fallback / (n_mtcnn_hit + n_resize_fallback)`
  in the prep stats JSON). Report both rates when reproducing.

## Layout expected by training scripts

```
$DATA_ROOT/
  ferplus/{train,val,test}/<class>/*.png
  expw/{train,val,test}/aligned/...
```

Point `config.local.yaml` → `data_root` at `$DATA_ROOT`. Never commit image folders
(see root `.gitignore`).
