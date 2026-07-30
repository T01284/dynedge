"""MTCNN face-align utilities used by ExpW preprocessing.

The batch-align path lives in prepare_expw.mtcnn_batch_align (facenet-pytorch MTCNN).
CLI: python -m dynedge.datasets.prepare_expw
Fallback resize (when MTCNN fails) is counted in stats; paper reports ~19.7% resize-fallback
on the formal ExpW cache — see datasets/README.md.
"""
from .prepare_expw import mtcnn_batch_align  # noqa: F401
