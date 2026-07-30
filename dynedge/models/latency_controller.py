"""C3 latency-budget early-exit controller (Linux devices only).

Per early-exit gate: 2-layer MLP, 8-dim input → {exit, continue} logits.
Final head is always taken if all gates continue.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


FEAT_DIM = 8  # locked design
N_GATES = 3  # exit2 / exit3 / exit4


def softmax_stats(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (max_prob, entropy_norm, margin) over class dim."""
    log_p = F.log_softmax(logits, dim=-1)
    p = log_p.exp()
    max_p, _ = p.max(dim=-1)
    # top-2 margin
    top2 = torch.topk(p, k=min(2, p.size(-1)), dim=-1).values
    margin = top2[..., 0] - top2[..., 1] if top2.size(-1) > 1 else top2[..., 0]
    ent = -(p * log_p).sum(dim=-1)
    ent_norm = ent / math.log(max(2, logits.size(-1)))
    return max_p, ent_norm, margin


class ExitGate(nn.Module):
    """2-layer MLP: 8 → hidden → 2 (exit / continue)."""

    def __init__(self, in_dim: int = FEAT_DIM, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LatencyBudgetController(nn.Module):
    """Three per-exit gates (paper: one MLP per exit point)."""

    def __init__(self, in_dim: int = FEAT_DIM, hidden: int = 32):
        super().__init__()
        self.gates = nn.ModuleList([ExitGate(in_dim, hidden) for _ in range(N_GATES)])
        self.in_dim = in_dim

    def build_features(
        self,
        exit_logits: Sequence[torch.Tensor],
        *,
        cum_ms: torch.Tensor,
        seg_ms: torch.Tensor,
        budget_ms: float,
    ) -> list[torch.Tensor]:
        """Build 8-dim features for each of the 3 early exits.

        Dims (locked):
          0 max_prob
          1 entropy / log(C)
          2 remaining_budget / B
          3 margin (p1−p2)
          4 spent / B
          5 next_seg / B
          6 exit_idx / 3
          7 budget_ms / 50
        """
        B = float(budget_ms)
        b_scale = torch.tensor(B / 50.0, device=exit_logits[0].device, dtype=exit_logits[0].dtype)
        feats: list[torch.Tensor] = []
        for i, logits in enumerate(exit_logits[:N_GATES]):
            max_p, ent_n, margin = softmax_stats(logits)
            spent = cum_ms[i]
            # remaining after this exit's cum cost
            rem = ((B - spent).clamp(min=0.0) / B).expand_as(max_p)
            spent_n = (spent / B).expand_as(max_p)
            next_seg = (seg_ms[i + 1] / B).expand_as(max_p)  # next segment cost
            idx_n = torch.full_like(max_p, float(i) / float(N_GATES))
            b_n = b_scale.expand_as(max_p)
            feats.append(
                torch.stack([max_p, ent_n, rem, margin, spent_n, next_seg, idx_n, b_n], dim=-1)
            )
        return feats

    def forward_gates(self, feats: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        return [gate(f) for gate, f in zip(self.gates, feats)]

    def soft_path(
        self,
        gate_logits: Sequence[torch.Tensor],
        *,
        tau: float,
        hard: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gumbel-Softmax sequential exit decisions.

        Returns:
          exit_probs: (B, 4) probability of exiting at exit2/3/4/final
          mixed used for CE weighting over 4 logits
        """
        bs = gate_logits[0].size(0)
        device = gate_logits[0].device
        surv = torch.ones(bs, device=device, dtype=gate_logits[0].dtype)
        probs = []
        for g in gate_logits:
            # columns: 0=exit, 1=continue
            soft = F.gumbel_softmax(g, tau=tau, hard=hard, dim=-1)
            p_exit = surv * soft[:, 0]
            surv = surv * soft[:, 1]
            probs.append(p_exit)
        probs.append(surv)  # final
        return torch.stack(probs, dim=-1), surv


def expected_latency_ms(
    exit_probs: torch.Tensor,
    cum_ms: torch.Tensor,
) -> torch.Tensor:
    """exit_probs (B,4), cum_ms (4,) → (B,) E[latency_ms]."""
    return (exit_probs * cum_ms.view(1, -1)).sum(dim=-1)


def mix_logits(
    exit_probs: torch.Tensor,
    all_logits: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Soft mixture of exit logits for differentiable CE."""
    stacked = torch.stack(list(all_logits), dim=1)  # (B, 4, C)
    w = exit_probs.unsqueeze(-1)
    return (stacked * w).sum(dim=1)
