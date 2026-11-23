"""The experiment matrix of the proposal (Table 8) as data. Pure Python, unit-tested against the proposal's totals:
E0..E4 = 102 runs and 82 standard-run equivalents (SRE) for the core plan."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

SYNTHETIC_REPORTED = ["lego", "chair"]
DEV = ["drums", "ficus"]
REAL = "real"
SWEEP_N = [4, 16, 32]


@dataclass(frozen=True)
class Run:
    exp: str
    scene: str
    arm: str
    n: object          # int number of views, or "dense"
    seed: int
    short: bool = False

    @property
    def run_id(self) -> str:
        return f"{self.scene}_{self.arm}_n{self.n}_s{self.seed}"

    @property
    def cost(self) -> float:
        if self.short:
            return 1 / 3
        return 5 / 3 if self.arm in ("E", "X2") else 1.0


def tuning_runs() -> List[Run]:
    """E0: 6 trials x 2 dev scenes for each tuned arm (A, D, E), as short runs."""
    return [Run("E0", s, arm, 8, t, short=True) for arm in ("A", "D", "E") for s in DEV for t in range(6)]


def core_runs(extensions: bool = False) -> List[Run]:
    scenes3 = SYNTHETIC_REPORTED + [REAL]
    runs: List[Run] = []
    runs += [Run("E1", s, a, 8, sd) for s in DEV for a in ("A", "D") for sd in range(3)]
    runs += [Run("E2a", s, a, 8, sd) for s in scenes3 for a in ("A", "D") for sd in range(3)]
    runs += [Run("E2b", s, a, 8, sd) for s in scenes3 for a in ("B", "C", "E") for sd in range(2)]
    runs += [Run("E3", s, "A", "dense", 0) for s in scenes3]
    runs += [Run("E3", s, a, n, 0) for s in SYNTHETIC_REPORTED for a in ("A", "D") for n in SWEEP_N]
    runs += [Run("E4", s, "Aprime", 8, 0) for s in scenes3]
    if extensions:
        runs += [Run("EXT", s, a, 8, 2) for s in scenes3 for a in ("B", "C", "E")]
        runs += [Run("EXT", "lego", a, 8, 0) for a in ("X1", "X2")]
    return runs


def total_cost(runs: List[Run]) -> float:
    return sum(r.cost for r in runs)


def select(runs: List[Run], only: Optional[str] = None, scene: Optional[str] = None) -> List[Run]:
    out = runs
    if only:
        wanted = {x.strip() for x in only.split(",")}
        out = [r for r in out if r.exp in wanted]
    if scene:
        out = [r for r in out if r.scene == scene]
    return out


def submission_runs() -> List[Run]:
    """Reduced matrix actually run for the final submission (free Kaggle GPU quota, deadline 29 Nov 2026).
    Every cut relative to core_runs() is listed in the report's 'Changes from the proposal' section."""
    runs: List[Run] = []
    runs += [Run("S-gate2", "drums", a, 8, 0) for a in ("A", "D")]                               # Gate 2 check, dev scene
    runs += [Run("S-main", s, a, 8, sd) for s in SYNTHETIC_REPORTED for a in ("A", "D") for sd in range(3)]
    runs += [Run("S-main", s, "A", "dense", 0) for s in SYNTHETIC_REPORTED]                      # rho denominator, C4
    runs += [Run("S-ablation", s, a, 8, 0) for s in SYNTHETIC_REPORTED for a in ("B", "C")]      # 2x2 factorial, 1 seed
    runs += [Run("S-ext", s, "Aprime", 8, 0) for s in SYNTHETIC_REPORTED]                        # untuned baseline (N6)
    runs += [Run("S-ext", "lego", a, n, 0) for a in ("A", "D") for n in (4, 16)]                 # view sweep for N_eq
    runs += [Run("S-real", REAL, a, 8, 0) for a in ("A", "D")] + [Run("S-real", REAL, "A", "dense", 0)]
    runs += [Run("S-opt", "lego", "E", 8, 0)]                                                     # comparator if time allows
    return runs
