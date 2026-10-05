from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
from scipy.optimize import linprog


@dataclass
class Instance:
    """MOILP + quadratic preference instance.

    The multiobjective problem is a maximization problem:
        VMAX C x
        s.t. A x <= b, x integer, 0 <= x <= u.

    The scalar preference is
        h(x) = x^T Q x + d^T x + alpha.

    Q is stored as an integer symmetric matrix in the generated benchmark.
    """

    A: np.ndarray
    b: np.ndarray
    C: np.ndarray
    Q: np.ndarray
    d: np.ndarray
    alpha: int | float = 0
    u: np.ndarray | None = None
    meta: Dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.A = np.asarray(self.A, dtype=np.int64)
        self.b = np.asarray(self.b, dtype=np.int64).reshape(-1)
        self.C = np.asarray(self.C, dtype=np.int64)
        self.Q = np.asarray(self.Q, dtype=np.int64)
        self.d = np.asarray(self.d, dtype=np.int64).reshape(-1)
        if self.u is not None:
            self.u = np.asarray(self.u, dtype=np.int64).reshape(-1)
        if self.meta is None:
            self.meta = {}
        self.validate_shapes()

    @property
    def m(self) -> int:
        return int(self.A.shape[0])

    @property
    def n(self) -> int:
        return int(self.A.shape[1])

    @property
    def p(self) -> int:
        return int(self.C.shape[0])

    def validate_shapes(self) -> None:
        m, n = self.A.shape
        if self.b.shape != (m,):
            raise ValueError(f"b has shape {self.b.shape}, expected {(m,)}")
        if self.C.ndim != 2 or self.C.shape[1] != n:
            raise ValueError("C must have n columns")
        if self.Q.shape != (n, n):
            raise ValueError("Q must be n x n")
        if self.d.shape != (n,):
            raise ValueError("d must have length n")
        if self.u is not None and self.u.shape != (n,):
            raise ValueError("u must have length n")

    def objective(self, x: np.ndarray) -> int | float:
        x = np.asarray(x, dtype=np.int64).reshape(-1)
        q = int(x @ self.Q @ x)
        lin = int(self.d @ x)
        return q + lin + self.alpha

    def criteria(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.int64).reshape(-1)
        return self.C @ x

    def is_feasible(self, x: np.ndarray, tol: float = 1e-8) -> bool:
        x = np.asarray(x)
        if x.shape != (self.n,):
            return False
        if np.max(np.abs(x - np.rint(x))) > tol:
            return False
        xi = np.rint(x).astype(np.int64)
        if np.any(xi < 0):
            return False
        if self.u is not None and np.any(xi > self.u):
            return False
        return bool(np.all(self.A @ xi <= self.b))

    def ensure_upper_bounds(self) -> np.ndarray:
        if self.u is None:
            self.u = compute_upper_bounds(self.A, self.b)
        return self.u

    def canonical_hash(self) -> str:
        h = sha256()
        for arr in (self.A, self.b, self.C, self.Q, self.d):
            a = np.ascontiguousarray(arr)
            h.update(str(a.shape).encode())
            h.update(a.dtype.str.encode())
            h.update(a.tobytes(order="C"))
        h.update(repr(float(self.alpha)).encode())
        if self.u is not None:
            h.update(np.ascontiguousarray(self.u).tobytes())
        return h.hexdigest()

    def save_npz(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta_json = json.dumps(self.meta or {}, ensure_ascii=False, sort_keys=True)
        np.savez_compressed(
            path,
            A=self.A,
            b=self.b,
            C=self.C,
            Q=self.Q,
            d=self.d,
            alpha=np.asarray([self.alpha]),
            u=self.ensure_upper_bounds(),
            meta=np.asarray([meta_json]),
        )

    @staticmethod
    def load_npz(path: str | Path) -> "Instance":
        with np.load(path, allow_pickle=False) as z:
            meta_raw = str(z["meta"][0]) if "meta" in z else "{}"
            return Instance(
                A=z["A"],
                b=z["b"],
                C=z["C"],
                Q=z["Q"],
                d=z["d"],
                alpha=float(z["alpha"][0]),
                u=z["u"],
                meta=json.loads(meta_raw),
            )


def compute_upper_bounds(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute finite valid integer upper bounds for x >= 0, A x <= b.

    For the nonnegative benchmark matrices, an analytic formula is exact enough
    for boundedness: u_j = min_i floor(b_i / A_ij) over A_ij > 0.
    For matrices containing negative coefficients, LP bounds are used.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    m, n = A.shape

    if np.all(A >= 0) and np.all(b >= 0):
        out = np.empty(n, dtype=np.int64)
        for j in range(n):
            pos = np.flatnonzero(A[:, j] > 0)
            if len(pos) == 0:
                raise ValueError(f"Variable x[{j}] is unbounded: column A[:,{j}] has no positive coefficient")
            out[j] = int(np.floor(np.min(b[pos] / A[pos, j]) + 1e-10))
            if out[j] < 0:
                raise ValueError("Feasible set is inconsistent with x >= 0")
        return out

    out = np.empty(n, dtype=np.int64)
    for j in range(n):
        c = np.zeros(n)
        c[j] = -1.0
        r = linprog(c, A_ub=A, b_ub=b, bounds=[(0, None)] * n, method="highs")
        if r.status == 3:
            raise ValueError(f"Variable x[{j}] is unbounded in continuous relaxation")
        if r.status == 2:
            raise ValueError("Continuous feasible set is empty")
        if not r.success:
            raise RuntimeError(f"LP bound computation failed for x[{j}]: {r.message}")
        out[j] = int(np.floor(float(r.x[j]) + 1e-8))
    return out


def standard_form(instance: Instance) -> dict[str, np.ndarray]:
    """Convert A x <= b to A x + s = b using nonnegative integer slacks.

    Returns the extended equality system and extended C/Q/d arrays. The original
    decision variables always occupy the first n positions.
    """
    u = instance.ensure_upper_bounds()
    A = instance.A
    b = instance.b
    m, n = A.shape

    Aeq = np.hstack([A, np.eye(m, dtype=np.int64)])
    min_ax = np.sum(np.minimum(A * u[np.newaxis, :], 0), axis=1)
    slack_u = np.maximum(0, b - min_ax).astype(np.int64)
    u_ext = np.concatenate([u, slack_u])

    C_ext = np.hstack([instance.C, np.zeros((instance.p, m), dtype=np.int64)])
    Q_ext = np.zeros((n + m, n + m), dtype=np.int64)
    Q_ext[:n, :n] = instance.Q
    d_ext = np.concatenate([instance.d, np.zeros(m, dtype=np.int64)])

    return {
        "Aeq": Aeq,
        "beq": b.copy(),
        "u_ext": u_ext,
        "C_ext": C_ext,
        "Q_ext": Q_ext,
        "d_ext": d_ext,
    }
