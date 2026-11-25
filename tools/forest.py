"""Portable evaluator with numeric weights; no pickled model or training runtime."""

import ctypes as c
from pathlib import Path
import numpy as np


def export_model(model, destination):
    trees = [stage[0].nodes for stage in model._predictors]
    shape = (len(trees), max(map(len, trees)))
    data = {
        k: np.zeros(shape, dtype=t)
        for k, t in [
            ("feature", "int32"),
            ("left", "int32"),
            ("right", "int32"),
            ("threshold", "float64"),
            ("value", "float64"),
        ]
    }
    for i, tree in enumerate(trees):
        for name, key in [
            ("left", "left"),
            ("right", "right"),
            ("threshold", "num_threshold"),
            ("value", "value"),
        ]:
            data[name][i, : len(tree)] = tree[key]
        data["feature"][i, : len(tree)] = np.where(
            tree["is_leaf"], -1, tree["feature_idx"].astype("int32")
        )
    np.savez_compressed(destination, **data, bias=model._baseline_prediction.item())


class Forest:
    def __init__(self):
        base = Path(__file__).resolve().parent
        with np.load(
            base.parent / "models/appearance.npz", allow_pickle=False
        ) as model:
            self.data = [
                model[k] for k in ["feature", "left", "right", "threshold", "value"]
            ]
            self.bias = float(model["bias"])
        self.lib = c.CDLL(str(base / "libforest.dylib")).evaluate_forest
        self.lib.argtypes = (
            [c.c_int, c.c_int, c.c_void_p, c.c_int, c.c_int]
            + [c.c_void_p] * 5
            + [c.c_double, c.c_void_p]
        )
        self.lib.restype = None

    def positive(self, x):
        x = np.ascontiguousarray(x, dtype=np.float32)
        y = np.empty(len(x), np.float32)
        self.lib(
            *x.shape,
            x.ctypes.data,
            *self.data[0].shape,
            *[m.ctypes.data for m in self.data],
            self.bias,
            y.ctypes.data,
        )
        return y
