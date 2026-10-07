from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace


class _NoOpTable:
    def __init__(self, *args, **kwargs):
        self.columns = kwargs.get("columns")
        self.data = kwargs.get("data", [])

    def add_data(self, *args):
        self.data.append(args)


class _NoOpImage:
    def __init__(self, image, *args, **kwargs):
        self.image = image


class _NoOpPlot:
    @staticmethod
    def bar(*args, **kwargs):
        return None


class _NoOpWandb:
    Table = _NoOpTable
    Image = _NoOpImage
    plot = _NoOpPlot()

    def __init__(self):
        self.run = SimpleNamespace(dir=tempfile.mkdtemp(prefix="sage-wandb-disabled-"), resumed=False)

    def init(self, *args, **kwargs):
        run_dir = kwargs.get("dir") or os.environ.get("WANDB_DIR") or self.run.dir
        os.makedirs(run_dir, exist_ok=True)
        self.run = SimpleNamespace(dir=run_dir, resumed=False)
        return self.run

    def log(self, *args, **kwargs):
        return None

    def save(self, *args, **kwargs):
        return None

    def finish(self, *args, **kwargs):
        return None

    def Api(self, *args, **kwargs):
        raise RuntimeError("wandb.Api is unavailable because wandb could not be imported.")


try:
    import wandb as wandb  # type: ignore
except Exception:
    wandb = _NoOpWandb()
