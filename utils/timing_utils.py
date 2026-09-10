"""Training-time analytics for train.py.

Writes <model_path>/training_time.json (machine readable, rewritten at every
save iteration and at the end, so a killed run still leaves a partial record)
and <model_path>/training_time.txt (one-look summary).

Wall-clock times come from time.perf_counter(); the per-iteration GPU time is
the CUDA event time train.py already measures around forward+backward.
"""
import json
import os
import platform
import socket
import time
from datetime import datetime

import torch


def _fmt_hms(sec):
    sec = int(round(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


class TrainTimer:
    def __init__(self, dataset, opt, first_iter=0):
        self.t0 = time.perf_counter()
        self.started_at = datetime.now().isoformat(timespec="seconds")
        self.model_path = None
        self.first_iter = first_iter
        self.total_iters = opt.iterations
        self.densify_until = opt.densify_until_iter
        self.args = {
            "source_path": dataset.source_path,
            "resolution": dataset.resolution,
            "iterations": opt.iterations,
            "densify_until_iter": opt.densify_until_iter,
            "densify_grad_threshold": opt.densify_grad_threshold,
            "optimizer_type": getattr(opt, "optimizer_type", None),
            "data_device": dataset.data_device,
        }
        self.setup_seconds = None
        self.loop_start = None
        self.densify_done_wall = None
        self.gpu_ms_total = 0.0          # forward+backward GPU time, summed
        self.eval_seconds = 0.0          # training_report (test renders + logging)
        self.save_seconds = 0.0          # scene.save + shadow ply
        self.milestones = []             # per save/test iteration
        self.per_1000 = []               # (iteration, wall_since_loop_start, n_gaussians)
        self.n_gaussians_final = None
        self.env = {
            "host": socket.gethostname(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch": torch.__version__,
            "python": platform.python_version(),
        }

    # --- phases -----------------------------------------------------------
    def set_model_path(self, model_path):
        self.model_path = model_path

    def mark_setup_done(self):
        self.setup_seconds = time.perf_counter() - self.t0
        self.loop_start = time.perf_counter()

    def _wall(self):
        return time.perf_counter() - self.loop_start

    def tick(self, iteration, gpu_ms, n_gaussians):
        self.gpu_ms_total += float(gpu_ms)
        if iteration == self.densify_until:
            self.densify_done_wall = self._wall()
        if iteration % 1000 == 0:
            self.per_1000.append([iteration, round(self._wall(), 2), int(n_gaussians)])

    def add_eval(self, seconds):
        self.eval_seconds += seconds

    def add_save(self, seconds):
        self.save_seconds += seconds

    def milestone(self, iteration, n_gaussians, kind):
        """Record wall time at a save/test iteration and rewrite the json."""
        self.milestones.append({
            "iteration": iteration,
            "kind": kind,
            "wall_seconds_since_loop_start": round(self._wall(), 2),
            "wall_hms": _fmt_hms(self._wall()),
            "n_gaussians": int(n_gaussians),
            "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if torch.cuda.is_available() else None,
        })
        self.write(final=False)

    def finish(self, n_gaussians):
        self.n_gaussians_final = int(n_gaussians)
        self.write(final=True)

    # --- output -----------------------------------------------------------
    def summary(self, final):
        loop = self._wall() if self.loop_start else 0.0
        total = time.perf_counter() - self.t0
        iters_done = (self.per_1000[-1][0] if self.per_1000 else 0) if not final else self.total_iters
        iters_run = max(iters_done - self.first_iter, 0)
        pure = max(loop - self.eval_seconds - self.save_seconds, 0.0)
        return {
            "status": "complete" if final else "in_progress",
            "started_at": self.started_at,
            "finished_at": datetime.now().isoformat(timespec="seconds") if final else None,
            "iterations_total": self.total_iters,
            "iterations_run": iters_run,
            "resumed_from_iteration": self.first_iter,
            "total_wall_seconds": round(total, 1),
            "total_wall_hms": _fmt_hms(total),
            "setup_seconds": round(self.setup_seconds, 1) if self.setup_seconds is not None else None,
            "train_loop_seconds": round(loop, 1),
            "train_loop_hms": _fmt_hms(loop),
            "train_loop_excl_eval_save_seconds": round(pure, 1),
            "eval_and_logging_seconds": round(self.eval_seconds, 1),
            "save_seconds": round(self.save_seconds, 1),
            "densification_phase_seconds": round(self.densify_done_wall, 1) if self.densify_done_wall is not None else None,
            "post_densification_seconds": round(loop - self.densify_done_wall, 1) if self.densify_done_wall is not None else None,
            "iterations_per_second": round(iters_run / pure, 2) if pure > 0 and iters_run else None,
            "gpu_ms_per_iteration_fwd_bwd": round(self.gpu_ms_total / iters_run, 2) if iters_run else None,
            "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if torch.cuda.is_available() else None,
            "n_gaussians_final": self.n_gaussians_final,
            "args": self.args,
            "env": self.env,
            "milestones": self.milestones,
            "per_1000_iterations": {"columns": ["iteration", "wall_seconds", "n_gaussians"], "rows": self.per_1000},
        }

    def write(self, final):
        if not self.model_path:
            return
        s = self.summary(final)
        with open(os.path.join(self.model_path, "training_time.json"), "w") as f:
            json.dump(s, f, indent=2)
        lines = [
            f"status                  {s['status']}",
            f"started                 {s['started_at']}",
            f"finished                {s['finished_at']}",
            f"host / gpu              {s['env']['host']} / {s['env']['gpu']}",
            f"source                  {s['args']['source_path']}  (resolution {s['args']['resolution']})",
            f"iterations              {s['iterations_run']} of {s['iterations_total']}",
            f"total wall time         {s['total_wall_hms']}  ({s['total_wall_seconds']} s, incl. setup {s['setup_seconds']} s)",
            f"train loop              {s['train_loop_hms']}  ({s['train_loop_seconds']} s)",
            f"  excl. eval + save     {s['train_loop_excl_eval_save_seconds']} s   (eval {s['eval_and_logging_seconds']} s, save {s['save_seconds']} s)",
            f"  densification phase   {s['densification_phase_seconds']} s  (until iter {s['args']['densify_until_iter']})",
            f"  after densification   {s['post_densification_seconds']} s",
            f"speed                   {s['iterations_per_second']} it/s   ({s['gpu_ms_per_iteration_fwd_bwd']} ms/it GPU fwd+bwd)",
            f"peak GPU memory         {s['peak_gpu_mem_gb']} GB",
            f"gaussians (final)       {s['n_gaussians_final']}",
            "",
            "milestones (iteration  kind  wall since loop start  gaussians  peak mem GB)",
        ]
        for m in s["milestones"]:
            lines.append(f"  {m['iteration']:>7}  {m['kind']:<5} {m['wall_hms']:>9}  {m['n_gaussians']:>9}  {m['peak_gpu_mem_gb']}")
        with open(os.path.join(self.model_path, "training_time.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
