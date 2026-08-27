"""A wall-clock deadline that can actually end the work.

The pipeline already caps a single rule application at five seconds and a single MCS at five,
through `signal.setitimer`. Those bound one call and bound nothing about a request: 7,581
templates at five seconds each is ten and a half hours with every cap working perfectly, and the
envelope sweep found eleven substrates of 106 that do not finish in 600 seconds at all.

A deadline that a service can rely on therefore cannot live inside the work. It has to be able
to end a process that is not cooperating, which means the work runs in a child and the parent
kills it. That is what this provides.

On the signal question, stated as measured rather than as believed: a Python signal handler runs
when the interpreter next regains control, so a long call inside a C extension is not interrupted
at the moment the alarm fires. Whether any call in this pipeline is long enough for that to
matter could not be shown -- MCS on the largest substrates in the deployment population returns
in under 0.01 s, and generation is a great many short calls between which the handler does run.
So the per-call caps are not demonstrated to be broken. They are demonstrated to be irrelevant to
the quantity a service cares about, which is the time the request takes.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import traceback
from dataclasses import dataclass
from typing import Any, Callable


class DeadlineExceeded(TimeoutError):
    """The work did not finish inside its wall-clock budget and the process was killed."""


@dataclass
class Outcome:
    value: Any = None
    error: str | None = None
    seconds: float = 0.0
    killed: bool = False


def _child(fn, args, kwargs, q):
    try:
        q.put(("ok", fn(*args, **kwargs)))
    except Exception:
        q.put(("err", traceback.format_exc()))


def run_with_deadline(fn: Callable, *args, seconds: float, **kwargs) -> Outcome:
    """Run `fn` in a child process and kill it if it outlives `seconds`.

    The child is killed with SIGKILL rather than asked to stop, because a process that has not
    returned by its deadline is precisely the one that cannot be relied on to honour a request.
    Returns an Outcome; it does not raise on timeout, so a caller can decide between an error
    response and a queued job without exception handling deciding it for them.
    """
    import time

    ctx = mp.get_context("spawn")     # fork would inherit torch and rdkit state mid-flight
    q = ctx.Queue()
    p = ctx.Process(target=_child, args=(fn, args, kwargs, q), daemon=True)
    t0 = time.perf_counter()
    p.start()
    p.join(seconds)
    if p.is_alive():
        try:
            os.kill(p.pid, 9)
        except ProcessLookupError:
            pass
        p.join(5)
        return Outcome(killed=True, seconds=time.perf_counter() - t0,
                       error=f"exceeded its {seconds:g}s deadline and was killed")
    elapsed = time.perf_counter() - t0
    if q.empty():
        return Outcome(error="the child produced no result", seconds=elapsed)
    kind, payload = q.get()
    if kind == "err":
        return Outcome(error=payload, seconds=elapsed)
    return Outcome(value=payload, seconds=elapsed)
