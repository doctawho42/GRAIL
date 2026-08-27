"""The request deadline must end work that a signal cannot.

These guard the distinction the serving module is built on: `signal.setitimer` bounds a call
only when the interpreter regains control, and nothing bounds the sum of many calls. The
deadline runner bounds the request itself.
"""
import signal
import time

import pytest

from grail_metabolism.serving.deadline import run_with_deadline


def _sleep(n):
    time.sleep(n)
    return "finished"


def _busy_python(n):
    """Many short steps, which is what generation looks like: no single long call."""
    end = time.perf_counter() + n
    x = 0
    while time.perf_counter() < end:
        x += 1
    return x


def test_work_inside_the_deadline_returns_its_value():
    out = run_with_deadline(_sleep, 0.1, seconds=10)
    assert out.value == "finished" and not out.killed and out.error is None


def test_work_past_the_deadline_is_killed_not_awaited():
    out = run_with_deadline(_sleep, 60, seconds=2)
    assert out.killed and out.value is None
    assert out.seconds < 20, "the parent waited for the child instead of killing it"


def test_a_sum_of_short_calls_is_bounded_too():
    """The per-call caps in preparation.py cannot bound this; the deadline can."""
    out = run_with_deadline(_busy_python, 60, seconds=2)
    assert out.killed and out.seconds < 20


def test_an_error_in_the_child_comes_back_as_text_not_a_crash():
    out = run_with_deadline(_sleep, "not a number", seconds=10)
    assert out.error and not out.killed and out.value is None


def test_a_signal_alarm_does_not_bound_a_sum_of_calls():
    """Why the deadline exists, held as a test rather than left as a claim.

    A per-call alarm fires and is handled, so each call is bounded; the loop then arms another
    one and keeps going. The total is unbounded no matter how well each individual cap works.
    """
    if not hasattr(signal, "setitimer"):
        pytest.skip("no setitimer on this platform")

    class Fired(Exception):
        pass

    def handler(sig, frame):
        raise Fired()

    prev = signal.signal(signal.SIGALRM, handler)
    fired = 0
    t0 = time.perf_counter()
    try:
        for _ in range(20):
            signal.setitimer(signal.ITIMER_REAL, 0.02)
            try:
                _busy_python(1.0)
            except Fired:
                fired += 1
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
    finally:
        signal.signal(signal.SIGALRM, prev)
    assert fired == 20, "each call was capped"
    assert time.perf_counter() - t0 > 0.2, "and the total still grew with the number of calls"
