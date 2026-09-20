# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compile-time memo for the schedule model.

A separate module, not a dict in ``_schedule.py``, and that is load-bearing rather
than stylistic. Triton's dependency finder deep-copies every global a traced function
references and re-checks it on *every* launch, so a module-level dict mutated during
tracing raises ``RuntimeError: Global variable ... has changed since we compiled this
kernel`` on the second launch -- long after the change that caused it. A module object
is skipped by that scan, so reaching the table through ``_schedule_cache.WAIT`` is
invisible to it.

Keys are :class:`~._schedule.ScheduleSpec` values, never tuning-configuration objects.
Triton builds those with ``eq_default=False``, so they hash by identity, hundreds of
distinct ones exist per compile for a single value, and several fields are unhashable
lists. The spec is also rebuilt from accessor results on every call, so a test that
mutates a configuration in place gets a different key rather than a stale answer.

This only ever avoids recomputing a pure function. It cannot change generated code,
and ``scripts/gluon_moe_asm_hash.py`` is the check that says so.
"""

#: ``(spec, stage, slot, drain, epilogue_groups, phase)`` -> wait immediate or None.
WAIT: dict = {}

#: Bound on retained entries. An autotuning sweep visits many configurations and
#: nothing here is worth holding across all of them; the win is within one compile,
#: where a handful of specs recur hundreds of times. Past the bound new results stop
#: being recorded rather than evicting, because eviction would need a method call on
#: this dict and Triton rejects any callable reachable from a traced body. Falling
#: back to recomputation is the safe direction.
LIMIT = 4096


def clear() -> None:
    """Drop every memoized result.

    Call from any fixture that mutates tuning configurations in place. Correctness
    does not depend on it -- the key is rebuilt from accessor results each time -- but
    it keeps one test's configurations from occupying the budget of the next.
    """
    WAIT.clear()


# No store helper on purpose, and no eviction: Triton's dependency finder rejects any
# callable reachable from a traced body -- a plain function and `WAIT.clear` alike,
# both with "Unsupported function referenced". The caller assigns through the module
# attribute, which is a subscript and not a call.
