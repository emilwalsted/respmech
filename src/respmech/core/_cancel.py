"""Cooperative cancellation for long-running compute.

``Cancelled`` is raised deep inside a numeric loop when a supplied ``cancel_check()``
returns True, so a superseded Preview job unwinds within tens of milliseconds instead of
running its whole (now-irrelevant) computation to completion. It subclasses
``BaseException`` — NOT ``Exception`` — so the many ``except Exception`` handlers in the
numeric code cannot accidentally swallow it.

GOLDEN-SAFE BY CONSTRUCTION: every checkpoint is guarded ``if cancel_check is not None and
cancel_check(): raise Cancelled``. The default ``cancel_check=None`` short-circuits before
the call, so the batch / CLI / golden path (which never passes a checker) runs the exact
same code with no extra float operations — byte-identical output.
"""


class Cancelled(BaseException):
    """Raised to abort a superseded compute early; only ever when a cancel_check is
    provided AND returns True."""


def check(cancel_check):
    """Raise :class:`Cancelled` iff a checker was supplied and reports cancellation.
    Inlined at hot-loop checkpoints; a no-op (single ``is not None`` test) by default."""
    if cancel_check is not None and cancel_check():
        raise Cancelled()
