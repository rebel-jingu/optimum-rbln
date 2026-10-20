# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Host traces for Perfetto and latency measurements separate from trace capture.

Host timestamps use ``perf_counter_ns``. RBLN ``.pb`` traces use a device clock;
do not concatenate their timestamps with these Chrome JSON events.
"""

import json
import os
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from threading import get_native_id
from time import perf_counter_ns


_ACTIVE = ContextVar("embeddinggemma2_trace_session", default=None)


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class TraceSession:
    """Collect nested host spans without importing Torch or the RBLN SDK.

    Spans are inclusive: a parent duration contains its children. Disabling
    ``capture_events`` retains stage durations without storing Chrome events.
    Nested sessions restore the preceding session on exit.
    """

    def __init__(self, output_path=None, *, capture_events=True):
        self.output_path = Path(output_path) if output_path is not None else None
        self.capture_events = capture_events
        self.events = []
        self._durations = defaultdict(list)
        self._token = None

    def __enter__(self):
        if self._token is not None:
            raise RuntimeError("A TraceSession cannot be entered twice concurrently")
        self._token = _ACTIVE.set(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        _ACTIVE.reset(self._token)
        self._token = None
        if self.output_path is not None:
            _write_json(self.output_path, self.chrome_trace())

    def chrome_trace(self):
        return {
            "traceEvents": self.events,
            "displayTimeUnit": "ms",
            "metadata": {"clock": "perf_counter_ns", "scope": "host", "stage_durations": "inclusive"},
        }

    def stage_timings_ms(self):
        return {name: list(samples) for name, samples in self._durations.items()}


@contextmanager
def trace_span(name, **args):
    """Record a host span; with no active session, read no clocks or SDK state."""
    session = _ACTIVE.get()
    if session is None:
        yield
        return
    start = perf_counter_ns()
    try:
        yield
    except BaseException as exc:
        # Exception messages may contain raw user inputs or paths.
        args = {**args, "error_type": type(exc).__name__}
        raise
    finally:
        duration = perf_counter_ns() - start
        session._durations[name].append(duration / 1e6)
        if session.capture_events:
            session.events.append(
                {
                    "name": name,
                    "cat": "embeddinggemma2.host",
                    "ph": "X",
                    "ts": start / 1e3,
                    "dur": duration / 1e3,
                    "pid": os.getpid(),
                    "tid": get_native_id(),
                    "args": args,
                }
            )
