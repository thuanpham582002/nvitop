# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
#
# Copyright 2021-2026 Xuehai Pan. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Hardware fault telemetry from NVML: XID events, ECC, remapped rows, PCIe replay, clock reasons."""

from __future__ import annotations

import logging
import threading
from typing import Any, Sequence

from nvitop import NA, libnvml


LOGGER = logging.getLogger(__name__)

_EVENT_WAIT_TIMEOUT_MS = 1000
_ERROR_BACKOFF_SECONDS = 5.0


def _query(func: str, *args: Any) -> Any:
    """Query NVML, returning ``NA`` on any failure (e.g. ``NOT_SUPPORTED`` on this card)."""
    try:
        return libnvml.nvmlQuery(func, *args, ignore_function_not_found=True)
    except libnvml.NVMLError:  # raised by the lazy NVML initialization, not by the query itself
        return NA


def read_faults(handle: Any) -> dict[str, Any]:
    """Read the polled fault counters of a physical GPU. Unsupported values are ``NA``."""
    remapped_rows = _query('nvmlDeviceGetRemappedRows', handle)
    if not isinstance(remapped_rows, tuple):
        remapped_rows = (NA, NA, NA, NA)

    clock_event_reasons = _query('nvmlDeviceGetCurrentClocksEventReasons', handle)
    if clock_event_reasons is NA:  # the former name of the function, for old `nvidia-ml-py`
        clock_event_reasons = _query('nvmlDeviceGetCurrentClocksThrottleReasons', handle)

    return {
        'ecc_volatile_corrected': _query(
            'nvmlDeviceGetTotalEccErrors',
            handle,
            libnvml.NVML_MEMORY_ERROR_TYPE_CORRECTED,
            libnvml.NVML_VOLATILE_ECC,
        ),
        'ecc_volatile_uncorrected': _query(
            'nvmlDeviceGetTotalEccErrors',
            handle,
            libnvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
            libnvml.NVML_VOLATILE_ECC,
        ),
        'remapped_rows_correctable': remapped_rows[0],
        'remapped_rows_uncorrectable': remapped_rows[1],
        'remapped_rows_pending': remapped_rows[2],
        'remapped_rows_failure': remapped_rows[3],
        'pcie_replay': _query('nvmlDeviceGetPcieReplayCounter', handle),
        'clock_event_reasons': clock_event_reasons,
    }


class XidEventMonitor:
    """Count XID critical errors of physical GPUs in a background thread.

    XID events only exist while somebody is waiting on an NVML event set, so an XID raised while the
    exporter is not running is lost and the counters restart from zero with the exporter.
    """

    def __init__(self, devices: Sequence[Any]) -> None:
        """Remember the physical devices to listen to. Nothing is registered until :meth:`start`."""
        self._devices = list(devices)
        self._lock = threading.Lock()
        self._counters: dict[str, tuple[int, int]] = {}  # uuid -> (last XID code, number of events)
        self._event_set: Any = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Register the devices and start listening. A device that cannot register stays unset."""
        if self._thread is not None:
            return
        try:
            self._event_set = libnvml.nvmlQuery('nvmlEventSetCreate', ignore_errors=False)
        except libnvml.NVMLError as ex:
            LOGGER.warning('XID events are not collected: nvmlEventSetCreate failed: %s', ex)
            return

        for device in self._devices:
            try:
                libnvml.nvmlQuery(
                    'nvmlDeviceRegisterEvents',
                    device.handle,
                    libnvml.nvmlEventTypeXidCriticalError,
                    self._event_set,
                    ignore_errors=False,
                )
            except libnvml.NVMLError as ex:
                LOGGER.warning('XID events are not collected on GPU %s: %s', device.uuid(), ex)
                continue
            with self._lock:
                self._counters[device.uuid()] = (0, 0)

        if not self._counters:
            self._free_event_set()
            return

        self._thread = threading.Thread(target=self._run, name='nvml-xid-events', daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop listening and release the event set."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def get(self, uuid: str) -> tuple[int, int] | None:
        """Return ``(last XID code, number of events)``, or :data:`None` if not listening."""
        with self._lock:
            return self._counters.get(uuid)

    def _run(self) -> None:
        wait = (
            'nvmlEventSetWait_v2' if hasattr(libnvml, 'nvmlEventSetWait_v2') else 'nvmlEventSetWait'
        )
        try:
            while not self._stop_event.is_set():
                try:
                    event = libnvml.nvmlQuery(
                        wait,
                        self._event_set,
                        _EVENT_WAIT_TIMEOUT_MS,
                        ignore_errors=False,
                    )
                except libnvml.NVMLError_Timeout:
                    continue
                except libnvml.NVMLError_Uninitialized:  # the interpreter is shutting down
                    return
                except libnvml.NVMLError as ex:  # e.g. GPU is lost: keep listening to the others
                    LOGGER.warning('Failed to wait for XID events: %s', ex)
                    self._stop_event.wait(_ERROR_BACKOFF_SECONDS)
                    continue
                self._record(event)
        finally:
            self._free_event_set()

    def _record(self, event: Any) -> None:
        if not event.eventType & libnvml.nvmlEventTypeXidCriticalError:
            return
        uuid = _query('nvmlDeviceGetUUID', event.device)
        with self._lock:
            counters = self._counters.get(uuid)
            if counters is not None:
                self._counters[uuid] = (int(event.eventData), counters[1] + 1)
                return
        LOGGER.warning('Dropped XID %s from unknown GPU %s', event.eventData, uuid)

    def _free_event_set(self) -> None:
        if self._event_set is not None:
            _query('nvmlEventSetFree', self._event_set)
            self._event_set = None
