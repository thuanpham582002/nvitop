"""Tests for the hardware fault metrics and the process series lifecycle (no GPU needed)."""

from __future__ import annotations

import contextlib
import datetime
import threading
import time
import types

import pytest
from prometheus_client import CollectorRegistry, generate_latest
from prometheus_client.openmetrics.exposition import generate_latest as generate_openmetrics

from nvitop import NA, libnvml
from nvitop_exporter import exporter as exporter_module
from nvitop_exporter import nvml_faults
from nvitop_exporter.exporter import PrometheusExporter
from nvitop_exporter.nvml_faults import XidEventMonitor, read_faults


LABELS = {'hostname': 'h1', 'index': '0', 'devicename': 'A100', 'uuid': 'GPU-1'}
FAULT_METRICS = {
    'gpu_xid_last_code': 79.0,
    'gpu_xid_events_total': 2.0,
    'gpu_ecc_volatile_corrected_total': 5.0,
    'gpu_ecc_volatile_uncorrected_total': 7.0,
    'gpu_remapped_rows_correctable': 11.0,
    'gpu_remapped_rows_uncorrectable': 13.0,
    'gpu_remapped_rows_pending': 1.0,
    'gpu_remapped_rows_failure': 0.0,
    'gpu_pcie_replay_total': 17.0,
    'gpu_clock_event_reasons': 0x28,
}


def fake_nvml(overrides=None):
    """Build a ``nvmlQuery`` replacement where ``overrides`` maps a function to a result or error."""
    results = {
        'nvmlDeviceGetTotalEccErrors': lambda _handle, kind, _location: (
            5 if kind == libnvml.NVML_MEMORY_ERROR_TYPE_CORRECTED else 7
        ),
        'nvmlDeviceGetRemappedRows': lambda _handle: (11, 13, 1, 0),
        'nvmlDeviceGetPcieReplayCounter': lambda _handle: 17,
        'nvmlDeviceGetCurrentClocksEventReasons': lambda _handle: 0x28,
        'nvmlDeviceGetCurrentClocksThrottleReasons': lambda _handle: NA,
    }
    results.update(overrides or {})

    def query(func, *args, default=NA, ignore_errors=True, **_kwargs):
        try:
            return results[func](*args)
        except libnvml.NVMLError:
            if ignore_errors:
                return default
            raise

    return query


def not_supported(*_args):
    raise libnvml.NVMLError(libnvml.NVML_ERROR_NOT_SUPPORTED)


class FakeDevice:
    handle = 'handle-0'
    index = 0
    mig = False
    procs = ()

    def __getattr__(self, _name):
        return lambda: 1000

    def __hash__(self):
        return 1

    def is_mig_device(self):
        return self.mig

    def name(self):
        return 'A100'

    def uuid(self):
        return 'GPU-1'

    def clock_infos(self):
        return types.SimpleNamespace(sm=1, memory=1, graphics=1, video=1)

    def nvlink_tx_throughput(self):
        return []

    nvlink_rx_throughput = nvlink_tx_throughput

    @contextlib.contextmanager
    def oneshot(self):
        yield

    def processes(self):
        return {process.pid: process for process in self.procs}


class FakeProcess:
    pid = 42
    pod = None

    def __init__(self, pod=None):
        self.pod = pod

    @contextlib.contextmanager
    def oneshot(self):
        yield

    def username(self):
        return 'root'

    def host_snapshot(self):
        return types.SimpleNamespace(
            status='R',
            command='python train.py',
            running_time=datetime.timedelta(seconds=1),
            cpu_percent=1.0,
            host_memory=1,
            memory_percent=1.0,
        )

    def gpu_memory(self):
        return 1

    def gpu_sm_utilization(self):
        return 1

    gpu_memory_utilization = gpu_encoder_utilization = gpu_decoder_utilization = gpu_sm_utilization

    def pod_name(self):
        return self.pod or NA_

    def pod_namespace(self):
        return 'team-a' if self.pod else NA_

    pod_uid = container_name = container_id = node_name = pod_namespace

    def pod_labels(self):
        return {} if self.pod else NA_

    def nvidia_gpu_requests(self):
        return 1 if self.pod else NA_

    nvidia_gpu_limits = nvidia_gpu_requests


@pytest.fixture
def make_exporter(monkeypatch):
    monkeypatch.setattr(exporter_module, 'PhysicalDevice', FakeDevice)
    monkeypatch.setattr(exporter_module.GpuProcess, 'failsafe', contextlib.nullcontext)
    monkeypatch.setattr(
        exporter_module,
        'Device',
        types.SimpleNamespace(
            count=lambda: 1,
            driver_version=lambda: '550',
            cuda_driver_version=lambda: '12.4',
        ),
    )

    def make(device=None):
        device = device or FakeDevice()
        return PrometheusExporter(
            [device],
            hostname='h1',
            registry=CollectorRegistry(),
            interval=5,
        ), device

    return make


def samples(exporter):
    return {
        sample.name: sample
        for metric in exporter.registry.collect()
        for sample in metric.samples
        if sample.name in FAULT_METRICS
    }


def test_metric_names_labels_and_values_match_the_contract(make_exporter, monkeypatch):
    monkeypatch.setattr(libnvml, 'nvmlQuery', fake_nvml())
    exporter, device = make_exporter()
    exporter.xid_monitor._counters['GPU-1'] = (79, 2)

    exporter.update_device(device)

    found = samples(exporter)
    assert {name: sample.value for name, sample in found.items()} == FAULT_METRICS
    assert all(sample.labels == LABELS for sample in found.values())
    text = generate_latest(exporter.registry).decode()
    assert 'gpu_ecc_volatile_uncorrected_total{' in text
    assert 'gpu_xid_events_total{' in text
    assert generate_openmetrics(exporter.registry)


def test_unsupported_card_has_no_series(make_exporter, monkeypatch):
    monkeypatch.setattr(
        libnvml,
        'nvmlQuery',
        fake_nvml(
            {
                'nvmlDeviceGetTotalEccErrors': not_supported,
                'nvmlDeviceGetRemappedRows': not_supported,
                'nvmlDeviceGetPcieReplayCounter': not_supported,
                'nvmlDeviceGetCurrentClocksEventReasons': not_supported,
                'nvmlDeviceGetCurrentClocksThrottleReasons': not_supported,
            },
        ),
    )
    exporter, device = make_exporter()

    exporter.update_device(device)

    assert samples(exporter) == {}


def test_series_is_dropped_when_a_value_becomes_unavailable(make_exporter, monkeypatch):
    monkeypatch.setattr(libnvml, 'nvmlQuery', fake_nvml())
    exporter, device = make_exporter()
    exporter.update_device(device)
    assert 'gpu_pcie_replay_total' in samples(exporter)

    monkeypatch.setattr(
        libnvml,
        'nvmlQuery',
        fake_nvml({'nvmlDeviceGetPcieReplayCounter': not_supported}),
    )
    exporter.update_device(device)

    assert 'gpu_pcie_replay_total' not in samples(exporter)
    assert 'gpu_remapped_rows_correctable' in samples(exporter)


def test_mig_device_has_no_fault_series(make_exporter, monkeypatch):
    monkeypatch.setattr(libnvml, 'nvmlQuery', fake_nvml())
    device = FakeDevice()
    device.mig = True
    exporter, device = make_exporter(device)

    exporter.update_device(device)

    assert samples(exporter) == {}
    assert exporter.xid_monitor._devices == []


def test_nvml_library_error_does_not_break_the_read(monkeypatch):
    def broken(*_args, **_kwargs):
        raise libnvml.NVMLError_LibraryNotFound

    monkeypatch.setattr(libnvml, 'nvmlQuery', broken)

    assert all(value is NA for value in read_faults('handle').values())


def test_clock_reasons_fall_back_to_the_old_function_name(monkeypatch):
    monkeypatch.setattr(
        libnvml,
        'nvmlQuery',
        fake_nvml(
            {
                'nvmlDeviceGetCurrentClocksEventReasons': not_supported,
                'nvmlDeviceGetCurrentClocksThrottleReasons': lambda _handle: 0x40,
            },
        ),
    )

    assert read_faults('handle')['clock_event_reasons'] == 0x40


def test_series_of_a_live_process_follows_its_pod_labels(make_exporter, monkeypatch):
    monkeypatch.setattr(libnvml, 'nvmlQuery', fake_nvml())
    exporter, device = make_exporter()

    def pod_names():
        return [
            sample.labels['pod_name']
            for metric in exporter.registry.collect()
            for sample in metric.samples
            if sample.name == 'process_gpu_memory_MiB'
        ]

    device.procs = [FakeProcess()]
    exporter.update_device(device)
    assert pod_names() == ['N/A']

    device.procs = [FakeProcess('llm-0')]
    exporter.update_device(device)
    assert pod_names() == ['llm-0']

    device.procs = []
    exporter.update_device(device)
    assert pod_names() == []


def xid_nvml(events, register=None):
    """Fake the NVML event API: ``events`` is drained by the wait, then it times out."""
    drained = threading.Event()

    def wait(_event_set, _timeout):
        if events:
            return events.pop(0)
        drained.set()
        time.sleep(0.01)
        raise libnvml.NVMLError(libnvml.NVML_ERROR_TIMEOUT)

    query = fake_nvml(
        {
            'nvmlEventSetCreate': lambda: 'event-set',
            'nvmlEventSetFree': lambda _event_set: None,
            'nvmlDeviceRegisterEvents': register or (lambda *_args: None),
            'nvmlEventSetWait_v2': wait,
            'nvmlEventSetWait': wait,
            'nvmlDeviceGetUUID': lambda handle: handle,
        },
    )
    return query, drained


def xid_event(xid, device='GPU-1', event_type=None):
    return types.SimpleNamespace(
        device=device,
        eventData=xid,
        eventType=libnvml.nvmlEventTypeXidCriticalError if event_type is None else event_type,
    )


def test_xid_monitor_counts_events_after_timeouts(monkeypatch):
    query, drained = xid_nvml([xid_event(48), xid_event(79), xid_event(13, device='GPU-other')])
    monkeypatch.setattr(libnvml, 'nvmlQuery', query)
    monitor = XidEventMonitor([FakeDevice()])

    assert monitor.get('GPU-1') is None
    monitor.start()
    assert drained.wait(5)
    monitor.stop()

    assert monitor.get('GPU-1') == (79, 2)
    assert monitor._event_set is None


def test_xid_monitor_starts_at_zero(monkeypatch):
    query, drained = xid_nvml([])
    monkeypatch.setattr(libnvml, 'nvmlQuery', query)
    monitor = XidEventMonitor([FakeDevice()])

    monitor.start()
    assert drained.wait(5)
    monitor.stop()

    assert monitor.get('GPU-1') == (0, 0)


def test_xid_monitor_does_not_listen_when_registration_fails(monkeypatch):
    query, _ = xid_nvml([], register=not_supported)
    monkeypatch.setattr(libnvml, 'nvmlQuery', query)
    monitor = XidEventMonitor([FakeDevice()])

    monitor.start()

    assert monitor._thread is None
    assert monitor.get('GPU-1') is None
    assert monitor._event_set is None


def test_xid_monitor_survives_wait_errors(monkeypatch):
    query, drained = xid_nvml([])
    errors = [libnvml.NVMLError(libnvml.NVML_ERROR_GPU_IS_LOST)]
    real_query = query

    def flaky(func, *args, **kwargs):
        if func.startswith('nvmlEventSetWait') and errors:
            raise errors.pop()
        return real_query(func, *args, **kwargs)

    monkeypatch.setattr(libnvml, 'nvmlQuery', flaky)
    monkeypatch.setattr(nvml_faults, '_ERROR_BACKOFF_SECONDS', 0.0)
    monitor = XidEventMonitor([FakeDevice()])

    monitor.start()
    assert drained.wait(5)
    monitor.stop()

    assert monitor.get('GPU-1') == (0, 0)
