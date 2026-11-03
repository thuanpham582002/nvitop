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
"""Prometheus exporter built on top of ``nvitop``."""

from __future__ import annotations

import math
import re
import time
from typing import Sequence

from prometheus_client import REGISTRY, CollectorRegistry, Gauge, Info

from nvitop import NA, Device, GpuProcess, MiB, MigDevice, PhysicalDevice, host
from nvitop_exporter.nvml_faults import XidEventMonitor, read_faults
from nvitop_exporter.utils import get_ip_address


_LABEL_NAME_INVALID = re.compile(r'[^a-zA-Z0-9_]')
_RESERVED_POD_LABELS = {'pod_name', 'pod_namespace', 'container_name', 'node_name'}
_MAX_POD_LABELS = 32
_MAX_LABEL_VALUE_LEN = 128


def _sanitize_pod_labels(labels: object) -> dict[str, str]:
    """Turn Kubernetes pod labels into valid Prometheus label names (bounded)."""
    if not isinstance(labels, dict):
        return {}
    result: dict[str, str] = {}
    for key, value in sorted(labels.items())[:_MAX_POD_LABELS]:
        name = _LABEL_NAME_INVALID.sub('_', str(key))
        if not name or name[0].isdigit() or name.startswith('__'):
            name = f'label_{name.lstrip("_")}'
        if name in _RESERVED_POD_LABELS:
            name = f'label_{name}'
        result[name] = str(value)[:_MAX_LABEL_VALUE_LEN]
    return result


class PrometheusExporter:  # pylint: disable=too-many-instance-attributes
    """Prometheus exporter built on top of ``nvitop``."""

    def __init__(  # pylint: disable=too-many-statements
        self,
        devices: Sequence[Device],
        hostname: str | None = None,
        *,
        registry: CollectorRegistry = REGISTRY,
        interval: float = 1.0,
    ) -> None:
        """Initialize the Prometheus exporter."""
        if not isinstance(devices, (list, tuple)):
            raise TypeError(f'Expected a list or tuple of devices, got {type(devices)}')
        devices = list(devices)

        for device in devices:
            if not isinstance(device, (PhysicalDevice, MigDevice)):
                raise TypeError(f'Expected a PhysicalDevice or MigDevice, got {type(device)}')

        self.devices = devices
        self.hostname = hostname or get_ip_address()
        self.registry = registry
        self.interval = interval
        self.alive_pids: dict[Device, set[tuple[int, str]]] = {
            device: set() for device in self.devices
        }
        # Kubernetes label values actually used for each alive process / pod, so that dead
        # series can be removed with the exact same label values (no stale-series leak).
        self.process_k8s_labels: dict[Device, dict[tuple[int, str], tuple[str, ...]]] = {
            device: {} for device in self.devices
        }
        self.pod_label_keys: dict[Device, set[tuple[str, str, str, str]]] = {
            device: set() for device in self.devices
        }

        self.info = Info(
            'nvitop',
            documentation='NVITOP Prometheus Exporter.',
            labelnames=['hostname'],
            registry=self.registry,
        )
        self.info.labels(hostname=self.hostname).info(
            {
                'device_count': str(Device.count()),
                'driver_version': Device.driver_version(),
                'cuda_driver_version': Device.cuda_driver_version(),
            },
        )

        # Create gauges for host metrics
        self.host_uptime = Gauge(
            name='host_uptime',
            documentation='Host uptime (s).',
            unit='Second',
            labelnames=['hostname'],
            registry=self.registry,
        )
        self.host_cpu_percent = Gauge(
            name='host_cpu_percent',
            documentation='Host CPU percent (%).',
            unit='Percentage',
            labelnames=['hostname'],
            registry=self.registry,
        )
        self.host_virtual_memory_total = Gauge(
            name='host_virtual_memory_total',
            documentation='Host virtual memory total (MiB).',
            unit='MiB',
            labelnames=['hostname'],
            registry=self.registry,
        )
        self.host_virtual_memory_used = Gauge(
            name='host_virtual_memory_used',
            documentation='Host virtual memory used (MiB).',
            unit='MiB',
            labelnames=['hostname'],
            registry=self.registry,
        )
        self.host_virtual_memory_free = Gauge(
            name='host_virtual_memory_free',
            documentation='Host virtual memory free (MiB).',
            unit='MiB',
            labelnames=['hostname'],
            registry=self.registry,
        )
        self.host_virtual_memory_percent = Gauge(
            name='host_virtual_memory_percent',
            documentation='Host virtual memory percent (%).',
            unit='Percentage',
            labelnames=['hostname'],
            registry=self.registry,
        )
        self.host_swap_memory_total = Gauge(
            name='host_swap_memory_total',
            documentation='Host swap total (MiB).',
            unit='MiB',
            labelnames=['hostname'],
            registry=self.registry,
        )
        self.host_swap_memory_used = Gauge(
            name='host_swap_memory_used',
            documentation='Host swap used (MiB).',
            unit='MiB',
            labelnames=['hostname'],
            registry=self.registry,
        )
        self.host_swap_memory_free = Gauge(
            name='host_swap_memory_free',
            documentation='Host swap free (MiB).',
            unit='MiB',
            labelnames=['hostname'],
            registry=self.registry,
        )
        self.host_swap_memory_percent = Gauge(
            name='host_swap_memory_percent',
            documentation='Host swap percent (%).',
            unit='Percentage',
            labelnames=['hostname'],
            registry=self.registry,
        )
        self.host_load_average_1m = Gauge(
            name='host_load_average_1m',
            documentation='Host load average for the last minute.',
            unit='Percentage',
            labelnames=['hostname'],
            registry=self.registry,
        )
        self.host_load_average_5m = Gauge(
            name='host_load_average_5m',
            documentation='Host load average for the last 5 minutes.',
            unit='Percentage',
            labelnames=['hostname'],
            registry=self.registry,
        )
        self.host_load_average_15m = Gauge(
            name='host_load_average_15m',
            documentation='Host load average for the last 15 minutes.',
            unit='Percentage',
            labelnames=['hostname'],
            registry=self.registry,
        )
        self.host_net_io_tx_data = Gauge(
            name='host_net_io_tx_data',
            documentation='Host network I/O transmitted data (MiB).',
            unit='MiB',
            labelnames=['hostname', 'interface'],
            registry=self.registry,
        )
        self.host_net_io_rx_data = Gauge(
            name='host_net_io_rx_data',
            documentation='Host network I/O received data (MiB).',
            unit='MiB',
            labelnames=['hostname', 'interface'],
            registry=self.registry,
        )
        self.host_net_io_tx_packets = Gauge(
            name='host_net_io_tx_packets',
            documentation='Host network I/O transmitted packets.',
            unit='Packet',
            labelnames=['hostname', 'interface'],
            registry=self.registry,
        )
        self.host_net_io_rx_packets = Gauge(
            name='host_net_io_rx_packets',
            documentation='Host network I/O received packets.',
            unit='Packet',
            labelnames=['hostname', 'interface'],
            registry=self.registry,
        )
        self.host_disk_io_read_data = Gauge(
            name='host_disk_io_read_data',
            documentation='Host disk I/O read data (MiB).',
            unit='MiB',
            labelnames=['hostname', 'partition'],
            registry=self.registry,
        )
        self.host_disk_io_write_data = Gauge(
            name='host_disk_io_write_data',
            documentation='Host disk I/O write data (MiB).',
            unit='MiB',
            labelnames=['hostname', 'partition'],
            registry=self.registry,
        )
        self.host_disk_usage_total = Gauge(
            name='host_disk_usage_total',
            documentation='Host disk usage total (MiB).',
            unit='MiB',
            labelnames=['hostname', 'mountpoint'],
            registry=self.registry,
        )
        self.host_disk_usage_used = Gauge(
            name='host_disk_usage_used',
            documentation='Host disk usage used (MiB).',
            unit='MiB',
            labelnames=['hostname', 'mountpoint'],
            registry=self.registry,
        )
        self.host_disk_usage_free = Gauge(
            name='host_disk_usage_free',
            documentation='Host disk usage free (MiB).',
            unit='MiB',
            labelnames=['hostname', 'mountpoint'],
            registry=self.registry,
        )
        self.host_disk_usage_percent = Gauge(
            name='host_disk_usage_percent',
            documentation='Host disk usage percent (%).',
            unit='Percentage',
            labelnames=['hostname', 'mountpoint'],
            registry=self.registry,
        )

        # Create gauges for GPU metrics
        self.gpu_utilization = Gauge(
            name='gpu_utilization',
            documentation='GPU utilization (%).',
            unit='Percentage',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_memory_utilization = Gauge(
            name='gpu_memory_utilization',
            documentation='GPU memory utilization (%).',
            unit='Percentage',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_encoder_utilization = Gauge(
            name='gpu_encoder_utilization',
            documentation='GPU encoder utilization (%).',
            unit='Percentage',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_decoder_utilization = Gauge(
            name='gpu_decoder_utilization',
            documentation='GPU decoder utilization (%).',
            unit='Percentage',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_memory_total = Gauge(
            name='gpu_memory_total',
            documentation='GPU memory total (MiB).',
            unit='MiB',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_memory_used = Gauge(
            name='gpu_memory_used',
            documentation='GPU memory used (MiB).',
            unit='MiB',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_memory_free = Gauge(
            name='gpu_memory_free',
            documentation='GPU memory free (MiB).',
            unit='MiB',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_memory_percent = Gauge(
            name='gpu_memory_percent',
            documentation='GPU memory percent (%).',
            unit='Percentage',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_clock_sm = Gauge(
            name='gpu_clock_sm',
            documentation='GPU SM clock (MHz).',
            unit='MHz',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_clock_memory = Gauge(
            name='gpu_clock_memory',
            documentation='GPU memory clock (MHz).',
            unit='MHz',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_clock_graphics = Gauge(
            name='gpu_clock_graphics',
            documentation='GPU graphics clock (MHz).',
            unit='MHz',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_clock_video = Gauge(
            name='gpu_clock_video',
            documentation='GPU video clock (MHz).',
            unit='MHz',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_power_usage = Gauge(
            name='gpu_power_usage',
            documentation='GPU power usage (W).',
            unit='W',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_power_limit = Gauge(
            name='gpu_power_limit',
            documentation='GPU power limit (W).',
            unit='W',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_temperature = Gauge(
            name='gpu_temperature',
            documentation='GPU temperature (C).',
            unit='C',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_fan_speed = Gauge(
            name='gpu_fan_speed',
            documentation='GPU fan speed (%).',
            unit='Percentage',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_pcie_tx_throughput = Gauge(
            name='gpu_pcie_tx_throughput',
            documentation='GPU PCIe transmit throughput (MiB/s).',
            unit='MiBps',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_pcie_rx_throughput = Gauge(
            name='gpu_pcie_rx_throughput',
            documentation='GPU PCIe receive throughput (MiB/s).',
            unit='MiBps',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_nvlink_total_tx_throughput = Gauge(
            name='gpu_nvlink_total_tx_throughput',
            documentation='GPU total NVLink transmit throughput (MiB/s).',
            unit='MiBps',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_nvlink_total_rx_throughput = Gauge(
            name='gpu_nvlink_total_rx_throughput',
            documentation='GPU total NVLink receive throughput (MiB/s).',
            unit='MiBps',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_nvlink_mean_tx_throughput = Gauge(
            name='gpu_nvlink_mean_tx_throughput',
            documentation='GPU mean NVLink transmit throughput (MiB/s).',
            unit='MiBps',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_nvlink_mean_rx_throughput = Gauge(
            name='gpu_nvlink_mean_rx_throughput',
            documentation='GPU mean NVLink receive throughput (MiB/s).',
            unit='MiBps',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_nvlink_tx_throughput = Gauge(
            name='gpu_nvlink_tx_throughput',
            documentation='GPU NVLink transmit throughput (MiB/s).',
            unit='MiBps',
            labelnames=['hostname', 'index', 'devicename', 'uuid', 'link'],
            registry=self.registry,
        )
        self.gpu_nvlink_rx_throughput = Gauge(
            name='gpu_nvlink_rx_throughput',
            documentation='GPU NVLink receive throughput (MiB/s).',
            unit='MiBps',
            labelnames=['hostname', 'index', 'devicename', 'uuid', 'link'],
            registry=self.registry,
        )

        # Create gauges for process metrics
        self.process_info = Info(
            name='process_info',
            documentation='Process information.',
            labelnames=[
                'hostname',
                'index',
                'devicename',
                'uuid',
                'pid',
                'username',
                'pod_name',
                'pod_namespace',
                'pod_uid',
                'container_name',
                'container_id',
                'node_name',
            ],
            registry=self.registry,
        )
        self.process_running_time = Gauge(
            name='process_running_time',
            documentation='Process running time (s).',
            unit='Second',
            labelnames=[
                'hostname',
                'index',
                'devicename',
                'uuid',
                'pid',
                'username',
                'pod_name',
                'pod_namespace',
                'pod_uid',
                'container_name',
                'container_id',
                'node_name',
            ],
            registry=self.registry,
        )
        self.process_cpu_percent = Gauge(
            name='process_cpu_percent',
            documentation='Process CPU percent (%).',
            unit='Percentage',
            labelnames=[
                'hostname',
                'index',
                'devicename',
                'uuid',
                'pid',
                'username',
                'pod_name',
                'pod_namespace',
                'pod_uid',
                'container_name',
                'container_id',
                'node_name',
            ],
            registry=self.registry,
        )
        self.process_rss_memory = Gauge(
            name='process_rss_memory',
            documentation='Process memory resident set size (MiB).',
            unit='MiB',
            labelnames=[
                'hostname',
                'index',
                'devicename',
                'uuid',
                'pid',
                'username',
                'pod_name',
                'pod_namespace',
                'pod_uid',
                'container_name',
                'container_id',
                'node_name',
            ],
            registry=self.registry,
        )
        self.process_memory_percent = Gauge(
            name='process_memory_percent',
            documentation='Process memory percent (%).',
            unit='Percentage',
            labelnames=[
                'hostname',
                'index',
                'devicename',
                'uuid',
                'pid',
                'username',
                'pod_name',
                'pod_namespace',
                'pod_uid',
                'container_name',
                'container_id',
                'node_name',
            ],
            registry=self.registry,
        )
        self.process_gpu_memory = Gauge(
            name='process_gpu_memory',
            documentation='Process GPU memory (MiB).',
            unit='MiB',
            labelnames=[
                'hostname',
                'index',
                'devicename',
                'uuid',
                'pid',
                'username',
                'pod_name',
                'pod_namespace',
                'pod_uid',
                'container_name',
                'container_id',
                'node_name',
            ],
            registry=self.registry,
        )
        self.process_gpu_sm_utilization = Gauge(
            name='process_gpu_sm_utilization',
            documentation='Process GPU SM utilization (%).',
            unit='Percentage',
            labelnames=[
                'hostname',
                'index',
                'devicename',
                'uuid',
                'pid',
                'username',
                'pod_name',
                'pod_namespace',
                'pod_uid',
                'container_name',
                'container_id',
                'node_name',
            ],
            registry=self.registry,
        )
        self.process_gpu_memory_utilization = Gauge(
            name='process_gpu_memory_utilization',
            documentation='Process GPU memory utilization (%).',
            unit='Percentage',
            labelnames=[
                'hostname',
                'index',
                'devicename',
                'uuid',
                'pid',
                'username',
                'pod_name',
                'pod_namespace',
                'pod_uid',
                'container_name',
                'container_id',
                'node_name',
            ],
            registry=self.registry,
        )
        self.process_gpu_encoder_utilization = Gauge(
            name='process_gpu_encoder_utilization',
            documentation='Process GPU encoder utilization (%).',
            unit='Percentage',
            labelnames=[
                'hostname',
                'index',
                'devicename',
                'uuid',
                'pid',
                'username',
                'pod_name',
                'pod_namespace',
                'pod_uid',
                'container_name',
                'container_id',
                'node_name',
            ],
            registry=self.registry,
        )
        self.process_gpu_decoder_utilization = Gauge(
            name='process_gpu_decoder_utilization',
            documentation='Process GPU decoder utilization (%).',
            unit='Percentage',
            labelnames=[
                'hostname',
                'index',
                'devicename',
                'uuid',
                'pid',
                'username',
                'pod_name',
                'pod_namespace',
                'pod_uid',
                'container_name',
                'container_id',
                'node_name',
            ],
            registry=self.registry,
        )

        # Kubernetes-specific metrics
        self.process_pod_labels = Info(
            name='process_pod_labels',
            documentation='Kubernetes pod labels for the process.',
            labelnames=[
                'pod_name',
                'pod_namespace',
                'container_name',
                'node_name',
            ],
            registry=self.registry,
        )
        self.process_nvidia_gpu_requests = Gauge(
            name='process_nvidia_gpu_requests',
            documentation='NVIDIA GPU requests for the process pod.',
            unit='Count',
            labelnames=[
                'pod_name',
                'pod_namespace',
                'container_name',
                'node_name',
            ],
            registry=self.registry,
        )
        self.process_nvidia_gpu_limits = Gauge(
            name='process_nvidia_gpu_limits',
            documentation='NVIDIA GPU limits for the process pod.',
            unit='Count',
            labelnames=[
                'pod_name',
                'pod_namespace',
                'container_name',
                'node_name',
            ],
            registry=self.registry,
        )
        # Hardware faults. These names are a contract with metrics-hub: do not pass `unit` here, it
        # would be appended to the metric name. Only physical GPUs that support them have series.
        self.gpu_xid_last_code = Gauge(
            name='gpu_xid_last_code',
            documentation='Last XID critical error code of the GPU (0 if none since the exporter started).',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_xid_events_total = Gauge(
            name='gpu_xid_events_total',
            documentation='Number of XID critical errors of the GPU since the exporter started.',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_ecc_volatile_corrected_total = Gauge(
            name='gpu_ecc_volatile_corrected_total',
            documentation='Corrected volatile ECC errors of the GPU since the driver loaded.',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_ecc_volatile_uncorrected_total = Gauge(
            name='gpu_ecc_volatile_uncorrected_total',
            documentation='Uncorrected volatile ECC errors of the GPU since the driver loaded.',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_remapped_rows_correctable = Gauge(
            name='gpu_remapped_rows_correctable',
            documentation='GPU memory rows remapped due to correctable errors.',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_remapped_rows_uncorrectable = Gauge(
            name='gpu_remapped_rows_uncorrectable',
            documentation='GPU memory rows remapped due to uncorrectable errors.',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_remapped_rows_pending = Gauge(
            name='gpu_remapped_rows_pending',
            documentation='Whether remapped rows are pending a GPU reset (0 or 1).',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_remapped_rows_failure = Gauge(
            name='gpu_remapped_rows_failure',
            documentation='Whether a row remapping has failed (0 or 1).',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_pcie_replay_total = Gauge(
            name='gpu_pcie_replay_total',
            documentation='PCIe replay counter of the GPU.',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_clock_event_reasons = Gauge(
            name='gpu_clock_event_reasons',
            documentation='Bitmask of the current clock event reasons of the GPU.',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.xid_monitor = XidEventMonitor(
            [device for device in self.devices if not device.is_mig_device()],
        )

    def collect(self) -> None:
        """Collect metrics."""
        self.xid_monitor.start()
        while True:
            next_update_time = time.monotonic() + self.interval
            self.update_host()
            for device in self.devices:
                self.update_device(device)
            time.sleep(max(0.0, next_update_time - time.monotonic()))

    def update_host(self) -> None:
        """Update metrics for the host."""
        load_average = host.load_average()
        if load_average is None:
            load_average = (0.0, 0.0, 0.0)  # type: ignore[unreachable]
        virtual_memory = host.virtual_memory()
        swap_memory = host.swap_memory()
        net_io_counters = host.net_io_counters(pernic=True)  # type: ignore[attr-defined]
        disk_io_counters = host.disk_io_counters(perdisk=True)  # type: ignore[attr-defined]

        for gauge, value in (
            (self.host_uptime, host.uptime()),
            (self.host_cpu_percent, host.cpu_percent()),
            (self.host_virtual_memory_total, virtual_memory.total / MiB),
            (self.host_virtual_memory_used, virtual_memory.used / MiB),
            (self.host_virtual_memory_free, virtual_memory.free / MiB),
            (self.host_virtual_memory_percent, virtual_memory.percent),
            (self.host_swap_memory_total, swap_memory.total / MiB),
            (self.host_swap_memory_used, swap_memory.used / MiB),
            (self.host_swap_memory_free, swap_memory.free / MiB),
            (self.host_swap_memory_percent, swap_memory.percent),
            (self.host_load_average_1m, load_average[0]),
            (self.host_load_average_5m, load_average[1]),
            (self.host_load_average_15m, load_average[2]),
        ):
            gauge.labels(self.hostname).set(value)

        for interface, net_io_counter in net_io_counters.items():
            for gauge, value in (
                (self.host_net_io_tx_data, net_io_counter.bytes_sent / MiB),
                (self.host_net_io_rx_data, net_io_counter.bytes_recv / MiB),
                (self.host_net_io_tx_packets, net_io_counter.packets_sent),
                (self.host_net_io_rx_packets, net_io_counter.packets_recv),
            ):
                gauge.labels(hostname=self.hostname, interface=interface).set(value)

        for partition, disk_io_counter in disk_io_counters.items():
            for gauge, value in (
                (self.host_disk_io_read_data, disk_io_counter.read_bytes / MiB),
                (self.host_disk_io_write_data, disk_io_counter.write_bytes / MiB),
            ):
                gauge.labels(hostname=self.hostname, partition=partition).set(value)

        for partition in host.disk_partitions():  # type: ignore[attr-defined]
            try:
                partition_usage = host.disk_usage(partition.mountpoint)  # type: ignore[attr-defined]
            except (OSError, host.PsutilError):
                continue
            for gauge, value in (
                (self.host_disk_usage_total, partition_usage.total / MiB),
                (self.host_disk_usage_used, partition_usage.used / MiB),
                (self.host_disk_usage_free, partition_usage.free / MiB),
                (self.host_disk_usage_percent, partition_usage.percent),
            ):
                gauge.labels(hostname=self.hostname, mountpoint=partition.mountpoint).set(value)

    def update_device(self, device: Device) -> None:  # pylint: disable=too-many-locals
        """Update metrics for a single device."""
        index = (
            str(device.index) if isinstance(device.index, int) else ':'.join(map(str, device.index))
        )
        name = device.name()
        uuid = device.uuid()

        if not device.is_mig_device():  # these faults only exist on the physical GPU
            self._update_device_faults(device, (self.hostname, index, name, uuid))

        with device.oneshot():
            for gauge, value in (
                (self.gpu_utilization, float(device.gpu_utilization())),
                (self.gpu_memory_utilization, float(device.memory_utilization())),
                (self.gpu_encoder_utilization, float(device.encoder_utilization())),
                (self.gpu_decoder_utilization, float(device.decoder_utilization())),
                (self.gpu_memory_total, device.memory_total() / MiB),
                (self.gpu_memory_used, device.memory_used() / MiB),
                (self.gpu_memory_free, device.memory_free() / MiB),
                (self.gpu_memory_percent, float(device.memory_percent())),
                (self.gpu_clock_sm, float(device.clock_infos().sm)),
                (self.gpu_clock_memory, float(device.clock_infos().memory)),
                (self.gpu_clock_graphics, float(device.clock_infos().graphics)),
                (self.gpu_clock_video, float(device.clock_infos().video)),
                (self.gpu_power_usage, device.power_usage() / 1000.0),
                (self.gpu_power_limit, device.power_limit() / 1000.0),
                (self.gpu_temperature, float(device.temperature())),
                (self.gpu_fan_speed, float(device.fan_speed())),
                (self.gpu_pcie_tx_throughput, device.pcie_tx_throughput() / 1024.0),
                (self.gpu_pcie_rx_throughput, device.pcie_rx_throughput() / 1024.0),
                (self.gpu_nvlink_total_tx_throughput, device.nvlink_total_tx_throughput() / 1024.0),
                (self.gpu_nvlink_total_rx_throughput, device.nvlink_total_rx_throughput() / 1024.0),
                (self.gpu_nvlink_mean_tx_throughput, device.nvlink_mean_tx_throughput() / 1024.0),
                (self.gpu_nvlink_mean_rx_throughput, device.nvlink_mean_rx_throughput() / 1024.0),
            ):
                gauge.labels(
                    hostname=self.hostname,
                    index=index,
                    devicename=name,
                    uuid=uuid,
                ).set(value)

            for gauge, nvlink_throughput in (
                (self.gpu_nvlink_tx_throughput, device.nvlink_tx_throughput()),
                (self.gpu_nvlink_rx_throughput, device.nvlink_rx_throughput()),
            ):
                for link, throughput in enumerate(nvlink_throughput):
                    gauge.labels(
                        hostname=self.hostname,
                        index=index,
                        devicename=name,
                        uuid=uuid,
                        link=link,
                    ).set(throughput / 1024.0)

        alive_pids = self.alive_pids[device]
        previous_alive_pids = alive_pids.copy()
        alive_pids.clear()
        current_pod_keys: set[tuple[str, str, str, str]] = set()

        with GpuProcess.failsafe():
            host_snapshots = {}
            for pid, process in device.processes().items():
                with process.oneshot():
                    username = process.username()
                    if (pid, username) not in host_snapshots:  # noqa: SIM401,RUF100
                        host_snapshot = host_snapshots[pid, username] = process.host_snapshot()
                    else:
                        host_snapshot = host_snapshots[pid, username]

                    # Collect Kubernetes information (never raises; N/A outside Kubernetes)
                    try:
                        k8s_values = tuple(
                            str(v)
                            for v in (
                                process.pod_name(),
                                process.pod_namespace(),
                                process.pod_uid(),
                                process.container_name(),
                                process.container_id(),
                                process.node_name(),
                            )
                        )
                        k8s_pod_labels = process.pod_labels()
                        k8s_gpu_requests = process.nvidia_gpu_requests()
                        k8s_gpu_limits = process.nvidia_gpu_limits()
                    except Exception:  # noqa: BLE001
                        k8s_values = ('N/A',) * 6
                        k8s_pod_labels = {}
                        k8s_gpu_requests = k8s_gpu_limits = NA
                    (
                        k8s_pod_name,
                        k8s_pod_namespace,
                        k8s_pod_uid,
                        k8s_container_name,
                        k8s_container_id,
                        k8s_node_name,
                    ) = k8s_values
                    # The labels of a live process can change (N/A -> pod name once the pod is
                    # resolved, or PID reuse): drop the old series or it keeps its last value forever
                    previous_k8s_values = self.process_k8s_labels[device].get((pid, username))
                    if previous_k8s_values not in (None, k8s_values):
                        self._remove_process_series(
                            (index, name, uuid, pid, username, *previous_k8s_values),
                        )
                    self.process_k8s_labels[device][pid, username] = k8s_values

                    self.process_info.labels(
                        hostname=self.hostname,
                        index=index,
                        devicename=name,
                        uuid=uuid,
                        pid=pid,
                        username=username,
                        pod_name=k8s_pod_name,
                        pod_namespace=k8s_pod_namespace,
                        pod_uid=k8s_pod_uid,
                        container_name=k8s_container_name,
                        container_id=k8s_container_id,
                        node_name=k8s_node_name,
                    ).info(
                        {
                            'status': host_snapshot.status,
                            'command': host_snapshot.command,
                        },
                    )
                    for gauge, value in (
                        (
                            self.process_running_time,
                            (
                                host_snapshot.running_time.total_seconds()
                                if host_snapshot.running_time
                                else math.nan
                            ),
                        ),
                        (self.process_cpu_percent, host_snapshot.cpu_percent),
                        (self.process_rss_memory, host_snapshot.host_memory / MiB),
                        (self.process_memory_percent, float(host_snapshot.memory_percent)),
                        (self.process_gpu_memory, process.gpu_memory() / MiB),
                        (
                            self.process_gpu_sm_utilization,
                            float(process.gpu_sm_utilization()),
                        ),
                        (
                            self.process_gpu_memory_utilization,
                            float(process.gpu_memory_utilization()),
                        ),
                        (
                            self.process_gpu_encoder_utilization,
                            float(process.gpu_encoder_utilization()),
                        ),
                        (
                            self.process_gpu_decoder_utilization,
                            float(process.gpu_decoder_utilization()),
                        ),
                    ):
                        gauge.labels(
                            hostname=self.hostname,
                            index=index,
                            devicename=name,
                            uuid=uuid,
                            pid=pid,
                            username=username,
                            pod_name=k8s_pod_name,
                            pod_namespace=k8s_pod_namespace,
                            pod_uid=k8s_pod_uid,
                            container_name=k8s_container_name,
                            container_id=k8s_container_id,
                            node_name=k8s_node_name,
                        ).set(value)

                    # Pod-level metrics: only for processes that belong to a resolved pod
                    if k8s_pod_name != 'N/A':
                        pod_key = (k8s_pod_name, k8s_pod_namespace, k8s_container_name, k8s_node_name)
                        current_pod_keys.add(pod_key)
                        self.process_pod_labels.labels(*pod_key).info(
                            _sanitize_pod_labels(k8s_pod_labels),
                        )
                        for gauge, value in (
                            (self.process_nvidia_gpu_requests, k8s_gpu_requests),
                            (self.process_nvidia_gpu_limits, k8s_gpu_limits),
                        ):
                            gauge.labels(*pod_key).set(
                                float(value) if isinstance(value, (int, float)) else math.nan,
                            )

        alive_pids.update(host_snapshots)
        k8s_labels = self.process_k8s_labels[device]
        for pid, username in previous_alive_pids.difference(alive_pids):
            label_values = k8s_labels.pop((pid, username), ('N/A',) * 6)
            self._remove_process_series((index, name, uuid, pid, username, *label_values))

        # Remove pod-level series that no device reports anymore
        previous_pod_keys = self.pod_label_keys[device]
        self.pod_label_keys[device] = current_pod_keys
        still_alive = set().union(*self.pod_label_keys.values())
        for pod_key in previous_pod_keys - still_alive:
            for collector in (
                self.process_pod_labels,
                self.process_nvidia_gpu_requests,
                self.process_nvidia_gpu_limits,
            ):
                try:
                    collector.remove(*pod_key)
                except KeyError:  # noqa: PERF203
                    pass

    def _remove_process_series(self, label_values: tuple[object, ...]) -> None:
        """Remove the series of a process from every per-process metric."""
        for collector in (
            self.process_info,
            self.process_running_time,
            self.process_cpu_percent,
            self.process_rss_memory,
            self.process_memory_percent,
            self.process_gpu_memory,
            self.process_gpu_sm_utilization,
            self.process_gpu_memory_utilization,
            self.process_gpu_encoder_utilization,
            self.process_gpu_decoder_utilization,
        ):
            try:
                collector.remove(self.hostname, *label_values)
            except KeyError:  # noqa: PERF203
                pass

    def _update_device_faults(self, device: Device, labels: tuple[str, str, str, str]) -> None:
        """Update the hardware fault metrics. A value NVML cannot give has no series."""
        faults = read_faults(device.handle)
        xid_last_code, xid_events = self.xid_monitor.get(labels[3]) or (NA, NA)
        for gauge, value in (
            (self.gpu_xid_last_code, xid_last_code),
            (self.gpu_xid_events_total, xid_events),
            (self.gpu_ecc_volatile_corrected_total, faults['ecc_volatile_corrected']),
            (self.gpu_ecc_volatile_uncorrected_total, faults['ecc_volatile_uncorrected']),
            (self.gpu_remapped_rows_correctable, faults['remapped_rows_correctable']),
            (self.gpu_remapped_rows_uncorrectable, faults['remapped_rows_uncorrectable']),
            (self.gpu_remapped_rows_pending, faults['remapped_rows_pending']),
            (self.gpu_remapped_rows_failure, faults['remapped_rows_failure']),
            (self.gpu_pcie_replay_total, faults['pcie_replay']),
            (self.gpu_clock_event_reasons, faults['clock_event_reasons']),
        ):
            if value is NA:  # missing is not zero: also drop a value exported earlier
                try:
                    gauge.remove(*labels)
                except KeyError:
                    pass
            else:
                gauge.labels(*labels).set(value)
