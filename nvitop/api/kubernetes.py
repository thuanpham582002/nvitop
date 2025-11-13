# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
#
# Copyright 2021-2025 Xuehai Pan. All Rights Reserved.
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
"""Kubernetes integration module for extracting pod information from processes."""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass
from typing import Any

from nvitop.api import host
from nvitop.api.utils import NA, NaType, ttl_cache


__all__ = [
    'KubernetesInfo',
    'KubernetesClient',
    'is_kubernetes_environment',
    'extract_pod_from_pid',
    'get_kubernetes_info',
]


@dataclass
class KubernetesInfo:
    """Container for Kubernetes pod and container information."""
    pod_name: str | NaType
    pod_namespace: str | NaType
    pod_uid: str | NaType
    container_name: str | NaType
    container_id: str | NaType
    node_name: str | NaType
    pod_labels: dict[str, str] | NaType
    nvidia_gpu_requests: int | NaType
    nvidia_gpu_limits: int | NaType


class KubernetesError(Exception):
    """Exception raised for Kubernetes-related errors."""
    pass


def is_kubernetes_environment() -> bool:
    """Check if the current process is running in a Kubernetes environment.

    Returns:
        True if running in Kubernetes, False otherwise.
    """
    # Check for Kubernetes service environment variables
    if os.getenv('KUBERNETES_SERVICE_HOST') is not None:
        return True

    # Check for service account token
    token_path = '/var/run/secrets/kubernetes.io/serviceaccount/token'
    if os.path.isfile(token_path):
        return True

    # Check for container runtime in cgroup
    try:
        if os.path.isfile('/proc/1/cgroup'):
            with open('/proc/1/cgroup', 'r') as f:
                cgroup_content = f.read()
                if 'docker' in cgroup_content or 'containerd' in cgroup_content or 'crio' in cgroup_content:
                    return True
    except (OSError, IOError):
        pass

    return False


def extract_pod_from_pid(pid: int) -> dict[str, str] | None:
    """Extract pod and container information from process PID using /proc filesystem.

    Args:
        pid: Process ID to extract information from.

    Returns:
        Dictionary containing pod info or None if not found.
    """
    try:
        # Read cgroup information to find container ID
        cgroup_path = f'/proc/{pid}/cgroup'
        if not os.path.isfile(cgroup_path):
            return None

        container_id = None
        with open(cgroup_path, 'r') as f:
            for line in f:
                line = line.strip()
                # cgroup v1 format: hierarchy-ID:subsystem-list:cgroup-path
                # cgroup v2 format: 0::cgroup-path
                if '::' in line:  # cgroup v2
                    _, cgroup_path = line.split('::', 1)
                else:  # cgroup v1
                    parts = line.split(':')
                    if len(parts) >= 3:
                        cgroup_path = parts[2]

                # Extract container ID from cgroup path
                # Typical format: .../kubepods/.../pod<uid>/<container_id>
                if 'kubepods' in cgroup_path:
                    # Extract container ID (last part of the path)
                    path_parts = cgroup_path.split('/')
                    if path_parts:
                        potential_id = path_parts[-1]
                        # Container ID is typically 64 characters
                        if len(potential_id) >= 12 and re.match(r'^[a-f0-9]{12,}', potential_id):
                            container_id = potential_id
                            break

        if container_id is None:
            return None

        # Try to get pod information from container ID
        # This is a simplified approach - in practice you'd query the Kubernetes API
        pod_info = {
            'container_id': container_id,
            'pod_uid': None,
            'pod_name': None,
            'namespace': None,
        }

        # Extract pod UID from container runtime path if available
        # This varies by container runtime, so we'll keep it simple
        return pod_info

    except (OSError, IOError, ValueError):
        return None


class KubernetesClient:
    """Minimal Kubernetes API client for pod information retrieval."""

    _instance: KubernetesClient | None = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> KubernetesClient:
        """Singleton pattern for Kubernetes client."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the Kubernetes client."""
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._token = None
            self._namespace = None
            self._api_host = None
            self._setup_client()

    def _setup_client(self) -> None:
        """Setup Kubernetes API client configuration."""
        try:
            # Get service account token
            token_path = '/var/run/secrets/kubernetes.io/serviceaccount/token'
            if os.path.isfile(token_path):
                with open(token_path, 'r') as f:
                    self._token = f.read().strip()

            # Get namespace from service account
            namespace_path = '/var/run/secrets/kubernetes.io/serviceaccount/namespace'
            if os.path.isfile(namespace_path):
                with open(namespace_path, 'r') as f:
                    self._namespace = f.read().strip()

            # Get API server host
            host = os.getenv('KUBERNETES_SERVICE_HOST')
            port = os.getenv('KUBERNETES_SERVICE_PORT', '443')
            if host:
                self._api_host = f'https://{host}:{port}'

        except (OSError, IOError):
            pass

    @property
    def is_available(self) -> bool:
        """Check if Kubernetes API is available."""
        return (self._token is not None and
                self._namespace is not None and
                self._api_host is not None)

    def _extract_nvidia_gpu_resources(self, pod_spec: dict, container_name: str | None = None) -> tuple[int, int]:
        """Extract NVIDIA GPU resources from pod specification.

        Args:
            pod_spec: Pod specification dictionary from Kubernetes API.
            container_name: Specific container name to extract from (if None, uses first container).

        Returns:
            Tuple of (gpu_requests, gpu_limits) as integers.
        """
        # Find the container
        containers = pod_spec.get('containers', [])
        if container_name:
            containers = [c for c in containers if c.get('name') == container_name]

        container = containers[0] if containers else {}
        resources = container.get('resources', {})

        # Extract NVIDIA GPU requests and limits
        requests = resources.get('requests', {})
        limits = resources.get('limits', {})

        gpu_requests = 0
        gpu_limits = 0

        # Parse nvidia.com/gpu values
        if 'nvidia.com/gpu' in requests:
            try:
                gpu_requests = int(requests['nvidia.com/gpu'])
            except (ValueError, TypeError):
                gpu_requests = 0

        if 'nvidia.com/gpu' in limits:
            try:
                gpu_limits = int(limits['nvidia.com/gpu'])
            except (ValueError, TypeError):
                gpu_limits = 0

        return gpu_requests, gpu_limits

    @ttl_cache(ttl=60)  # Cache for 60 seconds
    def get_pod_info(self, pod_name: str, namespace: str | None = None) -> KubernetesInfo:
        """Get pod information from Kubernetes API.

        Args:
            pod_name: Name of the pod.
            namespace: Namespace of the pod (defaults to service account namespace).

        Returns:
            KubernetesInfo object with pod details.
        """
        if not self.is_available:
            return KubernetesInfo(NA, NA, NA, NA, NA, NA, NA)

        if namespace is None:
            namespace = self._namespace

        try:
            import requests

            # Make API request to get pod information
            headers = {'Authorization': f'Bearer {self._token}'}
            url = f'{self._api_host}/api/v1/namespaces/{namespace}/pods/{pod_name}'

            response = requests.get(url, headers=headers, timeout=5, verify='/var/run/secrets/kubernetes.io/serviceaccount/ca.crt')
            response.raise_for_status()

            pod_data = response.json()

            # Extract relevant information
            metadata = pod_data.get('metadata', {})
            spec = pod_data.get('spec', {})

            # Extract NVIDIA GPU resources
            gpu_requests, gpu_limits = self._extract_nvidia_gpu_resources(spec)

            return KubernetesInfo(
                pod_name=metadata.get('name', NA),
                pod_namespace=metadata.get('namespace', NA),
                pod_uid=metadata.get('uid', NA),
                container_name=NA,  # Would need additional logic to determine container
                container_id=NA,
                node_name=spec.get('nodeName', NA),
                pod_labels=metadata.get('labels', NA) or {},
                nvidia_gpu_requests=gpu_requests,
                nvidia_gpu_limits=gpu_limits
            )

        except Exception:
            # Return NA values on any error
            return KubernetesInfo(NA, NA, NA, NA, NA, NA, NA, NA, NA)

    @ttl_cache(ttl=60)  # Cache for 60 seconds
    def get_pod_by_uid(self, pod_uid: str) -> KubernetesInfo:
        """Get pod information by UID using Kubernetes API.

        Args:
            pod_uid: UID of the pod.

        Returns:
            KubernetesInfo object with pod details.
        """
        if not self.is_available:
            return KubernetesInfo(NA, NA, NA, NA, NA, NA, NA, NA, NA)

        try:
            import requests

            # Search for pod by UID across all namespaces
            headers = {'Authorization': f'Bearer {self._token}'}
            url = f'{self._api_host}/api/v1/pods'

            response = requests.get(url, headers=headers, timeout=5, verify='/var/run/secrets/kubernetes.io/serviceaccount/ca.crt')
            response.raise_for_status()

            pods_data = response.json()

            # Find pod with matching UID
            for pod in pods_data.get('items', []):
                if pod.get('metadata', {}).get('uid') == pod_uid:
                    metadata = pod.get('metadata', {})
                    spec = pod.get('spec', {})

                    # Extract NVIDIA GPU resources
                    gpu_requests, gpu_limits = self._extract_nvidia_gpu_resources(spec)

                    return KubernetesInfo(
                        pod_name=metadata.get('name', NA),
                        pod_namespace=metadata.get('namespace', NA),
                        pod_uid=metadata.get('uid', NA),
                        container_name=NA,
                        container_id=NA,
                        node_name=spec.get('nodeName', NA),
                        pod_labels=metadata.get('labels', NA) or {},
                        nvidia_gpu_requests=gpu_requests,
                        nvidia_gpu_limits=gpu_limits
                    )

            # Pod not found
            return KubernetesInfo(NA, NA, NA, NA, NA, NA, NA, NA, NA)

        except Exception:
            return KubernetesInfo(NA, NA, NA, NA, NA, NA, NA, NA, NA)


# Global Kubernetes client instance
_kubernetes_client: KubernetesClient | None = None
_client_lock: threading.Lock = threading.Lock()


def _get_kubernetes_client() -> KubernetesClient:
    """Get the global Kubernetes client instance."""
    global _kubernetes_client
    if _kubernetes_client is None:
        with _client_lock:
            if _kubernetes_client is None:
                _kubernetes_client = KubernetesClient()
    return _kubernetes_client


@ttl_cache(ttl=30)  # Cache for 30 seconds
def get_kubernetes_info(pid: int) -> KubernetesInfo:
    """Get Kubernetes information for a given process PID.

    Args:
        pid: Process ID to get Kubernetes information for.

    Returns:
        KubernetesInfo object with pod/container details.
    """
    if not is_kubernetes_environment():
        return KubernetesInfo(NA, NA, NA, NA, NA, NA, NA, NA, NA)

    # Extract pod information from /proc
    pod_info = extract_pod_from_pid(pid)
    if pod_info is None:
        return KubernetesInfo(NA, NA, NA, NA, NA, NA, NA, NA, NA)

    # If we have a pod UID, try to get detailed information from API
    client = _get_kubernetes_client()
    if pod_info.get('pod_uid') and client.is_available:
        k8s_info = client.get_pod_by_uid(pod_info['pod_uid'])
        if k8s_info.container_id is NA:
            k8s_info.container_id = pod_info.get('container_id', NA)
        return k8s_info

    # Return basic information from /proc parsing
    return KubernetesInfo(
        pod_name=pod_info.get('pod_name', NA),
        pod_namespace=pod_info.get('namespace', NA),
        pod_uid=pod_info.get('pod_uid', NA),
        container_name=NA,
        container_id=pod_info.get('container_id', NA),
        node_name=NA,
        pod_labels={},
        nvidia_gpu_requests=NA,
        nvidia_gpu_limits=NA
    )