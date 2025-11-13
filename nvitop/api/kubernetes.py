"""Kubernetes integration module for extracting pod information from processes."""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass
from typing import Any


try:
    from typing import Self  # type: ignore
except ImportError:
    from typing_extensions import Self  # type: ignore

from nvitop.api.utils import NA, NaType, memoize_when_activated


__all__ = [
    'KubernetesClient',
    'KubernetesInfo',
    'extract_pod_from_pid',
    'get_kubernetes_client',
    'get_kubernetes_info',
    'is_kubernetes_environment',
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


def is_kubernetes_environment() -> bool:
    """Check if the current process is running in a Kubernetes environment.

    Returns:
        True if running in Kubernetes, False otherwise.
    """
    if os.getenv('KUBERNETES_SERVICE_HOST') is not None:
        return True

    token_path = '/var/run/secrets/kubernetes.io/serviceaccount/token'
    if os.path.isfile(token_path):
        return True

    try:
        if os.path.isfile('/proc/1/cgroup'):
            with open('/proc/1/cgroup') as f:
                cgroup_content = f.read()
                if (
                    'docker' in cgroup_content
                    or 'containerd' in cgroup_content
                    or 'crio' in cgroup_content
                ):
                    return True
    except OSError:
        pass

    return False


def extract_pod_from_pid(pid: int) -> dict[str, str | None] | None:
    """Extract pod and container information from process PID using /proc filesystem.

    Args:
        pid: Process ID to extract information from.

    Returns:
        Dictionary containing pod info or None if not found.
    """
    try:
        cgroup_path = f'/proc/{pid}/cgroup'
        if not os.path.isfile(cgroup_path):
            return None

        container_id = None
        pod_uid = None
        with open(cgroup_path) as f:
            for line in f:
                line = line.strip()
                if '::' in line:
                    _, cgroup_path = line.split('::', 1)
                else:
                    parts = line.split(':')
                    if len(parts) >= 3:
                        cgroup_path = parts[2]

                if 'kubepods' in cgroup_path:
                    # Extract pod UID using improved regex
                    pod_uid_pattern = r'pod([a-f0-9_-]+)\.slice'
                    pod_match = re.search(pod_uid_pattern, cgroup_path)
                    if pod_match:
                        pod_uid = pod_match.group(1)

                    # Extract container ID using improved regex
                    container_id_pattern = r'cri-[^-]+-([a-f0-9]{12,})'
                    container_match = re.search(container_id_pattern, cgroup_path)
                    if container_match:
                        container_id = container_match.group(1)

        if container_id is None:
            return None
        return {
            'container_id': container_id,
            'pod_uid': pod_uid,
            'pod_name': None,
            'namespace': None,
        }

    except (OSError, ValueError):
        return None


class KubernetesClient:
    """Minimal Kubernetes API client for pod information retrieval."""

    _instance: KubernetesClient | None = None
    _lock: threading.Lock = threading.Lock()

    def __new__(
        cls,
        kubeconfig_path: str | None = None,
        context: str | None = None,
        use_incluster_config: bool = True,
    ) -> Self:
        """Singleton pattern for Kubernetes client with configuration support."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        kubeconfig_path: str | None = None,
        context: str | None = None,
        use_incluster_config: bool = True,
    ) -> None:
        """Initialize the Kubernetes client with optional kubeconfig support.

        Args:
            kubeconfig_path: Path to kubeconfig file (defaults to ~/.kube/config or KUBECONFIG)
            context: Kubernetes context to use (defaults to current-context)
            use_incluster_config: Whether to fall back to in-cluster config
        """
        self._kubeconfig_path = kubeconfig_path
        self._context = context
        self._use_incluster_config = use_incluster_config

        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._k8s_loaded = False
            self._load_error: str | None = None
            self._setup_client()

    def _setup_client(self) -> None:
        try:
            from kubernetes import config

            if self._kubeconfig_path:
                config.load_kube_config(
                    config_file=self._kubeconfig_path,
                    context=self._context,
                )
            else:
                load_kwargs = {'context': self._context} if self._context else {}
                env_paths = [
                    os.getenv('KUBECONFIG'),
                    os.path.expanduser('~/.kube/config'),
                ]

                for path in env_paths:
                    if path and os.path.isfile(path):
                        config.load_kube_config(config_file=path, **load_kwargs)
                        break
                else:
                    if self._use_incluster_config or not is_kubernetes_environment():
                        config.load_config(**load_kwargs)

            self._k8s_loaded = True

        except (ImportError, OSError, KeyError) as e:
            self._load_error = str(e)
            self._k8s_loaded = False

    @property
    def is_available(self) -> bool:
        """Check if Kubernetes API is available."""
        return self._k8s_loaded

    @staticmethod
    def list_available_contexts(kubeconfig_path: str | None = None) -> list[str]:
        """List all available contexts in kubeconfig file.

        Args:
            kubeconfig_path: Path to kubeconfig file (defaults to KUBECONFIG or ~/.kube/config).

        Returns:
            List of context names, empty list if kubeconfig is not available or invalid.
        """
        try:
            from kubernetes import config

            if kubeconfig_path is None:
                kubeconfig_path = os.getenv('KUBECONFIG') or os.path.expanduser('~/.kube/config')

            if not os.path.isfile(kubeconfig_path):
                return []

            contexts, _ = config.list_kube_config_contexts(config_file=kubeconfig_path)
            return [ctx['name'] for ctx in contexts]

        except ImportError:
            return []
        except (OSError, KeyError, ValueError):
            return []

    @staticmethod
    def get_current_context(kubeconfig_path: str | None = None) -> str | None:
        """Get the currently active context from kubeconfig.

        Args:
            kubeconfig_path: Path to kubeconfig file (defaults to KUBECONFIG or ~/.kube/config).

        Returns:
            Current context name, or None if not available.
        """
        try:
            from kubernetes import config

            if kubeconfig_path is None:
                kubeconfig_path = os.getenv('KUBECONFIG') or os.path.expanduser('~/.kube/config')

            if not os.path.isfile(kubeconfig_path):
                return None

            _, current_context = config.list_kube_config_contexts(
                config_file=kubeconfig_path,
            )
            return current_context.get('name') if current_context else None

        except ImportError:
            return None
        except (OSError, KeyError, ValueError):
            return None

    def _extract_nvidia_gpu_resources(
        self,
        pod_spec: dict,
        container_name: str | None = None,
        container_id: str | None = None,
    ) -> tuple[int, int]:
        """Extract NVIDIA GPU resources from pod specification.

        Args:
            pod_spec: Pod specification dictionary from Kubernetes API.
            container_name: Specific container name to extract from (if None, uses first container).
            container_id: Container ID to match (if provided, prioritized over container_name).

        Returns:
            Tuple of (gpu_requests, gpu_limits) as integers.
        """
        containers = pod_spec.get('containers', [])

        if container_id:
            pass

        if container_name:
            containers = [c for c in containers if c.get('name') == container_name]

        container = containers[0] if containers else {}
        resources = container.get('resources', {})

        requests = resources.get('requests', {})
        limits = resources.get('limits', {})

        gpu_requests = 0
        gpu_limits = 0

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

    def _find_container_name_by_id(self, pod: Any, container_id: str) -> str | None:
        """Find container name by container ID using pod status information.

        Args:
            pod: Kubernetes pod object from API.
            container_id: Container ID to match (can be short or full ID).

        Returns:
            Container name if found, None otherwise.
        """
        try:
            if hasattr(pod.status, 'container_statuses') and pod.status.container_statuses:
                for container_status in pod.status.container_statuses:
                    if hasattr(container_status, 'container_id') and container_status.container_id:
                        k8s_container_id = container_status.container_id
                        if '://' in k8s_container_id:
                            k8s_container_id = k8s_container_id.split('://', 1)[1]

                        if (
                            k8s_container_id == container_id
                            or k8s_container_id.startswith(container_id)
                            or container_id.startswith(k8s_container_id[:12])
                        ):
                            return (
                                container_status.name if hasattr(container_status, 'name') else None
                            )
        except (AttributeError, TypeError, KeyError):
            pass

        return None

    @memoize_when_activated
    def get_pod_info(
        self,
        pod_name: str,
        namespace: str | None = None,
    ) -> KubernetesInfo:
        """Get pod information using official Kubernetes client.

        Args:
            pod_name: Name of the pod.
            namespace: Namespace of the pod (defaults to current namespace).

        Returns:
            KubernetesInfo object with pod details.
        """
        if not self.is_available:
            return KubernetesInfo(NA, NA, NA, NA, NA, NA, NA, NA, NA)

        try:
            from kubernetes.client import CoreV1Api

            api = CoreV1Api()
            pod = api.read_namespaced_pod(
                name=pod_name,
                namespace=namespace or 'default',
            )

            metadata = pod.metadata
            spec = pod.spec

            gpu_requests, gpu_limits = self._extract_nvidia_gpu_resources(
                spec.to_dict(),
            )

            return KubernetesInfo(
                pod_name=metadata.name,
                pod_namespace=metadata.namespace,
                pod_uid=metadata.uid,
                container_name=NA,  # Would need additional logic to determine container
                container_id=NA,
                node_name=spec.node_name,
                pod_labels=metadata.labels or {},
                nvidia_gpu_requests=gpu_requests,
                nvidia_gpu_limits=gpu_limits,
            )

        except (ImportError, OSError, KeyError, ValueError):
            return KubernetesInfo(NA, NA, NA, NA, NA, NA, NA, NA, NA)

    @memoize_when_activated
    def get_pod_by_uid(self, pod_uid: str) -> KubernetesInfo:
        """Get pod information by UID using official Kubernetes client.

        Args:
            pod_uid: UID of the pod.

        Returns:
            KubernetesInfo object with pod details.
        """
        if not self.is_available:
            return KubernetesInfo(NA, NA, NA, NA, NA, NA, NA, NA, NA)

        try:
            from kubernetes.client import CoreV1Api

            api = CoreV1Api()

            common_namespaces = ['default', 'kube-system', 'kube-public']

            for namespace in common_namespaces:
                try:
                    pods = api.list_namespaced_pod(namespace=namespace)
                    for pod in pods.items:
                        # Convert cgroup pod UID (underscores) to Kubernetes UID (dashes)
                        k8s_pod_uid = pod_uid.replace('_', '-')
                        if pod.metadata.uid == k8s_pod_uid:
                            gpu_requests, gpu_limits = self._extract_nvidia_gpu_resources(
                                pod.spec.to_dict(),
                            )

                            return KubernetesInfo(
                                pod_name=pod.metadata.name,
                                pod_namespace=pod.metadata.namespace,
                                pod_uid=pod.metadata.uid,
                                container_name=NA,
                                container_id=NA,
                                node_name=pod.spec.node_name,
                                pod_labels=pod.metadata.labels or {},
                                nvidia_gpu_requests=gpu_requests,
                                nvidia_gpu_limits=gpu_limits,
                            )
                except (ImportError, OSError, KeyError, ValueError):
                    continue

            try:
                from kubernetes.client import CoreV1Api

                api = CoreV1Api()

                namespaces = api.list_namespace()

                for ns in namespaces.items:
                    try:
                        pods = api.list_namespaced_pod(namespace=ns.metadata.name)
                        for pod in pods.items:
                            # Convert cgroup pod UID (underscores) to Kubernetes UID (dashes)
                            k8s_pod_uid = pod_uid.replace('_', '-')
                            if pod.metadata.uid == k8s_pod_uid:
                                gpu_requests, gpu_limits = self._extract_nvidia_gpu_resources(
                                    pod.spec.to_dict(),
                                )

                                return KubernetesInfo(
                                    pod_name=pod.metadata.name,
                                    pod_namespace=pod.metadata.namespace,
                                    pod_uid=pod.metadata.uid,
                                    container_name=NA,
                                    container_id=NA,
                                    node_name=pod.spec.node_name,
                                    pod_labels=pod.metadata.labels or {},
                                    nvidia_gpu_requests=gpu_requests,
                                    nvidia_gpu_limits=gpu_limits,
                                )
                    except (ImportError, OSError, KeyError, ValueError):
                        continue
            except (ImportError, OSError, KeyError, ValueError):
                pods = api.list_pod_for_all_namespaces()
                for pod in pods.items:
                    if pod.metadata.uid == pod_uid:
                        gpu_requests, gpu_limits = self._extract_nvidia_gpu_resources(
                            pod.spec.to_dict(),
                        )

                        return KubernetesInfo(
                            pod_name=pod.metadata.name,
                            pod_namespace=pod.metadata.namespace,
                            pod_uid=pod.metadata.uid,
                            container_name=NA,
                            container_id=NA,
                            node_name=pod.spec.node_name,
                            pod_labels=pod.metadata.labels or {},
                            nvidia_gpu_requests=gpu_requests,
                            nvidia_gpu_limits=gpu_limits,
                        )

            return KubernetesInfo(NA, NA, NA, NA, NA, NA, NA, NA, NA)

        except (ImportError, OSError, KeyError, ValueError):
            return KubernetesInfo(NA, NA, NA, NA, NA, NA, NA, NA, NA)


_kubernetes_client: KubernetesClient | None = None
_client_lock: threading.Lock = threading.Lock()


def _get_kubernetes_client(
    kubeconfig_path: str | None = None,
    context: str | None = None,
    use_incluster_config: bool = True,
) -> KubernetesClient:
    """Get the global Kubernetes client instance with optional configuration.

    Args:
        kubeconfig_path: Path to kubeconfig file.
        context: Kubernetes context to use.
        use_incluster_config: Whether to fall back to in-cluster config.

    Returns:
        KubernetesClient instance.
    """
    global _kubernetes_client
    if _kubernetes_client is None:
        with _client_lock:
            if _kubernetes_client is None:
                _kubernetes_client = KubernetesClient(
                    kubeconfig_path=kubeconfig_path,
                    context=context,
                    use_incluster_config=use_incluster_config,
                )
    return _kubernetes_client


def get_kubernetes_client(
    kubeconfig_path: str | None = None,
    context: str | None = None,
    use_incluster_config: bool = True,
) -> KubernetesClient:
    """Get a configured Kubernetes client instance.

    Args:
        kubeconfig_path: Path to kubeconfig file (defaults to KUBECONFIG or ~/.kube/config).
        context: Kubernetes context to use (defaults to current-context).
        use_incluster_config: Whether to fall back to in-cluster config.

    Returns:
        Configured KubernetesClient instance.

    Examples:
        >>> client = get_kubernetes_client()  # Use default kubeconfig
        >>> client = get_kubernetes_client(context="prod")  # Use specific context
        >>> client = get_kubernetes_client("/path/to/config", "staging")  # Use file and context
    """
    return KubernetesClient(
        kubeconfig_path=kubeconfig_path,
        context=context,
        use_incluster_config=use_incluster_config,
    )


_container_pod_cache: dict[str, KubernetesInfo] = {}
_cache_lock: threading.Lock = threading.Lock()


@memoize_when_activated
def get_kubernetes_info(pid: int) -> KubernetesInfo:
    """Get Kubernetes information for a given process PID.

    Args:
        pid: Process ID to get Kubernetes information for.

    Returns:
        KubernetesInfo object with pod/container details.
    """
    pod_info = extract_pod_from_pid(pid)
    if pod_info is None:
        return KubernetesInfo(NA, NA, NA, NA, NA, NA, NA, NA, NA)

    container_id = pod_info.get('container_id')

    if container_id:
        with _cache_lock:
            if container_id in _container_pod_cache:
                return _container_pod_cache[container_id]

    client = _get_kubernetes_client()
    pod_uid = pod_info.get('pod_uid')
    if pod_uid and client.is_available:
        k8s_info = client.get_pod_by_uid(pod_uid)

        if container_id and container_id is not NA and k8s_info.pod_name is not NA:
            try:
                from kubernetes.client import CoreV1Api

                api = CoreV1Api()
                pod = api.read_namespaced_pod(
                    name=k8s_info.pod_name,
                    namespace=k8s_info.pod_namespace,
                )

                container_name = client._find_container_name_by_id(pod, container_id)
                if container_name:
                    gpu_requests, gpu_limits = client._extract_nvidia_gpu_resources(
                        pod.spec.to_dict(),
                        container_name=container_name,
                    )
                    k8s_info.container_name = container_name
                    k8s_info.nvidia_gpu_requests = gpu_requests
                    k8s_info.nvidia_gpu_limits = gpu_limits

                if container_id:
                    with _cache_lock:
                        _container_pod_cache[container_id] = k8s_info

            except (ImportError, OSError, KeyError, ValueError):
                pass

        if k8s_info.container_id is NA:
            k8s_info.container_id = container_id or NA

        return k8s_info

    basic_info = KubernetesInfo(
        pod_name=pod_info.get('pod_name') or NA,
        pod_namespace=pod_info.get('namespace') or NA,
        pod_uid=pod_info.get('pod_uid') or NA,
        container_name=NA,
        container_id=container_id or NA,
        node_name=NA,
        pod_labels={},
        nvidia_gpu_requests=NA,
        nvidia_gpu_limits=NA,
    )

    if container_id:
        with _cache_lock:
            _container_pod_cache[container_id] = basic_info

    return basic_info
