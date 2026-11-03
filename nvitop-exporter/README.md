# nvitop-exporter

Prometheus exporter built on top of `nvitop`.

## Quickstart

Start the exporter with the following command:

```bash
uvx nvitop-exporter --bind-address 0.0.0.0 --port 5050
# or
pipx run nvitop-exporter --bind-address 0.0.0.0 --port 5050
```

Then you can access the metrics at [`http://localhost:5050/metrics`](http://localhost:5050/metrics).

You will need to configure Prometheus to scrape the metrics from the exporter.

```yaml
scrape_configs:
  - job_name: 'nvitop-exporter'
    static_configs:
      - targets: ['localhost:5050']
```

## TLS / mTLS

The exporter can serve metrics over HTTPS, optionally requiring client certificate authentication (mTLS). TLS support is provided by `prometheus_client` (>= 0.19.0) and configured entirely through CLI flags — no config file is involved.

### Plain HTTPS

Provide a server certificate and private key:

```bash
nvitop-exporter --bind-address 0.0.0.0 --port 5050 \
    --certfile /path/to/server.crt \
    --keyfile /path/to/server.key
```

The metrics endpoint is then served at [`https://localhost:5050/metrics`](https://localhost:5050/metrics). Update the Prometheus scrape config to use the `https` scheme, and point it at the CA that signed your server certificate:

```yaml
scrape_configs:
  - job_name: 'nvitop-exporter'
    scheme: https
    static_configs:
      - targets: ['localhost:5050']
    tls_config:
      ca_file: /path/to/server-ca.crt
```

### Mutual TLS (mTLS)

To require scrapers to present a valid client certificate, pass a CA bundle (`--client-cafile`) or CA directory (`--client-capath`) **and** `--client-auth-required`:

```bash
nvitop-exporter --bind-address 0.0.0.0 --port 5050 \
    --certfile /path/to/server.crt \
    --keyfile /path/to/server.key \
    --client-cafile /path/to/clients-ca.crt \
    --client-auth-required
```

`--client-cafile` / `--client-capath` and `--client-auth-required` must be specified together. Passing a CA without `--client-auth-required` is rejected by the CLI to avoid the silent "trust but don't verify" configuration that the underlying `prometheus_client` API would otherwise allow.

Configure Prometheus to present its client certificate when scraping:

```yaml
scrape_configs:
  - job_name: 'nvitop-exporter'
    scheme: https
    static_configs:
      - targets: ['localhost:5050']
    tls_config:
      ca_file: /path/to/server-ca.crt
      cert_file: /path/to/prometheus-client.crt
      key_file: /path/to/prometheus-client.key
```

### Authentication beyond mTLS

The exporter does not implement HTTP basic auth, OAuth, or IP allowlisting. Following the standard Prometheus exporter pattern, run the exporter behind a reverse proxy (`NGINX`, `Traefik`, `Caddy`, ...) if any of those are required.

## Hardware Fault Metrics

The exporter reads the following metrics from NVML for each physical GPU, with the labels `hostname`, `index`, `devicename` and `uuid`. MIG devices have none of them: their faults are reported on the parent GPU. A value that NVML cannot give (e.g. `NOT_SUPPORTED` on this card) has no series, so a missing series means "unknown", not zero.

| Metric | NVML function | Meaning |
| --- | --- | --- |
| `gpu_xid_last_code` | `nvmlEventSetWait`, event `XidCriticalError` | Last XID code, 0 if none since the exporter started |
| `gpu_xid_events_total` | as above | Number of XID events since the exporter started |
| `gpu_ecc_volatile_corrected_total`, `gpu_ecc_volatile_uncorrected_total` | `nvmlDeviceGetTotalEccErrors` | Volatile ECC errors since the driver loaded |
| `gpu_remapped_rows_correctable`, `gpu_remapped_rows_uncorrectable` | `nvmlDeviceGetRemappedRows` | Remapped memory rows |
| `gpu_remapped_rows_pending`, `gpu_remapped_rows_failure` | as above | A remap waits for a GPU reset; a remap has failed (0 or 1) |
| `gpu_pcie_replay_total` | `nvmlDeviceGetPcieReplayCounter` | PCIe replay counter |
| `gpu_clock_event_reasons` | `nvmlDeviceGetCurrentClocksEventReasons` | Bitmask of clock event reasons: `0x4` software power cap, `0x8` hardware slowdown, `0x20` software thermal, `0x40` hardware thermal, `0x80` power brake |

All of them are gauges holding the absolute value read from NVML, and the counters can go down when the driver reloads or the exporter restarts. An XID raised while the exporter is not running is lost.

## Grafana Dashboard

A Grafana dashboard is provided to visualize the metrics collected by the exporter.
The source of the dashboard is [`dashboard.json`](grafana/dashboard.json).
The Grafana dashboard can also be imported as by ID [22589](https://grafana.com/grafana/dashboards/22589-nvitop-dashboard).

If you are using [`docker-compose`](https://docs.docker.com/compose), you can start a dashboard at [`http://localhost:3000`](http://localhost:3000) with the following command:

```bash
cd nvitop-exporter/grafana
docker compose up --build --detach
```

<p align="center">
  <img width="100%" src="https://github.com/user-attachments/assets/e4867e64-2ca9-45bc-b524-929053f9673d" alt="Grafana Dashboard">
  <br/>
  The Grafana dashboard for the exporter.
</p>
