#!/usr/bin/env bash
set -euo pipefail

# Receive vLLM OTLP traces and save them to disk for offline inspection.
#
# Usage:
#   OUTPUT_DIR=/workspace/project/RL-learning/vllm-test/outputs/run1/otel_traces \
#     bash /workspace/project/RL-learning/vllm-test/scripts/run_otel_file_collector.sh
#
# Benchmark endpoint:
#   --otlp-traces-endpoint http://localhost:4317 --otlp-traces-protocol grpc

OUTPUT_DIR="${OUTPUT_DIR:-/workspace/project/RL-learning/vllm-test/outputs/otel_traces}"
TRACE_FILE="${TRACE_FILE:-${OUTPUT_DIR}/traces.json}"
OTLP_GRPC_PORT="${OTLP_GRPC_PORT:-4317}"
OTLP_HTTP_PORT="${OTLP_HTTP_PORT:-4318}"
OTEL_COLLECTOR_BIN="${OTEL_COLLECTOR_BIN:-}"
OTEL_COLLECTOR_AUTO_DOWNLOAD="${OTEL_COLLECTOR_AUTO_DOWNLOAD:-1}"
OTEL_COLLECTOR_VERSION="${OTEL_COLLECTOR_VERSION:-0.154.0}"
OTEL_COLLECTOR_TOOLS_DIR="${OTEL_COLLECTOR_TOOLS_DIR:-/workspace/project/RL-learning/vllm-test/.tools/otelcol-contrib}"
OTEL_COLLECTOR_DEBUG_EXPORTER="${OTEL_COLLECTOR_DEBUG_EXPORTER:-0}"

mkdir -p "${OUTPUT_DIR}"

echo "OTel trace output: ${TRACE_FILE}"
echo "OTLP gRPC endpoint: http://localhost:${OTLP_GRPC_PORT}"
echo "OTLP HTTP endpoint: http://localhost:${OTLP_HTTP_PORT}"

download_otelcol_contrib() {
  local os arch asset archive install_dir url

  case "$(uname -s)" in
    Linux) os="linux" ;;
    Darwin) os="darwin" ;;
    *)
      echo "Unsupported OS for auto-download: $(uname -s)" >&2
      return 1
      ;;
  esac

  case "$(uname -m)" in
    x86_64|amd64) arch="amd64" ;;
    aarch64|arm64) arch="arm64" ;;
    *)
      echo "Unsupported architecture for auto-download: $(uname -m)" >&2
      return 1
      ;;
  esac

  asset="otelcol-contrib_${OTEL_COLLECTOR_VERSION}_${os}_${arch}.tar.gz"
  url="https://github.com/open-telemetry/opentelemetry-collector-releases/releases/download/v${OTEL_COLLECTOR_VERSION}/${asset}"
  install_dir="${OTEL_COLLECTOR_TOOLS_DIR}/v${OTEL_COLLECTOR_VERSION}-${os}-${arch}"
  archive="${install_dir}/${asset}"

  mkdir -p "${install_dir}"
  if [[ ! -x "${install_dir}/otelcol-contrib" ]]; then
    if ! command -v curl >/dev/null 2>&1; then
      echo "curl is required to auto-download otelcol-contrib." >&2
      return 1
    fi
    echo "Downloading otelcol-contrib v${OTEL_COLLECTOR_VERSION}..." >&2
    echo "${url}" >&2
    curl -fL --retry 3 -o "${archive}" "${url}"
    tar -xzf "${archive}" -C "${install_dir}" otelcol-contrib
    chmod +x "${install_dir}/otelcol-contrib"
  fi

  echo "${install_dir}/otelcol-contrib"
}

if [[ -n "${OTEL_COLLECTOR_BIN}" ]]; then
  collector_bin="${OTEL_COLLECTOR_BIN}"
elif command -v otelcol-contrib >/dev/null 2>&1; then
  collector_bin="otelcol-contrib"
elif [[ "${OTEL_COLLECTOR_AUTO_DOWNLOAD}" == "1" ]]; then
  collector_bin="$(download_otelcol_contrib)"
else
  collector_bin=""
fi

if [[ -n "${collector_bin}" ]]; then
  local_config="${OUTPUT_DIR}/otel_file_collector.local.yaml"
  if [[ "${OTEL_COLLECTOR_DEBUG_EXPORTER}" == "1" ]]; then
    debug_exporter_config='
  debug:
    verbosity: basic'
    trace_exporters='[file, debug]'
  else
    debug_exporter_config=''
    trace_exporters='[file]'
  fi
  cat > "${local_config}" <<EOF
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:${OTLP_GRPC_PORT}
      http:
        endpoint: 0.0.0.0:${OTLP_HTTP_PORT}

processors:
  batch:

exporters:
  file:
    path: ${TRACE_FILE}
${debug_exporter_config}

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: ${trace_exporters}
EOF
  exec "${collector_bin}" --config "${local_config}"
fi

echo "No otelcol-contrib binary is available." >&2
echo "Set OTEL_COLLECTOR_BIN=/path/to/otelcol-contrib, or keep OTEL_COLLECTOR_AUTO_DOWNLOAD=1 with network access." >&2
exit 127
