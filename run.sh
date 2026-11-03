#!/usr/bin/env bash
set -euo pipefail
project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f /opt/ros/jazzy/setup.bash ]]; then
  printf 'ROS 2 Jazzy is missing from /opt/ros/jazzy.\n' >&2
  exit 1
fi
set +u
source /opt/ros/jazzy/setup.bash
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-52}"
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-LOCALHOST}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
if [[ "$RMW_IMPLEMENTATION" == "rmw_fastrtps_cpp" ]]; then
  export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-$project/config/fastdds.xml}"
fi
if [[ "${1:-}" == "--ros2" ]]; then
  shift
  exec ros2 "$@"
fi
export FUSION_STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/fusion_traversability"
export ROS_LOG_DIR="$FUSION_STATE_DIR/ros"
export PYTHONDONTWRITEBYTECODE=1
# WSLg needs the D3D12 Gallium driver explicitly on this host.
if grep -qi microsoft /proc/sys/kernel/osrelease; then
  export GALLIUM_DRIVER="${GALLIUM_DRIVER:-d3d12}"
  export MESA_D3D12_DEFAULT_ADAPTER_NAME="${MESA_D3D12_DEFAULT_ADAPTER_NAME:-NVIDIA}"
fi
# Thin XYZ products and OpenCV kernels are faster without per-process BLAS oversubscription.
export MALLOC_ARENA_MAX=2
export MALLOC_TRIM_THRESHOLD_=131072
export OPENBLAS_NUM_THREADS="${FUSION_NUMERICAL_THREADS:-1}"
export OMP_NUM_THREADS="${FUSION_NUMERICAL_THREADS:-1}"
export MKL_NUM_THREADS="${FUSION_NUMERICAL_THREADS:-1}"
export PYTHONPATH="$project/src${PYTHONPATH:+:$PYTHONPATH}"
export GZ_SIM_RESOURCE_PATH="$project/src/factory_sim/models${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"
mkdir -p -- "$ROS_LOG_DIR"
printf '查看 ROS 话题：%s/run.sh --ros2 topic list --no-daemon\n' "$project"
if [[ "${1:-}" == "--capture" ]]; then
  shift
  exec /usr/bin/python3 -m factory_sim.capture.session "$@"
fi
exec /usr/bin/python3 -m factory_sim.runtime.session "$@"
