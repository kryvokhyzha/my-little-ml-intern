#!/usr/bin/env bash
# Local compute probe. Prints key=value lines; ALWAYS exits 0 so auto-detect never crashes.
#   cuda=yes|no|unknown
#   mps=yes|no|unknown
#   gpu_count=<int>|unknown
#   gpu_name=<name>|none|unknown
#   vram_gb=<float>|unknown

set -u

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/../.." && pwd)"

if command -v nvidia-smi > /dev/null 2>&1; then
  echo "cuda=yes"
  echo "mps=no"
  smi="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2> /dev/null || true)"
  if [ -n "$smi" ]; then
    echo "gpu_count=$(printf '%s\n' "$smi" | grep -c .)"
    first="$(printf '%s\n' "$smi" | head -n 1)"
    name="$(printf '%s' "$first" | cut -d',' -f1 | sed 's/^ *//;s/ *$//')"
    mib="$(printf '%s' "$first" | cut -d',' -f2 | sed 's/^ *//;s/ *$//')"
    echo "gpu_name=${name}"
    echo "vram_gb=$(awk -v m="$mib" 'BEGIN { printf "%.1f", m / 1024 }')"
  else
    echo "gpu_count=unknown"
    echo "gpu_name=unknown"
    echo "vram_gb=unknown"
  fi
  exit 0
fi

# No nvidia-smi: ask the project's torch about CUDA and Apple MPS.
probe="$(
  cd "$project_root" && uv run python -c '
import torch

cuda = torch.cuda.is_available()
mps = torch.backends.mps.is_available()
print("cuda=" + ("yes" if cuda else "no"))
print("mps=" + ("yes" if mps else "no"))
if cuda:
    props = torch.cuda.get_device_properties(0)
    print("gpu_count=" + str(torch.cuda.device_count()))
    print("gpu_name=" + props.name)
    print("vram_gb=" + format(props.total_memory / 2**30, ".1f"))
elif mps:
    print("gpu_count=1")
    print("gpu_name=Apple Silicon (MPS)")
    try:
        print("vram_gb=" + format(torch.mps.recommended_max_memory() / 2**30, ".1f"))
    except Exception:
        print("vram_gb=unknown")
else:
    print("gpu_count=0")
    print("gpu_name=none")
    print("vram_gb=0")
' 2> /dev/null || true
)"

if [ -n "$probe" ]; then
  printf '%s\n' "$probe"
else
  echo "cuda=unknown"
  echo "mps=unknown"
  echo "gpu_count=unknown"
  echo "gpu_name=unknown"
  echo "vram_gb=unknown"
fi

exit 0
