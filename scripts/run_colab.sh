#!/usr/bin/env bash
set -euo pipefail

python run_ablation.py --config configs/ablation.yaml --mode all --seeds "${1:-3}"
