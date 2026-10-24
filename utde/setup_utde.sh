#!/usr/bin/env bash
# Fetch the UTDE reference implementation (Zhang et al., ICML 2023) at the commit we used and apply our adapter.
# Usage: bash utde/setup_utde.sh [target_dir=../MultimodalMIMIC]
set -euo pipefail
T=${1:-../MultimodalMIMIC}; HERE=$(cd "$(dirname "$0")" && pwd)
git clone https://github.com/XZhang97666/MultimodalMIMIC.git "$T"
cd "$T" && git checkout -q 8e513ca && git checkout -q -b sofa28-adapter
git apply "$HERE/MultimodalMIMIC_sofa28.patch"
mkdir -p Data/sofa28 && cp "$HERE/terms.txt" Data/sofa28/terms.txt
echo "UTDE adapter applied in $T (branch sofa28-adapter). Next: build Data/sofa28 with scripts/13_build_utde_dataset.py, then bash run_sofa28.sh"
