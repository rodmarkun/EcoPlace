#!/usr/bin/env bash
# fetch_data.sh — pinned acquisition of the Alibaba trace metadata, benchmark manifests and the
# EdgeSimPy reference copy. Everything lands under data/ (gitignored).
#
# Small things are fetched (READMEs, schemas, k8s/helm manifests, EdgeSimPy source). Multi-GB
# trace payloads are NOT downloaded: the script prints where to get them instead.
#
# Idempotent: re-running verifies pins and skips completed fetches.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$REPO_ROOT/data"
mkdir -p "$DATA"

# ---------------------------------------------------------------------------
# Pins (recorded 2026-07-13; verified by commit hash after fetch). Licenses: docs/DATA_LICENSES.md.
# ---------------------------------------------------------------------------
ALIBABA_URL=https://github.com/alibaba/clusterdata.git
ALIBABA_SHA=0d0f3f1efdbf1add6a7bcc63676eafbd1eb11f71

DSB_URL=https://github.com/delimitrou/DeathStarBench.git
DSB_SHA=6ecb09706140f8730b5385c08f1386c654c3c526

TEASTORE_URL=https://github.com/DescartesResearch/TeaStore.git
TEASTORE_SHA=34b37f7e7be433ce72d5f9455e66922a13116749

EDGESIMPY_URL=https://github.com/EdgeSimPy/EdgeSimPy.git
EDGESIMPY_SHA=76eb5ead74596bb4240759fa4336f1d6f190c70a

# fetch_pinned <url> <sha> <dest> [sparse-dir ...]
# Shallow-fetches exactly the pinned commit; with sparse dirs, only those trees'
# blobs are downloaded (partial clone). Verifies HEAD == pin.
fetch_pinned() {
  local url=$1 sha=$2 dest=$3
  shift 3
  if [ -e "$dest/.git" ]; then
    if [ "$(git -C "$dest" rev-parse HEAD)" = "$sha" ]; then
      echo "  [ok] $dest already at $sha"
      return 0
    fi
    echo "  [!!] $dest exists but is not at pin $sha — remove it and re-run." >&2
    return 1
  fi
  mkdir -p "$dest"
  git -C "$dest" init -q
  git -C "$dest" remote add origin "$url"
  if [ $# -gt 0 ]; then
    git -C "$dest" config extensions.partialClone origin
    git -C "$dest" sparse-checkout set --cone "$@"
    git -C "$dest" fetch -q --depth 1 --filter=blob:none origin "$sha"
  else
    git -C "$dest" fetch -q --depth 1 origin "$sha"
  fi
  git -C "$dest" checkout -q "$sha"
  echo "  [ok] $dest @ $(git -C "$dest" rev-parse HEAD)"
}

echo "== Alibaba cluster-trace-microservices v2021 + v2022 (metadata/schemas/scripts only)"
fetch_pinned "$ALIBABA_URL" "$ALIBABA_SHA" "$DATA/alibaba-clusterdata" \
  cluster-trace-microservices-v2021 cluster-trace-microservices-v2022
cat <<'EOF'
  NOTE — trace payloads are NOT downloaded (multi-GB):
    v2021 full trace  : ~50+ GB unpacked (MSResource / MSRTQps / MSCallGraph / Node tables)
    v2022 full trace  : ~90+ GB unpacked (10 clusters, call graphs + runtime metrics)
  The per-table download URLs (Aliyun OSS) and fetch scripts are inside:
    data/alibaba-clusterdata/cluster-trace-microservices-v2021/
    data/alibaba-clusterdata/cluster-trace-microservices-v2022/
  Download only the tables you need, onto a disk with >=200 GB free.
EOF

echo "== DeathStarBench (Social Network + Hotel Reservation manifests/charts)"
fetch_pinned "$DSB_URL" "$DSB_SHA" "$DATA/DeathStarBench" \
  socialNetwork/helm-chart hotelReservation/kubernetes hotelReservation/helm-chart
echo "  (full repo with datasets/media is ~GB-scale; sparse checkout keeps manifests only.)"

echo "== TeaStore (kubernetes + helm examples)"
fetch_pinned "$TEASTORE_URL" "$TEASTORE_SHA" "$DATA/TeaStore" \
  examples/kubernetes examples/helm

echo "== EdgeSimPy (historical reference copy)"
fetch_pinned "$EDGESIMPY_URL" "$EDGESIMPY_SHA" "$DATA/EdgeSimPy"

echo
echo "All pinned fetches complete. Licenses: docs/DATA_LICENSES.md"
