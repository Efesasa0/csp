#!/usr/bin/env bash
set -euo pipefail

URL="${1:-}"
OUT="01_contrastive_stubs_mlp_neurons.npy"
SHA256="b93078c650ea8e6f609eca91fa728a31fb6109a6965af757dab141978a3152ec"

if [[ -z "$URL" ]]; then
  echo "Usage: $0 <download-url>"
  exit 1
fi

curl -fL "$URL" -o "$OUT"
echo "$SHA256  $OUT" | shasum -a 256 -c -
echo "Downloaded and verified: $OUT"
