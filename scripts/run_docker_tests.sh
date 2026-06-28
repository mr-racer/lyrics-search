#!/usr/bin/env bash
# Run the live-stack test suite (tests/docker/) INSIDE the running musix
# container, against the real Qdrant service and real CLAP/dense models.
#
# The normal suite (tests/unit, tests/integration) mocks Qdrant and stubs the ML
# libs, so it runs anywhere with `pytest`. The tests under tests/docker/ are the
# opposite: they only mean anything with the real stack up. This wrapper:
#   1. requires the stack to be running   (docker compose up -d)
#   2. sets MUSIX_LIVE_STACK=1            (un-stubs torch/CLAP/ST, un-skips them)
#   3. execs pytest in the musix container, where tests/ is bind-mounted
#
# Usage:
#   docker compose up -d
#   scripts/run_docker_tests.sh                 # run the whole live suite
#   scripts/run_docker_tests.sh -k clap         # pass extra args through to pytest
#
# First run is slow: it downloads the embedding model (~1 GB) and the CLAP
# checkpoint (~2.3 GB) into the mounted volumes; later runs reuse them.
set -euo pipefail

SERVICE="${MUSIX_SERVICE:-musix}"

if ! docker compose ps --status running --services 2>/dev/null | grep -qx "$SERVICE"; then
  echo "error: '$SERVICE' container is not running. Start it first:" >&2
  echo "       docker compose up -d" >&2
  exit 1
fi

exec docker compose exec \
  -e MUSIX_LIVE_STACK=1 \
  "$SERVICE" \
  python -m pytest tests/docker/ -v "$@"
