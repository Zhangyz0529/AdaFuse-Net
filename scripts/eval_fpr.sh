#!/usr/bin/env bash
set -euo pipefail
python -m adafuse.cli.main calibrate "$@"
