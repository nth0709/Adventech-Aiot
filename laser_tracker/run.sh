#!/bin/bash
# run follow5.py; if the servo port open wedges (CH343 driver), kill and retry (max 4)
cd ~/aiot/laser_tracker
OUT=${1:-run.out}; shift
for a in 1 2 3 4; do
  python3 -u follow5.py "$@" > "$OUT" 2>&1 &
  P=$!
  for k in $(seq 50); do sleep 0.2; grep -q "moving to observe" "$OUT" && break; kill -0 $P 2>/dev/null || break; done
  if grep -q "moving to observe" "$OUT"; then echo "attempt $a: connected" >> "$OUT.log"; wait $P; echo "exit $?" >> "$OUT"; exit 0; fi
  echo "attempt $a: port open wedged - retry" >> "$OUT.log"; kill -9 $P 2>/dev/null; sleep 1
done
echo "FAILED: servo port would not open" >> "$OUT"
