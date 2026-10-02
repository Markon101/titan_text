#!/data/data/com.termux/files/usr/bin/bash
# RD-018b / vNext falsification battery: baseline vs Directed Carry Transport.
#
# Hypothesis under test: interior-slot failure at L=16 is a TRANSPORT bottleneck
# (a computed prefix-parity bit must travel 7-11 cells). Falsification: dedicate
# carry channels with ballistic transport speed s (cells/tick) and sweep s in
# {0,1,2,4}. If transport is the bottleneck, interior accuracy must rise with s
# and interior cells must become reachable at the predicted tick. If it does NOT,
# the DCT idea is dead and we learn transport is not the limiting factor.
#
# All arms share the FIXED substrate: seeded init (seed==seed label), advancing
# data stream, exhaustive 789-row heldout eval, gradient clip. Unit = trained run.
set -u
cd /data/data/com.termux/files/home/projects/titan_text
OUT=runs/vnext_battery
mkdir -p $OUT
BIN=target/release/titan_text

# arm -> flags. carry_channels Cc=16; skip_stride = transport speed s (cells/tick).
run() {
  local arm=$1; local seed=$2; shift 2
  local dir=$OUT/${arm}/seed_${seed}
  echo "=== TRAIN ${arm} seed ${seed} $(date +%T) ===" >> $OUT/driver.log
  if [ -f "$dir/manifest.json" ]; then echo "skip (exists)" >> $OUT/driver.log; return; fi
  $BIN train --task iterated-parity --epochs 1000 --seq-len 16 --dev-steps 16 \
    --batch-size 8 --seed "$seed" --zero-boundary --save-dir "$dir" "$@" \
    >> $OUT/${arm}_train_${seed}.log 2>&1
  [ $? -ne 0 ] && echo "TRAIN_FAIL ${arm} ${seed}" >> $OUT/driver.log
  python3 - "$dir/manifest.json" >> $OUT/driver.log 2>&1 <<'PYEOF'
import json,sys,math
m=json.load(open(sys.argv[1]))
acc,loss,gl=m.get('train_accuracy'),m.get('train_loss'),m.get('grad_norm')
vals=[v for v in (acc,loss,gl) if v is not None and v==v]
ok = acc not in (None,0) and len(vals)==3
print(f"MANIFEST_CHECK {'OK' if ok else 'DIVERGED'} acc={acc} loss={loss} grad={gl}")
PYEOF
  # Exhaustive heldout eval (789 rows) at the trained horizon + tick staircase.
  $BIN sweep --load-dir "$dir" --task iterated-parity --budgets 0,3,7,11,15,16 \
    --batch-size 789 --seeds 0 --output "$OUT/${arm}_eval_${seed}.json" \
    >> $OUT/${arm}_eval_${seed}.log 2>&1
  echo "=== DONE ${arm} seed ${seed} $(date +%T) ===" >> $OUT/driver.log
}

# Baseline: plain NCA, no carry channels (transport speed 0).
for s in 921 922 923 924 925; do run baseline_plain $s; done
# DCT speed 1 (1 cell/tick), speed 2, speed 4 (4 cells/tick). Cc=16, bidirectional.
for s in 926 927 928; do run dct_s1 $s --carry-channels 16 --carry-skip-stride 1 --carry-bidirectional; done
for s in 929 930 931; do run dct_s2 $s --carry-channels 16 --carry-skip-stride 2 --carry-bidirectional; done
for s in 932 933 934; do run dct_s4 $s --carry-channels 16 --carry-skip-stride 4 --carry-bidirectional; done
echo "BATTERY_COMPLETE $(date +%T)" >> $OUT/driver.log
