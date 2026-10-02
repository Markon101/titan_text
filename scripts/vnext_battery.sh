#!/data/data/com.termux/files/usr/bin/bash
# vNext Relay-Fold Transport (RFT) falsification battery.
#
# Runs at T=8 (dev-steps 8) — the horizon where transport genuinely binds: at
# T=16 every k covers the grid (k=1 reaches cell 15 in 15 ticks), so T=16 cannot
# attribute gains to transport. At T=8, slot2 (cell 11) needs k>=2 and slot3
# (cell 15) needs k>=3 to arrive in time.
#
# Falsifier: interior accuracy at T=8 + staircase onset. If k=4 interior stays at
# chance AND onset is arm-invariant across k, RFT is falsified (clean negative).
#
# All arms share the clean substrate: seeded init (seed==label), advancing stream,
# exhaustive 789-row heldout eval, grad clip. Unit = trained run. Fresh seeds.
set -u
cd /data/data/com.termux/files/home/projects/titan_text
OUT=runs/vnext_battery
mkdir -p $OUT
BIN=target/release/titan_text
# Use continuous/fold carry (full gradient). NEVER legacy ste_sign/ste_round (zero-grad bug).
FOLD_MODE=${FOLD_MODE:-fold}

run() {
  local arm=$1; local seed=$2; shift 2
  local dir=$OUT/${arm}/seed_${seed}
  echo "=== TRAIN ${arm} seed ${seed} T=8 $(date +%T) ===" >> $OUT/driver.log
  if [ -f "$dir/manifest.json" ]; then echo "skip (exists)" >> $OUT/driver.log; return; fi
  $BIN train --task iterated-parity --epochs 1000 --seq-len 16 --dev-steps 8 \
    --batch-size 8 --seed "$seed" --zero-boundary --state-norm bounded --save-dir "$dir" "$@" \
    >> $OUT/${arm}_train_${seed}.log 2>&1
  [ $? -ne 0 ] && echo "TRAIN_FAIL ${arm} ${seed}" >> $OUT/driver.log
  python3 - "$dir/manifest.json" >> $OUT/driver.log 2>&1 <<'PYEOF'
import json,sys
m=json.load(open(sys.argv[1]))
acc,loss,gl=m.get('train_accuracy'),m.get('train_loss'),m.get('grad_norm')
vals=[v for v in (acc,loss,gl) if v is not None and v==v]
print(f"MANIFEST_CHECK {'OK' if acc not in (None,0) and len(vals)==3 else 'DIVERGED'} acc={acc} loss={loss}")
PYEOF
  # tick-sweep eval on exhaustive heldout (789 rows); onset reveals the transport front
  $BIN sweep --load-dir "$dir" --task iterated-parity --budgets 0,4,6,8,12,16 \
    --batch-size 789 --seeds 0 --output "$OUT/${arm}_eval_${seed}.json" \
    >> $OUT/${arm}_eval_${seed}.log 2>&1
  # causal influence (transport front): far-prefix flip -> interior slot prediction change
  $BIN influence --load-dir "$dir" --task iterated-parity --budgets 4,6,8 \
    --output "$OUT/${arm}_influence_${seed}.json" >> $OUT/${arm}_eval_${seed}.log 2>&1
  echo "=== DONE ${arm} seed ${seed} $(date +%T) ===" >> $OUT/driver.log
}

# Baseline: plain NCA, no carry channels (transport speed 0).
for s in 921 922 923 924; do run baseline_plain $s; done
# RFT: carry Cc=16, fold write, uniform-k transport speed 1,2,4.
for s in 925 926 927 928; do run rft_k1 $s --carry-channels 16 --carry-skip-stride 1 --carry-bidirectional --carry-quantization $FOLD_MODE; done
for s in 929 930 931 932; do run rft_k2 $s --carry-channels 16 --carry-skip-stride 2 --carry-bidirectional --carry-quantization $FOLD_MODE; done
for s in 933 934 935 936; do run rft_k4 $s --carry-channels 16 --carry-skip-stride 4 --carry-bidirectional --carry-quantization $FOLD_MODE; done
echo "BATTERY_COMPLETE $(date +%T)" >> $OUT/driver.log
