#!/usr/bin/env bash
# One-command calibration: check setup, record N cycles (empty/sit/stand), train, report.
#
#   bash scripts/calibrate.sh SESSION_NAME [CYCLES=10] [SECONDS_PER_STATE=20]
#
# To only TEST an existing model on new cycles (no training):
#   MODEL=models/day1.joblib bash scripts/calibrate.sh hold_a 2
set -u
SESS="${1:?usage: bash scripts/calibrate.sh SESSION_NAME [CYCLES] [SECONDS]}"
CYCLES="${2:-10}"
DUR="${3:-20}"
IFACE="${IFACE:-wlan0}"
PY="${PY:-$(which python3)}"
RECORDER="${RECORDER:-scripts/record-csi-udp-raw.py}"
FEATURES="${FEATURES:-full}"
SUDO="${SUDO-sudo}"
WAIT_SCALE="${WAIT_SCALE:-1}"
MODEL="${MODEL:-}"
OUT="data/sessions/$SESS"
VARIED="${VARIED:-1}"
SIT_STYLES=("upright" "leaning back" "leaning forward" "legs crossed")
STAND_STYLES=("straight" "arms crossed" "phone in hand" "weight on one leg")
mkdir -p "$OUT" models

say() { spd-say -w "$1" 2>/dev/null || espeak "$1" 2>/dev/null || printf '\a'; }

[ -n "$SUDO" ] && sudo -v

echo "================ 1. SETUP CHECK (5 s) ================"
$SUDO "$PY" "$RECORDER" --interface "$IFACE" --port 5005 --duration 5 --output "$OUT/_chk" | grep -E "frames saved|Nodes seen"
FREQ=$(grep -oh '"freq_mhz":[0-9]*' "$OUT"/_chk/*.jsonl 2>/dev/null | sort | uniq -c | sort -rn | head -1 | awk -F: '{print $2}')
[ -n "$SUDO" ] && sudo chown -R "$USER:$USER" "$OUT"
"$PY" scripts/compare_sessions.py check="$OUT/_chk" | tail -8
if [ -z "$FREQ" ]; then echo "!!! No CSI data received. Check ESP power, router, laptop IP. Stopping."; exit 1; fi
echo "Radio channel: $FREQ MHz. Node signal (rssi) table above: write these numbers down."
echo "Press Enter to start recording (or Ctrl+C to stop)."; read -r _

rec() {   # name  seconds_to_get_ready  seconds_to_record  [spoken words]
  echo; echo ">>> $1 : $2 s to get ready   ${4:+($4)}"
  say "${4:-${1%%_*}}"
  sleep $(( $2 * WAIT_SCALE ))
  say "start"
  local res
  res=$($SUDO "$PY" "$RECORDER" --interface "$IFACE" --port 5005 --duration "$3" --output "$OUT/$1" | grep -E "frames saved|Nodes seen")
  echo "$res"
  local n; n=$(echo "$res" | awk '/frames saved/{print $NF}')
  if [ -z "$n" ] || [ "$n" -lt $(( $3 * 120 )) ]; then echo "!!! WARNING: only ${n:-0} frames (expected about $(( $3 * 175 ))). Redo this recording: rm -r $OUT/$1"; say "warning"; fi
  echo "$res" | grep -q "\[1, 2, 3, 4\]" || { echo "!!! WARNING: not all 4 nodes seen in $1"; say "warning"; }
  say "done"
}

echo "================ 2. RECORDING $CYCLES cycles ================"
for i in $(seq 1 "$CYCLES"); do
  n=$(printf %02d "$i")
  echo; echo "---------- cycle $i of $CYCLES ----------"
  si=$(( (i - 1) % 4 ))
  if [ "$VARIED" = "1" ]; then
    echo "cycle $n  sit: ${SIT_STYLES[$si]}   stand: ${STAND_STYLES[$si]}" >> "$OUT/styles.txt"
    rec "empty_c$n" 12 "$DUR"
    rec "sit_c$n"   12 "$DUR" "sit, ${SIT_STYLES[$si]}"
    rec "stand_c$n"  8 "$DUR" "stand, ${STAND_STYLES[$si]}"
  else
    rec "empty_c$n" 12 "$DUR"
    rec "sit_c$n"   12 "$DUR"
    rec "stand_c$n"  8 "$DUR"
  fi
done
[ -n "$SUDO" ] && sudo chown -R "$USER:$USER" "$OUT"
echo "Recording finished: $(date +%H:%M)"
FREQS=$(grep -oh '"freq_mhz":[0-9]*' "$OUT"/*/*.jsonl | sort | uniq -c | sed 's/"freq_mhz"://')
echo "Channels seen in this session (frames, MHz):"; echo "$FREQS"
[ "$(echo "$FREQS" | wc -l)" -gt 1 ] && echo "!!! WARNING: the radio channel changed during recording. Only $FREQ MHz data will be used."

if [ -n "$MODEL" ]; then
  echo "================ 3. TESTING $MODEL ================"
  "$PY" scripts/predict_folder.py --model "$MODEL" --data "$OUT"
else
  echo "================ 3. TRAINING ================"
  "$PY" scripts/train_posture.py --data "$OUT" --features "$FEATURES" --skip-zone --freq "$FREQ" --out "models/$SESS.joblib" | grep -E "SUMMARY|voting|SKIP"
  cp "models/$SESS.joblib" models/latest.joblib
  echo
  echo "Model saved: models/$SESS.joblib  (also models/latest.joblib)"
  echo "Run it live:   $SUDO $PY scripts/live_predict.py --model models/latest.joblib"
fi
