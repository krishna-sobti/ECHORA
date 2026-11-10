# ECHORA

Privacy-first human activity sensing using Wi-Fi Channel State Information (CSI).
No camera is used at runtime, so nobody's face or identity is ever captured.
ECHORA currently detects **sitting vs standing** and **room presence** live, using only Wi-Fi signals.

**Phase 1 (this release):** sitting/standing and presence detection. **Phase 2 (planned):** full-body skeleton tracking for multiple people. See [Roadmap](#roadmap).

ECHORA uses the ESP32 CSI firmware from [RuView](https://github.com/ruvnet/RuView) by rUv (MIT licensed).
The capture, training, live prediction and dashboard scripts in this repo are original work.

## Why

Cameras show what a person is doing, but they also show who the person is.
Wi-Fi signals change as they pass around a human body, so activity information can be
recovered without capturing identity.

## What works right now

- Live **sitting / standing detection** for a single person in a room
- **Presence detection** (is anyone in the room)
- Live web dashboard at `http://localhost:8000` (CSI heatmap, activity, current state)
- One-command calibration: record, train and test a model in one script
- Replay mode: run the dashboard from recorded data without any hardware

## How it works

1. **Capture:** 4 ESP32 nodes (one on each wall) send CSI frames over UDP to a Linux
   machine through a Wi-Fi router. `record-csi-udp-raw.py` receives and saves them.
2. **Calibrate:** `calibrate.sh` guides you through repeated sitting and standing
   cycles and records every state.
3. **Train:** `train_posture.py` extracts features from the recordings and trains
   a Random Forest classifier (scikit-learn).
4. **Predict live:** `live_predict.py` classifies a sliding window of live CSI,
   smooths the output by voting over the last 7 windows (about 4 seconds),
   and serves the dashboard.

## Results

Evaluation method: **leave-one-recording-out**. The posture data has 10 sitting and
10 standing recordings (about 38 windows each, 744 windows in total). Each recording is
held out in turn, the model is trained on all the other recordings, and then predicts the
held-out one. Presence uses 10 additional empty-room recordings.

| Features | Presence | Posture (sitting/standing) |
|----------|----------|----------------------------|
| full     | 94.6%    | 92.1%                      |
| compact  | 81.3%    | 86.6%                      |
| motion   | 73.9%    | 80.9%                      |

The `full` feature set (1032 features per window) is the best of the three.

**Important limit:** all recordings come from one session, one room, one person and one
node layout. These numbers show that the model works on new recordings from the same setup.
They do not show that it works on a different day, room, layout or person.
Cross-session testing for sitting/standing has not been done yet.

## Setup used and how it was tested

- 4 x ESP32 nodes, one on each wall; a TP-Link travel router on channel 6 (2437 MHz)
- Linux (Kali) laptop as the receiver
- Data recorded in repeated sitting / standing cycles with `calibrate.sh`
- Several feature sets (full, compact, motion-only) were trained and compared
- Live predictions were tested through the dashboard, in a real room, with real movement
- Cross-session behaviour was tested for multi-class activity recognition (see Known issues)

## Usage

Calibrate and train:

```bash
bash scripts/calibrate.sh SESSION_NAME [CYCLES] [SECONDS_PER_STATE]
```

Run live:

```bash
sudo python3 scripts/live_predict.py --model models/latest.joblib
```

Open `http://localhost:8000`. Without hardware, use `--replay` (and `--loop`) to see the dashboard.

## Requirements

- Python 3, numpy, scikit-learn, joblib
- Linux with a wireless interface and root access for capture
- ESP32 nodes with CSI firmware (see RuView)

## What is not published

Training data and trained models are not included, because the recordings contain
movement data of real people. Record your own with `calibrate.sh`.

## Roadmap

### Phase 1: Posture and presence (current release)

| Status | Item |
|--------|------|
| Done | Raw CSI capture over UDP from 4 ESP32 nodes |
| Done | Sitting / standing and presence detection, single person |
| Done | Live dashboard and one-command calibration |
| Done | Feature-set comparison (full / compact / motion) |

### Phase 2: Real skeletons, multiple people (planned)

The next step is to move from a single posture label to a real skeleton:

- **17-keypoint skeleton** (COCO-style body structure) estimated from Wi-Fi CSI, showing full body movements rather than just sitting or standing
- **Multi-person support**: track and draw more than one person in the room at the same time
- **Privacy stays the core idea:** camera-derived pose labels (e.g. MediaPipe) are used only to create training data, never at runtime. The live system uses Wi-Fi only
- Fine-tuning from pretrained pose weights instead of training from scratch
- More activities beyond sitting and standing
- Room zone detection (where in the room each person is)

Phase 2 is a work in progress. The skeleton output is experimental and not reliable yet.

## Known issues

- Single person only
- Sitting/standing accuracy is measured within one session only; cross-session performance for it is not measured yet
- Skeleton output is still experimental and not reliable
- Needs recalibration when the room or node layout changes
- Cross-session accuracy is low for multi-class activity recognition: in our experiments
  (5 activities, train on one session, test on another) accuracy ranged from about 15% to 70%.
  Always calibrate in the same room and layout you will use

## License

MIT. See [LICENSE](LICENSE).
