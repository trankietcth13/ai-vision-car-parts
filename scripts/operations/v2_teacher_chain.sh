#!/bin/bash
# Taxonomy v2 teacher baseline (docs/plans/teacher_all_components_workflow_2026-09-29.md, round 3 prerequisite).
# Same data sources and recipe as phase3_chain.sh, but labels mapped to the 27 taxonomy-v2 training classes
# (21 tier A incl. exhaust_manifold_heat_shield + 6 generic fallbacks), split by VEHICLE exactly like v8/p5
# (val 08/36/49/50/52/58, test 13/23/29) so it can be compared with teacher p5_reg avg5 (test mask mAP50-95 0.354).
# Gate: per-class mask AP of the 20 v1 classes on test not lower than p5_reg by more than 1 point on average.
#
# Code runs from a separate copy of the repo (new layout) so the running v1 chains are untouched:
#   (local) tar czf - scripts configs src | ssh admin@<dgx> 'mkdir -p ~/distillation_workspace/v2code && tar xzf - -C ~/distillation_workspace/v2code'
#   (local) sed -i 's/\r$//' scripts/operations/v2_teacher_chain.sh   # before copying: no CRLF on the DGX
# Launch (queues itself behind phase3_chain.sh; one GPU job at a time):
#   nohup bash v2code/scripts/operations/v2_teacher_chain.sh > runs/logs/v2_teacher_chain.log 2>&1 &
W=/home/admin/distillation_workspace
C=$W/v2code
cd $W
export CPATH=$HOME/pyheaders/usr/include/python3.12:$HOME/pyheaders/usr/include
PY=$W/.venv/bin/python
P=$W/data/engine_bay_full_v2
S=$W/runs/segment
T=engine_teacher_v2tax
run() { local name=$1; shift; echo "$(date +%H:%M:%S) START $name"; "$@" > runs/logs/$name.log 2>&1; local rc=$?; echo "$(date +%H:%M:%S) END $name rc=$rc"; return $rc; }

# wait for every running training / evaluation chain (DGX memory is shared with vLLM)
while pgrep -f "phase3_chain.sh|phase5_chain.sh|p5_kd_chain.sh|v8_chain.sh" >/dev/null; do sleep 300; done
[ -f $C/configs/taxonomy_v2.yaml ] || { echo "v2code not synced ($C)"; exit 1; }

EXTRA=""
for sp in train val test; do [ -d $W/data/engine_bay_reviewed/labels/$sp ] && EXTRA="$EXTRA $W/data/engine_bay_reviewed:$sp"; done
EMPTYFIX=""; [ -f $W/data/engine_bay_empty_fix/EMPTY_FIX_SUMMARY.json ] && EMPTYFIX="$W/data/engine_bay_empty_fix:train"

[ -f $P/data_full.yaml ] || run build_full_v2 $PY -u $C/scripts/data_pipeline/build_full_dataset.py \
  --base $W/data/engine_bay_train_v9 \
  --reviewed-train $EMPTYFIX $W/data/engine_bay_p2_verified:train \
  --reviewed $W/data/engine_bay_reviewed_p2:p2 $W/data/engine_bay_reviewed_hybrid:train \
  --taxonomy $C/configs/taxonomy_v2.yaml --extra-from $EXTRA \
  --ann-config $C/configs/data_engine_bay.yaml --train-classes $C/configs/engine_bay_train_classes.yaml \
  --candidates $W/data/engine_bay_phase2/CANDIDATES.jsonl --dup-groups $W/data/engine_bay_phase2/DUPLICATES.json \
  --folds 3 --out $P || exit 1
[ -f $P/data_vehicle.yaml ] || run vehicle_split_v2 $PY -u $C/scripts/data_pipeline/vehicle_split.py --pool $P || exit 1

# teacher: phase-5/phase-3 recipe; the vehicle val split is clean, so checkpoints are averaged top-5 by val
if [ ! -f $S/$T/weights/avg5.pt ]; then
  run $T $PY -u -c "from ultralytics import YOLO; YOLO(\"yolo11l-seg.pt\").train(data=\"$P/data_vehicle.yaml\", imgsz=640, batch=16, epochs=100, patience=100, cos_lr=True, copy_paste=0.3, mixup=0.1, degrees=5.0, close_mosaic=10, cache=\"ram\", workers=8, device=0, project=\"$S\", name=\"$T\", exist_ok=True, seed=0, save_period=5, weight_decay=0.001, scale=0.7, perspective=0.0005)" || exit 1
  run ${T}_avg $PY -u average_checkpoints.py --run $S/$T --top-k 5 --out $S/$T/weights/avg5.pt || exit 1
fi
# test on the 3 held-out vehicles (v2 classes); p5_reg is scored on the same images with its own v1 dataset
run qa_v2tax $PY -u qa_test.py --model teacher_v2tax_avg5=$S/$T/weights/avg5.pt@640 --data $P/data_vehicle.yaml --split test --out qa_results_v2tax
echo "$(date +%H:%M:%S) V2 TEACHER DONE (compare qa_results_v2tax per class with qa_results_p5_kd teacher_p5_reg_avg5)"
