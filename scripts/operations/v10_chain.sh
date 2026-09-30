#!/bin/bash
# v10 = v9/phase-3 data and the p5 recipe with ONE change: corrected labels (docs/plans/bottleneck_solutions_2026-09-30.md).
#   corrections: reservoir re-review rounds 1+2 (134 boxes) + D2 geometry audit (duct / terminal / battery, 39 images)
#   applied as geometry-matched patches (class + box IoU >= 0.8) to a NEW pool build; nothing older is edited.
# Split by vehicle like v8/p5 (val 08/36/49/50/52/58, test 13/23/29). The test labels are patched too = "test v2";
# every model (v10 and the p5 references) is scored on it, so old and new numbers must not be mixed.
# Gates (test v2, mask mAP50-95): teacher v10 >= p5_reg on test v2 + 0.01; KD v10 two-seed mean >= kd_n_p5t mean + 0.01.
#
# Needs on the DGX: v2code synced (scripts/configs/src), data/engine_bay_review_reservoir (CHANGES.json + verdicts),
# data/engine_bay_review_geometry/PATCHES.jsonl.
# Launch detached (waits for every other chain; one GPU job at a time):
#   setsid nohup bash v2code/scripts/operations/v10_chain.sh > runs/logs/v10_chain.log 2>&1 < /dev/null & disown
W=/home/admin/distillation_workspace
C=$W/v2code
cd $W
export CPATH=$HOME/pyheaders/usr/include/python3.12:$HOME/pyheaders/usr/include
export PYTHONPATH=$C/src:$C/scripts
PY=$W/.venv/bin/python
P=$W/data/engine_bay_full_v10
S=$W/runs/segment
R=$W/runs/train_kd
T=engine_teacher_v10
run() { local name=$1; shift; echo "$(date +%H:%M:%S) START $name"; "$@" > runs/logs/$name.log 2>&1; local rc=$?; echo "$(date +%H:%M:%S) END $name rc=$rc"; return $rc; }

while pgrep -f "phase3_chain.sh|phase5_chain.sh|p5_kd_chain.sh|v8_chain.sh|v2_teacher_chain.sh|e1_kd_chain.sh" >/dev/null; do sleep 300; done
[ -f $C/scripts/data_pipeline/patch_labels.py ] || { echo "v2code not synced ($C)"; exit 1; }

# 1) patches: built here because the reviewed ext / Phase 2 dirs exist only on the DGX
# the scripts resolve data/ and sam2.1_b.pt relative to the repo root (= v2code): point them at the workspace
[ -e $C/data ] || ln -s $W/data $C/data
[ -e $C/sam2.1_b.pt ] || ln -s $W/sam2.1_b.pt $C/sam2.1_b.pt
[ -f $W/data/label_patches_v10.jsonl ] || run build_patches_v10 $PY -u $C/scripts/data_pipeline/build_label_patches.py \
  --device 0 --out $W/data/label_patches_v10.jsonl || exit 1

# 2) pool build: identical arguments to phase3_chain.sh (v1 training ontology)
EMPTYFIX=""; [ -f $W/data/engine_bay_empty_fix/EMPTY_FIX_SUMMARY.json ] && EMPTYFIX="$W/data/engine_bay_empty_fix:train"
[ -f $P/data_full.yaml ] || run build_full_v10 $PY -u $C/scripts/data_pipeline/build_full_dataset.py \
  --base $W/data/engine_bay_train_v9 \
  --reviewed-train $EMPTYFIX $W/data/engine_bay_p2_verified:train \
  --reviewed $W/data/engine_bay_reviewed_p2:p2 $W/data/engine_bay_reviewed_hybrid:train \
  --ann-config $C/configs/data_engine_bay.yaml --train-classes $C/configs/engine_bay_train_classes.yaml \
  --candidates $W/data/engine_bay_phase2/CANDIDATES.jsonl --dup-groups $W/data/engine_bay_phase2/DUPLICATES.json \
  --folds 3 --out $P || exit 1

# 3) patch the new pool (labels/all and the labels/dataset links), then split by vehicle
if [ ! -f $P/labels/PATCH_REPORT_all.json ]; then
  run patch_v10 $PY -u $C/scripts/data_pipeline/patch_labels.py --labels $P/labels/all --data-yaml $P/data_full.yaml \
    --patches $W/data/label_patches_v10.jsonl --train-classes $C/configs/engine_bay_train_classes.yaml || exit 1
  [ -d $P/labels/dataset ] && run patch_v10_dataset $PY -u $C/scripts/data_pipeline/patch_labels.py --labels $P/labels/dataset \
    --data-yaml $P/data_full.yaml --patches $W/data/label_patches_v10.jsonl --train-classes $C/configs/engine_bay_train_classes.yaml
fi
[ -f $P/data_vehicle.yaml ] || run vehicle_split_v10 $PY -u $C/scripts/data_pipeline/vehicle_split.py --pool $P || exit 1

# 4) teacher (p5 recipe, best-by-val top-5 averaging on the clean vehicle val split)
if [ ! -f $S/$T/weights/avg5.pt ]; then
  run $T $PY -u -c "from ultralytics import YOLO; YOLO(\"yolo11l-seg.pt\").train(data=\"$P/data_vehicle.yaml\", imgsz=640, batch=16, epochs=100, patience=100, cos_lr=True, copy_paste=0.3, mixup=0.1, degrees=5.0, close_mosaic=10, cache=\"ram\", workers=8, device=0, project=\"$S\", name=\"$T\", exist_ok=True, seed=0, save_period=5, weight_decay=0.001, scale=0.7, perspective=0.0005)" || exit 1
  run ${T}_avg $PY -u average_checkpoints.py --run $S/$T --top-k 5 --out $S/$T/weights/avg5.pt || exit 1
fi

# 5) KD students, 2 seeds (kd_hyperparams_p5, as kd_n_p5t)
M="--model teacher_v10_avg5=$S/$T/weights/avg5.pt@640 --model teacher_p5_reg_avg5=$S/p5_reg/weights/avg5.pt@640"
M="$M --model kd_n_p5t_s0_avg5=$R/kd_n_p5t_s0/weights/avg5.pt@640 --model kd_n_p5t_s1_avg5=$R/kd_n_p5t_s1/weights/avg5.pt@640"
for s in 0 1; do
  n=kd_n_v10_s$s
  if [ ! -f $R/$n/weights/avg5.pt ]; then
    run $n $PY -u $C/scripts/training/train_kd.py --kd $C/configs/kd_hyperparams_p5.yaml --teacher $S/$T/weights/avg5.pt \
      --student yolo11n-seg.pt --data $P/data_vehicle.yaml --seed $s --device 0 --project $R --name $n || continue
    run ${n}_avg $PY -u average_checkpoints.py --run $R/$n --top-k 5 --strip-kd --out $R/$n/weights/avg5.pt
  fi
  [ -f $R/$n/weights/avg5.pt ] && M="$M --model ${n}_avg5=$R/$n/weights/avg5.pt@640"
done

# 6) everything on test v2 (patched labels of the 3 held-out vehicles)
run qa_v10 $PY -u qa_test.py $M --data $P/data_vehicle.yaml --split test --out qa_results_v10
echo "$(date +%H:%M:%S) V10 CHAIN DONE (qa_results_v10/QA_REPORT.md, patches: $P/labels/PATCH_REPORT_all.json)"
