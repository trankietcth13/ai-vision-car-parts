#!/bin/bash
# distill_v3: teacher AND student moved to YOLO26 (student yolo26s-seg = the model Nich uses), on the v10 data
# (already labelled and patched; nothing is relabelled). Same split by vehicle (val 08/36/49/50/52/58, test 13/23/29),
# same p5 recipe, top-5 checkpoint averaging; only the architectures change.
#   teacher   yolo26l-seg  (replaces yolo11l-seg engine_teacher_v10)
#   student   yolo26s-seg  KD from the yolo26l teacher, 2 seeds
#   baseline  yolo26s-seg  identical schedule without a teacher, 2 seeds (Student-only mode: shows what KD adds)
# Everything is scored on test v2 next to the v10 yolo11 models (qa_results_v26).
# YOLO26 heads have reg_max=1 (no DFL): box KD falls back to IoU distillation, and the deployed one2one head
# learns from the teacher through the features only.
#
# Needs on the DGX: v2code synced (scripts/configs/src), data/engine_bay_full_v10/data_vehicle.yaml (built by v10_chain.sh).
# Smoke test first (1 epoch, 10% of the data, results under runs/smoke_v26):  bash v2code/scripts/operations/v26_chain.sh --smoke
# Launch detached (waits for every other chain; one GPU job at a time):
#   setsid nohup bash v2code/scripts/operations/v26_chain.sh > runs/logs/v26_chain.log 2>&1 < /dev/null & disown
W=/home/admin/distillation_workspace
C=$W/v2code
cd $W
export CPATH=$HOME/pyheaders/usr/include/python3.12:$HOME/pyheaders/usr/include
export PYTHONPATH=$C/src:$C/scripts
PY=$W/.venv/bin/python
P=$W/data/engine_bay_full_v10
S=$W/runs/segment
R=$W/runs/train_kd
T=engine_teacher_v26l
KD=$C/configs/kd_hyperparams_p5.yaml
TEPOCHS=100; TSAVE=5; TFRAC=1.0
SEEDS="0 1"
QA_OUT=qa_results_v26
LOGS=runs/logs
if [ "$1" = "--smoke" ]; then
  S=$W/runs/smoke_v26/segment; R=$W/runs/smoke_v26/train_kd; LOGS=runs/smoke_v26/logs; mkdir -p $LOGS
  TEPOCHS=1; TSAVE=-1; TFRAC=0.1
  sed -e 's/^  epochs: .*/  epochs: 1/' -e 's/^  save_period: .*/  save_period: -1/' $KD > $W/runs/smoke_v26/kd_smoke.yaml
  echo "  fraction: 0.1" >> $W/runs/smoke_v26/kd_smoke.yaml
  KD=$W/runs/smoke_v26/kd_smoke.yaml
  SEEDS="0"
  QA_OUT=runs/smoke_v26/qa
fi
run() { local name=$1; shift; echo "$(date +%H:%M:%S) START $name"; "$@" > $LOGS/$name.log 2>&1; local rc=$?; echo "$(date +%H:%M:%S) END $name rc=$rc"; return $rc; }
# average the top-5 checkpoints by val; a smoke run has a single epoch, so best.pt stands in
avg() { local mode=$1 dir=$2; shift 2
  if [ $mode = smoke ]; then cp $dir/weights/best.pt "${@: -1}"; else $PY -u average_checkpoints.py --run $dir "$@"; fi; }
MODE=full; [ "$1" = "--smoke" ] && MODE=smoke

if [ $MODE = full ]; then
  while pgrep -f "phase3_chain.sh|phase5_chain.sh|p5_kd_chain.sh|v8_chain.sh|v2_teacher_chain.sh|e1_kd_chain.sh|v10_chain.sh" >/dev/null; do sleep 300; done
fi
[ -f $P/data_vehicle.yaml ] || { echo "v10 data missing ($P/data_vehicle.yaml): run v10_chain.sh first"; exit 1; }

# 1) teacher yolo26l-seg, p5 recipe (identical arguments to engine_teacher_v10)
if [ ! -f $S/$T/weights/avg5.pt ]; then
  run $T $PY -u -c "from ultralytics import YOLO; YOLO(\"yolo26l-seg.pt\").train(data=\"$P/data_vehicle.yaml\", imgsz=640, batch=16, epochs=$TEPOCHS, fraction=$TFRAC, patience=100, cos_lr=True, copy_paste=0.3, mixup=0.1, degrees=5.0, close_mosaic=10, cache=\"ram\", workers=8, device=0, project=\"$S\", name=\"$T\", exist_ok=True, seed=0, save_period=$TSAVE, weight_decay=0.001, scale=0.7, perspective=0.0005)" || exit 1
  run ${T}_avg avg $MODE $S/$T --top-k 5 --out $S/$T/weights/avg5.pt || exit 1
fi

# 2) students yolo26s-seg: KD from the yolo26l teacher, and the same schedule without a teacher
M="--model teacher_v26l_avg5=$S/$T/weights/avg5.pt@640 --model teacher_v10_avg5=$W/runs/segment/engine_teacher_v10/weights/avg5.pt@640"
M="$M --model kd_n_v10_s0_avg5=$W/runs/train_kd/kd_n_v10_s0/weights/avg5.pt@640 --model kd_n_v10_s1_avg5=$W/runs/train_kd/kd_n_v10_s1/weights/avg5.pt@640"
for s in $SEEDS; do
  for kind in kd base; do
    n=${kind}_26s_v26_s$s
    if [ ! -f $R/$n/weights/avg5.pt ]; then
      if [ $kind = kd ]; then
        run $n $PY -u $C/scripts/training/train_kd.py --kd $KD --teacher $S/$T/weights/avg5.pt \
          --student yolo26s-seg.pt --data $P/data_vehicle.yaml --seed $s --device 0 --project $R --name $n || continue
      else
        run $n $PY -u $C/scripts/training/train_kd.py --no-kd --kd $KD \
          --student yolo26s-seg.pt --data $P/data_vehicle.yaml --seed $s --device 0 --project $R --name $n || continue
      fi
      run ${n}_avg avg $MODE $R/$n --top-k 5 --strip-kd --out $R/$n/weights/avg5.pt
    fi
    [ -f $R/$n/weights/avg5.pt ] && M="$M --model ${n}_avg5=$R/$n/weights/avg5.pt@640"
  done
done

# 3) everything on test v2 (patched labels of the 3 held-out vehicles)
run qa_v26 $PY -u qa_test.py $M --data $P/data_vehicle.yaml --split test --out $QA_OUT
echo "$(date +%H:%M:%S) V26 CHAIN DONE ($MODE, $QA_OUT/QA_REPORT.md)"
