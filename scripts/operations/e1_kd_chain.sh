#!/bin/bash
# Experiment E1 (docs/plans/small_component_detection_research_2026-09-29.md): FGD scale-aware mask.
# Same as p5_kd_chain.sh (teacher p5_reg avg5, data v8, 2 seeds, top-5 by val averaging) with ONE change:
# configs/kd_hyperparams_e1.yaml = kd_hyperparams_p5.yaml + distillation.fgd.scale_aware: true.
# Reference (test, 3 vehicles, mask mAP50-95): kd_n_p5t_s0_avg5 0.302, kd_n_p5t_s1_avg5 0.270 (mean 0.286).
# Gate: mean of the 2 E1 seeds >= 0.296 (+1 point) AND small-class APs (oil_dipstick, oil_filler_cap,
# battery_terminal, radiator_cap) not lower on average.
#
# Needs the new code in ~/distillation_workspace/v2code (resync after changes):
#   (local) tar czf - --exclude=__pycache__ scripts configs src | ssh admin@<dgx> 'tar xzf - -C ~/distillation_workspace/v2code'
# Launch detached (waits for every other chain; one GPU job at a time):
#   setsid nohup bash v2code/scripts/operations/e1_kd_chain.sh > runs/logs/e1_kd_chain.log 2>&1 < /dev/null & disown
W=/home/admin/distillation_workspace
C=$W/v2code
cd $W
export CPATH=$HOME/pyheaders/usr/include/python3.12:$HOME/pyheaders/usr/include
export PYTHONPATH=$C/src:$C/scripts
PY=$W/.venv/bin/python
D=$W/data/engine_bay_train_v8/data_engine_bay_train.yaml
T=$W/runs/segment/p5_reg/weights/avg5.pt
R=$W/runs/train_kd
run() { local name=$1; shift; echo "$(date +%H:%M:%S) START $name"; "$@" > runs/logs/$name.log 2>&1; local rc=$?; echo "$(date +%H:%M:%S) END $name rc=$rc"; return $rc; }

while pgrep -f "phase3_chain.sh|phase5_chain.sh|p5_kd_chain.sh|v8_chain.sh|v2_teacher_chain.sh" >/dev/null; do sleep 300; done
[ -f $C/configs/kd_hyperparams_e1.yaml ] || { echo "v2code not synced ($C)"; exit 1; }

M="--model kd_n_p5t_s0_avg5=$R/kd_n_p5t_s0/weights/avg5.pt@640 --model kd_n_p5t_s1_avg5=$R/kd_n_p5t_s1/weights/avg5.pt@640"
for s in 0 1; do
  n=kd_n_e1_s$s
  if [ ! -f $R/$n/weights/avg5.pt ]; then
    run $n $PY -u $C/scripts/training/train_kd.py --kd $C/configs/kd_hyperparams_e1.yaml --teacher $T --student yolo11n-seg.pt \
      --data $D --seed $s --device 0 --project $R --name $n || continue
    run ${n}_avg $PY -u average_checkpoints.py --run $R/$n --top-k 5 --strip-kd --out $R/$n/weights/avg5.pt
  fi
  [ -f $R/$n/weights/avg5.pt ] && M="$M --model ${n}_avg5=$R/$n/weights/avg5.pt@640"
done
run qa_e1 $PY -u qa_test.py $M --data $D --split test --out qa_results_e1
echo "$(date +%H:%M:%S) E1 CHAIN DONE (qa_results_e1/QA_REPORT.md)"
