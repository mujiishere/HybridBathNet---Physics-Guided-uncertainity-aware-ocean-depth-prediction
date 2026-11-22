#!/bin/bash
cd /home/claude/hbn; export PYTHONPATH=.
while pgrep -f "run_study cv --config full" >/dev/null; do sleep 30; done
for t in seribu_id langebaan_za ponce_pr; do
  python3 -m hbn2.run_study loso --target $t --members 2 --out runs_v2/loso >> runs_v2/loso.out 2>&1
  python3 -m hbn2.run_study loso --target $t --adapt --members 2 --out runs_v2/loso >> runs_v2/loso.out 2>&1
done
for c in cnn_only no_physics_loss no_physics_features no_film; do
  [ -f runs_v2/ABL_SKIP_$c ] && continue
  python3 -m hbn2.run_study cv --config $c --members 1 --out runs_v2/abl_$c >> runs_v2/abl.out 2>&1
done
echo ALLDONE >> runs_v2/queue.log
