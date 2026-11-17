#!/bin/bash
# (series 4: JPEG copies for Foxglove; NAV_ODOM from the environment; series 3: per field:lane_rows file names, new tracker code, optional planned pass + path watchdog, camera stall detection, sim start retry)
# Isaac field test of the visual navigation, ONE parameter file for every field (only lane_rows per bed layout).
# Run on the VM host: ~/isaac_sim_stack/vnav_ws/tmp/isaac_field_test.sh "beds3:3 seed3:3 weeds2:4 mid4:4 late4:4"
# Per field: nav start (VNAV_CAM=1), show_field (truth), bag (odom, status, truth error, crops), navigator driving
# /gui_cmd_vel until ROW_END / FAILED / 30 min, then everything stopped. Results in vnav_ws/tmp/itest/.
W=/home/brainy/isaac_sim_stack/vnav_ws
N=/home/brainy/isaac_sim_stack/laser_sim/navfield
OUT=${OUT:-$W/tmp/itest2}; mkdir -p $OUT
PARAMS=${PARAMS:-$W/params_isaac_unified.yaml}
FIELDS=${1:-"beds3:0 seed3:0 weeds2:0 mid4:0 late4:0"}
PLAN=${PLAN:-}   # e.g. "0.0": publish the planned pass (pub_plan.py) and turn the path watchdog on
inC() { docker exec eterry_host bash -lc "source $W/env.sh >/dev/null 2>&1; $*"; }
stopAll() {
  inC "pkill -INT -f vs_navigator.py; pkill -INT -f 'ros2 bag record'; pkill -f show_field.py; pkill -f pub_plan.py; pkill -f '[r]epublish raw compressed'; pkill -x republish" >/dev/null 2>&1
  sleep 5
  inC "pkill -9 -f vs_navigator.py; pkill -9 -f show_field.py" >/dev/null 2>&1
}
set -- $FIELDS; declare -A RETRIED
while [ $# -gt 0 ]; do
  fr=$1; shift; stuck=0
  F=${fr%%:*}; R=${fr#*:}
  echo "=== $(date +%H:%M:%S) field $F, lane_rows $R, params $PARAMS"
  stopAll
  # Isaac can crash while starting (renderer segfault, exit 139): checked on its clock and Nav2 ("inactive"
  # also contains "active"), one more try
  started=0
  for attempt in 1 2; do
    (cd $N && ./nav stop) > $OUT/${F}_r${R}_navstop.log 2>&1; sleep 10
    (cd $N && VNAV_CAM=1 timeout 900 ./nav start $F) > $OUT/${F}_r${R}_navstart.log 2>&1
    if grep -q "sim clock" $OUT/${F}_r${R}_navstart.log && grep -q "bt_navigator: active" $OUT/${F}_r${R}_navstart.log; then started=1; break; fi
    echo "  $(date +%H:%M:%S) sim did not start (attempt $attempt): $(grep -m1 -iE "no /clock|error" $OUT/${F}_r${R}_navstart.log)"
    cp $OUT/${F}_r${R}_navstart.log $OUT/${F}_r${R}_navstart_fail$attempt.log
  done
  if [ $started = 0 ]; then echo "  sim did not start, field skipped, see $OUT/${F}_r${R}_navstart.log"; continue; fi
  docker exec -d eterry_host bash -lc "source $W/env.sh >/dev/null 2>&1; exec python3 $W/show_field.py $F > $OUT/${F}_r${R}_show.log 2>&1"
  sleep 8
  rm -rf $OUT/${F}_r${R}_bag
  EXTRA=""
  if [ -n "$PLAN" ]; then
    docker exec -d eterry_host bash -lc "source $W/env.sh >/dev/null 2>&1; exec python3 $W/tmp/pub_plan.py $PLAN /vnav/plan > $OUT/${F}_r${R}_plan.log 2>&1"
    EXTRA="-p reference_path_topic:=/vnav/plan -p max_path_deviation:=0.7"
    sleep 3
  fi
  docker exec -d eterry_host bash -lc "source $W/env.sh >/dev/null 2>&1; exec ros2 bag record -o $OUT/${F}_r${R}_bag /odom /vs_nav/status /vnav/lateral_error /field/crops /vs_nav/path_deviation > $OUT/${F}_r${R}_bag.log 2>&1"
  sleep 4
  docker exec -d eterry_host bash -lc "source $W/env.sh >/dev/null 2>&1; exec ros2 run visual_multi_crop_row_navigation vs_navigator.py --ros-args -r __node:=vs_navigation --params-file $PARAMS -p use_sim_time:=true -p cmd_vel_topic:=/gui_cmd_vel -p lane_rows:=$R $EXTRA > $OUT/${F}_r${R}_nav.log 2>&1"
  # JPEG copies for Foxglove (layout vnav_ws/foxglove_vnav_layout.json), as run_vnav.sh
  for t in /visual_nav_camera/rgb/image_raw:/visual_nav_camera/view /vs_nav/graphic:/vs_nav/graphic/view; do
    docker exec -d eterry_host bash -lc "source $W/env.sh >/dev/null 2>&1; exec ros2 run image_transport republish raw compressed --ros-args -r in:=${t%%:*} -r out/compressed:=${t#*:}/compressed -p out.jpeg_quality:=50 > /dev/null 2>&1"
  done
  t0=$(date +%s); seen=0
  lastx=x; still=0
  while [ $(( $(date +%s) - t0 )) -lt 2400 ]; do
    sleep 20
    # state transitions from the navigator log (a status poll can miss the last one)
    tr=$(grep -E "\] \[vs_navigation\]: [A-Z_]+ -> [A-Z_]+" $OUT/${F}_r${R}_nav.log 2>/dev/null | sed -E "s/.*\]: //")
    n=$(echo "$tr" | grep -c .)
    if [ "$n" -gt "$seen" ]; then echo "$tr" | tail -n +$((seen+1)) | grep -vE "^(SCAN -> ADVANCE|ADVANCE -> SCAN)" | sed "s/^/  $(date +%H:%M:%S) /" | cut -c1-170; seen=$n; fi
    echo "$tr" | grep -qE -e "-> (ROW_END|FAILED)" && break
    # shorter runs: stop once the robot has driven MAXX m along the row (0: to the row end)
    xf=$(inC "timeout 6 ros2 topic echo --once /odom 2>/dev/null | grep -A1 position: | tail -1" | awk '{printf "%.1f", $2}')
    if echo "$tr" | grep -q "SCAN -> TRACKING" && [ -n "$xf" ]; then
      if [ "$xf" = "$lastx" ]; then still=$((still + 1)); else still=0; fi
      lastx=$xf
      if [ $still -ge 9 ]; then echo "  $(date +%H:%M:%S) ROBOT NOT MOVING (odom x $xf for 3 min): field abandoned"; stuck=1; break; fi
    fi
    if [ "${MAXX:-25}" != 0 ]; then
      xo=${xf%%.*}
      if [ -n "$xo" ] && [ "$xo" -ge "${MAXX:-25}" ]; then echo "  $(date +%H:%M:%S) reached ${MAXX:-25} m: run stopped (MAXX)"; break; fi
    fi
    # the sim's camera can stall (rendering): no image for 2 min -> give up this field
    st=$(inC "timeout 6 ros2 topic echo --once /vs_nav/status --qos-durability transient_local 2>/dev/null | head -1")
    secs=$(echo "$st" | sed -nE 's/.*no camera image for ([0-9]+)\.[0-9]+ s.*/\1/p')
    if [ -n "$secs" ] && [ "$secs" -gt 120 ]; then echo "  $(date +%H:%M:%S) CAMERA STALLED ($secs s): field abandoned"; break; fi
  done
  [ $(( $(date +%s) - t0 )) -ge 2400 ] && echo "  $(date +%H:%M:%S) timeout (40 min)"
  stopAll
  echo "=== $(date +%H:%M:%S) field $F done"
  if [ $stuck = 1 ] && [ -z "${RETRIED[$fr]}" ]; then RETRIED[$fr]=1; echo "  field $fr queued again (robot did not move)"; set -- "$@" "$fr"; fi
done
(cd $N && ./nav stop) > $OUT/final_navstop.log 2>&1
echo "ALL DONE"
