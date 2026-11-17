# Test tools

## `bench/`: offline bench (no ROS, minutes)

`bench.py` drives the crop-row tracker (`groundTracker.py`) with pure pursuit on a simulated bed field, seen through
the Isaac ZED camera model, and measures against the truth: lateral error, distance driven off the lane (more than half
a furrow from the true lane centre), losses, stops (FAILED) and the wheel-to-crop distance (gen5 wheels, tyre 0.315 m).
Settings are environment variables, listed in its docstring.

```
python3 -I tools/bench/bench.py 7                                    # one run, seed 7, field beds3
FIELD=weeds2 TRACE=15,22 python3 -I tools/bench/bench.py 9           # tracker state between 15 and 22 m
python3 tools/bench/run_seeds.py FIELD=weeds2 WD=8 WP=3              # 30 seeds of one configuration
python3 tools/bench/run_seeds.py --set core                          # regression sets: core, stress, all
python3 tools/bench/run_seeds.py --tracker /tmp/gt_new.py --set core # another version of the tracker
```

A change to the tracker is kept only if the core set has no new off-lane run and no more stops, and the case it
targets improves. `CTRL=oracle` drives a perfect follower on the true lane: what is left then comes from the field
(row wander, plant scatter, bends), not from the navigation.

## `isaac/`: Isaac Sim runs on the VM (`ssh isaac`, ~17 min per field)

Copies of the scripts used on the VM (`~/isaac_sim_stack/vnav_ws/tmp/`):

- `isaac_field_test4.sh "weeds2:0 beds3:3"`: per field (`field:lane_rows`), starts the sim (`nav start`, one retry
  if Isaac crashes), the truth publisher, a bag and the navigator; stops at the row end, FAILED, `MAXX` metres
  (default 25, 0: the whole row) or when the robot does not move for 3 min (field retried once). Env: `OUT`,
  `PARAMS`, `PLAN` (straight plan + path watchdog), `NAV_ODOM=fused` (wheel + gyro odometry instead of the truth).
- `itest_analyze.py <dir> field_r0:4`: true lateral error, metres off lane, losses, final state.
  A max around 1.4 m on a single sample is an artefact (the neighbour bed taken as reference).
- `itest_wheels.py <dir> field_r0`: closest wheel axis to crop plant distance per wheel.
- `itest_series.py`, `itest_status.py`: per metre / per status change views of a bag.
- `grab_along.py`, `grab_graphic.py`: camera and tracking images saved along a run or once.
- `pub_plan.py`: a straight planned pass for the path watchdog.
