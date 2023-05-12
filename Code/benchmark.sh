#!/bin/bash
nohup \
  python -u benchmark.py \
    --codecs JPEG2000 \
    --qps 1 2 3 4 5 10 15 20 \
    -j 12 \
&> logs/nohup.log &
echo $! > logs/pid.log
exit
