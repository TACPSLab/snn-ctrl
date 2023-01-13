#!/bin/bash

while :
do
    mem=$(/usr/bin/awk '$1=="MemTotal:"{t=$2} $1=="MemAvailable:"{f=$2} END{printf "%d", (t-f)/(t/100)}' /proc/meminfo)

    # If the memory usage is less than %
    if [ $mem -lt 35 ]
    then
        python3 ./launch.py
        break
    fi

    sleep 60
done
