#!/bin/bash
for job in $(qstat -r | grep "Full jobname" | cut -c 26-50 | sort | uniq); do
    echo "==============================================================="
    echo "JOB: $job"
    bash exp/check.sh exp/${job}
    echo
done
