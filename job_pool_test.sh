#!/bin/bash

. job_pool.sh

job_pool_init 3 0

export TEST=asdf
job_pool_run sleep 1
job_pool_run sleep 1
job_pool_run sleep 1
job_pool_run echo ${TEST}

for i in 1 2 3 4 5; do
export TEST=asdf$i
job_pool_run echo ${TEST}
job_pool_run sleep 1
done
#job_pool_run sleep 6 && echo 'hello2'
#job_pool_run sleep 3 && echo 'hello3'

job_pool_wait