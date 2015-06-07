RUN_CMD="qsub -m abe -M lukas@zilka.me -hard -l mem_free=${MEM_PER_WORKER}G,act_mem_free=${MEM_PER_WORKER}G,h_vmem=${MEM_PER_WORKER}G -pe smp ${SMP} -N ${eid} -o ${STDOUT_LOCATION} -e ${STDERR_LOCATION} -cwd"

if [ -e "qsub_cmd.sh.local" ]; then
    . qsub_cmd.sh.local
fi
