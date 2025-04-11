#!/bin/bash
#tag="exp1_p8_seeds"
PROJ_DIR="${HOME}/locrepos/s4"
RUN_FILE="completed_runs.log"
echo ${PROJ_DIR}
# Function to check if a run_id is completed
check_completed() {
    run_id=$1
    if grep -q "^$run_id$" "$RUN_FILE"; then
        return 0  # Run already completed
    else
        return 1  # Run not completed yet
    fi
}

# Function to track a completed run
log_run() {
    run_id=$1
    echo "$run_id" >> "$RUN_FILE"  
    echo "Run $run_id completed and logged."
}

# PARAMS
ds=mnist
model=DeepRNN
# model=S4

lr=1e-3
Ls=(1 2 4)
seeds=(2 3 4 5 6 7 8 9)
hss=(128) # suggested, not sure what other values would be suitable

# dropout=0.2
# epoch=100
# bs=64 # batch_size
# prenorm=true # only for S4 currently
# criterion="CrossEntropy" #"L1" "MSE"

params=()
for seed in "${seeds[@]}";do
for L in "${Ls[@]}";do
for hs in "${hss[@]}";do
      params+=("$seed,$L,$hs")
done
done
done

ii=0
for param in "${params[@]}"; do
	ii=$((ii+1))
	run_id=$ii  
	if check_completed "$run_id"; then
	    echo "Run $run_id already completed."
	else
	    echo "Running ID $run_id..."
		IFS=',' read -r seed L hs <<< "$param"
		params_chain="L-${L}_hs-${hs}"
		printf "=====================\n=\n=\n=\n=\n=\n"
		echo $params_chain
		printf "=\n=\n=\n=\n=\n=====================\n"
		python "${PROJ_DIR}/example.py" --model $model --n_layers $L --d_model $hs --seed $seed --dataset $ds --lr $lr
    	if [[ $? -eq 0 ]]; then
	    	log_run "$run_id"
	    else
	    	echo "Run failed for ${params_chain}"
	    fi
	fi
done
