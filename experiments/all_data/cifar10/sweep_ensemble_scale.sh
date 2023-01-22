# Directory of project
project_dir=~/active-learning

script=$(realpath "$0")
script_dir=$(dirname "$script")
script_name=$(basename "$script" .sh)
echo "Name of script: ${script_name}"

# Name of experiment
experiment_name=${script%.sh}
experiment_name=${experiment_name//\//_}
experiment_name=${experiment_name#*_experiments_}
echo "Name of experiment: ${experiment_name}"

# Folder for storing results
data_folder="${script_dir}/${script_name}_data"
mkdir -p ${data_folder}
echo "Data directory: ${data_folder}"

# BSUB arguments
queue="gpuv100"
n_cores="2"
n_gpus="1"
wall_time="12:00"
memory="4GB"

base_bsub_args=`echo \
    -q ${queue} \
    -n ${n_cores} \
    -R "span[hosts=1]" \
    -gpu "num=${n_gpus}:mode=exclusive_process" \
    -W ${wall_time} \
    -R "rusage[mem=${memory}]"`

base_command="\
    cd ~/;\
    source .virtualenvs/al/bin/activate;\
    cd ~/active-learning;\
    python3 ${script_dir}/train_cifar10.py"

# Seeds
seeds=(0 1 2 3 4)

# Build commands
command_list=()
bsub_args_list=()

method="laplace_ensemble"
scales=() #(0.75 0.5 0.25 0.1 1e-2 1e-3 1e-4 1e-5 1e-6)
for seed in "${seeds[@]}"; do
    for scale in "${scales[@]}"; do
        data_path="${data_folder}/${experiment_name}_${seed}_${method}_${scale}.dill"
        full_command="${base_command} seed=${seed} inference_key=${method} data_path=${data_path} \
            laplace_ensemble.fit_hparams.covar_scale=${scale}"
        command_list+=("${full_command}")

        job_name=`echo ${experiment_name}_${seed}_${method}_${scale}`
        std_out=`echo ${project_dir}/misc_files/${job_name}.out`
        std_err=`echo ${project_dir}/misc_files/${job_name}.err`
        bsub_args=`echo ${base_bsub_args} -J ${job_name} -o ${std_out} -e ${std_err}`
        bsub_args_list+=("${bsub_args}")
    done
done

# Unscaled covar but 100 mc samples
scale=1.0
for seed in "${seeds[@]}"; do
    data_path="${data_folder}/${experiment_name}_${seed}_${method}_${scale}_100mc.dill"
    full_command="${base_command} seed=${seed} inference_key=${method} data_path=${data_path} \
        laplace_ensemble.fit_hparams.covar_scale=${scale} mc_samples=100"
    command_list+=("${full_command}")

    job_name=`echo ${experiment_name}_${seed}_${method}_${scale}_100mc`
    std_out=`echo ${project_dir}/misc_files/${job_name}.out`
    std_err=`echo ${project_dir}/misc_files/${job_name}.err`
    bsub_args=`echo ${base_bsub_args} -J ${job_name} -o ${std_out} -e ${std_err}`
    bsub_args_list+=("${bsub_args}")
done

# Execute commands
for ((i = 0; i < ${#command_list[@]}; i++)); do
    full_command="${command_list[$i]}"
    bsub_args="${bsub_args_list[$i]}"
    
    if [[ $1 = "bsub" ]]
    then
        echo "Now submitting to BSUB:"
        echo ${full_command}
        bsub $bsub_args $full_command
        sleep 1
    # Run interactively
    else
        echo "Now running:"
        echo ${full_command}
        eval ${full_command}
    fi
done