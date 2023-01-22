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

methods=("deterministic" "swag_svi" "swag") #("deterministic" "ensemble" "laplace" "laplace_ensemble" "swag" "swag_svi")
for seed in "${seeds[@]}"; do
    for method in "${methods[@]}"; do
        data_path="${data_folder}/${experiment_name}_${seed}_${method}.dill"
        full_command="${base_command} seed=${seed} inference_key=${method} data_path=${data_path} \
            model.model_class=WideResNetFRN \
            swag_svi.fit_hparams.fit_covar_hparams.svi_epochs=10"
        command_list+=("${full_command}")

        job_name=`echo ${experiment_name}_${seed}_${method}`
        std_out=`echo ${project_dir}/misc_files/${job_name}.out`
        std_err=`echo ${project_dir}/misc_files/${job_name}.err`
        bsub_args=`echo ${base_bsub_args} -J ${job_name} -o ${std_out} -e ${std_err}`
        bsub_args_list+=("${bsub_args}")
    done
done

method="laplace_nf"
n_transforms=() #(1 5 10 30)
for seed in "${seeds[@]}"; do
    for nt in "${n_transforms[@]}"; do
        data_path="${data_folder}/${experiment_name}_${seed}_${method}_${nt}.dill"
        full_command="${base_command} seed=${seed} inference_key=${method} data_path=${data_path} \
            laplace_nf.fit_hparams.fit_flow_hparams.n_transforms=${nt} model.model_class=WideResNetFRN"
        command_list+=("${full_command}")

        job_name=`echo ${experiment_name}_${seed}_${method}_${nt}`
        std_out=`echo ${project_dir}/misc_files/${job_name}.out`
        std_err=`echo ${project_dir}/misc_files/${job_name}.err`
        bsub_args=`echo ${base_bsub_args} -J ${job_name} -o ${std_out} -e ${std_err}`
        bsub_args_list+=("${bsub_args}")
    done
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