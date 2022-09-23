project_dir=~/active-learning

# BSUB arguments
queue="gpuv100"
job_name="gpu_job"
n_cores="4"
n_gpus="1"
wall_time="02:00"
memory="8GB"
std_out="${project_dir}/misc_files/gpu_%J.out"
std_err="${project_dir}/misc_files/gpu_%J.err"

bsub_args=`echo \
    -q ${queue} \
    -J ${job_name} \
    -n ${n_cores} \
    -R "span[hosts=1]" \
    -gpu "num=${n_gpus}:mode=exclusive_process" \
    -W ${wall_time} \
    -R "rusage[mem=${memory}]" \
    -o ${std_out} \
    -e ${std_err}`

# Parameters to execute over
seeds=(0 1 2 3 4 5 6 7 8 9)
acquisition_functions=(random_acquisition max_entropy bald)

for acquisition_function in "${acquisition_functions[@]}"; do
    for seed in "${seeds[@]}"; do    
        echo $seed $acquisition_function
        
        python_command="python3 ${project_dir}/src/models/active_learning.py \
            al_seed=${seed} \
            al.n_acquisitions=100 \
            al.acquisition_size=10 \
            al.acquisition_function=${acquisition_function} \
            al.balanced_initial_set=True \
            al.init_acquisition_size=20 \
            al.pool_subsample=2000 \
            model.model_class=DropoutCNN \
            model.model_hparams.n_posterior_samples=100 \
            model.model_hparams.device=null \
            data.data_class=MNISTDataset \
            data.data_hparams.batch_size=128 \
            data.data_hparams.seed=${seed} \
            data.data_hparams.n_val=100 \
            inference.inference_hparams.n_epochs=50 \
            inference.inference_hparams.lr=1e-3 \
            inference.inference_hparams.weight_decay=2.5 \
            inference.inference_hparams.dynamic_weight_decay=True \
            inference.inference_hparams.optim_class=Adam \
            misc.result_path=${project_dir}/experiments/02_02_mnist_al_curve_subsample_pool/${acquisition_function}_${seed}.dill"
        
        full_command="\
            cd ~/;\
            source .virtualenvs/al/bin/activate;\
            cd ~/active-learning;\
            ${python_command}"
        
        # Submit job
        bsub $bsub_args $full_command
        sleep 1

    done
done