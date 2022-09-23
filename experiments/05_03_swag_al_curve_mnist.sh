# Directory of project
project_dir=~/active-learning

# Experiment name is name of this shell script
experiment_name=`basename "$0"`
experiment_name=${experiment_name::-3}

# Create experiment folder
experiment_folder="${project_dir}/experiments/${experiment_name}"
mkdir -p ${experiment_folder}

# Create plots folder and empty file for visualization
mkdir -p "${experiment_folder}_plots"
touch "${experiment_folder}_visualize.py"

# BSUB arguments
queue="gpuv100"
n_cores="2"
n_gpus="1"
wall_time="07:00"
memory="4GB"

# Parameters to execute over
lr_start=1e-7
lr_end=1
n_lrs=40
swag_lrs=`python3 -c \
    "import numpy as np; import sys; \
    lrs = np.geomspace(${lr_start},${lr_end},${n_lrs}); \
    lrs = ['{:.3e}'.format(lr) for lr in lrs]; \
    print('[' + ','.join(lrs) + ']', file=sys.stdout)"`
seeds=(0 1 2 3 4 5 6 7 8 9)
acquisition_functions=(random_acquisition max_entropy bald)

for seed in "${seeds[@]}"; do
    for acquisition_function in "${acquisition_functions[@]}"; do
            
        echo $seed $acquisition_function
        # Some parameter dependent bsub args
        job_name=`echo ${experiment_name}_${seed}_${acquisition_function}`
        std_out=`echo ${project_dir}/misc_files/${job_name}.out`
        std_err=`echo ${project_dir}/misc_files/${job_name}.err`

        # All bsubs args
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
        
        # Python command
        python_command="python3 ${project_dir}/src/models/active_learning.py \
            model=mnist_conv_net \
            data=mnist \
            inference=swag \
            al_seed=${seed} \
            al.n_acquisitions=100 \
            al.acquisition_size=10 \
            al.acquisition_function=${acquisition_function} \
            al.balanced_initial_set=True \
            al.init_acquisition_size=20 \
            al.pool_subsample=null \
            model.model_class=MNISTConvNet \
            model.model_hparams.drop_probs=[0,0] \
            model.model_hparams.prior=False \
            model.model_hparams.prior_var=1 \
            model.model_hparams.hyperprior=False \
            model.model_hparams.device=cuda \
            data.data_class=MNISTDataset \
            data.data_hparams.batch_size=128 \
            data.data_hparams.seed=${seed} \
            data.data_hparams.n_val=100 \
            inference.inference_class=SWAG \
            inference.init_hparams.K=50 \
            inference.init_hparams.n_posterior_samples=100 \
            inference.init_hparams.with_gamma=False \
            inference.init_hparams.sequential_samples=False \
            inference.fit_hparams.optimize_covar=False \
            inference.fit_hparams.fit_model_hparams.n_epochs=50 \
            inference.fit_hparams.fit_model_hparams.lr=1e-3 \
            inference.fit_hparams.fit_model_hparams.weight_decay=2.5 \
            inference.fit_hparams.fit_model_hparams.dynamic_weight_decay=True \
            inference.fit_hparams.fit_swag_hparams.swag_batch_size=null \
            inference.fit_hparams.fit_swag_hparams.swag_steps=1000 \
            inference.fit_hparams.fit_swag_hparams.swag_lr=${swag_lrs} \
            inference.fit_hparams.fit_swag_hparams.update_freq=10 \
            inference.fit_hparams.fit_swag_hparams.clip_value=null \
            inference.fit_hparams.fit_swag_hparams.save_iterates=False \
            inference.fit_hparams.fit_swag_hparams.val_criterion=accuracy \
            misc.result_path=${experiment_folder}/${job_name}.dill"
        
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