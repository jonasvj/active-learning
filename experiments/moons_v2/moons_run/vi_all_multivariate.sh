# Directory of project
project_dir=~/active-learning

# Experiment name is name of this shell script
experiment_name=`basename -s .sh "$0"`

# Folder where this script is located
run_folder=`dirname -- "$0"`

# Folder for data and plots
data_folder="${run_folder%_run}_data"
plot_folder="${run_folder%_run}_plots"

mkdir -p ${data_folder}
mkdir -p ${plot_folder}

# Data Parameters
data_params="\
    data.data_class=MoonsDataset \
    data.data_hparams.batch_size=100 \
    data.data_hparams.n_train=100 \
    data.data_hparams.n_val=100 \
    data.data_hparams.n_test=100 \
    data.data_hparams.noise=0.1"

# Model parameters
model_params="\
    model.model_class=ClassificationFNN \
    model.model_hparams.n_in=2 \
    model.model_hparams.n_classes=2 \
    model.model_hparams.hidden_sizes=[5] \
    model.model_hparams.drop_probs=[0.05] \
    model.model_hparams.prior_scale_bias=1.0 \
    model.model_hparams.prior_scale_weight=1.0 \
    model.model_hparams.scale_weight_prior_by_dim=False \
    model.model_hparams.device=cuda"

# Inference parameters
inference_params="\
    inference=vi \
    inference.inference_class=VI \
    inference.init_hparams.n_posterior_samples=500 \
    inference.init_hparams.subset_of_weights=all \
    
    inference.fit_hparams.fit_vi_hparams.n_epochs=100000 \
    inference.fit_hparams.fit_vi_hparams.lr=1e-3 \
    inference.fit_hparams.fit_vi_hparams.guide=multivariate \
    inference.fit_hparams.fit_vi_hparams.num_particles=32 \
    inference.fit_hparams.fit_vi_hparams.init_scale=1e-4"

# Data path
data_path="data_path=${data_folder}/${experiment_name}"

python_command="python3 ${project_dir}/experiments/moons/moons.py \
    ${data_params} \
    ${model_params} \
    ${inference_params} \
    ${data_path}"

# BSUB arguments
queue="gpuv100"
n_cores="2"
n_gpus="1"
wall_time="23:59"
memory="4GB"

job_name=`basename ${run_folder}`
job_name="${job_name%_run}_${experiment_name}"
std_out=`echo ${project_dir}/misc_files/${job_name}.out`
std_err=`echo ${project_dir}/misc_files/${job_name}.err`

# All BSUB args
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

full_command="\
    cd ~/;\
    source .virtualenvs/al/bin/activate;\
    cd ~/active-learning;\
    ${python_command}"


if [[ $1 = "bsub" ]]
then
    # Submit to BSUB
    echo "Now submitting to BSUB:"
    echo ${python_command}
    bsub $bsub_args $full_command
else
    # Run command interactively
    echo "Now running:"
    echo ${python_command}
    ${python_command}
fi