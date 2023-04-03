project_dir=~/active-learning

seeds=(9)
acquisition_functions=(bald)

for acquisition_function in "${acquisition_functions[@]}"; do
    for seed in "${seeds[@]}"; do    
        echo $seed $acquisition_function
        python3 $project_dir/src/models/active_learning.py \
            al_seed=$seed \
            al.n_acquisitions=100 \
            al.acquisition_size=10 \
            al.acquisition_function=$acquisition_function \
            al.balanced_initial_set=True \
            al.init_acquisition_size=20 \
            al.pool_subsample=null \
            model.model_class=DropoutCNN \
            model.model_hparams.n_posterior_samples=100 \
            model.model_hparams.device=null \
            data.data_class=FashionMNISTDataset \
            data.data_hparams.batch_size=128 \
            data.data_hparams.seed=$seed \
            data.data_hparams.n_val=100 \
            inference.inference_hparams.n_epochs=50 \
            inference.inference_hparams.lr=1e-3 \
            inference.inference_hparams.weight_decay=2.5 \
            inference.inference_hparams.dynamic_weight_decay=True \
            inference.inference_hparams.optim_class=Adam \
            misc.result_path=$project_dir/experiments/fmnist_al_curve/"${acquisition_function}_${seed}.dill"
    done
done