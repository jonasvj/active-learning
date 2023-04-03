python3 src/models/active_learning.py \
    model=mnist_conv_net \
    data=mnist inference=swag \
    al_seed=8 al.n_acquisitions=100 \
    al.acquisition_size=10 \
    al.acquisition_function=max_entropy \
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
    data.data_hparams.seed=8 \
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
    inference.fit_hparams.fit_swag_hparams.swag_lr=[1.000e-06,3.594e-06,1.292e-05,4.642e-05,1.668e-04,5.995e-04,2.154e-03,7.743e-03,2.783e-02,1.000e-01] \
    inference.fit_hparams.fit_swag_hparams.update_freq=10 \
    inference.fit_hparams.fit_swag_hparams.clip_value=null \
    inference.fit_hparams.fit_swag_hparams.save_iterates=False \
    inference.fit_hparams.fit_swag_hparams.val_criterion=accuracy
