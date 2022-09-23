python3 src/models/active_learning.py \
    model.model_hparams.drop_probs=[0,0] \
    inference=swag \
    inference.init_hparams.with_gamma=True \
    inference.fit_hparams.optimize_covar=True \
    inference.fit_hparams.fit_swag_hparams.swag_lr=1e-2 \
    inference.fit_hparams.fit_covar_hparams.mini_batch=True \
    inference.fit_hparams.fit_covar_hparams.n_variational_samples=10