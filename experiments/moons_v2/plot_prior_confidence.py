from src.models import ClassificationFNN
from src.data import MoonsDataset
from src.inference import HMC
from pyro.infer import Predictive
import torch
from src.utils import default_width
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def pred_w_prior(hmc_model, x):
    hmc_model.bayesian_model.eval()
    
    pred_dist = Predictive(
        model=hmc_model.bayesian_model.model,
        posterior_samples={},
        return_sites=['model_output'],
        parallel=False,
        num_samples=2000
    )

    return pred_dist(x)['model_output'].squeeze()

def pred_w_post(hmc_model, x):
   return hmc_model.predict(x).squeeze()


def get_prior_conf(hmc_model, x):
    preds = pred_w_prior(hmc_model, x)          # S x N x C
    probs = torch.softmax(preds, dim=-1)        # S x N x C
    avg_probs = torch.mean(probs, dim=0)        #     N x C
    confidence = torch.max(avg_probs, dim=-1).values

    return confidence


def get_post_conf(hmc_model, x):
    preds = pred_w_post(hmc_model, x)          # S x N x C
    probs = torch.softmax(preds, dim=-1)        # S x N x C
    avg_probs = torch.mean(probs, dim=0)        #     N x C
    confidence = torch.max(avg_probs, dim=-1).values

    return confidence

def plot_conf(data, confidence):

    fig, ax = plt.subplots(
        figsize=(default_width, default_width)
    )
    cnt = ax.contourf(data.x1_grid, data.x2_grid, confidence, levels=200)

    # Remove contour lines
    for c in cnt.collections:
        c.set_edgecolor('face') 

    # Create axis for colorbar
    divider = make_axes_locatable(ax)
    ax_cb = divider.append_axes('right', size='5%', pad=0.2)
    fig.add_axes(ax_cb)

    # Create colorbar and change font
    plt.colorbar(cnt, cax=ax_cb, format='${x:.2f}$')
    for l in ax_cb.yaxis.get_ticklabels():
        l.set_family(plt.rcParams['font.family'])

    # Plot training data points
    X_train = data.X[data.train_indices]
    y_train = data.y[data.train_indices]

    # First class
    ax.scatter(
        X_train[y_train==0, 0], X_train[y_train==0, 1], marker='+', color='red'
    )
    # Second class
    ax.scatter(
        X_train[y_train==1, 0], X_train[y_train==1, 1], marker='+', color='green'
    )

    # Set aspect and labels
    ax.set_aspect('equal')

    return fig, ax

def plot_preds(data, preds):
    # preds: # S x N x C
    
    fig, ax = plt.subplots(
        figsize=(default_width, default_width)
    )
    
    probs = torch.softmax(preds, dim=-1)[:,:,0]

    for i in range(probs.shape[0]):
        cnt_probs = probs[i].detach().cpu().numpy().reshape(
            data.points_per_axis, data.points_per_axis
        )
        cnt = ax.contour(data.x1_grid, data.x2_grid, cnt_probs, levels=[0.5], linewidths=1)

    X_train = data.X[data.train_indices]
    y_train = data.y[data.train_indices]
    ax.scatter(
        X_train[y_train==0, 0], X_train[y_train==0, 1], marker='+', color='red'
    )
    # Second class
    ax.scatter(
        X_train[y_train==1, 0], X_train[y_train==1, 1], marker='+', color='green'
    )
    return fig, ax
 

def main():
    from math import sqrt
    model = ClassificationFNN(
        n_train=100,
        n_in=2,
        n_classes=2,
        hidden_sizes=[50],
        drop_probs=[0.05],
        sigma_b=4.,
        sigma_w=4.,
        sigma_default=2.,
        scale_sigma_w_by_dim=False,
        use_prior=True, 
        device='cuda'
    )
    print('Num params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    data = MoonsDataset(
        batch_size=100,
        n_train=100,
        n_val=100,
        n_test=100,
        noise=0.1,
    )

    hmc_model = HMC(model, subset_of_weights='all', n_posterior_samples=2000)
    hmc_model.fit_hmc(
        data.train_dataloader(),
        warmup_steps=2000,
        num_chains=1,
        max_tree_depth=5,
    )

    prior_conf = get_prior_conf(
        hmc_model,
        torch.from_numpy(data.inputs_flattened).to(model.device, torch.float)
    )
    prior_conf = prior_conf.cpu().numpy().reshape(
        data.points_per_axis, data.points_per_axis
    )
    
    fig, ax = plot_conf(data, prior_conf)
    plt.tight_layout()
    fig.savefig('experiments/moons_v2/prior_conf.pdf')

    prior_preds = pred_w_prior(
        hmc_model,
        torch.from_numpy(data.inputs_flattened).to(model.device, torch.float)
    )
    fig, ax = plot_preds(data, prior_preds)
    plt.tight_layout()
    fig.savefig('experiments/moons_v2/prior_preds.pdf')

    post_conf = get_post_conf(
        hmc_model,
        torch.from_numpy(data.inputs_flattened).to(model.device, torch.float)
    )
    post_conf = post_conf.cpu().numpy().reshape(
        data.points_per_axis, data.points_per_axis
    )
    
    fig, ax = plot_conf(data, post_conf)
    plt.tight_layout()
    fig.savefig('experiments/moons_v2/post_conf.pdf')

    post_preds = pred_w_post(
        hmc_model,
        torch.from_numpy(data.inputs_flattened).to(model.device, torch.float)
    )
    fig, ax = plot_preds(data, post_preds)
    plt.tight_layout()
    fig.savefig('experiments/moons_v2/post_preds.pdf')



if __name__ == '__main__':
    main()