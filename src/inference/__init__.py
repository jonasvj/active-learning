from .inference_base import InferenceBase
from .bayesian_nets import FullyBayesianNet, LastLayerBayesianNet
from .deterministic import Deterministic
from .monte_carlo_dropout import MonteCarloDropout
from .hmc import HMC
from .vi import VI
from .swag import SWAG
from .swag_svi import SWAGSVI
from .laplace_approximation import LaplaceApproximation
from .nf_refined_last_layer_laplace import NFRefinedLastLayerLaplace
from .ensemble import Ensemble
from .laplace_ensemble import LaplaceEnsemble