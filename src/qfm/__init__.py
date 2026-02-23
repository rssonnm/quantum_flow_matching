from .ansatz import EHA_Circuit
from .channels import *
from .qfim import *
from .expressibility import *
from .lindblad import *
from .metrics import *
from .trainer import QFMTrainer
from .utils import *
from .dynamical_systems import *
from .optimizers import *
from .noise_models import *
from .adaptive_grid import *
from .convergence_bounds import (
    lipschitz_constant_qfm, lipschitz_trajectory,
    discretization_error_bound, discretization_error_vs_steps,
    fit_convergence_rate, action_optimality_ratio,
    expressivity_lower_bound, fidelity_convergence_curve,
    bures_convergence_curve,
)
from .quantum_ot import (
    bures_geodesic_interpolation, quantum_w2_distance, quantum_w2_squared,
    trajectory_vs_geodesic_analysis, transport_efficiency,
    discrete_benamou_brenier_energy, total_benamou_brenier_cost,
    quantum_speed_limit_bound, qsl_efficiency, mean_geodesic_deviation,
)
from .error_mitigation import (
    gate_decomposition_cost, effective_noise_rate,
    apply_depolarizing_noise, apply_dephasing_noise,
    zero_noise_extrapolation, zne_fidelity_mitigation,
    noisy_trajectory, mitigated_fidelity_trajectory,
)
from .entanglement_scaling import (
    partial_trace, entanglement_entropy_bipartite, mutual_information_bipartite,
    concurrence, formation_entropy, ghz_density_matrix, w_density_matrix,
    cluster_state_density_matrix, ghz_fidelity, w_fidelity, cluster_fidelity,
    entanglement_generation_rate, scaling_analysis,
)
from .classical_baseline import (
    rho_to_vector, vector_to_rho, ClassicalVectorField,
    ClassicalFlowMatchingBaseline, compare_qfm_vs_cfm,
)
from .flow_geometry import (
    quantum_vector_field, hilbert_schmidt_norm, vector_field_magnitude,
    generator_from_consecutive, generator_spectrum_trajectory,
    sectional_curvature_bures, curvature_along_trajectory,
    geodesic_curvature, parallel_transport_deviation, flow_divergence,
    geometric_phase_estimate, full_geometry_report,
)
