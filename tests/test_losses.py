import math

import numpy as np

from freenerf_ns.losses import depth_smoothness_loss, occlusion_loss


def render_weights(sigma, delta):
    """Eq. (2): w_i = T_i (1 - exp(-sigma_i delta_i)), T_i = exp(-sum_{j<i} sigma_j delta_j)."""
    a = sigma * delta
    T = np.exp(-(np.cumsum(a, axis=-1) - a))
    return T * (1 - np.exp(-a))


def test_occlusion_loss_matches_definition():
    d = np.zeros((4, 100)); d[:, :5] = 2.0
    assert math.isclose(float(occlusion_loss(d, 20)), 5 * 2.0 / 100)       # (1/K) sum_{k<=M} sigma_k
    assert float(occlusion_loss(d, 3)) == 3 * 2.0 / 100                    # window M smaller than the mass
    assert float(occlusion_loss(np.zeros((3, 50)), 20)) == 0.0


def test_remark1_billboard_cost_bound():
    """Remark 1: a plane inside the first M samples that lets through fraction eps costs >= ln(1/eps)/(K*delta_max)."""
    K, M, near, far = 192, 20, 2.0, 6.0
    delta = np.full(K, (far - near) / K)
    for eps in (0.5, 0.1, 0.01):
        sigma = np.zeros((1, K))
        sigma[0, 5] = -math.log(eps) / delta[5]                 # transmittance behind the plane is exactly eps
        T_after = np.exp(-(sigma * delta).sum())
        assert math.isclose(T_after, eps, rel_tol=1e-9)
        loss = float(occlusion_loss(sigma, M))
        bound = math.log(1 / eps) / (K * delta.max())
        assert loss >= bound - 1e-12


def test_billboard_field_identities_C9():
    """C9: a billboard (opaque plane in front of the camera) gives Omega_M ~ 1, an empty field gives 0."""
    K, M = 192, 20
    delta = np.full((8, K), 4.0 / K)
    sigma = np.zeros((8, K)); sigma[:, 3] = 1e4                  # opaque plane at sample 3
    w = render_weights(sigma, delta)
    assert abs(w[:, :M].sum(axis=1).mean() - 1.0) < 1e-6
    w0 = render_weights(np.zeros((8, K)), delta)
    assert w0.sum() == 0.0


def test_depth_smoothness_zero_on_fronto_parallel_plane():
    d = np.full((3, 8, 8), 4.2)                                  # a billboard has constant depth
    assert float(depth_smoothness_loss(d)) == 0.0


def test_depth_smoothness_positive_and_counts_both_directions():
    d = np.zeros((1, 3, 3)); d[0, 0, 1] = 1.0                   # one bump at row 0, column 1
    v = float(depth_smoothness_loss(d))
    # Eq. (9) over i,j in {0,1}: (0,0) right diff 1; (0,1) right diff 1 and down diff 1  ->  1 + 1 + 1
    assert math.isclose(v, 3.0)
    ramp = np.tile(np.arange(5.0), (1, 5, 1))                    # slope along x only
    assert float(depth_smoothness_loss(ramp)) > 0


def test_depth_smoothness_alpha_weights_remove_background_edges():
    d = np.zeros((1, 4, 4)); d[0, :, 2:] = 4.0                  # sharp depth step (silhouette against background)
    acc_bg = np.zeros((1, 4, 4)); acc_bg[0, :, 2:] = 1.0        # left half is empty background
    plain = float(depth_smoothness_loss(d))
    weighted = float(depth_smoothness_loss(d, acc_bg))
    assert plain > 0 and weighted == 0.0                          # step between empty and object is not penalised
    acc_full = np.ones((1, 4, 4))
    assert math.isclose(float(depth_smoothness_loss(d, acc_full)), plain)   # fully opaque -> literal paper form
