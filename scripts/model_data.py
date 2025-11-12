import astropy.constants as const
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from optax import adam
from spectracles import (
    FourierGP,
    Kernel,
    Parameter,
    PhaseConfig,
    SpatialDataLVM,
    SpatialModel,
    SpectralSpatialModel,
    l_bounded,
)

A_LOWER = 1e-5

C_KMS = const.c.to_value("km/s")

rng = np.random.default_rng(0)


def neg_ln_posterior(model, velocities, xy_data, data, u_data, mask):
    # Model predictions
    pred = jax.vmap(model, in_axes=(0, None))(velocities, xy_data)
    # Likelihood
    ln_like = jnp.sum(
        jnp.where(
            mask,
            jax.scipy.stats.norm.logpdf(x=pred, loc=data, scale=u_data),
            0.0,
        )
    )
    ln_prior_line1 = model.line1.peak_raw.prior_logpdf() + model.line1.velocity.prior_logpdf()
    ln_prior_line2 = model.line2.peak_raw.prior_logpdf() + model.line2.velocity.prior_logpdf()
    return -1 * (ln_like + ln_prior_line1 + ln_prior_line2)


class GaussianLine(SpectralSpatialModel):
    # Model components / line quantities
    peak_raw: SpatialModel  # Peak intensity in K
    velocity: SpatialModel  # Radial velocity in rest frame in km/s
    broadening_raw: SpatialModel  # Line broadening in km/s
    # Global parameters
    v_syst: Parameter  # Systemic velocity in km/s
    w_min: Parameter  # Minimum line width in km/s

    def __call__(self, velocities: Array, spatial_data: SpatialDataLVM) -> Array:
        peak = self.peak(spatial_data)
        v_obs = self.velocity_obs(spatial_data)
        w2_obs = self.w2_obs(spatial_data) ** 2
        return peak * jnp.exp(-0.5 * (velocities - v_obs) ** 2 / w2_obs)

    def peak(self, s) -> Array:
        return l_bounded(self.peak_raw(s), lower=0.0)

    def velocity_obs(self, s) -> Array:
        return self.velocity(s) + self.v_syst.val

    def width(self, s) -> Array:
        return l_bounded(self.broadening_raw(s), lower=0.0)

    def w2_obs(self, s) -> Array:
        return self.width(s) + self.w_min.val


# TODO: Generalise to K lines
# class KLineMixture(SpectralSpatialModel):
#     # Model components
#     lines: dict[str, GaussianLine]  # line models

#     def __init__(
#         self,
#         K: int,
#         n_modes: tuple[int, int],
#         peak_kernels: list[Kernel],
#         velocity_kernels: list[Kernel],
#         broadening_kernels: list[Kernel],
#         v_systs: list[Parameter],
#         w_min: Parameter,
#     ):
#         self.lines = {}
#         for k in range(K):
#             self.lines[f"line{k + 1}"] = GaussianLine(
#                 peak_raw=FourierGP(n_modes=n_modes, kernel=peak_kernels[k]),
#                 velocity=FourierGP(n_modes=n_modes, kernel=velocity_kernels[k]),
#                 broadening_raw=FourierGP(n_modes=n_modes, kernel=broadening_kernels[k]),
#                 v_syst=v_systs[k],
#                 w_min=w_min,
#             )

#     def __call__(self, velocities, spatial_data):
#         return sum(
#             jax.vmap(line, in_axes=(0, None))(velocities, spatial_data)
#             for line in self.lines.values()
#         )


class TwoLineMixture(SpectralSpatialModel):
    # Model components
    line1: GaussianLine  # line models
    line2: GaussianLine  # line models

    def __init__(
        self,
        n_modes: tuple[int, int],
        peak_kernels: list[Kernel],
        velocity_kernels: list[Kernel],
        broadening_kernels: list[Kernel],
        v_systs: list[Parameter],
        w_min: Parameter,
    ):
        self.line1 = GaussianLine(
            peak_raw=FourierGP(n_modes=n_modes, kernel=peak_kernels[0]),
            velocity=FourierGP(n_modes=n_modes, kernel=velocity_kernels[0]),
            broadening_raw=FourierGP(n_modes=n_modes, kernel=broadening_kernels[0]),
            v_syst=v_systs[0],
            w_min=w_min,
        )
        self.line2 = GaussianLine(
            peak_raw=FourierGP(n_modes=n_modes, kernel=peak_kernels[1]),
            velocity=FourierGP(n_modes=n_modes, kernel=velocity_kernels[1]),
            broadening_raw=FourierGP(n_modes=n_modes, kernel=broadening_kernels[1]),
            v_syst=v_systs[1],
            w_min=w_min,
        )

    def __call__(self, velocities, spatial_data):
        line1 = self.line1(velocities, spatial_data)
        line2 = self.line2(velocities, spatial_data)
        return line1 + line2


Δloss = 1e-2

N_STEPS = 2000


def get_phases(n_modes: tuple[int, int]) -> list[PhaseConfig]:
    A_coeffs_init = PhaseConfig(
        n_steps=N_STEPS,
        optimiser=adam(1e-2),
        Δloss_criterion=Δloss,
        fix_status_updates={
            # Allowed to vary:
            "line1.peak_raw.coefficients": False,
            "line2.peak_raw.coefficients": False,
            # Fixed:
            "line1.w_min": True,
            "line2.w_min": True,
            "line1.velocity.coefficients": True,
            "line2.velocity.coefficients": True,
            "line1.broadening_raw.coefficients": True,
            "line2.broadening_raw.coefficients": True,
        },
        param_val_updates={
            "line1.peak_raw.coefficients": jnp.array(rng.standard_normal(n_modes)),
            "line2.peak_raw.coefficients": jnp.array(rng.standard_normal(n_modes)),
        },
    )
    v_coeffs_init = PhaseConfig(
        n_steps=N_STEPS,
        optimiser=adam(1e-2),
        Δloss_criterion=Δloss,
        fix_status_updates={
            # Allowed to vary:
            "line1.velocity.coefficients": False,
            "line2.velocity.coefficients": False,
            # Fixed:
            "line1.w_min": True,
            "line2.w_min": True,
            "line1.peak_raw.coefficients": True,
            "line2.peak_raw.coefficients": True,
            "line1.broadening_raw.coefficients": True,
            "line2.broadening_raw.coefficients": True,
        },
        param_val_updates={
            "line1.velocity.coefficients": jnp.array(rng.standard_normal(n_modes)),
            "line2.velocity.coefficients": jnp.array(rng.standard_normal(n_modes)),
        },
    )
    w_coeffs_init = PhaseConfig(
        n_steps=N_STEPS,
        optimiser=adam(1e-2),
        Δloss_criterion=Δloss,
        fix_status_updates={
            # Allowed to vary:
            "line1.broadening_raw.coefficients": False,
            "line2.broadening_raw.coefficients": False,
            # Fixed:
            "line1.w_min": True,
            "line2.w_min": True,
            "line1.peak_raw.coefficients": True,
            "line2.peak_raw.coefficients": True,
            "line1.velocity.coefficients": True,
            "line2.velocity.coefficients": True,
        },
        param_val_updates={
            "line1.broadening_raw.coefficients": jnp.array(rng.standard_normal(n_modes)),
            "line2.broadening_raw.coefficients": jnp.array(rng.standard_normal(n_modes)),
        },
    )
    both_coeffs = PhaseConfig(
        n_steps=N_STEPS,
        optimiser=adam(1e-2),
        Δloss_criterion=Δloss,
        fix_status_updates={
            # Allowed to vary:
            "line1.peak_raw.coefficients": False,
            "line2.peak_raw.coefficients": False,
            "line1.velocity.coefficients": False,
            "line2.velocity.coefficients": False,
            # Fixed:
            "line1.w_min": True,
            "line2.w_min": True,
            "line1.broadening_raw.coefficients": True,
            "line2.broadening_raw.coefficients": True,
        },
    )
    all_coeffs = PhaseConfig(
        n_steps=N_STEPS,
        optimiser=adam(1e-2),
        Δloss_criterion=Δloss,
        fix_status_updates={
            # Allowed to vary:
            "line1.peak_raw.coefficients": False,
            "line2.peak_raw.coefficients": False,
            "line1.velocity.coefficients": False,
            "line2.velocity.coefficients": False,
            "line1.broadening_raw.coefficients": False,
            "line2.broadening_raw.coefficients": False,
            # Fixed:
            "line1.w_min": True,
            "line2.w_min": True,
        },
    )
    A_coeffs = PhaseConfig(
        n_steps=N_STEPS,
        optimiser=adam(1e-3),
        Δloss_criterion=Δloss,
        fix_status_updates={
            # Allowed to vary:
            "line1.peak_raw.coefficients": False,
            "line2.peak_raw.coefficients": False,
            # Fixed:
            "line1.w_min": True,
            "line2.w_min": True,
            "line1.velocity.coefficients": True,
            "line2.velocity.coefficients": True,
            "line1.broadening_raw.coefficients": True,
            "line2.broadening_raw.coefficients": True,
        },
    )
    v_coeffs = PhaseConfig(
        n_steps=N_STEPS,
        optimiser=adam(1e-3),
        Δloss_criterion=Δloss,
        fix_status_updates={
            # Allowed to vary:
            "line1.velocity.coefficients": False,
            "line2.velocity.coefficients": False,
            # Fixed:
            "line1.w_min": True,
            "line2.w_min": True,
            "line1.peak_raw.coefficients": True,
            "line2.peak_raw.coefficients": True,
            "line1.broadening_raw.coefficients": True,
            "line2.broadening_raw.coefficients": True,
        },
    )
    w_coeffs = PhaseConfig(
        n_steps=N_STEPS,
        optimiser=adam(1e-3),
        Δloss_criterion=Δloss,
        fix_status_updates={
            # Allowed to vary:
            "line1.broadening_raw.coefficients": False,
            "line2.broadening_raw.coefficients": False,
            # Fixed:
            "line1.w_min": True,
            "line2.w_min": True,
            "line1.peak_raw.coefficients": True,
            "line2.peak_raw.coefficients": True,
            "line1.velocity.coefficients": True,
            "line2.velocity.coefficients": True,
        },
    )
    # adapt_w_min = PhaseConfig(
    #     n_steps=N_STEPS,
    #     optimiser=adam(1e-3),
    #     Δloss_criterion=Δloss,
    #     fix_status_updates={
    #         # Allowed to vary:
    #         "line1.w_min": False,
    #         "line2.w_min": False,
    #         # Fixed:
    #         "line1.peak_raw.coefficients": True,
    #         "line2.peak_raw.coefficients": True,
    #         "line1.velocity.coefficients": True,
    #         "line2.velocity.coefficients": True,
    #         "line1.broadening_raw.coefficients": True,
    #         "line2.broadening_raw.coefficients": True,
    #     },
    # )
    return [
        A_coeffs_init,
        v_coeffs_init,
        w_coeffs_init,
        both_coeffs,
        all_coeffs,
        A_coeffs,
        v_coeffs,
        w_coeffs,
    ]
