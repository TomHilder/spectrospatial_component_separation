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
    ln_prior_line1 = (
        model.line1.peak_raw.prior_logpdf()
        + model.line1.velocity.prior_logpdf()
        + model.line1.broadening_raw.prior_logpdf()
    )
    ln_prior_line2 = (
        model.line2.peak_raw.prior_logpdf()
        + model.line2.velocity.prior_logpdf()
        + model.line2.broadening_raw.prior_logpdf()
    )
    ln_prior_line3 = (
        model.line3.peak_raw.prior_logpdf()
        + model.line3.velocity.prior_logpdf()
        + model.line3.broadening_raw.prior_logpdf()
    )
    ln_prior_line4 = (
        model.line4.peak_raw.prior_logpdf()
        + model.line4.velocity.prior_logpdf()
        + model.line4.broadening_raw.prior_logpdf()
    )
    ln_prior_line5 = (
        model.line5.peak_raw.prior_logpdf()
        + model.line5.velocity.prior_logpdf()
        + model.line5.broadening_raw.prior_logpdf()
    )
    return -1 * (
        ln_like + ln_prior_line1 + ln_prior_line2 + ln_prior_line3 + ln_prior_line4 + ln_prior_line5
    )


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


class ThreeLineMixture(SpectralSpatialModel):
    # Model components
    line1: GaussianLine  # line models
    line2: GaussianLine  # line models
    line3: GaussianLine  # line models

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
        self.line3 = GaussianLine(
            peak_raw=FourierGP(n_modes=n_modes, kernel=peak_kernels[2]),
            velocity=FourierGP(n_modes=n_modes, kernel=velocity_kernels[2]),
            broadening_raw=FourierGP(n_modes=n_modes, kernel=broadening_kernels[2]),
            v_syst=v_systs[2],
            w_min=w_min,
        )

    def __call__(self, velocities, spatial_data):
        line1 = self.line1(velocities, spatial_data)
        line2 = self.line2(velocities, spatial_data)
        line3 = self.line3(velocities, spatial_data)
        return line1 + line2 + line3


class FiveLineMixture(SpectralSpatialModel):
    # Model components
    line1: GaussianLine  # line models
    line2: GaussianLine  # line models
    line3: GaussianLine  # line models
    line4: GaussianLine  # line models
    line5: GaussianLine  # line models

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
        self.line3 = GaussianLine(
            peak_raw=FourierGP(n_modes=n_modes, kernel=peak_kernels[2]),
            velocity=FourierGP(n_modes=n_modes, kernel=velocity_kernels[2]),
            broadening_raw=FourierGP(n_modes=n_modes, kernel=broadening_kernels[2]),
            v_syst=v_systs[2],
            w_min=w_min,
        )
        self.line4 = GaussianLine(
            peak_raw=FourierGP(n_modes=n_modes, kernel=peak_kernels[3]),
            velocity=FourierGP(n_modes=n_modes, kernel=velocity_kernels[3]),
            broadening_raw=FourierGP(n_modes=n_modes, kernel=broadening_kernels[3]),
            v_syst=v_systs[3],
            w_min=w_min,
        )
        self.line5 = GaussianLine(
            peak_raw=FourierGP(n_modes=n_modes, kernel=peak_kernels[4]),
            velocity=FourierGP(n_modes=n_modes, kernel=velocity_kernels[4]),
            broadening_raw=FourierGP(n_modes=n_modes, kernel=broadening_kernels[4]),
            v_syst=v_systs[4],
            w_min=w_min,
        )

    def __call__(self, velocities, spatial_data):
        line1 = self.line1(velocities, spatial_data)
        line2 = self.line2(velocities, spatial_data)
        line3 = self.line3(velocities, spatial_data)
        line4 = self.line4(velocities, spatial_data)
        line5 = self.line5(velocities, spatial_data)
        return line1 + line2 + line3 + line4 + line5


class KLineMixture(SpectralSpatialModel):
    # Model components
    lines: dict[str, GaussianLine]  # line models

    def __init__(
        self,
        K: int,
        n_modes: tuple[int, int],
        peak_kernels: list[Kernel],
        velocity_kernels: list[Kernel],
        broadening_kernels: list[Kernel],
        v_systs: list[Parameter],
        w_min: Parameter,
    ):
        self.lines = {}
        for k in range(K):
            self.lines[f"line{k + 1}"] = GaussianLine(
                peak_raw=FourierGP(n_modes=n_modes, kernel=peak_kernels[k]),
                velocity=FourierGP(n_modes=n_modes, kernel=velocity_kernels[k]),
                broadening_raw=FourierGP(n_modes=n_modes, kernel=broadening_kernels[k]),
                v_syst=v_systs[k],
                w_min=w_min,
            )

    def __call__(self, velocities, spatial_data):
        # Convert dict to list of modules for vmapping
        lines_list = list(self.lines.values())

        # Apply each line to all velocities, then sum
        def apply_line(line):
            return jax.vmap(line, in_axes=(0, None))(velocities, spatial_data)

        # vmap over the lines and sum results
        results = jax.vmap(apply_line)(lines_list)
        return jnp.sum(results, axis=0)


Δloss = 1e-2

N_STEPS = 200


def merge_dicts(dict_list):
    out = {}
    for d in dict_list:
        out.update(d)
    return out


def get_phases(n_modes: tuple[int, int], n_components: int) -> list[PhaseConfig]:
    lr_1 = 1e-1
    lr_2 = 1e-2

    def get_phase(
        n_steps=N_STEPS,
        lr=lr_1,
        fix_peak=True,
        fix_velocity=True,
        fix_broadening=True,
        fix_w_min=True,
        init_peak=False,
        init_velocity=False,
        init_broadening=False,
    ) -> PhaseConfig:
        init_dicts = []
        if init_peak:
            init_dicts.append(
                {
                    f"line{k + 1}.peak_raw.coefficients": jnp.array(
                        rng.standard_normal(n_modes),
                    )
                    for k in range(n_components)
                }
            )
        if init_velocity:
            init_dicts.append(
                {
                    f"line{k + 1}.velocity.coefficients": jnp.array(
                        rng.standard_normal(n_modes),
                    )
                    for k in range(n_components)
                }
            )
        if init_broadening:
            init_dicts.append(
                {
                    f"line{k + 1}.broadening_raw.coefficients": jnp.array(
                        rng.standard_normal(n_modes),
                    )
                    for k in range(n_components)
                }
            )
        return PhaseConfig(
            n_steps=n_steps,
            optimiser=adam(lr),
            Δloss_criterion=Δloss,
            fix_status_updates=merge_dicts(
                [
                    {f"line{k + 1}.peak_raw.coefficients": fix_peak for k in range(n_components)},
                    {
                        f"line{k + 1}.velocity.coefficients": fix_velocity
                        for k in range(n_components)
                    },
                    {
                        f"line{k + 1}.broadening_raw.coefficients": fix_broadening
                        for k in range(n_components)
                    },
                    {f"line{k + 1}.w_min": fix_w_min for k in range(n_components)},
                ]
            ),
            param_val_updates=merge_dicts(init_dicts),
        )

    peak_coeffs_init = get_phase(fix_peak=False, init_peak=True)
    velocity_coeffs_init = get_phase(fix_velocity=False, init_velocity=True)
    broadening_coeffs_init = get_phase(fix_broadening=False, init_broadening=True)
    both_coeffs = get_phase(fix_peak=False, fix_velocity=False)
    all_coeffs = get_phase(fix_peak=False, fix_velocity=False, fix_broadening=False)
    peak_coeffs = get_phase(lr=lr_2, fix_peak=False)
    velocity_coeffs = get_phase(lr=lr_2, fix_velocity=False)
    broadening_coeffs = get_phase(lr=lr_2, fix_broadening=False)

    return [
        peak_coeffs_init,
        velocity_coeffs_init,
        broadening_coeffs_init,
        both_coeffs,
        all_coeffs,
        peak_coeffs,
        velocity_coeffs,
        broadening_coeffs,
    ]
