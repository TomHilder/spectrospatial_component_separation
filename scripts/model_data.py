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
from spectracles_extension import dict_to_module

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

    # Prior contributions from all lines
    ln_prior = 0.0
    for k in range(model.K):
        line = getattr(model.lines, f"line{k + 1}")
        ln_prior += (
            line.peak_raw.prior_logpdf()
            + line.velocity.prior_logpdf()
            + line.broadening_raw.prior_logpdf()
            + line.skew_raw.prior_logpdf()
            + line.kurtosis_raw.prior_logpdf()
        )

    return -1 * (ln_like + ln_prior)


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


def hermite3(x: Array) -> Array:
    return x**3 - 3.0 * x


def hermite4(x: Array) -> Array:
    return x**4 - 6.0 * x**2 + 3.0


class GaussHermiteLine(SpectralSpatialModel):
    # Model components / line quantities
    peak_raw: SpatialModel  # Peak intensity in K
    velocity: SpatialModel  # Radial velocity in rest frame in km/s
    broadening_raw: SpatialModel  # Line broadening in km/s

    skew_raw: SpatialModel  # Skew field (unitless I think)
    kurtosis_raw: SpatialModel  # Kurtosis field (unitless I think)

    # Global parameters
    v_syst: Parameter  # Systemic velocity in km/s
    w_min: Parameter  # Minimum line width in km/s

    h3_max: Parameter  # max h3
    h4_max: Parameter  # max h4

    def __call__(self, velocities: Array, spatial_data: SpatialDataLVM) -> Array:
        peak = self.peak(spatial_data)  # same shape as before
        v_obs = self.velocity_obs(spatial_data)
        w2_obs = self.w2_obs(spatial_data) ** 2  # same semantics as your GaussianLine

        # Dimensionless coordinate: use sigma = sqrt(w2_obs)
        x = (velocities - v_obs) / jnp.sqrt(w2_obs)

        # Effective GH coefficients from transformed GPs
        h3 = self.h3(spatial_data)
        h4 = self.h4(spatial_data)

        # Gaussian line profile
        gaussian = jnp.exp(-0.5 * (velocities - v_obs) ** 2 / w2_obs)

        # Hermite polynomials
        H3 = hermite3(x)
        H4 = hermite4(x)

        # Gauss–Hermite line profile
        return peak * gaussian * (1.0 + h3 * H3 + h4 * H4)

    def peak(self, s) -> Array:
        return l_bounded(self.peak_raw(s), lower=0.0)

    def velocity_obs(self, s) -> Array:
        return self.velocity(s) + self.v_syst.val

    def width(self, s) -> Array:
        return l_bounded(self.broadening_raw(s), lower=0.0)

    def w2_obs(self, s) -> Array:
        return self.width(s) + self.w_min.val

    def h3(self, s) -> Array:
        z3 = self.skew_raw(s)
        return self.h3_max.val * jnp.tanh(z3)

    def h4(self, s) -> Array:
        z4 = self.kurtosis_raw(s)
        return self.h4_max.val * jnp.tanh(z4)


class KLineMixture(SpectralSpatialModel):
    # Model components
    lines: dict[str, GaussHermiteLine]  # line models
    # Number of lines
    K: int

    def __init__(
        self,
        K: int,
        n_modes: tuple[int, int],
        peak_kernels: list[Kernel],
        velocity_kernels: list[Kernel],
        broadening_kernels: list[Kernel],
        skew_kernels: list[Kernel],
        kurt_kernels: list[Kernel],
        v_systs: list[Parameter],
        w_min: Parameter,
        h3_max: Parameter,
        h4_max: Parameter,
    ):
        self.K = K
        lines = {}
        for k in range(K):
            lines[f"line{k + 1}"] = GaussHermiteLine(
                peak_raw=FourierGP(n_modes=n_modes, kernel=peak_kernels[k]),
                velocity=FourierGP(n_modes=n_modes, kernel=velocity_kernels[k]),
                broadening_raw=FourierGP(n_modes=n_modes, kernel=broadening_kernels[k]),
                skew_raw=FourierGP(n_modes=n_modes, kernel=skew_kernels[k]),
                kurtosis_raw=FourierGP(n_modes=n_modes, kernel=kurt_kernels[k]),
                v_syst=v_systs[k],
                w_min=w_min,
                h3_max=h3_max,
                h4_max=h4_max,
            )
        # print(lines)
        self.lines = dict_to_module(lines, module_name="LineCollection")
        # print("==============================")
        # print(self.lines)

    def __call__(self, velocities, spatial_data):
        # Explicitly collect lines in order: line1, line2, ..., lineK
        lines_list = [getattr(self.lines, f"line{k + 1}") for k in range(self.K)]

        # Apply each line and sum
        result = jnp.zeros_like(velocities)
        for line in lines_list:
            result += line(velocities, spatial_data)

        return result


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
        fix_skew=True,
        fix_kurtosis=True,
        fix_w_min=True,
        init_peak=False,
        init_velocity=False,
        init_broadening=False,
        init_skew=False,
        init_kurtosis=False,
    ) -> PhaseConfig:
        init_dicts = []
        if init_peak:
            init_dicts.append(
                {
                    f"lines.line{k + 1}.peak_raw.coefficients": jnp.array(
                        rng.standard_normal(n_modes),
                    )
                    for k in range(n_components)
                }
            )
        if init_velocity:
            init_dicts.append(
                {
                    f"lines.line{k + 1}.velocity.coefficients": jnp.array(
                        rng.standard_normal(n_modes),
                    )
                    for k in range(n_components)
                }
            )
        if init_broadening:
            init_dicts.append(
                {
                    f"lines.line{k + 1}.broadening_raw.coefficients": jnp.array(
                        rng.standard_normal(n_modes),
                    )
                    for k in range(n_components)
                }
            )
        if init_skew:
            init_dicts.append(
                {
                    f"lines.line{k + 1}.skew_raw.coefficients": jnp.array(
                        rng.standard_normal(n_modes),
                    )
                    for k in range(n_components)
                }
            )
        if init_kurtosis:
            init_dicts.append(
                {
                    f"lines.line{k + 1}.kurtosis_raw.coefficients": jnp.array(
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
                    {
                        f"lines.line{k + 1}.peak_raw.coefficients": fix_peak
                        for k in range(n_components)
                    },
                    {
                        f"lines.line{k + 1}.velocity.coefficients": fix_velocity
                        for k in range(n_components)
                    },
                    {
                        f"lines.line{k + 1}.broadening_raw.coefficients": fix_broadening
                        for k in range(n_components)
                    },
                    {
                        f"lines.line{k + 1}.skew_raw.coefficients": fix_skew
                        for k in range(n_components)
                    },
                    {
                        f"lines.line{k + 1}.kurtosis_raw.coefficients": fix_kurtosis
                        for k in range(n_components)
                    },
                    {f"lines.line{k + 1}.w_min": fix_w_min for k in range(n_components)},
                ]
            ),
            param_val_updates=merge_dicts(init_dicts),
        )

    peak_coeffs_init = get_phase(fix_peak=False, init_peak=True)
    velocity_coeffs_init = get_phase(fix_velocity=False, init_velocity=True)
    broadening_coeffs_init = get_phase(
        fix_broadening=False,
        init_broadening=True,
        fix_skew=False,
        init_skew=True,
        fix_kurtosis=False,
        init_kurtosis=True,
    )
    both_coeffs = get_phase(fix_peak=False, fix_velocity=False)
    all_coeffs = get_phase(
        fix_peak=False,
        fix_velocity=False,
        fix_broadening=False,
        fix_skew=False,
        fix_kurtosis=False,
    )
    peak_coeffs = get_phase(lr=lr_2, fix_peak=False)
    velocity_coeffs = get_phase(lr=lr_2, fix_velocity=False)
    broadening_coeffs = get_phase(
        lr=lr_2,
        fix_broadening=False,
        fix_skew=False,
        fix_kurtosis=False,
    )

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
