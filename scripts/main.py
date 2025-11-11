from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mpl_drip  # noqa: F401
import numpy as np
from model import TwoLineMixture, all_phases, neg_ln_posterior
from numpy import pi as π
from spectracles import (
    Matern32,
    OptimiserSchedule,
    Parameter,
    SpatialDataGeneric,
    build_model,
)

jax.config.update("jax_enable_x64", True)
plt.style.use("mpl_drip.custom")
rng = np.random.default_rng(0)

PLOTS_DIR = Path("plots")
PLOTS_DIR.exists()
if not PLOTS_DIR.exists():
    PLOTS_DIR.mkdir()
SAVEFIG_KWARGS = dict(dpi=300, bbox_inches="tight")
SAVE = True

NX = 200
NY = 200
Nλ = 50
λMIN = 4990
λMAX = 5015

PAD_FAC = 0.8

LINE_CENTRE1 = 5000
LINE_CENTRE2 = 5002

# Number of modes in each spatial dimension
# NOTE: These needs to be higher if you make the length scales smaller
# NOTE: They also both must be odd numbers for arcane reasons
N_MODES = (101, 101)

NOISE_STD = 0.05

x_grid = jnp.linspace(-PAD_FAC * π, PAD_FAC * π, NX)
y_grid = jnp.linspace(-PAD_FAC * π, PAD_FAC * π, NY)
λ_grid = jnp.linspace(λMIN, λMAX, Nλ)

x_points, y_points = np.meshgrid(x_grid, y_grid)

spatial_data = SpatialDataGeneric(x=x_points, y=y_points, idx=jnp.arange(NX * NY))

# Hyperparameters
n_spaxels = NX * NY

μ_line1 = Parameter(initial=LINE_CENTRE1, fixed=True)
μ_line2 = Parameter(initial=LINE_CENTRE2, fixed=True)

# Kernel variances
A1_variance = Parameter(initial=1.0, fixed=True)
A2_variance = Parameter(initial=1.0, fixed=True)
v1_variance = Parameter(initial=1000.0, fixed=True)
v2_variance = Parameter(initial=1000.0, fixed=True)

# Kernel lengthscales
A1_lengthscale = Parameter(initial=0.25, fixed=True)
A2_lengthscale = Parameter(initial=0.25, fixed=True)
v1_lengthscale = Parameter(initial=0.5, fixed=True)
v2_lengthscale = Parameter(initial=0.5, fixed=True)

# Kernels
A1_kernel = Matern32(variance=A1_variance, length_scale=A1_lengthscale)
A2_kernel = Matern32(variance=A2_variance, length_scale=A2_lengthscale)
v1_kernel = Matern32(variance=v1_variance, length_scale=v1_lengthscale)
v2_kernel = Matern32(variance=v2_variance, length_scale=v2_lengthscale)

# LSF
σ_lsf1 = Parameter(initial=0.6, fixed=True)
σ_lsf2 = Parameter(initial=0.6, fixed=True)

# Build the true model
my_model = build_model(
    TwoLineMixture,
    line_centre1=μ_line1,
    line_centre2=μ_line2,
    n_modes=N_MODES,
    A_kernel1=A1_kernel,
    A_kernel2=A2_kernel,
    v_kernel1=v1_kernel,
    v_kernel2=v2_kernel,
    σ_lsf1=σ_lsf1,
    σ_lsf2=σ_lsf2,
)

# Gen and set some true coefficients
mean = 3
A1_coeffs_true = rng.standard_normal(N_MODES)
A1_coeffs_true[N_MODES[0] // 2, N_MODES[1] // 2] = (
    np.abs(A1_coeffs_true[N_MODES[0] // 2, N_MODES[1] // 2]) + mean
)  # Ensure positive mean
v1_coeffs_true = rng.standard_normal(N_MODES)
A2_coeffs_true = rng.standard_normal(N_MODES)
A2_coeffs_true[N_MODES[0] // 2, N_MODES[1] // 2] = (
    np.abs(A2_coeffs_true[N_MODES[0] // 2, N_MODES[1] // 2]) + mean
)  # Ensure positive mean
v2_coeffs_true = rng.standard_normal(N_MODES)

# Convert to jax arrays
A1_coeffs_true = jnp.array(A1_coeffs_true)
v1_coeffs_true = jnp.array(v1_coeffs_true)
A2_coeffs_true = jnp.array(A2_coeffs_true)
v2_coeffs_true = jnp.array(v2_coeffs_true)

true_model = my_model.set(
    [
        "line1.A_raw.coefficients",
        "line1.v.coefficients",
        "line2.A_raw.coefficients",
        "line2.v.coefficients",
    ],
    [
        A1_coeffs_true,
        v1_coeffs_true,
        A2_coeffs_true,
        v2_coeffs_true,
    ],
)
true_model = true_model.get_locked_model()
true_model_A1 = true_model.line1.A(spatial_data)
true_model_A2 = true_model.line2.A(spatial_data)
true_model_v1 = true_model.line1.v(spatial_data)
true_model_v2 = true_model.line2.v(spatial_data)

# Get the true line centres
true_μ1 = true_model.line1.μ_obs(spatial_data)
true_μ2 = true_model.line2.μ_obs(spatial_data)

# Plot the true fields
fig, axes = plt.subplots(2, 2, figsize=(8, 8), layout="compressed")

A_kwargs = dict(vmin=0, vmax=3, cmap="viridis")
v_kwargs = dict(vmin=-100, vmax=100, cmap="RdBu")

im1 = axes[0, 0].imshow(true_model_A1.reshape(NY, NX), **A_kwargs)
axes[0, 0].set_title("True A1 field")
fig.colorbar(im1, ax=axes[0, 0])
im2 = axes[0, 1].imshow(true_model_v1.reshape(NY, NX), **v_kwargs)
axes[0, 1].set_title("True v1 field")
fig.colorbar(im2, ax=axes[0, 1])
im3 = axes[1, 0].imshow(true_model_A2.reshape(NY, NX), **A_kwargs)
axes[1, 0].set_title("True A2 field")
fig.colorbar(im3, ax=axes[1, 0])
im4 = axes[1, 1].imshow(true_model_v2.reshape(NY, NX), **v_kwargs)
axes[1, 1].set_title("True v2 field")
fig.colorbar(im4, ax=axes[1, 1])
plt.show()

# Plot the (signed) line centre difference in units of lsf at each spaxel
line_centre_diff = (true_μ2 - true_μ1) / jnp.sqrt(
    true_model.line1.σ2_obs(spatial_data) + true_model.line2.σ2_obs(spatial_data)
)
fig, ax = plt.subplots(figsize=(6, 5), layout="compressed")
im = ax.imshow(line_centre_diff.reshape(NY, NX), cmap="RdBu", vmin=-4, vmax=4)
ax.set_title("Difference of line centres \n(component 2 - component 1)")
fig.colorbar(im, ax=ax, label=r"units of LSF $\sigma$")
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel(r"$x$ sky [arbitrary units]")
ax.set_ylabel(r"$y$ sky [arbitrary units]")
if SAVE:
    plt.savefig(PLOTS_DIR / "true_line_centre_diff.pdf", **SAVEFIG_KWARGS)
plt.show()

# Simulate data
true_model_pred = jax.vmap(true_model, in_axes=(0, None))(λ_grid, spatial_data)
noise = NOISE_STD * rng.standard_normal(true_model_pred.shape)
data_cube = true_model_pred + noise

# Plot the integrated flux in each spaxel
fig, ax = plt.subplots(figsize=(8, 8), layout="compressed")
im = ax.imshow(data_cube.sum(axis=0).reshape(NY, NX), cmap="viridis")
ax.set_title("Simulated data: Integrated flux per spaxel (moment 0)")
fig.colorbar(im, ax=ax)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel(r"$x$ sky [arbitrary units]")
ax.set_ylabel(r"$y$ sky [arbitrary units]")
if SAVE:
    plt.savefig(PLOTS_DIR / "simulated_integrated_flux.pdf", **SAVEFIG_KWARGS)
plt.show()

# Plot the first moment (flux-weighted velocity), first do a unit conversion
eff_line_centre = (LINE_CENTRE1 + LINE_CENTRE2) / 2
velocity_grid = (λ_grid - eff_line_centre) / eff_line_centre * 3e5  # km/s
first_moment = jnp.sum(data_cube * velocity_grid[:, None], axis=0) / jnp.sum(
    data_cube,
    axis=0,
)
fig, ax = plt.subplots(figsize=(8, 8), layout="compressed")
im = ax.imshow(first_moment.reshape(NY, NX), cmap="RdBu", vmin=-100, vmax=100)
ax.set_title("Simulated data: flux-weighted velocity (moment 1)")
fig.colorbar(im, ax=ax, label="km/s")
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel(r"$x$ sky [arbitrary units]")
ax.set_ylabel(r"$y$ sky [arbitrary units]")
if SAVE:
    plt.savefig(PLOTS_DIR / "simulated_first_moment.pdf", **SAVEFIG_KWARGS)
plt.show()

# Plot the integrated spectrum
integrated_spectrum = data_cube.sum(axis=1)
fig, ax = plt.subplots(figsize=(8, 5), layout="compressed")
ax.plot(λ_grid, integrated_spectrum, color="black")
ax.set_xlabel("Wavelength")
ax.set_ylabel("Integrated flux")
ax.set_title("Simulated data: Integrated spectrum")
if SAVE:
    plt.savefig(PLOTS_DIR / "simulated_integrated_spectrum.pdf", **SAVEFIG_KWARGS)
plt.show()


# Reinitialise the model
init_model = my_model.copy()
init_model = init_model.set(
    [
        "line1.A_raw.coefficients",
        "line1.v.coefficients",
        "line2.A_raw.coefficients",
        "line2.v.coefficients",
    ],
    [
        jnp.zeros(N_MODES),
        jnp.zeros(N_MODES),
        jnp.zeros(N_MODES),
        jnp.zeros(N_MODES),
    ],
)


schedule = OptimiserSchedule(
    model=my_model,
    loss_fn=neg_ln_posterior,
    phase_configs=all_phases,
)
schedule.run_all(
    λ=λ_grid,
    xy_data=spatial_data,
    data=data_cube,
    u_data=NOISE_STD * jnp.ones_like(data_cube),
    mask=jnp.ones_like(data_cube, dtype=bool),
)


plt.figure()
plt.plot(schedule.loss_history)
plt.show()

pred_model = schedule.model_history[-1].get_locked_model()

# Plot the inferred fields next to the true fields
pred_model_A1 = pred_model.line1.A(spatial_data)
pred_model_A2 = pred_model.line2.A(spatial_data)
pred_model_v1 = pred_model.line1.v(spatial_data)
pred_model_v2 = pred_model.line2.v(spatial_data)

fig, axes = plt.subplots(4, 2, figsize=(8, 16), layout="compressed")
fs = 14
axes[0, 0].text(
    10,
    15,
    r"\textbf{Flux (component 1)}",
    va="center",
    ha="left",
    rotation=0,
    fontsize=fs,
    c="white",
)
axes[1, 0].text(
    10,
    15,
    r"\textbf{Velocity (component 1)}",
    va="center",
    ha="left",
    rotation=0,
    fontsize=fs,
    c="black",
)
axes[2, 0].text(
    10,
    15,
    r"\textbf{Flux (component 2)}",
    va="center",
    ha="left",
    rotation=0,
    fontsize=fs,
    c="white",
)
axes[3, 0].text(
    10,
    15,
    r"\textbf{Velocity (component 2)}",
    va="center",
    ha="left",
    rotation=0,
    fontsize=fs,
    c="black",
)
axes[0, 0].set_title("True")
axes[0, 1].set_title("Inferred")
im1 = axes[0, 0].imshow(true_model_A1.reshape(NY, NX), **A_kwargs)
im2 = axes[0, 1].imshow(pred_model_A1.reshape(NY, NX), **A_kwargs)
im3 = axes[1, 0].imshow(true_model_v1.reshape(NY, NX), **v_kwargs)
im4 = axes[1, 1].imshow(pred_model_v1.reshape(NY, NX), **v_kwargs)
im5 = axes[2, 0].imshow(true_model_A2.reshape(NY, NX), **A_kwargs)
im6 = axes[2, 1].imshow(pred_model_A2.reshape(NY, NX), **A_kwargs)
im7 = axes[3, 0].imshow(true_model_v2.reshape(NY, NX), **v_kwargs)
im8 = axes[3, 1].imshow(pred_model_v2.reshape(NY, NX), **v_kwargs)

for ax in axes.flatten():
    ax.set_xticks([])
    ax.set_yticks([])
axes[-1, 0].set_xlabel(r"$x$ sky [arbitrary units]")
axes[-1, 1].set_xlabel(r"$x$ sky [arbitrary units]")
axes[-1, 0].set_ylabel(r"$y$ sky [arbitrary units]")

fig.colorbar(im2, ax=axes[0, :], location="right", label="[arbitrary units]")
fig.colorbar(im4, ax=axes[1, :], location="right", label="[km/s]")
fig.colorbar(im6, ax=axes[2, :], location="right", label="[arbitrary units]")
fig.colorbar(im8, ax=axes[3, :], location="right", label="[km/s]")
if SAVE:
    plt.savefig(PLOTS_DIR / "inferred_fields.pdf", **SAVEFIG_KWARGS)
plt.show()


# For some random spectra, plot the data vs the model prediction
n_spectra_to_plot = 5
spaxel_indices = rng.choice(NX * NY, size=n_spectra_to_plot, replace=False)
λ_dense = np.linspace(λ_grid.min(), λ_grid.max(), 200)

pred_data = jax.vmap(pred_model, in_axes=(0, None))(
    λ_dense,
    spatial_data,
)
pred_line1 = jax.vmap(pred_model.line1, in_axes=(0, None))(
    λ_dense,
    spatial_data,
)
pred_line2 = jax.vmap(pred_model.line2, in_axes=(0, None))(
    λ_dense,
    spatial_data,
)

fig, axes = plt.subplots(
    n_spectra_to_plot,
    1,
    figsize=(12, 2 * n_spectra_to_plot),
    layout="compressed",
)
for i, spaxel_idx in enumerate(spaxel_indices):
    ax = axes[i]
    data_spectrum = data_cube[:, spaxel_idx]
    pred_spectrum = pred_data[:, spaxel_idx]
    ax.plot(λ_grid, data_spectrum, color="black", label="Data")
    ax.plot(λ_dense, pred_spectrum, color="red", label="Model prediction")
    ax.plot(
        λ_dense,
        pred_line1[:, spaxel_idx],
        color="C0",
        linestyle="--",
        label="Component 1",
    )
    ax.plot(
        λ_dense,
        pred_line2[:, spaxel_idx],
        color="C1",
        linestyle="--",
        label="Component 2",
    )
    if i != n_spectra_to_plot - 1:
        ax.set_xticklabels([])
    ax.set_title(f"Spaxel index {spaxel_idx}")
    ax.set_ylabel("Flux")
axes[0].legend(bbox_to_anchor=(1.05, 1))
axes[-1].set_xlabel("Wavelength (Angstroms)")
if SAVE:
    plt.savefig(PLOTS_DIR / "example_spectra.pdf", **SAVEFIG_KWARGS)
plt.show()
