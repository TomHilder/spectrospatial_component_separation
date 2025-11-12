from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mpl_drip  # noqa: F401
import numpy as np
from model_data import TwoLineMixture, get_phases, neg_ln_posterior
from mpl_drip import colormaps  # noqa: F401
from numpy import pi as π
from spectracles import (
    Matern32,
    Matern52,
    OptimiserSchedule,
    Parameter,
    SpatialDataGeneric,
    build_model,
)

plt.style.use("mpl_drip.custom")
rng = np.random.default_rng(0)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_FNAME = "ic1613_C+D+tp_hi21cm_0p8kms_30as.fits"
TRUNC_DATA_FNAME = "ic1613_hi21cm_truncated_data.npz"
DATA_PATH = DATA_DIR / DATA_FNAME
TRUNC_DATA_PATH = DATA_DIR / TRUNC_DATA_FNAME
assert DATA_PATH.exists(), f"Data file not found: {DATA_PATH}"
assert TRUNC_DATA_PATH.exists(), f"Truncated data file not found: {TRUNC_DATA_PATH}"


PLOTS_DIR = Path("plots_ic1613_two_line_mixture")
PLOTS_DIR.exists()
if not PLOTS_DIR.exists():
    PLOTS_DIR.mkdir()
SAVEFIG_KWARGS = dict(dpi=300, bbox_inches="tight")
SAVE = True


# Read in the saved truncated data
file = np.load(TRUNC_DATA_PATH)
intensities = file["intensities"]
velocities = file["velocities"]
rms = file["rms"]

data = jnp.array(intensities)
u_data = jnp.array(rms) * jnp.ones_like(data)
vels = jnp.array(velocities)

# Normalise the data by the peak intensity
peak_intensity = jnp.nanmax(data)
data /= peak_intensity
u_data /= peak_intensity

# Initial guess for v_syst by finding the max intensity channel
peak_velocity_idx = jnp.nanargmax(data, axis=0)
init_v_syst = vels[peak_velocity_idx].mean()

# Initial guess for w_in by estimating the second moment of the average spectrum
mean_spectrum = jnp.nanmean(data, axis=(1, 2))
spectrum_mean = jnp.sum(vels * mean_spectrum) / jnp.sum(mean_spectrum)
spectrum_var = jnp.sum(((vels - spectrum_mean) ** 2) * mean_spectrum) / jnp.sum(mean_spectrum)
# init_w_min = jnp.sqrt(spectrum_var) / 3  # There are two components
init_w_min = 2
print(f"Initial w_min guess: {init_w_min:.2f} km/s")

# Assemble a pixel grid
PAD_FAC = 0.8
nλ, ny, nx = data.shape
x_grid = jnp.linspace(-PAD_FAC * π, PAD_FAC * π, nx)
y_grid = jnp.linspace(-PAD_FAC * π, PAD_FAC * π, ny)
x_points, y_points = np.meshgrid(x_grid, y_grid)
spatial_data = SpatialDataGeneric(x=x_points, y=y_points, idx=jnp.arange(ny * ny))

# Modes
kernel_peak = Matern32
kernel_velocity = Matern32
kernel_broadening = Matern52
n_modes = (201, 201)
# Need to account for if I use different larger nx, ny which is MORE image, and MORE sky area
# But I want the same physical scale, so length scale in pixels should decrease as nx increases
ls_kwargs_pv = dict(initial=π / 7 / (nx / 100), fixed=True)
ls_kwargs_w = dict(initial=π / 5 / (nx / 100), fixed=True)
var_kwargs_pv = dict(initial=1.0, fixed=True)
var_kwargs_w = dict(initial=1.0, fixed=True)
peak_kernels = [
    kernel_peak(length_scale=Parameter(**ls_kwargs_pv), variance=Parameter(**var_kwargs_pv))
    for _ in range(2)
]
velocity_kernels = [
    kernel_velocity(length_scale=Parameter(**ls_kwargs_pv), variance=Parameter(**var_kwargs_pv))
    for _ in range(2)
]
broadening_kernels = [
    kernel_broadening(length_scale=Parameter(**ls_kwargs_w), variance=Parameter(**var_kwargs_w))
    for _ in range(2)
]
v_systs = [
    Parameter(initial=init_v_syst - 1, fixed=False),
    Parameter(initial=init_v_syst + 1, fixed=False),
]
w_min = Parameter(initial=init_w_min, fixed=True)

# Build the model
my_model = build_model(
    TwoLineMixture,
    n_modes=n_modes,
    peak_kernels=peak_kernels,
    velocity_kernels=velocity_kernels,
    broadening_kernels=broadening_kernels,
    v_systs=v_systs,
    w_min=w_min,
)
phases = get_phases(n_modes)
init_model = my_model.get_locked_model()

schedule = OptimiserSchedule(model=my_model, loss_fn=neg_ln_posterior, phase_configs=phases)
data_shape = (nλ, ny * nx)
schedule.run_all(
    velocities=vels,
    xy_data=spatial_data,
    data=data.reshape(data_shape),
    u_data=u_data.reshape(data_shape),
    mask=jnp.ones(data_shape, dtype=bool),  # Nothing is masked here
)

plt.figure()
plt.plot(schedule.loss_history)
plt.show()


pred_model = schedule.model_history[-1].get_locked_model()

# Plot the inferred fields next to the true fields
pred_model_A1 = pred_model.line1.peak(spatial_data) * peak_intensity
pred_model_A2 = pred_model.line2.peak(spatial_data) * peak_intensity
pred_model_v1 = pred_model.line1.velocity_obs(spatial_data) - init_v_syst
pred_model_v2 = pred_model.line2.velocity_obs(spatial_data) - init_v_syst
pred_model_σ1 = pred_model.line1.width(spatial_data) + init_w_min
pred_model_σ2 = pred_model.line2.width(spatial_data) + init_w_min

A_max = max(pred_model_A1.max(), pred_model_A2.max())
v_max = max(jnp.abs(pred_model_v1).max(), jnp.abs(pred_model_v2).max())
w_max = max(pred_model_σ1.max(), pred_model_σ2.max())
A_kwargs = dict(cmap="viridis", origin="lower", vmin=0, vmax=A_max)
v_kwargs = dict(cmap="RdBu_r", origin="lower", vmin=-v_max, vmax=v_max)
w_kwargs = dict(cmap="magma", origin="lower", vmin=0, vmax=w_max)

fig, axes = plt.subplots(3, 2, figsize=(8, 16), layout="compressed")
fs = 14

axes[0, 0].set_title("Component 1")
axes[0, 1].set_title("Component 2")

im00 = axes[0, 0].imshow(pred_model_A1.reshape(ny, nx), **A_kwargs, interpolation="gaussian")
im01 = axes[0, 1].imshow(pred_model_A2.reshape(ny, nx), **A_kwargs, interpolation="gaussian")
im10 = axes[1, 0].imshow(pred_model_v1.reshape(ny, nx), **v_kwargs, interpolation="gaussian")
im11 = axes[1, 1].imshow(pred_model_v2.reshape(ny, nx), **v_kwargs, interpolation="gaussian")
im20 = axes[2, 0].imshow(pred_model_σ1.reshape(ny, nx), **w_kwargs, interpolation="gaussian")
im21 = axes[2, 1].imshow(pred_model_σ2.reshape(ny, nx), **w_kwargs, interpolation="gaussian")

for ax in axes.flatten():
    ax.set_xticks([])
    ax.set_yticks([])
axes[-1, 0].set_xlabel(r"$x$ sky [pix]")
axes[-1, 1].set_xlabel(r"$x$ sky [pix]")
axes[-1, 0].set_ylabel(r"$y$ sky [pix]")

fig.colorbar(im00, ax=axes[0, :], location="right", label="Line peak [K]")
fig.colorbar(im11, ax=axes[1, :], location="right", label="Line centre [km/s]")
fig.colorbar(im20, ax=axes[2, :], location="right", label="Line width [km/s]")
if SAVE:
    plt.savefig(PLOTS_DIR / "inferred_fields.pdf", **SAVEFIG_KWARGS)
plt.show()


# Plot some random spectra with peak > 1x RMS and their fits
rms_thresh = 10

n_spectra = 24

mask = jnp.nanmax(data, axis=0) > rms_thresh * np.nanmean(u_data)
y_indices, x_indices = jnp.where(mask)
selected_indices = rng.choice(len(x_indices), size=n_spectra, replace=False)

pred_spectra = jax.vmap(pred_model, in_axes=(0, None))(vels, spatial_data)
component_1 = jax.vmap(pred_model.line1, in_axes=(0, None))(vels, spatial_data)
component_2 = jax.vmap(pred_model.line2, in_axes=(0, None))(vels, spatial_data)

fig, ax = plt.subplots(
    n_spectra, 1, figsize=(10, 2 * n_spectra), layout="compressed", sharex=True, sharey=False
)
for i, idx in enumerate(selected_indices):
    y = y_indices[idx]
    x = x_indices[idx]
    spectrum = data[:, y, x]
    pred_spectrum = pred_spectra[:, y * nx + x]
    ax[i].plot(
        vels,
        spectrum * peak_intensity,
        drawstyle="steps-mid",
        alpha=1,
        label="Data",
        lw=2.5,
        c="k",
    )
    ax[i].plot(
        vels,
        pred_spectrum * peak_intensity,
        alpha=1,
        label="Model",
        lw=2.5,
        c="red",
    )
    ax[i].plot(
        vels,
        component_1[:, y * nx + x] * peak_intensity,
        alpha=1,
        label="Component 1",
        lw=2.5,
        ls="--",
        c="C0",
    )
    ax[i].plot(
        vels,
        component_2[:, y * nx + x] * peak_intensity,
        alpha=1,
        label="Component 2",
        lw=2.5,
        ls="--",
        c="C1",
    )
    residuals = spectrum - pred_spectrum
    ax[i].plot(
        vels,
        residuals * peak_intensity,
        drawstyle="steps-mid",
        alpha=1,
        label="Residuals",
        lw=2,
        c="gray",
        zorder=-1,
    )
    ax[i].set_xlim(init_v_syst - 45, init_v_syst + 45)
ax[-1].set_xlabel("Velocity [km/s]")
ax[-1].set_ylabel("Intensity [K]")
# ax[0].set_title("Spectral Fits at Selected Pixels")
# Add a legend to the first subplot only
ax[0].legend(loc="upper right", fontsize=14)
if SAVE:
    plt.savefig(PLOTS_DIR / "spectral_fits.pdf", **SAVEFIG_KWARGS)
plt.show()

# Print the inferred global parameters
print("Global parameters:")
for i, line in enumerate([pred_model.line1, pred_model.line2], start=1):
    print(f"Line {i}:")
    print(f"  v_syst = {line.v_syst.val[0]:.2f} km/s")
    print(f"  w_min = {line.w_min.val[0]:.2f} km/s")


# Channel maps plot
# Description:
# - each row is a different velocity channel, labelled in the first column with plt.text by the vel
# - first column is the data, second column is the model, third column is the residuals
# - shared colorbar for the first two columns which we will place at the top of the figure
#   covering the first two columns
# - colorbar for residuals column also at the top of the figure above the residuals column,
#   and using the red_white_blue colormap from mpl_drip
# Do 10 rows/channels evenly spaced between -250 and -220 km/s
channel_velocities = jnp.linspace(-250, -220, 10)
channel_indices = jnp.array([jnp.argmin(jnp.abs(vels - v)) for v in channel_velocities])
channel_velocities_actual = vels[channel_indices]

fig, axes = plt.subplots(10, 3, figsize=(10, 18), layout="compressed", dpi=100)
for i, channel_idx in enumerate(channel_indices):
    vel = vels[channel_idx]
    data_channel = data[channel_idx, :, :].reshape(ny, nx) * peak_intensity
    pred_channel = pred_spectra[channel_idx, :].reshape(ny, nx) * peak_intensity
    residuals_channel = (
        data[channel_idx, :, :].reshape(ny, nx) - pred_spectra[channel_idx, :].reshape(ny, nx)
    ) * peak_intensity

    im0 = axes[i, 0].imshow(data_channel, origin="lower", cmap="viridis", vmin=0, vmax=A_max)
    im1 = axes[i, 1].imshow(pred_channel, origin="lower", cmap="viridis", vmin=0, vmax=A_max)
    im2 = axes[i, 2].imshow(
        residuals_channel, origin="lower", cmap="red_white_blue_r", vmin=-A_max / 5, vmax=A_max / 5
    )

    for j in range(3):
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])

    axes[i, 0].text(
        0.05 * nx,
        0.95 * nx,
        f"{vel:.1f} km/s",
        color="white",
        fontsize=12,
        va="top",
        ha="left",
        bbox=dict(facecolor="black", alpha=0.1, pad=2, edgecolor=None, linewidth=0),
    )

# Add colorbars at the top
cbar0 = fig.colorbar(
    im0, ax=axes[:, 0:2], location="bottom", label="Intensity [K]", aspect=20, pad=0.01
)
cbar1 = fig.colorbar(
    im2, ax=axes[:, 2], location="bottom", label="Residuals [K]", aspect=10, pad=0.01
)
axes[0, 0].set_title("Data")
axes[0, 1].set_title("Model")
axes[0, 2].set_title("Residuals")
if SAVE:
    plt.savefig(PLOTS_DIR / "channel_maps.pdf", **SAVEFIG_KWARGS)
plt.show()


# Now a version where instead of the model channel maps, we plot the individual components
# So the columns are: data, component 1, component 2, residuals
fig, axes = plt.subplots(10, 4, figsize=(12, 18), layout="compressed", dpi=100)
for i, channel_idx in enumerate(channel_indices):
    vel = vels[channel_idx]
    data_channel = data[channel_idx, :, :].reshape(ny, nx) * peak_intensity
    comp1_channel = component_1[channel_idx, :].reshape(ny, nx) * peak_intensity
    comp2_channel = component_2[channel_idx, :].reshape(ny, nx) * peak_intensity
    residuals_channel = (
        data[channel_idx, :, :].reshape(ny, nx) - pred_spectra[channel_idx, :].reshape(ny, nx)
    ) * peak_intensity

    im0 = axes[i, 0].imshow(data_channel, origin="lower", cmap="viridis", vmin=0, vmax=A_max)
    im1 = axes[i, 1].imshow(comp1_channel, origin="lower", cmap="viridis", vmin=0, vmax=A_max)
    im2 = axes[i, 2].imshow(comp2_channel, origin="lower", cmap="viridis", vmin=0, vmax=A_max)
    im3 = axes[i, 3].imshow(
        residuals_channel, origin="lower", cmap="red_white_blue_r", vmin=-A_max / 5, vmax=A_max / 5
    )

    for j in range(4):
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])

    axes[i, 0].text(
        0.05 * nx,
        0.95 * nx,
        f"{vel:.1f} km/s",
        color="white",
        fontsize=12,
        va="top",
        ha="left",
        bbox=dict(facecolor="black", alpha=0.1, pad=2, edgecolor=None, linewidth=0),
    )
# Add colorbars at the top
cbar0 = fig.colorbar(
    im0, ax=axes[:, 0:3], location="bottom", label="Intensity [K]", aspect=20, pad=0.01
)
cbar1 = fig.colorbar(
    im3, ax=axes[:, 3], location="bottom", label="Residuals [K]", aspect=10, pad=0.01
)
axes[0, 0].set_title("Data")
axes[0, 1].set_title("Component 1")
axes[0, 2].set_title("Component 2")
axes[0, 3].set_title("Residuals")
if SAVE:
    plt.savefig(PLOTS_DIR / "channel_maps_components.pdf", **SAVEFIG_KWARGS)
plt.show()


# Residuals cube
residuals_cube = data - pred_spectra.reshape((nλ, ny, nx))
weighted_residuals_cube = residuals_cube / u_data
averageλ_abs_residual = jnp.nanmean(jnp.abs(residuals_cube), axis=0)

# Spectrally summed residuals (not abs)
sumλ_residual = jnp.nansum(residuals_cube, axis=0)

# Put the above two plots side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 6), layout="compressed")
im0 = axes[0].imshow(averageλ_abs_residual, origin="lower", cmap="magma")
fig.colorbar(im0, ax=axes[0])
axes[0].set_title("Spectrally-Averaged Abs Weighted Residual", fontsize=16)
axes[0].set_xlabel("Pixel X")
axes[0].set_ylabel("Pixel Y")
im1 = axes[1].imshow(sumλ_residual, origin="lower", cmap="red_white_blue_r")
fig.colorbar(im1, ax=axes[1])
axes[1].set_title("Spectrally-Summed Weighted Residuals", fontsize=16)
axes[1].set_xlabel("Pixel X")
axes[1].set_ylabel("Pixel Y")
if SAVE:
    plt.savefig(PLOTS_DIR / "spectrally_collapsed_residuals.pdf", **SAVEFIG_KWARGS)
plt.show()
