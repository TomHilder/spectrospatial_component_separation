from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mpl_drip  # noqa: F401
import numpy as np
from model_data import (
    KLineMixture,
    get_phases,
    neg_ln_posterior,
)
from mpl_drip import colormaps  # noqa: F401
from numpy import pi as π
from numpy import save
from read_data import DATA_FNAME
from spectracles import (
    Matern32,
    Matern52,
    OptimiserSchedule,
    Parameter,
    SpatialDataGeneric,
    build_model,
    load_model,
    save_model,
)

plt.style.use("mpl_drip.custom")
rng = np.random.default_rng(6742)

if __name__ == "__main__":
    DATA_DIR = Path(__file__).resolve().parent.parent / "data"
    # TRUNC_DATA_FNAME = "ic1613_hi21cm_truncated_data.npz"
    TRUNC_DATA_FNAME = DATA_FNAME.split(".")[0] + "_truncated_data.npz"
    DATA_PATH = DATA_DIR / DATA_FNAME
    TRUNC_DATA_PATH = DATA_DIR / TRUNC_DATA_FNAME
    assert DATA_PATH.exists(), f"Data file not found: {DATA_PATH}"
    assert TRUNC_DATA_PATH.exists(), f"Truncated data file not found: {TRUNC_DATA_PATH}"

    # PLOTS_DIR = Path("plots_ic1613_two_line_mixture")
    PLOTS_DIR = Path("m33")
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
    init_w_min = 1
    # print(f"Initial w_min guess: {init_w_min:.2f} km/s")

    # Assemble a pixel grid
    PAD_FAC = 0.5
    nλ, ny, nx = data.shape
    x_grid = jnp.linspace(-PAD_FAC * π, PAD_FAC * π, nx)
    y_grid = jnp.linspace(-PAD_FAC * π, PAD_FAC * π, ny)
    x_points, y_points = np.meshgrid(x_grid, y_grid)
    spatial_data = SpatialDataGeneric(x=x_points, y=y_points, idx=jnp.arange(ny * ny))

    # Number of components to fit
    N_COMPONENTS = 3

    # Modes
    n_modes = (151, 151)

    # Kernels
    # kernel_peak = Matern52
    # kernel_velocity = Matern52
    # kernel_broadening = Matern52
    # kernel_skew = Matern52
    # kernel_kurtosis = Matern52
    kernel_peak = Matern32
    kernel_velocity = Matern32
    kernel_broadening = Matern32
    kernel_skew = Matern32
    kernel_kurtosis = Matern32

    # Need to account for if I use different larger nx, ny which is MORE image, and MORE sky area
    # But I want the same physical scale, so length scale in pixels should decrease as nx increases
    ls_kwargs_pv = dict(initial=PAD_FAC * π / 19 / (nx / 100), fixed=True)
    ls_kwargs_w = dict(initial=PAD_FAC * π / 13 / (nx / 100), fixed=True)
    ls_kwargs_s = dict(initial=PAD_FAC * π / 11 / (nx / 100), fixed=True)  # skew
    ls_kwargs_k = dict(initial=PAD_FAC * π / 11 / (nx / 100), fixed=True)  # kurtosis
    var_kwargs_pv = dict(initial=1.0, fixed=True)
    var_kwargs_w = dict(initial=1.0, fixed=True)
    var_kwargs_s = dict(initial=0.1, fixed=True)  # skew
    var_kwargs_k = dict(initial=0.1, fixed=True)  # kurtosis
    peak_kernels = [
        kernel_peak(
            length_scale=Parameter(**ls_kwargs_pv),
            variance=Parameter(**var_kwargs_pv),
        )
        for _ in range(N_COMPONENTS)
    ]
    velocity_kernels = [
        kernel_velocity(
            length_scale=Parameter(**ls_kwargs_pv),
            variance=Parameter(**var_kwargs_pv),
        )
        for _ in range(N_COMPONENTS)
    ]
    broadening_kernels = [
        kernel_broadening(
            length_scale=Parameter(**ls_kwargs_w),
            variance=Parameter(**var_kwargs_w),
        )
        for _ in range(N_COMPONENTS)
    ]
    skew_kernels = [
        kernel_skew(
            length_scale=Parameter(**ls_kwargs_s),
            variance=Parameter(**var_kwargs_s),
        )
        for _ in range(N_COMPONENTS)
    ]
    kurt_kernels = [
        kernel_kurtosis(
            length_scale=Parameter(**ls_kwargs_k),
            variance=Parameter(**var_kwargs_k),
        )
        for _ in range(N_COMPONENTS)
    ]
    if N_COMPONENTS == 2:
        v_syst_offs = jnp.array([-1.0, 1.0])
    elif N_COMPONENTS == 3:
        v_syst_offs = jnp.array([-1.0, 0.0, 1.0])
    elif N_COMPONENTS == 4:
        v_syst_offs = jnp.array([-5.0, -1.0, 1.0, 5.0])
    elif N_COMPONENTS == 5:
        v_syst_offs = jnp.array([-5.0, -1.0, 0.0, 1.0, 5.0])
    elif N_COMPONENTS == 6:
        v_syst_offs = jnp.array([-10.0, -5.0, -1.0, 1.0, 5.0, 10.0])
    elif N_COMPONENTS == 7:
        v_syst_offs = jnp.array([-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0])
    else:
        raise NotImplementedError(
            "Woah buddy, you know how many components you're carrying there? Maybe think about toning it down a bit."
        )

    v_systs = [
        Parameter(initial=init_v_syst + v_syst_offs[i], fixed=False) for i in range(N_COMPONENTS)
    ]
    w_min = Parameter(initial=init_w_min, fixed=True)
    h3_max = Parameter(initial=0.3, fixed=True)
    h4_max = Parameter(initial=0.3, fixed=True)

    # Build the model
    model_cls = KLineMixture
    my_model = build_model(
        model_cls,
        K=N_COMPONENTS,
        n_modes=n_modes,
        peak_kernels=peak_kernels,
        velocity_kernels=velocity_kernels,
        broadening_kernels=broadening_kernels,
        skew_kernels=skew_kernels,
        kurt_kernels=kurt_kernels,
        v_systs=v_systs,
        w_min=w_min,
        h3_max=h3_max,
        h4_max=h4_max,
    )
    phases = get_phases(n_modes, n_components=N_COMPONENTS)
    init_model = my_model.get_locked_model()

    # Optionally print the nested model structure
    # my_model.print_model_tree()

    # Optionally plot the model graph (first turn off latex plotting)
    # with plt.rc_context({"text.usetex": False}):
    #     fig, ax = plt.subplots(figsize=(40, 25))
    #     # Turn off the axes
    #     ax.axis("off")
    #     fig_model = my_model.plot_model_graph(ax=ax)
    #     plt.show()

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
    # plt.yscale("log")
    plt.show()

    save_model(schedule.model_history[-1], PLOTS_DIR / "fitted_model.pkl", overwrite=True)

    pred_model = schedule.model_history[-1].get_locked_model()

    # Plot the inferred fields next to the true fields
    lines_pred_funcs = [getattr(pred_model.lines, f"line{k + 1}") for k in range(pred_model.K)]

    pred_model_As = [
        lines_pred_funcs[i].peak(spatial_data) * peak_intensity for i in range(N_COMPONENTS)
    ]
    pred_model_vs = [
        lines_pred_funcs[i].velocity_obs(spatial_data) - init_v_syst for i in range(N_COMPONENTS)
    ]
    pred_model_σs = [
        lines_pred_funcs[i].width(spatial_data) + init_w_min for i in range(N_COMPONENTS)
    ]
    pred_model_ss = [lines_pred_funcs[i].h3(spatial_data) for i in range(N_COMPONENTS)]
    pred_model_ks = [lines_pred_funcs[i].h4(spatial_data) for i in range(N_COMPONENTS)]

    def max_of_components(component_list, abs=False):
        if abs:
            tran_f = jnp.abs
        else:
            tran_f = lambda x: x  # noqa: E731

        return max([tran_f(component_list[i]).max() for i in range(N_COMPONENTS)])

    A_max = max_of_components(pred_model_As)
    v_max = max_of_components(pred_model_vs, abs=True)
    w_max = max_of_components(pred_model_σs)
    s_max = max_of_components(pred_model_ss, abs=True)
    k_max = max_of_components(pred_model_ks, abs=True)

    A_kwargs = dict(cmap="viridis", origin="lower", vmin=0, vmax=A_max)
    v_kwargs = dict(cmap="RdBu_r", origin="lower", vmin=-v_max, vmax=v_max)
    w_kwargs = dict(cmap="magma", origin="lower", vmin=0, vmax=w_max)
    s_kwargs = dict(cmap="PiYG", origin="lower", vmin=-0.9 * s_max, vmax=0.9 * s_max)
    k_kwargs = dict(cmap="PuOr", origin="lower", vmin=-0.9 * k_max, vmax=0.9 * k_max)

    # Plot some random spectra with peak > 1x RMS and their fits
    rms_thresh = 5

    n_spectra = 24

    mask = jnp.nanmax(data, axis=0) > rms_thresh * np.nanmean(u_data)
    y_indices, x_indices = jnp.where(mask)
    selected_indices = rng.choice(len(x_indices), size=n_spectra, replace=False)

    pred_spectra = jax.vmap(pred_model, in_axes=(0, None))(vels, spatial_data)

    components = [
        jax.vmap(lines_pred_funcs[i], in_axes=(0, None))(vels, spatial_data)
        for i in range(N_COMPONENTS)
    ]

    fig, ax = plt.subplots(
        n_spectra, 1, figsize=(12, 2 * n_spectra), layout="compressed", sharex=True, sharey=False
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
        for j in range(N_COMPONENTS):
            ax[i].plot(
                vels,
                components[j][:, y * nx + x] * peak_intensity,
                alpha=1,
                label=f"Component {j + 1}",
                lw=2.5,
                ls="--",
                c=f"C{j}",
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
        ax[i].set_xlim(init_v_syst - 65, init_v_syst + 65)
    ax[-1].set_xlabel("Velocity [km/s]")
    ax[-1].set_ylabel("Intensity [K]")
    # ax[0].set_title("Spectral Fits at Selected Pixels")
    # Add a legend to the first subplot only
    ax[0].legend(loc="upper right", bbox_to_anchor=(1.25, 1.0), fontsize=14)
    if SAVE:
        plt.savefig(PLOTS_DIR / "spectral_fits.pdf", **SAVEFIG_KWARGS)
    plt.show()

    # Channel maps plot
    v_c_ch = -180
    v_pad_ch = 15
    channel_velocities = jnp.linspace(v_c_ch - v_pad_ch, v_c_ch + v_pad_ch, 10)
    channel_indices = jnp.array([jnp.argmin(jnp.abs(vels - v)) for v in channel_velocities])
    channel_velocities_actual = vels[channel_indices]

    # fig, axes = plt.subplots(10, 3, figsize=(10, 18), layout="compressed", dpi=100)
    # for i, channel_idx in enumerate(channel_indices):
    #     vel = vels[channel_idx]
    #     data_channel = data[channel_idx, :, :].reshape(ny, nx) * peak_intensity
    #     pred_channel = pred_spectra[channel_idx, :].reshape(ny, nx) * peak_intensity
    #     residuals_channel = (
    #         data[channel_idx, :, :].reshape(ny, nx) - pred_spectra[channel_idx, :].reshape(ny, nx)
    #     ) * peak_intensity

    #     im0 = axes[i, 0].imshow(data_channel, origin="lower", cmap="viridis", vmin=0, vmax=A_max)
    #     im1 = axes[i, 1].imshow(pred_channel, origin="lower", cmap="viridis", vmin=0, vmax=A_max)
    #     im2 = axes[i, 2].imshow(
    #         residuals_channel,
    #         origin="lower",
    #         cmap="red_white_blue_r",
    #         vmin=-A_max / 5,
    #         vmax=A_max / 5,
    #     )

    #     for j in range(3):
    #         axes[i, j].set_xticks([])
    #         axes[i, j].set_yticks([])

    #     axes[i, 0].text(
    #         0.05 * nx,
    #         0.95 * nx,
    #         f"{vel:.1f} km/s",
    #         color="white",
    #         fontsize=12,
    #         va="top",
    #         ha="left",
    #         bbox=dict(facecolor="black", alpha=0.1, pad=2, edgecolor=None, linewidth=0),
    #     )

    # # Add colorbars at the top
    # cbar0 = fig.colorbar(
    #     im0, ax=axes[:, 0:2], location="bottom", label="Intensity [K]", aspect=20, pad=0.01
    # )
    # cbar1 = fig.colorbar(
    #     im2, ax=axes[:, 2], location="bottom", label="Residuals [K]", aspect=10, pad=0.01
    # )
    # axes[0, 0].set_title("Data")
    # axes[0, 1].set_title("Model")
    # axes[0, 2].set_title("Residuals")
    # if SAVE:
    #     plt.savefig(PLOTS_DIR / "channel_maps.pdf", **SAVEFIG_KWARGS)
    # plt.show()

    # # Now a version where instead of the model channel maps, we plot the individual components
    # # So the columns are: data, component 1, component 2, residuals
    # fig, axes = plt.subplots(
    #     10, 2 + N_COMPONENTS, figsize=(12, 10 + 4 * N_COMPONENTS), layout="compressed", dpi=100
    # )
    # for i, channel_idx in enumerate(channel_indices):
    #     vel = vels[channel_idx]
    #     data_channel = data[channel_idx, :, :].reshape(ny, nx) * peak_intensity

    #     comp_channels = [
    #         components[i][channel_idx, :].reshape(ny, nx) * peak_intensity
    #         for i in range(N_COMPONENTS)
    #     ]
    #     residuals_channel = (
    #         data[channel_idx, :, :].reshape(ny, nx) - pred_spectra[channel_idx, :].reshape(ny, nx)
    #     ) * peak_intensity

    #     im0 = axes[i, 0].imshow(data_channel, origin="lower", cmap="viridis", vmin=0, vmax=A_max)
    #     for j, comp_channel in enumerate(comp_channels):
    #         im1 = axes[i, 1 + j].imshow(
    #             comp_channel, origin="lower", cmap="viridis", vmin=0, vmax=A_max
    #         )
    #     im2 = axes[i, 1 + N_COMPONENTS].imshow(
    #         residuals_channel,
    #         origin="lower",
    #         cmap="red_white_blue_r",
    #         vmin=-A_max / 5,
    #         vmax=A_max / 5,
    #     )

    #     for j in range(2 + N_COMPONENTS):
    #         axes[i, j].set_xticks([])
    #         axes[i, j].set_yticks([])

    #     axes[i, 0].text(
    #         0.05 * nx,
    #         0.95 * nx,
    #         f"{vel:.1f} km/s",
    #         color="white",
    #         fontsize=12,
    #         va="top",
    #         ha="left",
    #         bbox=dict(facecolor="black", alpha=0.1, pad=2, edgecolor=None, linewidth=0),
    #     )
    # # Add colorbars at the top
    # cbar0 = fig.colorbar(
    #     im0,
    #     ax=axes[:, 0 : 1 + N_COMPONENTS],
    #     location="bottom",
    #     label="Intensity [K]",
    #     aspect=(N_COMPONENTS + 1) * 10,
    #     pad=0.01,
    # )
    # cbar1 = fig.colorbar(
    #     im2,
    #     ax=axes[:, 1 + N_COMPONENTS],
    #     location="bottom",
    #     label="Residuals [K]",
    #     aspect=10,
    #     pad=0.01,
    # )
    # axes[0, 0].set_title("Data")
    # for j in range(N_COMPONENTS):
    #     axes[0, 1 + j].set_title(f"Component {j + 1}")
    # axes[0, 1 + N_COMPONENTS].set_title("Residuals")
    # if SAVE:
    #     plt.savefig(PLOTS_DIR / "channel_maps_components.pdf", **SAVEFIG_KWARGS)
    # plt.show()

    # Residuals cube
    residuals_cube = data - pred_spectra.reshape((nλ, ny, nx))
    weighted_residuals_cube = residuals_cube / u_data
    averageλ_abs_residual = jnp.nanmean(jnp.abs(residuals_cube), axis=0)

    # Spectrally summed residuals (not abs)
    sumλ_residual = jnp.nansum(residuals_cube, axis=0)

    sumλ_max = jnp.max(jnp.abs(sumλ_residual))

    # Put the above two plots side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), layout="compressed")
    im0 = axes[0].imshow(averageλ_abs_residual, origin="lower", cmap="magma")
    fig.colorbar(im0, ax=axes[0])
    axes[0].set_title("Spectrally-Averaged Abs Weighted Residual", fontsize=16)
    axes[0].set_xlabel("Pixel X")
    axes[0].set_ylabel("Pixel Y")
    im1 = axes[1].imshow(
        sumλ_residual, origin="lower", cmap="red_white_blue_r", vmin=-sumλ_max, vmax=sumλ_max
    )
    fig.colorbar(im1, ax=axes[1])
    axes[1].set_title("Spectrally-Summed Weighted Residuals", fontsize=16)
    axes[1].set_xlabel("Pixel X")
    axes[1].set_ylabel("Pixel Y")
    if SAVE:
        plt.savefig(PLOTS_DIR / "spectrally_collapsed_residuals.pdf", **SAVEFIG_KWARGS)
    plt.show()

    # Another channels plot but with all the columns data, model, components, residuals
    fig, axes = plt.subplots(
        10, 3 + N_COMPONENTS, figsize=(15, 10 + 4 * N_COMPONENTS), layout="compressed", dpi=100
    )
    for i, channel_idx in enumerate(channel_indices):
        vel = vels[channel_idx]
        data_channel = data[channel_idx, :, :].reshape(ny, nx) * peak_intensity
        pred_channel = pred_spectra[channel_idx, :].reshape(ny, nx) * peak_intensity
        residuals_channel = (
            data[channel_idx, :, :].reshape(ny, nx) - pred_spectra[channel_idx, :].reshape(ny, nx)
        ) * peak_intensity

        im0 = axes[i, 0].imshow(data_channel, origin="lower", cmap="viridis", vmin=0, vmax=A_max)
        im1 = axes[i, 1].imshow(pred_channel, origin="lower", cmap="viridis", vmin=0, vmax=A_max)
        for j in range(N_COMPONENTS):
            comp_channel = components[j][channel_idx, :].reshape(ny, nx) * peak_intensity
            axes[i, 2 + j].imshow(comp_channel, origin="lower", cmap="viridis", vmin=0, vmax=A_max)
        im2 = axes[i, 2 + N_COMPONENTS].imshow(
            residuals_channel,
            origin="lower",
            cmap="red_white_blue_r",
            vmin=-A_max / 5,
            vmax=A_max / 5,
        )

        for j in range(3 + N_COMPONENTS):
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
        im0,
        ax=axes[:, 0:5],
        location="bottom",
        label="Intensity [K]",
        aspect=(N_COMPONENTS + 2) * 10,
        pad=0.01,
    )
    cbar1 = fig.colorbar(
        im2,
        ax=axes[:, 2 + N_COMPONENTS],
        location="bottom",
        label="Residuals [K]",
        aspect=10,
        pad=0.01,
    )
    axes[0, 0].set_title("Data")
    axes[0, 1].set_title("Model")
    for j in range(N_COMPONENTS):
        axes[0, 2 + j].set_title(f"Component {j + 1}")
    axes[0, 2 + N_COMPONENTS].set_title("Residuals")
    if SAVE:
        plt.savefig(PLOTS_DIR / "channel_maps_full.pdf", **SAVEFIG_KWARGS)
    plt.show()

    # ==

    pred_model_Is = [
        pred_model_As[i] * pred_model_σs[i] * jnp.sqrt(2 * jnp.pi) for i in range(N_COMPONENTS)
    ]

    I_min = min(pred_model_Is[i].min() for i in range(N_COMPONENTS))
    I_max = max(pred_model_Is[i].max() for i in range(N_COMPONENTS)) * 0.9
    w_max = max(pred_model_σs[i].max() for i in range(N_COMPONENTS))

    I_kwargs = dict(cmap="viridis", origin="lower", vmin=I_min, vmax=I_max)
    fig, axes = plt.subplots(6, N_COMPONENTS, figsize=(16, 16), layout="compressed")
    fs = 14

    for i in range(N_COMPONENTS):
        axes[0, i].set_title(f"Component {i + 1}")
        pred_model_A = pred_model_As[i]
        pred_model_v = pred_model_vs[i]
        pred_model_σ = pred_model_σs[i]
        pred_model_s = pred_model_ss[i]
        pred_model_k = pred_model_ks[i]
        pred_model_I = pred_model_Is[i]
        im00 = axes[0, i].imshow(pred_model_I.reshape(ny, nx), **I_kwargs, interpolation="gaussian")
        im10 = axes[1, i].imshow(pred_model_A.reshape(ny, nx), **A_kwargs, interpolation="gaussian")
        im20 = axes[2, i].imshow(pred_model_v.reshape(ny, nx), **v_kwargs, interpolation="gaussian")
        im30 = axes[3, i].imshow(pred_model_σ.reshape(ny, nx), **w_kwargs, interpolation="gaussian")
        im40 = axes[4, i].imshow(pred_model_s.reshape(ny, nx), **s_kwargs, interpolation="gaussian")
        im50 = axes[5, i].imshow(pred_model_k.reshape(ny, nx), **k_kwargs, interpolation="gaussian")

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
    axes[-1, 0].set_xlabel(r"x sky [pix]")
    axes[-1, 0].set_ylabel(r"y sky [pix]")

    fig.colorbar(im00, ax=axes[0, :], location="right", label="Int Intensity [K km/s]")
    fig.colorbar(im10, ax=axes[1, :], location="right", label="Line peak [K]")
    fig.colorbar(im20, ax=axes[2, :], location="right", label="Line centre [km/s]")
    fig.colorbar(im30, ax=axes[3, :], location="right", label="Line width [km/s]")
    fig.colorbar(im40, ax=axes[4, :], location="right", label="Line skew [h3]")
    fig.colorbar(im50, ax=axes[5, :], location="right", label="Line kurtosis [h4]")

    if SAVE:
        plt.savefig(PLOTS_DIR / "inferred_fields.pdf", **SAVEFIG_KWARGS)
    plt.show()

    coeffs_kwargs = dict(cmap="RdBu", origin="lower", vmin=-3.0, vmax=3.0)

    # Plot the Fourier coefficients for each component
    fig, axes = plt.subplots(N_COMPONENTS, 5, figsize=(20, 4 * N_COMPONENTS), layout="compressed")
    for i in range(N_COMPONENTS):
        lines_pred_func = lines_pred_funcs[i]
        peak_coeffs = lines_pred_func.peak_raw.coefficients.val
        velocity_coeffs = lines_pred_func.velocity.coefficients.val
        broadening_coeffs = lines_pred_func.broadening_raw.coefficients.val
        skew_coeffs = lines_pred_func.skew_raw.coefficients.val
        kurtosis_coeffs = lines_pred_func.kurtosis_raw.coefficients.val

        cm = axes[i, 0].imshow(peak_coeffs, **coeffs_kwargs)
        axes[i, 1].imshow(velocity_coeffs, **coeffs_kwargs)
        axes[i, 2].imshow(broadening_coeffs, **coeffs_kwargs)
        axes[i, 3].imshow(skew_coeffs, **coeffs_kwargs)
        axes[i, 4].imshow(kurtosis_coeffs, **coeffs_kwargs)

        for j in range(5):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    axes[0, 0].set_title("Peak Coefficients")
    axes[0, 1].set_title("Velocity Coefficients")
    axes[0, 2].set_title("Broadening Coefficients")
    axes[0, 3].set_title("Skew Coefficients")
    axes[0, 4].set_title("Kurtosis Coefficients")

    plt.colorbar(
        cm,
        ax=axes[:, :],
        location="bottom",
        label="Z coeffs (not kernel PSD scaled, whitened in some sense)",
        aspect=40,
        pad=1e-2,
    )

    if SAVE:
        plt.savefig(PLOTS_DIR / "fourier_coefficients.pdf", **SAVEFIG_KWARGS)
    plt.show()

    # Plot of dominant component in terms of integrated intensity per spaxel
    dominant_component_idx = jnp.argmax(
        jnp.array([pred_model_Is[i] for i in range(N_COMPONENTS)]), axis=0
    ).reshape(ny, nx)
    print("Dominant component index shape:", dominant_component_idx.shape)

    from matplotlib.colors import ListedColormap

    base_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    component_colors = base_colors[:N_COMPONENTS]
    discrete_cmap = ListedColormap(component_colors)

    plt.figure(figsize=(8, 6))
    im = plt.imshow(dominant_component_idx, cmap=discrete_cmap, origin="lower")
    plt.colorbar(im, label="Dominant Component Index", ticks=jnp.arange(N_COMPONENTS))
    plt.title("Dominant Spectral Component per Spaxel")
    plt.xlabel("Pixel X")
    plt.ylabel("Pixel Y")
    plt.xticks([])
    plt.yticks([])
    if SAVE:
        plt.savefig(PLOTS_DIR / "dominant_component_map.pdf", **SAVEFIG_KWARGS)
    plt.show()

    # Assemble masks for each component based on integrated intensity above some threshold
    intensity_thresholds = [0.35 * jnp.max(pred_model_Is[i]) for i in range(N_COMPONENTS)]
    component_masks = [
        (pred_model_Is[i].reshape(ny, nx) > intensity_thresholds[i]) for i in range(N_COMPONENTS)
    ]

    # Plot the masks
    fig, axes = plt.subplots(1, N_COMPONENTS, figsize=(4 * N_COMPONENTS, 4), layout="compressed")
    for i in range(N_COMPONENTS):
        im = axes[i].imshow(component_masks[i], cmap="gray", origin="lower")
        axes[i].set_title(f"Component {i + 1} Mask")
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    if SAVE:
        plt.savefig(PLOTS_DIR / "component_masks.pdf", **SAVEFIG_KWARGS)
    plt.show()

    inferred_v_systs = np.array([lines_pred_funcs[i].v_syst.val[0] for i in range(N_COMPONENTS)])
    print("Inferred systemic velocities for each component:", inferred_v_systs)

    # Make the inferred fields plot from before, but now overlay the masks with transparency
    # The not masked regions shouldn't plot white on top, they should just be transparent
    fig, axes = plt.subplots(6, N_COMPONENTS, figsize=(16, 16), layout="compressed")
    fs = 14

    mask_alpha = 0.95

    dv_kwargs = dict(cmap="RdBu_r", origin="lower", vmin=-5, vmax=5)

    for i in range(N_COMPONENTS):
        axes[0, i].set_title(f"Component {i + 1}")
        pred_model_A = pred_model_As[i]
        pred_model_v = pred_model_vs[i] - inferred_v_systs[i] + init_v_syst
        pred_model_σ = pred_model_σs[i]
        pred_model_s = pred_model_ss[i]
        pred_model_k = pred_model_ks[i]
        pred_model_I = pred_model_Is[i]
        im00 = axes[0, i].imshow(pred_model_I.reshape(ny, nx), **I_kwargs, interpolation="gaussian")
        im10 = axes[1, i].imshow(pred_model_A.reshape(ny, nx), **A_kwargs, interpolation="gaussian")
        im20 = axes[2, i].imshow(
            pred_model_v.reshape(ny, nx), **dv_kwargs, interpolation="gaussian"
        )
        im30 = axes[3, i].imshow(pred_model_σ.reshape(ny, nx), **w_kwargs, interpolation="gaussian")
        im40 = axes[4, i].imshow(pred_model_s.reshape(ny, nx), **s_kwargs, interpolation="gaussian")
        im50 = axes[5, i].imshow(pred_model_k.reshape(ny, nx), **k_kwargs, interpolation="gaussian")

        # Overlay the mask
        mask_overlay = jnp.where(component_masks[i], np.nan, 1.0)
        axes[0, i].imshow(mask_overlay, cmap="gray", origin="lower", alpha=mask_alpha)
        axes[1, i].imshow(mask_overlay, cmap="gray", origin="lower", alpha=mask_alpha)
        axes[2, i].imshow(mask_overlay, cmap="gray", origin="lower", alpha=mask_alpha)
        axes[3, i].imshow(mask_overlay, cmap="gray", origin="lower", alpha=mask_alpha)
        axes[4, i].imshow(mask_overlay, cmap="gray", origin="lower", alpha=mask_alpha)
        axes[5, i].imshow(mask_overlay, cmap="gray", origin="lower", alpha=mask_alpha)

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
    axes[-1, 0].set_xlabel(r"x sky [pix]")
    axes[-1, 0].set_ylabel(r"y sky [pix]")
    fig.colorbar(im00, ax=axes[0, :], location="right", label="Int Intensity [K km/s]")
    fig.colorbar(im10, ax=axes[1, :], location="right", label="Line peak [K]")
    fig.colorbar(im20, ax=axes[2, :], location="right", label="Line centre [km/s]")
    fig.colorbar(im30, ax=axes[3, :], location="right", label="Line width [km/s]")
    fig.colorbar(im40, ax=axes[4, :], location="right", label="Line skew [h3]")
    fig.colorbar(im50, ax=axes[5, :], location="right", label="Line kurtosis [h4]")
    if SAVE:
        plt.savefig(PLOTS_DIR / "inferred_fields_with_masks.pdf", **SAVEFIG_KWARGS)
    plt.show()
