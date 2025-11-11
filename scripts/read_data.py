from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

plt.style.use("mpl_drip.custom")
rng = np.random.default_rng(0)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_FNAME = "ic1613_C+D+tp_hi21cm_0p8kms_30as.fits"
DATA_PATH = DATA_DIR / DATA_FNAME
assert DATA_PATH.exists(), f"Data file not found: {DATA_PATH}"

# Open and read
with fits.open(DATA_PATH) as hdul:
    hdul.info()
    data = hdul[0].data
    header = hdul[0].header


def velocities_from_header(header):
    # Check that it's m/s
    assert header["CUNIT3"] == "m/s", "Velocity unit is not m/s"
    return header["CRVAL3"] + header["CDELT3"] * (
        np.arange(1, header["NAXIS3"] + 1) - header["CRPIX3"]
    )


# Calculate velocity axis in km/s
vels = velocities_from_header(header)
vels /= 1000  # Convert to km/s

# Plot integrated intensity map
plt.figure(figsize=(10, 6))
plt.imshow(np.nanmax(data, axis=0), origin="lower")
plt.colorbar(label="[K]")
plt.xlabel("Pixel X")
plt.ylabel("Pixel Y")
plt.title("Peak Intensity Map")
plt.show()

# Plot average spectrum
spectrum = np.nanmean(data, axis=(1, 2))
plt.figure(figsize=(10, 6))
plt.plot(vels, spectrum, drawstyle="steps-mid")
plt.xlabel("Velocity [m/s]")
plt.ylabel("Intensity [K]")
plt.title("Average Spectrum")
plt.show()

# Plot some random spectra with peak > 6x RMS
rms = np.nanstd(data[0:50, :, :], axis=0)
rms_thresh = 6
mask = np.nanmax(data, axis=0) > rms_thresh * rms
y_indices, x_indices = np.where(mask)
selected_indices = rng.choice(len(x_indices), size=5, replace=False)
plt.figure(figsize=(10, 6))
for idx in selected_indices:
    y = y_indices[idx]
    x = x_indices[idx]
    plt.plot(
        vels,
        data[:, y, x],
        drawstyle="steps-mid",
        alpha=0.95,
        label=f"Pixel ({x}, {y})",
        lw=1,
    )
plt.xlabel("Velocity [m/s]")
plt.ylabel("Intensity [K]")
plt.title(rf"Random Spectra with Peak $>$ {rms_thresh}$\times$RMS")
plt.legend()
plt.show()


# Plot rms as a function of velocities by calculating in windows
window_size = 50
rms_values = []
for i in range(0, data.shape[0], window_size):
    window_data = data[i : i + window_size, :, :]
    rms_window = np.nanstd(window_data, axis=(1, 2))
    rms_values.extend(rms_window)
plt.figure(figsize=(10, 6))
plt.plot(vels, rms_values, drawstyle="steps-mid")
plt.xlabel("Velocity [m/s]")
plt.ylabel("RMS [K]")
plt.title("RMS as a Function of Velocity")
plt.show()
