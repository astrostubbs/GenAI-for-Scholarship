import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

df = pd.read_csv('rubin_mirror_temps.csv', parse_dates=['timestamp'])

temp = df['temperature'].values
N = len(temp)
dt = 15.0 / 60.0  # sampling interval in hours

# Remove mean before FFT
temp_centered = temp - np.mean(temp)

# Compute FFT
fft_vals = np.fft.rfft(temp_centered)
power = np.abs(fft_vals) ** 2
freqs = np.fft.rfftfreq(N, d=dt)  # in cycles per hour

# Convert frequency to period in days
with np.errstate(divide='ignore'):
    periods_days = 1.0 / (freqs * 24.0)

# Skip DC component (index 0)
freqs = freqs[1:]
power = power[1:]
periods_days = periods_days[1:]

# Smooth the power spectrum to estimate local background
# Use a wide window in log-frequency space for a gentle smooth
log_power = np.log10(power + 1e-30)
smooth_window = 201  # wide enough to average over individual peaks
background = uniform_filter1d(log_power, size=smooth_window, mode='nearest')
background_linear = 10.0 ** background

# Ratio of power to local smoothed background
snr = power / background_linear

# Find peaks in the SNR spectrum
peak_indices, properties = find_peaks(snr, height=2.0, distance=5)

# Sort by SNR
sorted_peaks = peak_indices[np.argsort(snr[peak_indices])[::-1]]

print("Spectral peaks ranked by significance above local background:")
print(f"{'Rank':<6} {'Period (days)':<18} {'SNR (power/background)'}")
print("-" * 50)
for rank, idx in enumerate(sorted_peaks[:15], 1):
    print(f"{rank:<6} {periods_days[idx]:<18.4f} {snr[idx]:.2f}")

# Plot
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Panel 1: Power spectrum (log-log)
mask = periods_days > 0.05
ax = axes[0]
ax.loglog(periods_days[mask], power[mask], linewidth=0.5, alpha=0.7, label='Power')
ax.loglog(periods_days[mask], background_linear[mask], 'r-', linewidth=1.5, label='Smoothed background')
ax.set_xlabel('Period (days)')
ax.set_ylabel('Power')
ax.set_title('Power Spectrum with Local Background')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: SNR (signal above local background)
ax = axes[1]
ax.semilogx(periods_days[mask], snr[mask], linewidth=0.5)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Period (days)')
ax.set_ylabel('Power / Background')
ax.set_title('Signal-to-Background Ratio')
ax.grid(True, alpha=0.3)

# Mark the top peaks
for idx in sorted_peaks[:5]:
    if periods_days[idx] > 0.05:
        ax.annotate(f'{periods_days[idx]:.2f} d',
                     xy=(periods_days[idx], snr[idx]),
                     xytext=(0, 10), textcoords='offset points',
                     ha='center', fontsize=9, color='red',
                     arrowprops=dict(arrowstyle='->', color='red', lw=0.8))

# Panel 3: Zoom on 0.3 to 3 day range
mask3 = (periods_days > 0.3) & (periods_days < 3.0)
ax = axes[2]
ax.plot(periods_days[mask3], snr[mask3], linewidth=0.8)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Period (days)')
ax.set_ylabel('Power / Background')
ax.set_title('Signal-to-Background Ratio — Zoom on 0.3–3 Day Range')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
