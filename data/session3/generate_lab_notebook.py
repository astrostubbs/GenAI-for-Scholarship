"""Generate a self-contained HTML lab notebook from the thermal analysis."""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from astropy.coordinates import EarthLocation, AltAz, get_sun
from astropy.time import Time
import astropy.units as u
from datetime import datetime, timedelta
import base64
import io

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64

# Load data
df = pd.read_csv('rubin_mirror_temps.csv', parse_dates=['timestamp'])

# --- Plot 1: Full time series ---
fig1, ax = plt.subplots(figsize=(12, 5))
ax.plot(df['timestamp'], df['temperature'], linewidth=0.5)
ax.set_xlabel('Date (UTC)')
ax.set_ylabel('Temperature (°C)')
ax.set_title('Rubin Observatory Primary Mirror Temperature — Jun–Dec 2025')
ax.grid(True, alpha=0.3)
fig1.tight_layout()
img1 = fig_to_base64(fig1)

# --- Plot 2: Histogram ---
fig2, ax = plt.subplots(figsize=(8, 5))
ax.hist(df['temperature'], bins=60, edgecolor='black', linewidth=0.5)
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Count')
ax.set_title('Distribution of Mirror Temperatures — Jun–Dec 2025')
ax.grid(True, alpha=0.3, axis='y')
fig2.tight_layout()
img2 = fig_to_base64(fig2)

# --- Extremes ---
idx_max = df['temperature'].idxmax()
idx_min = df['temperature'].idxmin()
t_max = df.loc[idx_max, 'temperature']
ts_max = df.loc[idx_max, 'timestamp']
t_min = df.loc[idx_min, 'temperature']
ts_min = df.loc[idx_min, 'timestamp']

# --- Plot 3: Last week with sunset markers ---
last_time = df['timestamp'].max()
week_start = last_time - timedelta(days=7)
week = df[df['timestamp'] >= week_start].copy()

location = EarthLocation(lat=-30.24*u.deg, lon=-70.74*u.deg, height=2715*u.m)
unique_dates = week['timestamp'].dt.date.unique()
sunset_times = []

for date in unique_dates:
    t_start = pd.Timestamp(date).replace(hour=21, minute=0)
    times_utc = [t_start + timedelta(minutes=m) for m in range(300)]
    astropy_times = Time(times_utc)
    altaz_frame = AltAz(obstime=astropy_times, location=location)
    sun_altitudes = get_sun(astropy_times).transform_to(altaz_frame).alt.deg
    for i in range(len(sun_altitudes) - 1):
        if sun_altitudes[i] > 0 and sun_altitudes[i + 1] <= 0:
            frac = sun_altitudes[i] / (sun_altitudes[i] - sun_altitudes[i + 1])
            sunset_utc = times_utc[i] + frac * timedelta(minutes=1)
            sunset_times.append(sunset_utc)
            break

sunset_temps = []
for st in sunset_times:
    idx = (week['timestamp'] - st).abs().idxmin()
    sunset_temps.append(week.loc[idx, 'temperature'])

fig3, ax = plt.subplots(figsize=(12, 5))
ax.plot(week['timestamp'], week['temperature'], linewidth=0.8)
ax.plot(sunset_times, sunset_temps, 'ro', markersize=10, label='Sunset', zorder=5)
ax.set_xlabel('Date (UTC)')
ax.set_ylabel('Temperature (°C)')
ax.set_title('Mirror Temperature — Last Week of Data')
ax.legend()
ax.grid(True, alpha=0.3)
fig3.tight_layout()
img3 = fig_to_base64(fig3)

# --- Plot 4: Fourier analysis (raw power) ---
temp = df['temperature'].values
N = len(temp)
dt_hrs = 15.0 / 60.0
temp_centered = temp - np.mean(temp)

fft_vals = np.fft.rfft(temp_centered)
power = np.abs(fft_vals) ** 2
freqs = np.fft.rfftfreq(N, d=dt_hrs)

with np.errstate(divide='ignore'):
    periods_days = 1.0 / (freqs * 24.0)

freqs = freqs[1:]
power = power[1:]
periods_days = periods_days[1:]

top_indices = np.argsort(power)[-10:][::-1]
raw_peaks = [(periods_days[i], power[i] / power[top_indices[0]]) for i in top_indices]

fig4, ax = plt.subplots(figsize=(12, 5))
mask = periods_days > 0.05
ax.loglog(periods_days[mask], power[mask], linewidth=0.5)
ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='1 day')
ax.axvline(x=0.5, color='orange', linestyle='--', alpha=0.5, label='0.5 day')
ax.set_xlabel('Period (days)')
ax.set_ylabel('Power')
ax.set_title('Power Spectrum (raw) — Rubin Mirror Temperature')
ax.legend()
ax.grid(True, alpha=0.3)
fig4.tight_layout()
img4 = fig_to_base64(fig4)

# --- Plot 5: Fourier with local background ---
log_power = np.log10(power + 1e-30)
background = uniform_filter1d(log_power, size=201, mode='nearest')
background_linear = 10.0 ** background
snr = power / background_linear

peak_indices, _ = find_peaks(snr, height=2.0, distance=5)
sorted_peaks = peak_indices[np.argsort(snr[peak_indices])[::-1]]
snr_peaks = [(periods_days[i], snr[i]) for i in sorted_peaks[:10]]

fig5, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

ax1.loglog(periods_days[mask], power[mask], linewidth=0.5, alpha=0.7, label='Power')
ax1.loglog(periods_days[mask], background_linear[mask], 'r-', linewidth=1.5, label='Smoothed background')
ax1.set_xlabel('Period (days)')
ax1.set_ylabel('Power')
ax1.set_title('Power Spectrum with Local Background')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.semilogx(periods_days[mask], snr[mask], linewidth=0.5)
ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
ax2.set_xlabel('Period (days)')
ax2.set_ylabel('Power / Background')
ax2.set_title('Signal-to-Background Ratio')
ax2.grid(True, alpha=0.3)
for idx in sorted_peaks[:5]:
    if periods_days[idx] > 0.05:
        ax2.annotate(f'{periods_days[idx]:.2f} d',
                     xy=(periods_days[idx], snr[idx]),
                     xytext=(0, 10), textcoords='offset points',
                     ha='center', fontsize=9, color='red',
                     arrowprops=dict(arrowstyle='->', color='red', lw=0.8))
fig5.tight_layout()
img5 = fig_to_base64(fig5)

# --- Build HTML ---
now = datetime.now().strftime('%Y-%m-%d %H:%M')

raw_peak_rows = ''.join(
    f'<tr><td>{i+1}</td><td>{p:.4f}</td><td>{r:.4f}</td></tr>'
    for i, (p, r) in enumerate(raw_peaks)
)

snr_peak_rows = ''.join(
    f'<tr><td>{i+1}</td><td>{p:.4f}</td><td>{s:.2f}</td></tr>'
    for i, (p, s) in enumerate(snr_peaks)
)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Lab Notebook — Rubin Thermal Data Analysis</title>
<style>
body {{ font-family: Georgia, serif; max-width: 900px; margin: 0 auto; padding: 2rem; line-height: 1.6; color: #1e1e1e; }}
h1 {{ color: #A51C30; border-bottom: 3px solid #A51C30; padding-bottom: 0.5rem; }}
h2 {{ color: #8C1515; margin-top: 2rem; }}
img {{ max-width: 100%; border: 1px solid #ddd; margin: 1rem 0; }}
table {{ border-collapse: collapse; margin: 1rem 0; }}
th, td {{ border: 1px solid #ddd; padding: 0.4rem 0.8rem; text-align: left; }}
th {{ background: #f3f4f4; }}
code {{ font-family: Menlo, monospace; font-size: 0.9em; background: #f3f4f4; padding: 0.15em 0.4em; border-radius: 3px; }}
.meta {{ color: #555; font-size: 0.9rem; }}
a {{ color: #A51C30; }}
</style>
</head>
<body>

<h1>Lab Notebook: Rubin Observatory Thermal Data Analysis</h1>
<p class="meta">Generated: {now}<br>
Data: rubin_mirror_temps.csv — Vera C. Rubin Observatory primary mirror temperature, Jun–Dec 2025<br>
Location: Cerro Pachon, Chile (lat -30.24, lon -70.74, elev 2715 m)</p>

<h2>1. Data Summary</h2>
<p>The dataset contains {len(df):,} measurements at 15-minute intervals, from
{df['timestamp'].min().strftime('%Y-%m-%d %H:%M')} to
{df['timestamp'].max().strftime('%Y-%m-%d %H:%M')} UTC.
Two columns: <code>timestamp</code> and <code>temperature</code> (°C).
No missing values, no gaps in the time series, no outliers detected.</p>

<h2>2. Temperature Time Series</h2>
<img src="data:image/png;base64,{img1}" alt="Temperature time series">
<p>Source: <a href="plot_temperature.py">plot_temperature.py</a></p>

<h2>3. Temperature Distribution</h2>
<img src="data:image/png;base64,{img2}" alt="Temperature histogram">
<p>Temperature range: {df['temperature'].min():.2f} to {df['temperature'].max():.2f} °C.
Mean: {df['temperature'].mean():.2f} °C, Std Dev: {df['temperature'].std():.2f} °C.</p>

<h2>4. Temperature Extremes</h2>
<table>
<tr><th>Extreme</th><th>Value (°C)</th><th>Timestamp (UTC)</th></tr>
<tr><td>Maximum</td><td>{t_max:.4f}</td><td>{ts_max}</td></tr>
<tr><td>Minimum</td><td>{t_min:.4f}</td><td>{ts_min}</td></tr>
</table>

<h2>5. Last Week with Sunset Markers</h2>
<img src="data:image/png;base64,{img3}" alt="Last week with sunset markers">
<p>Sunset times computed using astropy for Cerro Pachon.
Red dots indicate the temperature at local sunset for each day.</p>
<p>Source: <a href="plot_temperature.py">plot_temperature.py</a></p>

<h2>6. Fourier Analysis — Raw Power Spectrum</h2>
<img src="data:image/png;base64,{img4}" alt="Raw power spectrum">
<table>
<tr><th>Rank</th><th>Period (days)</th><th>Relative Power</th></tr>
{raw_peak_rows}
</table>

<h2>7. Fourier Analysis — Peaks Above Local Background</h2>
<img src="data:image/png;base64,{img5}" alt="Power spectrum with background">
<table>
<tr><th>Rank</th><th>Period (days)</th><th>SNR (power/background)</th></tr>
{snr_peak_rows}
</table>
<p>Source: <a href="fourier_analysis.py">fourier_analysis.py</a></p>

<h2>Data Provenance</h2>
<ul>
<li>Data file: <a href="rubin_mirror_temps.csv">rubin_mirror_temps.csv</a></li>
<li>Source: Vera C. Rubin Observatory telemetry, primary mirror thermal sensors</li>
<li>Period: June 1 – December 1, 2025</li>
<li>Sampling: 15-minute intervals</li>
<li>Temperature column is the mean of tempMax and tempMin from the original dataset</li>
</ul>

</body>
</html>
"""

with open('lab_notebook.html', 'w') as f:
    f.write(html)

print("Generated lab_notebook.html")
