import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import EarthLocation, AltAz
from astropy.time import Time
from astropy.coordinates import get_sun
import astropy.units as u
from datetime import timedelta

df = pd.read_csv('rubin_mirror_temps.csv', parse_dates=['timestamp'])

# Filter to most recent week
last_time = df['timestamp'].max()
week_start = last_time - timedelta(days=7)
week = df[df['timestamp'] >= week_start].copy()

# Cerro Pachon location
location = EarthLocation(lat=-30.24 * u.deg, lon=-70.74 * u.deg, height=2715 * u.m)

# Compute sunset time for each day in the week
unique_dates = week['timestamp'].dt.date.unique()
sunset_times = []

for date in unique_dates:
    # Scan from 21:00 to 01:00 UTC next day (late afternoon/evening local)
    # in 1-minute steps to find when sun altitude crosses zero going down
    t_start = pd.Timestamp(date).replace(hour=21, minute=0)
    times_utc = [t_start + timedelta(minutes=m) for m in range(300)]
    astropy_times = Time(times_utc)
    altaz_frame = AltAz(obstime=astropy_times, location=location)
    sun_altitudes = get_sun(astropy_times).transform_to(altaz_frame).alt.deg

    # Find where altitude crosses from positive to negative
    found = False
    for i in range(len(sun_altitudes) - 1):
        if sun_altitudes[i] > 0 and sun_altitudes[i + 1] <= 0:
            # Linear interpolation for more precise sunset time
            frac = sun_altitudes[i] / (sun_altitudes[i] - sun_altitudes[i + 1])
            sunset_utc = times_utc[i] + frac * timedelta(minutes=1)
            sunset_times.append(sunset_utc)
            found = True
            break
    if not found:
        print(f"Warning: no sunset found for {date}")

# Interpolate temperature at each sunset time
sunset_temps = []
for st in sunset_times:
    # Find nearest timestamp in data
    idx = (week['timestamp'] - st).abs().idxmin()
    sunset_temps.append(week.loc[idx, 'temperature'])

# Plot last week with sunset markers
plt.figure(figsize=(12, 5))
plt.plot(week['timestamp'], week['temperature'], linewidth=0.8)
plt.plot(sunset_times, sunset_temps, 'ro', markersize=10, label='Sunset', zorder=5)
plt.xlabel('Date (UTC)')
plt.ylabel('Temperature (°C)')
plt.title('Rubin Observatory Mirror Temperature — Last Week of Data')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Histogram
plt.figure(figsize=(8, 5))
plt.hist(df['temperature'], bins=60, edgecolor='black', linewidth=0.5)
plt.xlabel('Temperature (°C)')
plt.ylabel('Count')
plt.title('Distribution of Mirror Temperatures — Jun–Dec 2025')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()

plt.show()
