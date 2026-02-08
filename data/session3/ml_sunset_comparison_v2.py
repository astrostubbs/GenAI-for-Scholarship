"""
Advanced ML comparison for sunset temperature prediction.
Uses full time history up to 3 hours before sunset.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from astropy.coordinates import EarthLocation, AltAz, get_sun
from astropy.time import Time
import astropy.units as u
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.optimize import curve_fit
from scipy.fft import rfft, rfftfreq
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
df = pd.read_csv('rubin_mirror_temps.csv', parse_dates=['timestamp'])

# Cerro Pachon location
location = EarthLocation(lat=-30.24*u.deg, lon=-70.74*u.deg, height=2715*u.m)

print("Computing sunset times...")
unique_dates = df['timestamp'].dt.date.unique()
sunset_dict = {}

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
            sunset_dict[date] = sunset_utc
            break

print(f"Found {len(sunset_dict)} sunset times")

# Build dataset: for each day, extract time history and sunset temperature
dataset = []
for date, sunset_time in sunset_dict.items():
    cutoff_time = sunset_time - timedelta(hours=3)

    # Get all data for this day up to cutoff
    day_start = pd.Timestamp(date)
    day_data = df[(df['timestamp'] >= day_start) & (df['timestamp'] < cutoff_time)]

    # Need at least 4 days of prior data for Fourier method
    four_days_ago = day_start - timedelta(days=4)
    history_data = df[(df['timestamp'] >= four_days_ago) & (df['timestamp'] < cutoff_time)]

    # Get sunset temperature
    idx_sunset = (df['timestamp'] - sunset_time).abs().idxmin()
    if abs((df.loc[idx_sunset, 'timestamp'] - sunset_time).total_seconds()) < 900:
        if len(day_data) > 10 and len(history_data) > 100:  # Need enough data
            dataset.append({
                'date': date,
                'day_of_month': date.day,
                'sunset_time': sunset_time,
                'cutoff_time': cutoff_time,
                'day_temps': day_data['temperature'].values,
                'day_times': day_data['timestamp'].values,
                'history_temps': history_data['temperature'].values,
                'history_times': history_data['timestamp'].values,
                'temp_at_sunset': df.loc[idx_sunset, 'temperature']
            })

print(f"Built dataset with {len(dataset)} days")

# Split into train (even) and test (odd)
train_data = [d for d in dataset if d['day_of_month'] % 2 == 0]
test_data = [d for d in dataset if d['day_of_month'] % 2 == 1]
print(f"Train: {len(train_data)} days, Test: {len(test_data)} days")

y_test = np.array([d['temp_at_sunset'] for d in test_data])

# ============================================================
# Method 1: Simple Mean Baseline
# ============================================================
print("\n=== Method 1: 24-Hour Mean Baseline ===")
predictions_mean = []
for d in test_data:
    # Use mean of last 24 hours
    recent = d['history_temps'][-96:]  # last 24 hours (96 points at 15-min intervals)
    pred = np.mean(recent) if len(recent) > 0 else np.mean(d['history_temps'])
    predictions_mean.append(pred)

y_pred_mean = np.array(predictions_mean)
mae_mean = mean_absolute_error(y_test, y_pred_mean)
rmse_mean = np.sqrt(mean_squared_error(y_test, y_pred_mean))
r2_mean = r2_score(y_test, y_pred_mean)
print(f"MAE:  {mae_mean:.4f} °C")
print(f"RMSE: {rmse_mean:.4f} °C")
print(f"R²:   {r2_mean:.4f}")

# ============================================================
# Method 2: Linear Trend Extrapolation
# ============================================================
print("\n=== Method 2: Linear Trend Extrapolation (last 6 hours) ===")
predictions_linear = []
for d in test_data:
    # Use last 6 hours (24 points)
    recent_temps = d['history_temps'][-24:]
    recent_times = d['history_times'][-24:]

    if len(recent_temps) >= 10:
        # Convert times to hours since first point
        t0 = pd.Timestamp(recent_times[0])
        times_hours = np.array([(pd.Timestamp(t) - t0).total_seconds() / 3600.0 for t in recent_times])

        # Fit linear trend
        coeffs = np.polyfit(times_hours, recent_temps, 1)

        # Extrapolate to sunset
        dt_to_sunset = (d['sunset_time'] - pd.Timestamp(recent_times[-1])).total_seconds() / 3600.0
        pred = coeffs[0] * (times_hours[-1] + dt_to_sunset) + coeffs[1]
    else:
        pred = recent_temps[-1] if len(recent_temps) > 0 else d['history_temps'][-1]

    predictions_linear.append(pred)

y_pred_linear = np.array(predictions_linear)
mae_linear = mean_absolute_error(y_test, y_pred_linear)
rmse_linear = np.sqrt(mean_squared_error(y_test, y_pred_linear))
r2_linear = r2_score(y_test, y_pred_linear)
print(f"MAE:  {mae_linear:.4f} °C")
print(f"RMSE: {rmse_linear:.4f} °C")
print(f"R²:   {r2_linear:.4f}")

# ============================================================
# Method 3: Fourier Series (prior 4 days)
# ============================================================
print("\n=== Method 3: Fourier Series Extrapolation (4 days) ===")

def fourier_model(t, *params):
    """Fourier series: a0 + sum(a_n*cos(n*w*t) + b_n*sin(n*w*t))"""
    n_harmonics = len(params) // 2
    result = params[0]  # DC component
    w = 2 * np.pi / 24.0  # 24-hour period
    for n in range(1, n_harmonics + 1):
        a_n = params[2*n - 1]
        b_n = params[2*n]
        result += a_n * np.cos(n * w * t) + b_n * np.sin(n * w * t)
    return result

predictions_fourier = []
n_harmonics = 3  # Use first 3 harmonics

for d in test_data:
    temps = d['history_temps']
    times = d['history_times']

    # Convert to hours since start
    t0 = pd.Timestamp(times[0])
    times_hours = np.array([(pd.Timestamp(t) - t0).total_seconds() / 3600.0 for t in times])

    # Fit Fourier series
    try:
        # Initial guess: DC + small harmonics
        p0 = [np.mean(temps)] + [0.5, 0.5] * n_harmonics
        popt, _ = curve_fit(fourier_model, times_hours, temps, p0=p0, maxfev=5000)

        # Extrapolate to sunset
        t_sunset = (d['sunset_time'] - t0).total_seconds() / 3600.0
        pred = fourier_model(t_sunset, *popt)
    except:
        # Fallback if fit fails
        pred = temps[-1]

    predictions_fourier.append(pred)

y_pred_fourier = np.array(predictions_fourier)
mae_fourier = mean_absolute_error(y_test, y_pred_fourier)
rmse_fourier = np.sqrt(mean_squared_error(y_test, y_pred_fourier))
r2_fourier = r2_score(y_test, y_pred_fourier)
print(f"MAE:  {mae_fourier:.4f} °C")
print(f"RMSE: {rmse_fourier:.4f} °C")
print(f"R²:   {r2_fourier:.4f}")

# ============================================================
# Method 4: XGBoost with Engineered Features
# ============================================================
print("\n=== Method 4: XGBoost with Engineered Features ===")

try:
    import xgboost as xgb

    def extract_features(data_dict):
        """Extract statistical features from time history"""
        temps = data_dict['history_temps']
        features = {
            'mean_24h': np.mean(temps[-96:]) if len(temps) >= 96 else np.mean(temps),
            'std_24h': np.std(temps[-96:]) if len(temps) >= 96 else np.std(temps),
            'min_24h': np.min(temps[-96:]) if len(temps) >= 96 else np.min(temps),
            'max_24h': np.max(temps[-96:]) if len(temps) >= 96 else np.max(temps),
            'current': temps[-1],
            'trend_6h': (temps[-1] - temps[-24]) if len(temps) >= 24 else 0,
            'mean_6h': np.mean(temps[-24:]) if len(temps) >= 24 else np.mean(temps),
        }
        return list(features.values())

    X_train = np.array([extract_features(d) for d in train_data])
    y_train = np.array([d['temp_at_sunset'] for d in train_data])
    X_test_xgb = np.array([extract_features(d) for d in test_data])

    model_xgb = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    model_xgb.fit(X_train, y_train)
    y_pred_xgb = model_xgb.predict(X_test_xgb)

    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    r2_xgb = r2_score(y_test, y_pred_xgb)
    print(f"MAE:  {mae_xgb:.4f} °C")
    print(f"RMSE: {rmse_xgb:.4f} °C")
    print(f"R²:   {r2_xgb:.4f}")
    has_xgb = True
except ImportError:
    print("XGBoost not installed, skipping")
    has_xgb = False

# ============================================================
# Method 5: NBEATSx-Ridge (windowed Ridge with time features)
# ============================================================
print("\n=== Method 5: NBEATSx-Ridge (Time Series Ridge) ===")

# Extract sliding window features + time-of-day for Ridge
def extract_nbeats_features(history_temps, history_times, sunset_time, window_size=12):
    """
    NBEATSx-inspired features: recent windows + time features
    window_size = number of recent measurements to use (12 = 3 hours at 15-min intervals)
    """
    recent = history_temps[-window_size:] if len(history_temps) >= window_size else history_temps

    # Pad if needed
    if len(recent) < window_size:
        recent = np.pad(recent, (window_size - len(recent), 0), mode='edge')

    # Time features
    sunset_hour = sunset_time.hour + sunset_time.minute / 60.0

    # Combine: recent temps + time-of-day + basic stats
    features = list(recent) + [
        sunset_hour,
        np.mean(recent),
        np.std(recent),
        recent[-1] - recent[0],  # trend over window
    ]
    return features

# Build training data
X_train_nbeats = []
y_train_nbeats = []
for d in train_data:
    feats = extract_nbeats_features(d['history_temps'], d['history_times'], d['sunset_time'])
    X_train_nbeats.append(feats)
    y_train_nbeats.append(d['temp_at_sunset'])

X_train_nbeats = np.array(X_train_nbeats)
y_train_nbeats = np.array(y_train_nbeats)

# Train Ridge model
from sklearn.linear_model import Ridge
ridge_nbeats = Ridge(alpha=10.0)
ridge_nbeats.fit(X_train_nbeats, y_train_nbeats)

# Predict on test
predictions_nbeats = []
for d in test_data:
    feats = extract_nbeats_features(d['history_temps'], d['history_times'], d['sunset_time'])
    pred = ridge_nbeats.predict([feats])[0]
    predictions_nbeats.append(pred)

y_pred_nbeats = np.array(predictions_nbeats)
mae_nbeats = mean_absolute_error(y_test, y_pred_nbeats)
rmse_nbeats = np.sqrt(mean_squared_error(y_test, y_pred_nbeats))
r2_nbeats = r2_score(y_test, y_pred_nbeats)
print(f"MAE:  {mae_nbeats:.4f} °C")
print(f"RMSE: {rmse_nbeats:.4f} °C")
print(f"R²:   {r2_nbeats:.4f}")
has_nbeats = True

# ============================================================
# Summary
# ============================================================
print("\n" + "="*70)
print("PERFORMANCE SUMMARY")
print("="*70)
print(f"{'Method':<35} {'MAE (°C)':<12} {'RMSE (°C)':<12} {'R²':<8}")
print("-"*70)
print(f"{'24-Hour Mean Baseline':<35} {mae_mean:<12.4f} {rmse_mean:<12.4f} {r2_mean:<8.4f}")
print(f"{'Linear Trend (6h)':<35} {mae_linear:<12.4f} {rmse_linear:<12.4f} {r2_linear:<8.4f}")
print(f"{'Fourier Series (4 days)':<35} {mae_fourier:<12.4f} {rmse_fourier:<12.4f} {r2_fourier:<8.4f}")
if has_xgb:
    print(f"{'XGBoost + Engineered Features':<35} {mae_xgb:<12.4f} {rmse_xgb:<12.4f} {r2_xgb:<8.4f}")
if has_nbeats:
    print(f"{'NBEATSx-Ridge':<35} {mae_nbeats:<12.4f} {rmse_nbeats:<12.4f} {r2_nbeats:<8.4f}")
print("="*70)

# Find best
models = {
    'Mean': mae_mean,
    'Linear': mae_linear,
    'Fourier': mae_fourier
}
if has_xgb:
    models['XGBoost'] = mae_xgb
if has_nbeats:
    models['NBEATSx-Ridge'] = mae_nbeats

best = min(models.items(), key=lambda x: x[1])
print(f"\nBest: {best[0]} (MAE = {best[1]:.4f} °C)\n")

# ============================================================
# Visualization
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Predicted vs Actual (Fourier)
ax = axes[0, 0]
ax.scatter(y_test, y_pred_fourier, alpha=0.5, s=25, color='purple')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel('Actual Sunset Temp (°C)')
ax.set_ylabel('Predicted (°C)')
ax.set_title(f'Fourier Series\nMAE = {mae_fourier:.3f} °C')
ax.grid(True, alpha=0.3)

# Panel 2: Predicted vs Actual (XGBoost if available, else Linear)
ax = axes[0, 1]
if has_xgb:
    ax.scatter(y_test, y_pred_xgb, alpha=0.5, s=25, color='green')
    ax.set_title(f'XGBoost\nMAE = {mae_xgb:.3f} °C')
else:
    ax.scatter(y_test, y_pred_linear, alpha=0.5, s=25, color='orange')
    ax.set_title(f'Linear Trend\nMAE = {mae_linear:.3f} °C')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel('Actual Sunset Temp (°C)')
ax.set_ylabel('Predicted (°C)')
ax.grid(True, alpha=0.3)

# Panel 3: Residuals (Fourier)
ax = axes[1, 0]
residuals = y_test - y_pred_fourier
ax.scatter(y_pred_fourier, residuals, alpha=0.5, s=25, color='purple')
ax.axhline(y=0, color='r', linestyle='--', lw=2)
ax.set_xlabel('Predicted Temp (°C)')
ax.set_ylabel('Residual (°C)')
ax.set_title('Fourier Series Residuals')
ax.grid(True, alpha=0.3)

# Panel 4: Model comparison
ax = axes[1, 1]
names = list(models.keys())
maes = list(models.values())
colors = ['gray', 'orange', 'purple', 'green', 'blue'][:len(names)]
bars = ax.bar(names, maes, color=colors, alpha=0.7)
ax.set_ylabel('MAE (°C)')
ax.set_title('Model Performance Comparison')
ax.set_ylim(0, max(maes) * 1.15)
ax.grid(True, alpha=0.3, axis='y')
for bar, mae_val in zip(bars, maes):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
            f'{mae_val:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('ml_comparison_v2.png', dpi=150, bbox_inches='tight')
print("Saved ml_comparison_v2.png")

# ============================================================
# Comprehensive Residual Analysis for Best Model
# ============================================================
from scipy import stats

# Use the best model's predictions
if has_nbeats:
    y_pred_best = y_pred_nbeats
    best_name = 'NBEATSx-Ridge'
    mae_best = mae_nbeats
elif has_xgb:
    y_pred_best = y_pred_xgb
    best_name = 'XGBoost'
    mae_best = mae_xgb
else:
    y_pred_best = y_pred_mean
    best_name = 'Mean Baseline'
    mae_best = mae_mean

residuals_best = y_test - y_pred_best
errors_best = np.abs(residuals_best)

fig2, axes = plt.subplots(2, 3, figsize=(18, 10))

# Panel 1: Histogram overlay - actual temps vs prediction errors
ax = axes[0, 0]
ax.hist(y_test, bins=30, alpha=0.6, label='Actual Sunset Temps', color='blue', edgecolor='black')
ax.hist(errors_best, bins=30, alpha=0.6, label='Prediction Errors', color='red', edgecolor='black')
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Count')
ax.set_title(f'Actual Temps vs Prediction Errors\n{best_name}')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: Residuals vs Predicted
ax = axes[0, 1]
ax.scatter(y_pred_best, residuals_best, alpha=0.5, s=20, color='blue')
ax.axhline(y=0, color='r', linestyle='--', lw=2)
ax.set_xlabel('Predicted Temperature (°C)')
ax.set_ylabel('Residual (°C)')
ax.set_title(f'Residuals vs Predicted\n{best_name}')
ax.grid(True, alpha=0.3)

# Panel 3: Residuals vs Actual
ax = axes[0, 2]
ax.scatter(y_test, residuals_best, alpha=0.5, s=20, color='green')
ax.axhline(y=0, color='r', linestyle='--', lw=2)
ax.set_xlabel('Actual Temperature (°C)')
ax.set_ylabel('Residual (°C)')
ax.set_title(f'Residuals vs Actual\n{best_name}')
ax.grid(True, alpha=0.3)

# Panel 4: Histogram of residuals with normal fit
ax = axes[1, 0]
ax.hist(residuals_best, bins=25, density=True, alpha=0.6, color='skyblue', edgecolor='black')
mu, sigma = residuals_best.mean(), residuals_best.std()
x = np.linspace(residuals_best.min(), residuals_best.max(), 100)
ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2, label=f'N({mu:.2f}, {sigma:.2f}²)')
ax.set_xlabel('Residual (°C)')
ax.set_ylabel('Density')
ax.set_title(f'Residual Distribution\n{best_name}')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 5: Q-Q plot
ax = axes[1, 1]
stats.probplot(residuals_best, dist="norm", plot=ax)
ax.set_title(f'Q-Q Plot\n{best_name}')
ax.grid(True, alpha=0.3)

# Panel 6: Residuals vs Time Order
ax = axes[1, 2]
test_indices = np.arange(len(residuals_best))
ax.scatter(test_indices, residuals_best, alpha=0.5, s=20, color='purple')
ax.axhline(y=0, color='r', linestyle='--', lw=2)
ax.set_xlabel('Test Sample Index')
ax.set_ylabel('Residual (°C)')
ax.set_title(f'Residuals vs Time Order\n{best_name}')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ml_residuals_v2.png', dpi=150, bbox_inches='tight')
print("Saved ml_residuals_v2.png")

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)
print(f"The {best[0]} method achieves the best performance (MAE = {best[1]:.4f} °C).")
print("\nKey insights:")
print("- Fourier series captures the diurnal cycle naturally")
print("- XGBoost with features is robust and fast")
print("- NBEATSx-Ridge uses windowed time series features effectively")
print("- Simple methods (mean, linear trend) provide useful baselines")
print("="*70)
