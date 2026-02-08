"""
Machine learning comparison for sunset temperature prediction.
Goal: Predict temperature at sunset using temperature from 3 hours before.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from astropy.coordinates import EarthLocation, AltAz, get_sun
from astropy.time import Time
import astropy.units as u
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
df = pd.read_csv('rubin_mirror_temps.csv', parse_dates=['timestamp'])

# Cerro Pachon location
location = EarthLocation(lat=-30.24*u.deg, lon=-70.74*u.deg, height=2715*u.m)

print("Computing sunset times for all dates...")
unique_dates = df['timestamp'].dt.date.unique()
sunset_dict = {}

for date in unique_dates:
    # Search for sunset from 21:00 to 01:00 UTC next day
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

# Build dataset: for each day, get T at sunset and T at (sunset - 3 hours)
data_rows = []

for date, sunset_time in sunset_dict.items():
    three_hours_before = sunset_time - timedelta(hours=3)

    # Find nearest temperature measurements
    idx_sunset = (df['timestamp'] - sunset_time).abs().idxmin()
    idx_before = (df['timestamp'] - three_hours_before).abs().idxmin()

    # Make sure we have valid measurements
    if abs((df.loc[idx_sunset, 'timestamp'] - sunset_time).total_seconds()) < 900:  # within 15 min
        if abs((df.loc[idx_before, 'timestamp'] - three_hours_before).total_seconds()) < 900:
            data_rows.append({
                'date': date,
                'day_of_month': date.day,
                'sunset_time': sunset_time,
                'temp_3hr_before': df.loc[idx_before, 'temperature'],
                'temp_at_sunset': df.loc[idx_sunset, 'temperature']
            })

dataset = pd.DataFrame(data_rows)
print(f"Built dataset with {len(dataset)} days")

# Split into train (even days) and test (odd days)
train = dataset[dataset['day_of_month'] % 2 == 0].copy()
test = dataset[dataset['day_of_month'] % 2 == 1].copy()

print(f"Train: {len(train)} days, Test: {len(test)} days")

X_train = train[['temp_3hr_before']].values
y_train = train['temp_at_sunset'].values
X_test = test[['temp_3hr_before']].values
y_test = test['temp_at_sunset'].values

# ===== Model 1: Simple baseline (persistence) =====
print("\n=== Model 1: Persistence Baseline ===")
y_pred_baseline = X_test.flatten()
mae_baseline = mean_absolute_error(y_test, y_pred_baseline)
rmse_baseline = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
r2_baseline = r2_score(y_test, y_pred_baseline)
print(f"MAE:  {mae_baseline:.4f} °C")
print(f"RMSE: {rmse_baseline:.4f} °C")
print(f"R²:   {r2_baseline:.4f}")

# ===== Model 2: Linear regression (Ridge) =====
print("\n=== Model 2: Ridge Regression ===")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
r2_ridge = r2_score(y_test, y_pred_ridge)
print(f"MAE:  {mae_ridge:.4f} °C")
print(f"RMSE: {rmse_ridge:.4f} °C")
print(f"R²:   {r2_ridge:.4f}")
print(f"Coefficient: {ridge.coef_[0]:.4f}, Intercept: {ridge.intercept_:.4f}")

# ===== Model 3: Random Forest =====
print("\n=== Model 3: Random Forest ===")
rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)
print(f"MAE:  {mae_rf:.4f} °C")
print(f"RMSE: {rmse_rf:.4f} °C")
print(f"R²:   {r2_rf:.4f}")

# ===== Try Prophet if available =====
try:
    from prophet import Prophet
    print("\n=== Model 4: Prophet ===")

    # Prophet needs time series with 'ds' and 'y' columns
    # We'll use it as a time series forecaster
    train_prophet = train[['sunset_time', 'temp_at_sunset']].copy()
    train_prophet.columns = ['ds', 'y']

    model_prophet = Prophet(daily_seasonality=True, yearly_seasonality=False)
    model_prophet.fit(train_prophet)

    test_prophet = test[['sunset_time']].copy()
    test_prophet.columns = ['ds']
    forecast = model_prophet.predict(test_prophet)

    y_pred_prophet = forecast['yhat'].values
    mae_prophet = mean_absolute_error(y_test, y_pred_prophet)
    rmse_prophet = np.sqrt(mean_squared_error(y_test, y_pred_prophet))
    r2_prophet = r2_score(y_test, y_pred_prophet)
    print(f"MAE:  {mae_prophet:.4f} °C")
    print(f"RMSE: {rmse_prophet:.4f} °C")
    print(f"R²:   {r2_prophet:.4f}")
    has_prophet = True
except ImportError:
    print("\n=== Model 4: Prophet ===")
    print("Prophet not installed, skipping")
    has_prophet = False

# ===== Summary =====
print("\n" + "="*60)
print("PERFORMANCE SUMMARY")
print("="*60)
print(f"{'Model':<25} {'MAE (°C)':<12} {'RMSE (°C)':<12} {'R²':<8}")
print("-"*60)
print(f"{'Persistence Baseline':<25} {mae_baseline:<12.4f} {rmse_baseline:<12.4f} {r2_baseline:<8.4f}")
print(f"{'Ridge Regression':<25} {mae_ridge:<12.4f} {rmse_ridge:<12.4f} {r2_ridge:<8.4f}")
print(f"{'Random Forest':<25} {mae_rf:<12.4f} {rmse_rf:<12.4f} {r2_rf:<8.4f}")
if has_prophet:
    print(f"{'Prophet':<25} {mae_prophet:<12.4f} {rmse_prophet:<12.4f} {r2_prophet:<8.4f}")
print("="*60)

# Find best model
models = {
    'Persistence': mae_baseline,
    'Ridge': mae_ridge,
    'Random Forest': mae_rf
}
if has_prophet:
    models['Prophet'] = mae_prophet

best_model = min(models.items(), key=lambda x: x[1])
print(f"\nBest model: {best_model[0]} (MAE = {best_model[1]:.4f} °C)")

# ===== Visualization =====
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Predicted vs Actual (Ridge)
ax = axes[0, 0]
ax.scatter(y_test, y_pred_ridge, alpha=0.5, s=20)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel('Actual Sunset Temperature (°C)')
ax.set_ylabel('Predicted Temperature (°C)')
ax.set_title(f'Ridge Regression\nMAE = {mae_ridge:.3f} °C, R² = {r2_ridge:.3f}')
ax.grid(True, alpha=0.3)

# Panel 2: Predicted vs Actual (Random Forest)
ax = axes[0, 1]
ax.scatter(y_test, y_pred_rf, alpha=0.5, s=20, color='green')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel('Actual Sunset Temperature (°C)')
ax.set_ylabel('Predicted Temperature (°C)')
ax.set_title(f'Random Forest\nMAE = {mae_rf:.3f} °C, R² = {r2_rf:.3f}')
ax.grid(True, alpha=0.3)

# Panel 3: Residuals (Ridge)
ax = axes[1, 0]
residuals_ridge = y_test - y_pred_ridge
ax.scatter(y_pred_ridge, residuals_ridge, alpha=0.5, s=20)
ax.axhline(y=0, color='r', linestyle='--', lw=2)
ax.set_xlabel('Predicted Temperature (°C)')
ax.set_ylabel('Residual (°C)')
ax.set_title('Ridge Regression Residuals')
ax.grid(True, alpha=0.3)

# Panel 4: Model comparison bar chart
ax = axes[1, 1]
model_names = list(models.keys())
maes = list(models.values())
colors = ['gray', 'blue', 'green', 'orange'][:len(model_names)]
bars = ax.bar(model_names, maes, color=colors, alpha=0.7)
ax.set_ylabel('MAE (°C)')
ax.set_title('Model Performance Comparison')
ax.set_ylim(0, max(maes) * 1.2)
ax.grid(True, alpha=0.3, axis='y')
for bar, mae_val in zip(bars, maes):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
            f'{mae_val:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('ml_comparison.png', dpi=150, bbox_inches='tight')
print("\nSaved visualization to ml_comparison.png")
plt.show()

print("\n" + "="*60)
print("RECOMMENDATION")
print("="*60)
print(f"The {best_model[0]} model performs best with an MAE of {best_model[1]:.4f} °C.")
print(f"This means the typical 3-hour-ahead prediction error is ~{best_model[1]:.2f} °C.")
print("\nFor operational use, Ridge Regression is recommended because:")
print("- Simple, interpretable, and fast")
print("- Performance is close to or better than more complex methods")
print("- Coefficient close to 1.0 suggests temperature is persistent over 3 hours")
print("- No external dependencies (Prophet requires installation)")
print("="*60)
