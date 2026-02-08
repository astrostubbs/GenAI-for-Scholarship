"""Generate an HTML summary report for the ML sunset prediction comparison."""

import base64
from datetime import datetime

def image_to_base64(filepath):
    with open(filepath, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

# Performance metrics from the analysis
results = [
    {'method': '24-Hour Mean Baseline', 'mae': 2.0273, 'rmse': 2.6791, 'r2': 0.6852},
    {'method': 'Linear Trend (6h)', 'mae': 3.8790, 'rmse': 4.1906, 'r2': 0.2297},
    {'method': 'Fourier Series (4 days)', 'mae': 3.6032, 'rmse': 4.6689, 'r2': 0.0439},
    {'method': 'XGBoost + Engineered Features', 'mae': 1.1449, 'rmse': 1.5062, 'r2': 0.9005},
    {'method': 'NBEATSx-Ridge', 'mae': 0.8945, 'rmse': 1.1222, 'r2': 0.9448},
]

# Embed images
img1_b64 = image_to_base64('ml_comparison_v2.png')
img2_b64 = image_to_base64('ml_residuals_v2.png')

now = datetime.now().strftime('%Y-%m-%d %H:%M')

# Build performance table rows
table_rows = '\n'.join([
    f'<tr><td>{r["method"]}</td><td>{r["mae"]:.4f}</td><td>{r["rmse"]:.4f}</td><td>{r["r2"]:.4f}</td></tr>'
    for r in results
])

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Machine Learning for Sunset Temperature Prediction</title>
<style>
body {{
    font-family: Georgia, serif;
    max-width: 1000px;
    margin: 0 auto;
    padding: 2rem;
    line-height: 1.6;
    color: #1e1e1e;
}}
h1 {{
    color: #A51C30;
    border-bottom: 3px solid #A51C30;
    padding-bottom: 0.5rem;
}}
h2 {{
    color: #8C1515;
    margin-top: 2rem;
}}
h3 {{
    color: #333;
    margin-top: 1.5rem;
}}
img {{
    max-width: 100%;
    border: 1px solid #ddd;
    margin: 1rem 0;
}}
table {{
    border-collapse: collapse;
    width: 100%;
    margin: 1rem 0;
}}
th, td {{
    border: 1px solid #ddd;
    padding: 0.5rem 0.8rem;
    text-align: left;
}}
th {{
    background: #f3f4f4;
    font-weight: bold;
}}
.highlight {{
    background: #ffffcc;
}}
.meta {{
    color: #555;
    font-size: 0.9rem;
}}
code {{
    font-family: Menlo, monospace;
    font-size: 0.9em;
    background: #f3f4f4;
    padding: 0.15em 0.4em;
    border-radius: 3px;
}}
.recommendation {{
    background: #f0f8ff;
    border-left: 4px solid #A51C30;
    padding: 1rem;
    margin: 1rem 0;
}}
</style>
</head>
<body>

<h1>Machine Learning for Sunset Temperature Prediction</h1>
<p class="meta">Generated: {now}<br>
Dataset: Rubin Observatory primary mirror temperature, Jun–Dec 2025<br>
Prediction task: Forecast temperature at sunset using full time history up to 3 hours before</p>

<h2>1. Introduction</h2>
<p>
Predicting the temperature of the Rubin Observatory primary mirror at sunset is critical for
operational planning. Thermal gradients across the mirror surface can degrade image quality,
so knowing the sunset temperature 3 hours in advance allows time to implement thermal
control strategies.
</p>

<p>
This analysis compares five machine learning approaches for sunset temperature forecasting,
using the complete time history of temperature measurements from the start of each day up to
3 hours before sunset. The training data (even days of the month) comprises 90 days, and
the test data (odd days) comprises 92 days.
</p>

<h2>2. Methods Tested</h2>

<h3>Method 1: 24-Hour Mean Baseline</h3>
<p>
Simple baseline that predicts sunset temperature as the mean of the last 24 hours of measurements.
Provides a naive persistence forecast assuming recent thermal behavior continues.
</p>

<h3>Method 2: Linear Trend Extrapolation (6 hours)</h3>
<p>
Fits a linear regression to the last 6 hours of temperature data and extrapolates forward to
the sunset time. Captures short-term trends but sensitive to noise and may extrapolate poorly
beyond the fitting window.
</p>

<h3>Method 3: Fourier Series (4 days)</h3>
<p>
Models the temperature as a Fourier series with 24-hour fundamental period and 3 harmonics,
fitting to the prior 4 days of data. Captures the diurnal temperature cycle naturally but
struggles with transient weather events and extrapolation instability.
</p>

<h3>Method 4: XGBoost with Engineered Features</h3>
<p>
Gradient-boosted decision tree ensemble trained on engineered statistical features:
mean, std, min, max, current value, 6-hour trend, and 6-hour mean. Robust to noise
and non-linear relationships, with good generalization via regularization.
</p>

<h3>Method 5: NBEATSx-Ridge (Windowed Ridge Regression)</h3>
<p>
Inspired by the NBEATSx neural architecture, this approach uses a sliding window of the last
12 measurements (3 hours) as input features, plus time-of-day and basic statistics. A Ridge
regression model (L2 regularization, alpha=10.0) learns the mapping from windowed history
to sunset temperature. Fast inference, interpretable linear model.
</p>

<h2>3. Results</h2>

<h3>Performance Comparison</h3>
<img src="data:image/png;base64,{img1_b64}" alt="Model performance comparison">

<table>
<tr>
    <th>Method</th>
    <th>MAE (°C)</th>
    <th>RMSE (°C)</th>
    <th>R²</th>
</tr>
{table_rows}
</table>

<h3>Residual Analysis</h3>
<p>
Comprehensive diagnostics for the best-performing model (NBEATSx-Ridge):
</p>
<img src="data:image/png;base64,{img2_b64}" alt="Residual analysis">

<p><strong>Residual plot interpretation:</strong></p>
<ul>
<li><strong>Actual Temps vs Prediction Errors:</strong> Prediction errors are much smaller
than the range of actual temperatures, indicating good predictive skill.</li>

<li><strong>Residuals vs Predicted:</strong> Residuals are randomly scattered around zero
with no clear trend, suggesting the model is unbiased and has captured the relationship well.</li>

<li><strong>Residuals vs Actual:</strong> No systematic pattern, confirming the model performs
equally well across the temperature range.</li>

<li><strong>Residual Distribution:</strong> Approximately normal distribution (slight positive
skew), consistent with well-behaved prediction errors.</li>

<li><strong>Q-Q Plot:</strong> Residuals follow the theoretical normal distribution closely,
with slight deviations in the tails indicating occasional larger errors.</li>

<li><strong>Residuals vs Time Order:</strong> No temporal autocorrelation or drift,
suggesting the model is stable and doesn't have systematic time-dependent biases.</li>
</ul>

<h2>4. Recommendation</h2>

<div class="recommendation">
<p><strong>Recommended Method: NBEATSx-Ridge</strong></p>

<p>
The NBEATSx-Ridge windowed regression achieves the best performance (MAE = 0.89°C, R² = 0.94)
and is recommended for operational use. Key advantages:
</p>

<ul>
<li><strong>Best accuracy:</strong> 22% lower MAE than XGBoost, the next-best method</li>
<li><strong>Fast inference:</strong> Single linear model, no iterative optimization</li>
<li><strong>Interpretable:</strong> Linear weights show importance of each time lag</li>
<li><strong>Robust:</strong> Residuals show no systematic bias or patterns</li>
<li><strong>Simple deployment:</strong> Minimal dependencies (NumPy, scikit-learn)</li>
</ul>

<p>
The typical 3-hour-ahead forecast error is less than 1°C, providing operationally useful
predictions for thermal control planning.
</p>
</div>

<h2>5. Key Insights</h2>

<ul>
<li>Time series methods benefit from leveraging the full temporal history, not just a single
point 3 hours before sunset.</li>

<li>Windowed features capture local patterns effectively without requiring explicit periodic
models (like Fourier series).</li>

<li>XGBoost and NBEATSx-Ridge both outperform traditional time series methods (mean, trend,
Fourier), highlighting the value of engineered features.</li>

<li>Linear extrapolation performs poorly due to short-term noise amplification when
extrapolating 3 hours ahead.</li>

<li>Fourier series struggles because transient weather patterns break the assumption of
strict periodicity.</li>
</ul>

<h2>6. Data Provenance</h2>
<ul>
<li>Source: Vera C. Rubin Observatory primary mirror thermal telemetry</li>
<li>Period: June 1 – December 1, 2025</li>
<li>Location: Cerro Pachon, Chile (lat -30.24°, lon -70.74°, elev 2715 m)</li>
<li>Sampling: 15-minute intervals</li>
<li>Training: Even days of month (90 days)</li>
<li>Testing: Odd days of month (92 days)</li>
<li>Code: <a href="ml_sunset_comparison_v2.py">ml_sunset_comparison_v2.py</a></li>
</ul>

</body>
</html>
"""

with open('ml_summary.html', 'w') as f:
    f.write(html)

print("Generated ml_summary.html")
