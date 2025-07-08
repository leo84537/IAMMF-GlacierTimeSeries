# IAMMF-GlacierTimeSeries

**IAMMF-GlacierTimeSeries** is a Python-based time series pipeline purpose-built for analyzing satellite-derived glacier albedo signals. It applies an optimized Iterative Adaptive Moving Median Filter (IAMMF) to MODIS NDSI time series to enable robust smoothing and intelligent gap-filling, even in the presence of noisy or sparse data.

### Key Features
- 🚀 **Accelerated Filtering**: Numba-compiled algorithm for fast, scalable processing
- 🧠 **Adaptive Smoothing**: Dynamically adjusts window sizes using asymmetric constraints and statistical convergence
- 📉 **Signal Denoising**: Reduces noise and outliers using MAD-based convergence criteria
- 🛰 **Remote Sensing Ready**: Handles irregular sampling, cloud interference, and seasonal glacier behavior
- 🛠 **Extensible**: Modular, maintainable codebase for researchers and engineers

### Technical Stack
- Python (NumPy, Pandas, Numba)
- Time Series Analysis
- Statistical Signal Processing
- Remote Sensing (MODIS Albedo / NDSI)
- Data Imputation + Noise Reduction

---

