# DFT, Matrix-DFT and FFT Benchmark for Power-Quality Signals

Code and raw measurement data accompanying the paper *Comparative Computational Analysis of Naive, Vectorized, and Fast Fourier Transform Algorithms for Power Quality Signal Processing*.

## Implementations compared

| Label | Description |
|---|---|
| `loop` | Pure-Python double-loop DFT |
| `matrix_total` | DFT matrix `W` built on every call, then `W @ x` |
| `matrix_build` | Construction of `W` only |
| `matrix_pre` | `W` precomputed outside the timed region; `W @ x` only |
| `fft` | `numpy.fft.fft` |

## Requirements

- Python 3.13.3 and NumPy 2.5.1 (tested versions)
- `threadpoolctl` (optional; required for BLAS thread information and the single-thread pass)

```
pip install numpy threadpoolctl
```

## Data

The real-data experiment uses the SEED Power Quality Disturbance Dataset (Kaggle):
https://www.kaggle.com/datasets/sumairaziz/seed-power-quality-disturbance-dataset

Place the following four files in one folder:

```
Pure_Sinusoidal.csv
Harmonics.csv
Sag.csv
Sag_with_Harmonics.csv
```

Rows 0-9 of each file (no header, 100 samples per row) are concatenated to N = 1000 samples. One row spans one fundamental cycle.

## Usage

```
python benchmark_v2.py --out run1 --data-dir <folder_with_csv_files>
```

| Option | Description |
|---|---|
| `--out DIR` | Output folder (default: `results`) |
| `--data-dir DIR` | Folder containing the four SEED files (default: `.`) |
| `--skip-real` | Skip the real-data experiment |
| `--skip-single-thread` | Skip the one-BLAS-thread pass |
| `--max-n N` | Omit signal lengths above N |
| `--quick` | Three repeats only (smoke test) |

## Experimental settings

- Sampling frequency 6400 Hz (128 samples per 50 Hz cycle); noise seed 42.
- Signal lengths N = 64, 128, 256, 512, 640, 1024, 1280, 2560 (windows of 1, 2, 5, 10 and 20 cycles at N = 128, 256, 640, 1280, 2560).
- Signal-condition test at N = 512: pure sinusoid, harmonics, harmonics with noise, voltage sag with harmonics. Real data at N = 1000.
- Timing: two warm-up calls; timed blocks calibrated to at least 20 ms; garbage collection disabled; 30 blocks per measurement (`loop`: 30 for N <= 256, 10 for N <= 1024, 5 above); median and interquartile range per measurement.
- Two BLAS configurations: default thread count (`default`) and one thread (`blas_1thread`). `loop` does not use BLAS and is measured in the default configuration only.
- Real-time deadline: the duration of the analysis window, N / f_s.
- The benchmark was executed three times (`run1`, `run2`, `run3`). The paper reports, for each measurement, the median of the three per-run medians.

## Output files (per run)

| File | Content |
|---|---|
| `raw_timings.csv` | All measurements: configuration, experiment, condition, N, implementation, repeats, inner loop count, median, Q1, Q3, mean, std, min, max (seconds) |
| `summary_scalability.csv` | Per N: median times, speedup ratios, size of `W` in bytes, `tracemalloc` peak while building `W` |
| `realtime.csv` | Per window: computation time and window duration, feasibility flags (median and worst case) |
| `accuracy.csv` | Maximum absolute error and relative L2 error against `numpy.fft.fft` |
| `env.json` | Operating system, CPU, Python, NumPy, BLAS library and thread information, settings, command-line arguments |

## Repository contents

```
benchmark_v2.py
results/run1/    results/run2/    results/run3/
```

The `results` folders contain the raw output of the three runs reported in the paper (Intel Core i5-12450HX, Windows 11, Python 3.13.3, NumPy 2.5.1, OpenBLAS 0.3.33, 12 threads). Absolute timings depend on the machine, BLAS build and thread count; ratios and trends are the reproducible quantities.
