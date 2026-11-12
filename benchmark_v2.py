#!/usr/bin/env python3
"""
benchmark_v2.py -- reviewer-proof benchmark for loop-DFT / matrix-DFT / FFT.

Addresses ICRPSET2026 reviewer comments:
  * warm-up runs, GC off, median/IQR over many repeats, batched timing
  * DFT-matrix construction timed SEPARATELY from the matrix-vector product
      matrix_build  = construct W only
      matrix_pre    = W precomputed, time only W @ x   (realistic fixed-N monitor)
      matrix_total  = construct W + W @ x             (what a naive call costs)
  * memory of W (bytes) and tracemalloc peak during construction
  * BLAS config + thread info logged; second pass with BLAS threads = 1
  * per-window real-time deadline (window duration = deadline), plus 200 ms column
  * fixed seed, explicit signal definitions (fs = 6400 Hz for ALL experiments)
  * environment dump (CPU, OS, Python, NumPy, BLAS)

Usage:
  pip install numpy threadpoolctl
  python benchmark_v2.py --out results --data-dir <folder with the 4 real CSVs>
  python benchmark_v2.py --quick --max-n 256 --skip-real --out test   # smoke test
"""
import argparse, csv, contextlib, gc, io, json, os, platform, sys, time, tracemalloc
import numpy as np

try:
    from threadpoolctl import threadpool_info, threadpool_limits
except ImportError:
    threadpool_info = None
    threadpool_limits = None

F_GRID = 50          # Hz
FS = 6400            # Hz  (128 samples / cycle)  -- used for every experiment
SEED = 42
NS = [64, 128, 256, 512, 640, 1024, 1280, 2560]   # union of scalability + RT windows
CYCLES = [1, 2, 5, 10, 20]
REAL_FILES = {
    'Pure Sinusoidal': 'Pure_Sinusoidal.csv',
    'Harmonics': 'Harmonics.csv',
    'Sag': 'Sag.csv',
    'Sag+Harmonics': 'Sag_with_Harmonics.csv',
}
REAL_ROWS = 10       # rows 0..9 (0-based) x 100 columns -> N = 1000


# ----------------------------------------------------------------- algorithms
def loop_dft(x):
    """Pure-Python double loop (same as the paper's original implementation)."""
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X


def build_W(N):
    """DFT matrix -- identical to the paper's original matrix_dft construction."""
    n = np.arange(N)
    k = n.reshape((N, 1))
    return np.exp(-2j * np.pi * k * n / N)


def matrix_total(x):
    return build_W(len(x)) @ x


# -------------------------------------------------------------------- signals
def make_signal(kind, N, rng):
    t = np.arange(N) / FS
    s50 = np.sin(2 * np.pi * 50 * t)
    s150 = 0.15 * np.sin(2 * np.pi * 150 * t)
    s250 = 0.08 * np.sin(2 * np.pi * 250 * t)
    if kind == 'pure':
        return s50
    if kind == 'harmonics':
        return s50 + s150 + s250
    if kind == 'harm_noise':
        return s50 + s150 + s250 + 0.05 * rng.standard_normal(N)
    if kind == 'sag_harm':
        frac = np.arange(N) / N                      # position inside the record
        sag = (frac >= 0.3) & (frac <= 0.7)
        return np.where(sag, 0.5 * s50 + s150, s50 + s150)
    raise ValueError(kind)


# --------------------------------------------------------------------- timing
def time_fn(fn, x, repeats, warmup, min_block=0.02):
    """Batched timing. Returns (stats dict, last output)."""
    gc.collect()
    gc.disable()
    try:
        for _ in range(warmup):
            fn(x)
        inner = 1                                    # calibrate: block >= min_block
        while True:
            t0 = time.perf_counter()
            for _ in range(inner):
                fn(x)
            if time.perf_counter() - t0 >= min_block or inner >= (1 << 20):
                break
            inner *= 2
        samples, out = [], None
        for _ in range(repeats):
            t0 = time.perf_counter()
            for _ in range(inner):
                out = fn(x)
            samples.append((time.perf_counter() - t0) / inner)
    finally:
        gc.enable()
    a = np.array(samples)
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    return dict(repeats=repeats, inner=inner, median_s=med, q1_s=q1, q3_s=q3,
                mean_s=a.mean(), std_s=a.std(ddof=1) if len(a) > 1 else 0.0,
                min_s=a.min(), max_s=a.max()), out


def repeats_for(name, N, quick):
    if name == 'loop':
        r = 30 if N <= 256 else (10 if N <= 1024 else 5)
    else:
        r = 30
    return min(r, 3) if quick else r


def mem_profile(N):
    x = np.zeros(N)
    tracemalloc.start()
    W = build_W(N)
    _ = W @ x
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return W.nbytes, peak


def measure(cfg, exp, cond, x, include_loop, quick, rows, acc):
    N = len(x)
    W = build_W(N)
    impls = {}
    if include_loop:
        impls['loop'] = (loop_dft, 0)
    impls['matrix_total'] = (matrix_total, 2)
    impls['matrix_build'] = (lambda v: build_W(len(v)), 2)
    impls['matrix_pre'] = (lambda v: W @ v, 2)
    impls['fft'] = (np.fft.fft, 2)
    stats, outs = {}, {}
    for name, (fn, wu) in impls.items():
        s, o = time_fn(fn, x, repeats_for(name, N, quick), wu)
        stats[name] = s
        if name != 'matrix_build':
            outs[name] = o
        rows.append(dict(config=cfg, experiment=exp, condition=cond, N=N, impl=name, **s))
        print(f"[{cfg}] {exp}/{cond} N={N:5d} {name:13s} median={s['median_s']:.3e}s "
              f"(IQR {s['q1_s']:.2e}-{s['q3_s']:.2e}, inner={s['inner']}, r={s['repeats']})", flush=True)
    ref = outs['fft']
    for name in ('loop', 'matrix_total', 'matrix_pre'):
        if name in outs:
            d = outs[name] - ref
            acc.append(dict(config=cfg, experiment=exp, condition=cond, N=N, impl=name,
                            E_abs_max=float(np.max(np.abs(d))),
                            E_rel_l2=float(np.linalg.norm(d) / np.linalg.norm(ref))))
    return stats


# ---------------------------------------------------------------------- misc
def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def env_info(args):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            np.show_config()
        except Exception as e:
            print('show_config failed:', e)
    info = dict(
        platform=platform.platform(), machine=platform.machine(),
        processor=platform.processor(), cpu_count_logical=os.cpu_count(),
        python=sys.version, numpy=np.__version__,
        threadpool_info=threadpool_info() if threadpool_info else 'threadpoolctl not installed',
        numpy_show_config=buf.getvalue(),
        env_threads={k: os.environ.get(k) for k in
                     ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
        settings=dict(F_GRID=F_GRID, FS=FS, SEED=SEED, NS=NS, CYCLES=CYCLES,
                      real_rows=f'0..{REAL_ROWS - 1} (0-based, header=None)', args=vars(args)),
    )
    return info


def run_suite(cfg, limit, args, rows, acc, summ, rt, mem):
    include_loop = (cfg == 'default')                # loop-DFT does not use BLAS
    ctx = (threadpool_limits(limits=limit, user_api='blas')
           if (limit and threadpool_limits) else contextlib.nullcontext())
    with ctx:
        rng = np.random.default_rng(SEED)
        Ns = [n for n in NS if n <= args.max_n]
        # 1) scalability + real-time windows (harmonic signal)
        for N in Ns:
            x = make_signal('harmonics', N, rng)
            st = measure(cfg, 'sweep', 'harmonics', x, include_loop, args.quick, rows, acc)
            if cfg == 'default':
                mem[N] = mem_profile(N)
            T = {k: v['median_s'] for k, v in st.items()}
            r = dict(config=cfg, N=N, **{f'T_{k}': v for k, v in T.items()})
            if 'loop' in T:
                r['S_loop/fft'] = T['loop'] / T['fft']
                r['S_loop/matrix_pre'] = T['loop'] / T['matrix_pre']
            r['S_matrix_total/fft'] = T['matrix_total'] / T['fft']
            r['S_matrix_pre/fft'] = T['matrix_pre'] / T['fft']
            r['W_bytes'] = N * N * 16
            summ.append(r)
            # real-time table: deadline = window duration
            if (N * F_GRID) % FS == 0 and N * F_GRID // FS in CYCLES:
                win = N / FS
                for name, s in st.items():
                    if name == 'matrix_build':
                        continue
                    rt.append(dict(config=cfg, cycles=N * F_GRID // FS, N=N, impl=name,
                                   window_ms=win * 1e3, median_ms=s['median_s'] * 1e3,
                                   max_ms=s['max_s'] * 1e3,
                                   ratio_median_over_window=s['median_s'] / win,
                                   feasible_vs_window=s['median_s'] <= win,
                                   feasible_vs_window_worstcase=s['max_s'] <= win,
                                   within_fixed_200ms=s['median_s'] <= 0.2))
        # 2) signal conditions at N = 512
        if 512 <= args.max_n:
            for kind in ('pure', 'harmonics', 'harm_noise', 'sag_harm'):
                x = make_signal(kind, 512, np.random.default_rng(SEED))
                measure(cfg, 'signals', kind, x, include_loop, args.quick, rows, acc)
        # 3) real data
        if not args.skip_real:
            for name, fname in REAL_FILES.items():
                p = os.path.join(args.data_dir, fname)
                if not os.path.exists(p):
                    print('MISSING real file, skipped:', p)
                    continue
                x = np.loadtxt(p, delimiter=',', max_rows=REAL_ROWS).flatten().astype(float)
                measure(cfg, 'real', name, x, include_loop, args.quick, rows, acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results')
    ap.add_argument('--data-dir', default='.')
    ap.add_argument('--quick', action='store_true', help='3 repeats (smoke test only)')
    ap.add_argument('--max-n', type=int, default=10 ** 9)
    ap.add_argument('--skip-real', action='store_true')
    ap.add_argument('--skip-single-thread', action='store_true')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if threadpool_limits is None:
        print('WARNING: threadpoolctl missing -> BLAS thread info/limit unavailable '
              '(pip install threadpoolctl)')

    rows, acc, summ, rt, mem = [], [], [], [], {}
    configs = [('default', None)]
    if threadpool_limits and not args.skip_single_thread:
        configs.append(('blas_1thread', 1))
    for cfg, lim in configs:
        run_suite(cfg, lim, args, rows, acc, summ, rt, mem)

    for r in summ:
        if r['N'] in mem:
            r['W_tracemalloc_peak_bytes'] = mem[r['N']][1]
    write_csv(os.path.join(args.out, 'raw_timings.csv'), rows)
    write_csv(os.path.join(args.out, 'accuracy.csv'), acc)
    write_csv(os.path.join(args.out, 'summary_scalability.csv'), summ)
    write_csv(os.path.join(args.out, 'realtime.csv'), rt)
    with open(os.path.join(args.out, 'env.json'), 'w') as f:
        json.dump(env_info(args), f, indent=2, default=str)
    print('\nDone. Files in', args.out)


if __name__ == '__main__':
    main()
