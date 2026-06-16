<p align="center">
  <img src="docs/images/dashboard_icon.png" width="140" alt="Trading dashboard icon" />
</p>

<h1 align="center">High-Frequency Trading Simulator</h1>

<p align="center">
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.10%2B-3776ab.svg?logo=python&logoColor=white" alt="Python 3.10+" />
  </a>
  <a href="https://cmake.org/">
    <img src="https://img.shields.io/badge/CMake-3.15%2B-064F8C.svg?logo=cmake&logoColor=white" alt="CMake 3.15+" />
  </a>
  <a href="https://streamlit.io/">
    <img src="https://img.shields.io/badge/Streamlit-App-ff4b4b.svg?logo=streamlit&logoColor=white" alt="Streamlit app" />
  </a>
  <a href="docs/USAGE.md#execute-tests">
    <img src="https://img.shields.io/badge/tests-Catch2-brightgreen.svg?logo=github" alt="Catch2 tests" />
  </a>
</p>

<p align="center">
  <a href="https://high-frequency-trading-simulator-5yzi7yyjfvmckxa6ovjcje.streamlit.app/">
    <img src="https://img.shields.io/badge/Open%20in-Streamlit-1f6feb?logo=streamlit&logoColor=white" alt="Open Streamlit app" />
  </a>
  <a href="docs/USAGE.md">
    <img src="https://img.shields.io/badge/Docs-Usage%20Guide-0a84ff.svg" alt="Usage guide" />
  </a>
</p>

A practical sandbox for market microstructure research. Explore how clustered order flow emerges from Hawkes processes, prototype execution logic on a deterministic C++ limit order book, and surface results through notebooks, scripts, and a guided Streamlit front end.

## At a Glance
- **Deterministic order book core** – Modern C++20 engine with price-time priority kept intentionally readable for experimentation.
- **Shared Hawkes kernels** – Exponential and power-law intensity implementations exposed to both C++ and Python.
- **Analytics & visualization** – Python package with thinning simulators, diagnostics, plots, and export utilities.
- **Deterministic backtester** – C++ order book bridged into Python for reproducible order/fill replays and structured metrics.
- **Interactive Streamlit app** – Visualise timelines, compare kernels, and download simulated order flow without touching a compiler.
- **Reproducible artefacts** – Synthetic datasets, plots, and experiment outputs tracked under `data/` and `docs/`.

## Table of Contents
- [Quick Start](#quick-start)
- [Architecture & Data Flow](#architecture--data-flow)
- [Illustrated Analytics](#illustrated-analytics)
- [Interactive Streamlit App](#interactive-streamlit-app)
- [Documentation](#documentation)
- [Research Benchmarks](#research-benchmarks)
- [Repository Layout](#repository-layout)
- [Working with Data & Plots](#working-with-data--plots)
- [Theory Snapshot](#theory-snapshot)
- [Example Outputs](#example-outputs)
- [Development Notes](#development-notes)
- [Roadmap Ideas](#roadmap-ideas)

## Quick Start
Need more context? The step-by-step guide in `docs/USAGE.md` covers the full workflow end to end.

## Architecture & Data Flow

<p align="center">
  <img src="docs/images/timeline_exponential.png" width="720" alt="Event timeline illustration" />
</p>

The simulator stitches together four stages, mirroring the reference architecture described by Cartea et al. (2015) and Gatheral & Schied (2013):

1. **Hawkes-driven order flow** – exponential/power-law kernels generate clustered market/limit-order timing scenarios. Timeline plots (above) illustrate self-excitation during liquidity shocks.
2. **Deterministic matching engine** – the C++20 order book enforces price-time priority and stores resting orders in intrusive FIFO lists at each price level.
3. **Risk, PnL, and backtesting services** – Python orchestrators replay fills, compute realised/unrealised PnL, and stream metrics to dashboards.
4. **Visualization & research surfaces** – notebooks and Streamlit panels expose the same artefacts for exploratory analysis or reporting.

### How data moves through the stack

| Stage | Input | Output | Notes |
| --- | --- | --- | --- |
| Feed ingestion | Hawkes samples / recorded CSV | Normalised event arrays | Supports Binance, LOBSTER, and synthetic datasets. |
| Matching | Feed events, strategy orders | Executions, book snapshots | Deterministic, regression-tested (`tests/order_tests.cpp`). |
| Risk engine | Executions, snapshots | Inventory, PnL, alerts | Snapshots logged under `logs/` for dashboards. |
| Analytics | Risk snapshots, raw fills | Plots, CSVs, Streamlit widgets | Artefacts saved in `results/week*/`. |

A single `cmake --build` step compiles both the matching engine and Hawkes bridges; scripts under `python/scripts/` then orchestrate batch experiments and reports.

## Illustrated Analytics

<p align="center">
  <img src="docs/images/exponential_kernel_hawkes_intensity.png" width="360" alt="Exponential Hawkes intensity" />
  <img src="docs/images/arrivals_acf_bins_0.5.png" width="360" alt="Arrivals ACF bins" />
</p>

- **Intensity tracking** – exponential kernels adapt quickly to surges, while power-law kernels retain memory. The figures above (auto-generated via `python/demo.py`) help compare how different λ choices affect self-excitation, echoing Bouchaud et al. (2009) on supply/demand digestion.
- **Autocorrelation diagnostics** – arrivals ACFs quantify clustering. Values closer to zero after a few bins suggest weaker residual dependence; persistent autocorrelation suggests the need for heavier tails, seasonality, or regime switching.

<p align="center">
  <img src="docs/images/hawkes_rescale_qq.png" width="360" alt="Rescale QQ" />
  <img src="docs/images/qq_ks_btcusdt.png" width="360" alt="QQ/KS Binance BTCUSDT" />
</p>

- **Goodness-of-fit diagnostics** – rescaled QQ and KS plots (stored in `docs/images/`) diagnose how close fitted or simulated arrivals are to the exponential residual benchmark. Tail deviations highlight where adaptive intensity, seasonality, or exogenous shocks may be needed.
- **Ready-to-use assets** – these plots are referenced by reports in `docs/week7_*` and `docs/week8_*`, making it easy to embed them in presentations or papers without regenerating graphics manually.

### Copy & Paste Quickstart
```bash
# Clone and configure out-of-tree build
git clone https://github.com/sohaibelkarmi/High-Frequency-Trading-Simulator.git
cd High-Frequency-Trading-Simulator
cmake -S . -B build -DENABLE_DOCS=ON

# Build core library + docs
cmake --build build
cmake --build build --target docs

# Create Python env for analytics/backtester
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r python/requirements.txt

# Run deterministic backtester replay
python -m backtester.run --config configs/backtest_demo.json
```
The docs target drops HTML into `build/docs/html/index.html` ready for review.


### Prerequisites
- CMake >= 3.15 and a C++20-capable compiler (Clang, GCC, or MSVC).
- Python 3.10+ with `pip` for the analytics layer and Streamlit app.

### Build the C++ Simulator
```bash
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release --target hft_sim
./build/release/hft_sim
```

### Run the Hawkes Example (C++)
```bash
cmake --build build/release --target hawkes_example
./build/release/hawkes_example
```

### Execute Tests
```bash
cmake -S . -B build/tests -DHFT_ENABLE_TESTS=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build/tests --target order_tests
ctest --test-dir build/tests --output-on-failure
```

### Explore the Python Package & Demos
```bash
cd python
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
MPLCONFIGDIR=.matplotlib python3 -m demo
```
The demo prints branching ratios, generates intensity/ACF plots saved to `docs/images/`, and exports event streams to `data/runs/`.

Generate the order book showcase GIF straight from the repository root:

```bash
make demo
```
This replays the miniature feed in `data/sample/` and writes `assets/demo.gif` for quick sharing.

### Run the Backtester
```bash
python -m backtester.run --config configs/backtest_demo.json
```
Logs land under `logs/` (JSONL + optional SQLite) and drive the Streamlit diagnostics in `python/streamlit_app_backtest.py`.

## Interactive Streamlit App

Launch the pedagogy-first Streamlit interface to experiment with Hawkes processes visually:

```bash
cd python
streamlit run streamlit_app.py
```

Inside the app you can:
- Pick preset market regimes (Calm Market, Frenzy, Flash Crash) or define your own parameters.
- Toggle between exponential and power-law kernels and overlay comparison runs.
- Inspect branching ratios with criticality warnings and view order-size histograms.
- Export simulated order flow as CSV or download calibration notebooks directly from the sidebar.

<p align="center">
  <img src="docs/images/dashboard_icon.png" width="200" alt="Streamlit Hawkes dashboard icon" />
</p>

The app bridges directly to the native C++ kernels via `bridge_utils.ensure_bridge_path`, so ensure build artefacts exist under `build/lib` (or set `HFT_HAWKES_BRIDGE`).

## Documentation
- **Usage Guide** — `docs/USAGE.md` deep-dives into build configuration, order book APIs, Hawkes simulators, Python tooling, and troubleshooting tips.
- **Backtester design notes** — `docs/backtester/pnl.md` derive PnL formulas and risk settings for deterministic replays.
- **Research Primer** — `docs/research_primer.md` covers Hawkes theory, branching interpretation, and diagnostics for market microstructure.
- **Benchmark Protocol** — `docs/benchmark_protocol.md` explains how to reproduce multi-symbol experiments and report comparable metrics.
- **Notebooks** — `docs/notebooks/` contains ready-to-run calibration notebooks for synthetic and Binance datasets.
- **Paper snippets** — `docs/paper/results.tex` provides LaTeX-ready tables/figures plus a reproducibility checklist.

## Research Benchmarks

### Prepare Datasets
- **Binance BTCUSDT (2025-09-21)**
  ```bash
  python scripts/pack_binance_npz.py \
    --input-dir data/runs/events \
    --symbol BTCUSDT \
    --days 2025-09-21 \
    --output data/runs/events/binance_btcusdt_2025-09-21.npz
  ```
  Produces the NPZ plus a companion metadata JSON describing symbol/day and preprocessing options.
- **LOBSTER AAPL (2012-06-21)**
  ```bash
  python scripts/preprocess_lobster.py \
    --messages data/lobster/LOBSTER_SampleFile_AAPL_2012-06-21_10/\\
      AAPL_2012-06-21_34200000_57600000_message_10.csv \
    --symbol AAPL \
    --date 2012-06-21 \
    --output data/runs/events/lobster_aapl_2012-06-21_sample.npz
  ```
  Adapt the path if you download additional sessions; a `.meta.json` file records the seed and filters used.

### Train GRU and Transformer Backbones
```bash
export PYTHONPATH=.
PYTHONPATH=. python experiments/run_matrix.py \
  --config experiments/configs/binance_backbones.json \
  --results-dir experiments/results \
  --run-dir experiments/runs

PYTHONPATH=. python experiments/run_matrix.py \
  --config experiments/configs/lobster_backbones.json \
  --results-dir experiments/results \
  --run-dir experiments/runs
```
Each run logs deterministic seeds and checkpoints. Artefacts land in `experiments/runs/<experiment_id>/`:

- `metrics.json` summarises Train/Val/Test NLL, MAE, accuracy, KS stats, runtime, and parameter count.
- `curves/` stores CSVs for loss and calibration bins.
- `figs/` holds paper-ready loss/QQ/KS/calibration plots (PNG, 300 dpi).

### Aggregate & Summaries
- Collect per-run metrics into a single CSV:
  ```bash
  python scripts/collect_runs.py \
    --run-dir experiments/runs \
    --output experiments/summary/benchmarks.csv
  ```
- Generate the markdown table, ablation figure, and copy best plots:
  ```bash
  python scripts/prepare_summary_assets.py
  ```
  Outputs appear under `experiments/summary/` and can be dropped straight into a manuscript.

### Additional Research Utilities
- `neural_hawkes.py` exposes a JSON-driven `run_experiment` function with CLI logging (`--output`) and diagnostics (KS/QQ statistics, runtime).
- `experiments/run_matrix.py` executes matrices of configs (see `experiments/configs/`); summarise with `experiments/aggregate_results.py`.
- Diagnostics span time-rescaling QQ/KS plots, runtime comparisons, and branching statistics for publication-ready benchmarking.

## Repository Layout
- `src/` — C++ sources (`OrderBook`, Hawkes kernels, example apps).
- `tests/` — Catch2-based regression tests for order handling.
- `python/` — Hawkes simulation package (`kernels.py`, `simulate.py`, `viz.py`, Streamlit app, etc.).
- `data/` — Sample CSV/JSON runs produced by the demos.
- `docs/images/` — Generated figures used for reporting or documentation.
- `experiments/` — Configuration grids, scripts, and aggregated results for neural Hawkes benchmarks.

## Working with Data & Plots
- Regenerate datasets via `python -m demo`; outputs are deterministic with the seeded RNGs.
- High-resolution figures are committed for convenience; consider Git LFS if you plan to add many binary assets.
- `python/io.py` centralizes CSV/JSON serialization so you can swap in alternative storage formats with minimal changes.

## Theory Snapshot
- **Limit-order dynamics** — the C++ core models submissions, cancellations, and executions with price-time priority, letting you observe queue evolution as a discrete-event system.
- **Hawkes intensity** — arrivals follow `λ(t) = μ + \sum_i φ(t - T_i, V_i)`, capturing self-excitation where past trades raise the probability of near-future activity.
- **Kernel choices** — the exponential kernel `φ(u,v)=α v e^{-βu}` yields Markovian state updates; the power-law alternative `φ(u,v)=α v (u+c)^{-γ}` captures longer memory but requires `γ>1` to stay integrable.
- **Branching ratio** — expected offspring per event, `n = E[φ]`; keeping `n < 1` gives the standard subcritical Hawkes regime with finite stationary mean intensity.
- **Marks** — random volumes (log-normal, exponential, deterministic) feed back into intensity, providing a stylised link between trade size and subsequent activity.

## Example Outputs

Below are sample outputs from the Hawkes simulator, comparing exponential and power-law kernels.

### Order book replay demo

![Order book depth and trade tape](assets/demo.gif)


### Intensity Paths
- **Exponential kernel Hawkes**
  ![Exponential kernel Hawkes Intensity](docs/images/exponential_kernel_hawkes_intensity.png)

- **Power-law (rough) kernel Hawkes**
  ![Power-law kernel Hawkes Intensity](docs/images/power-law_rough_kernel_hawkes_intesity.png)

### Autocorrelation Functions
- **Volume marks ACF**
  ![Volume Marks ACF](docs/images/volume_marks_acf.png)

- **Volume marks ACF (alt run)**
  ![Volume Marks ACF bis](docs/images/volume_marks_acf_bis.png)

- **Arrival process ACF**
  ![Arrivals ACF](docs/images/arrivals_acf_bins_0.5.png)

## Development Notes
- Keep commits focused (parameter tuning, new kernels, plotting tweaks).
- Record RNG seeds alongside configuration in `data/runs/*.json` for reproducibility.
- When contributing kernels or strategies, add tests under `tests/` and plots/examples under `docs/` so results stay reproducible.

## Roadmap Ideas
1. Order-flow modelling: baseline Poisson generator, Hawkes kernels, and calibration notebooks (see `docs/5_intro.md`).
2. Extend the order book with latency models and queue-position analytics.
3. Expose a REST/gRPC shim for streaming orders to the simulator.
4. Package the Python tooling for pip installation and add notebook tutorials.
5. Wire CI (Catch2 + lint + demo smoke test) to keep the repo maintainable.

Feel free to fork and adapt—this project is meant to be a sandbox for experimentation as much as a reference implementation.
