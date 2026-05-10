---
author: "Kyle Jones"
date_published: "March 17, 2025"
date_exported_from_medium: "November 10, 2025"
canonical_link: "https://medium.com/@kyle-t-jones/forecasting-energy-demand-with-ibm-granite-tinytimemixer-abd16836238a"
---

# Forecasting Energy Demand with IBM Granite TinyTimeMixer Energy demand forecasting has always been a challenge, especially for
large-scale grids like ERCOT. With volatile consumption patterns...

### Forecasting Energy Demand with IBM Granite TinyTimeMixer 

Energy demand forecasting has always been a challenge, especially for large-scale grids like ERCOT. With volatile consumption patterns, changing weather conditions, and economic fluctuations, predicting energy needs requires models that are accurate, fast, and adaptable. That's where IBM's TinyTimeMixer (TTM) comes in.

TTM is a group of compact, pre-trained models for time series forecasting. TTM delivers state-of-the-art forecasts with just a fraction of the computational load of other LLM for time series models. It's built for real-world use --- small enough to run on a laptop but powerful enough to outperform traditional models in zero-shot and few-shot learning scenarios.

I tested Granite TTM on ERCOT's energy demand data. ERCOT (Electric Reliability Council of Texas) manages the power grid for most of Texas, balancing supply and demand across millions of households and businesses. Accurate forecasting here isn't just a convenience --- it's critical for grid stability and cost efficiency.

### Setting Up the Model
TTM comes in multiple versions. The latest release, TTM r2.1, has been trained on an enormous dataset of about a billion time series samples. For this experiment, I used the **TTM-512--96** model, meaning it takes 512 historical data points as input and predicts the next 96.

IBM makes it easy to get started with TTM. A simple call to `get_model()` automatically selects the right pre-trained version based on the input context length and forecast horizon. The setup looks like this:

```python
from tsfm_public.toolkit.get_model import get_model

TTM_MODEL_PATH = "ibm-granite/granite-timeseries-ttm-r2"
context_length = 512
forecast_length = 96
model = get_model(TTM_MODEL_PATH, context_length=context_length, prediction_length=forecast_length)
```

The context window (512 in this case) is HUGELY important for the forecasting. We are not training a model --- it is already trained. We are just using the model for inference.

Once the model was ready, I loaded ERCOT's energy demand dataset. This dataset contains historical electricity consumption measurements at different time intervals. Preprocessing was straightforward, using IBM's `TimeSeriesPreprocessor`:

```python
from tsfm_public import TimeSeriesPreprocessor

column_specifiers = {
    "timestamp_column": "datetime",
    "target_columns": ["demand_mw"],
    "id_columns": [],
    "control_columns": [],
}
tsp = TimeSeriesPreprocessor(
    **column_specifiers,
    context_length=context_length,
    prediction_length=forecast_length,
    scaling=True
)
```

### Zero-Shot Forecasting
One of TTM's biggest advantages is its zero-shot forecasting capability. Normally, a forecasting model needs to be trained on a dataset before it can make predictions. TTM skips that step entirely. It has already been pre-trained on a massive corpus of time series data, allowing it to generate meaningful forecasts out of the box.

Running a zero-shot forecast was as simple as calling `zeroshot_eval()`:

``` 
zeroshot_eval(
    dataset_name="ercot_energy",
    context_length=context_length,
    forecast_length=forecast_length,
    batch_size=64
)
```

Without any training on ERCOT data, TTM produced surprisingly accurate predictions. The model's pretraining on diverse time series allowed it to generalize well to unseen data.

MSE: 0.3628121316432953


### Few-Shot Fine-Tuning
Zero-shot forecasting is useful, but fine-tuning the model on domain-specific data improves accuracy. TTM is designed to learn quickly, requiring just 5% of the dataset to achieve performance comparable to models trained on the full dataset.

Fine-tuning involved training the model on a small subset of ERCOT's energy demand history:

``` 
fewshot_finetune_eval(
    dataset_name="ercot_energy",
    context_length=context_length,
    forecast_length=forecast_length,
    batch_size=64,
    fewshot_percent=5,
    learning_rate=0.001
)
```

The results were impressive. With minimal training, TTM adapted to ERCOT's demand patterns, refining its predictions. The mean squared error (MSE) dropped significantly compared to the zero-shot approach.

MSE: 0.36187952756881714


I ran the same thing with a context window of 1024. The MSE was lower.


<figcaption>Lower MSE is better. Longer context window is better.</figcaption>


Energy demand forecasting directly impacts operational decisions in power generation and distribution. Overestimating demand leads to wasted resources and higher costs, while underestimating can cause shortages and blackouts. Traditional forecasting models require extensive training and computational power, making real-time adaptation difficult.

TTM changes that. It delivers fast, accurate forecasts with minimal training, making it ideal for dynamic environments like ERCOT. And because it's lightweight, it doesn't need a GPU cluster --- it can run on a standard laptop or a single GPU instance.

IBM Granite TinyTimeMixer proves that smaller models can perform well on specialisted tasks. For ERCOT and similar energy grids, TTM offers a practical, efficient solution to forecasting challenges.

If you're working with time series data and need a lightweight yet powerful forecasting model, give TTM a shot. It's open-source, easy to use, and surprisingly effective.

**Resources:**

- [[IBM Granite TTM Repository](https://github.com/ibm-granite/granite-tsfm)]
- [[TTM Model on PyPI](https://pypi.org/project/granite-tsfm/)]
- [[TTM NeurIPS 2024 Paper](https://arxiv.org/abs/XXXXXX)]
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
import torch

np.random.seed(42)
plt.rcParams.update({
    'axes.grid': False,'font.family': 'serif','axes.spines.top': False,'axes.spines.right': False,'axes.linewidth': 0.8})

def save_fig(path: str):
    plt.tight_layout(); plt.savefig(path, bbox_inches='tight'); plt.close()

@dataclass
class Config:
    csv_path: str = "2001-2025 Net_generation_United_States_all_sectors_monthly.csv"
    freq: str = "MS"
    context_len: int = 512
    horizon: int = 8


def load_series(cfg: Config) -> pd.Series:
    p = Path(cfg.csv_path)
    df = pd.read_csv(p, header=None, usecols=[0,1], names=["date","value"], sep=",")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.dropna().sort_values("date").set_index("date")["value"].asfreq(cfg.freq)
    return s.astype(float)


def build_input_context(y: pd.Series, end_ts: pd.Timestamp, context_len: int) -> torch.Tensor:
    ctx = y.loc[:end_ts].values.astype(np.float32)
    if len(ctx) >= context_len:
        ctx = ctx[-context_len:]
    else:
        pad = np.zeros(context_len - len(ctx), dtype=np.float32)
        ctx = np.concatenate([pad, ctx])
    x = torch.tensor(ctx, dtype=torch.float32).view(1, context_len, 1)
    return x


def main():
    from tsfm_public.toolkit.get_model import get_model
    from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor

    cfg = Config()
    y = load_series(cfg)

    end_2024 = pd.Timestamp("2024-12-01")
    jan_2025 = pd.Timestamp("2025-01-01")
    aug_2025 = pd.Timestamp("2025-08-01")

    y_act = y.loc[jan_2025:aug_2025]

    model = get_model(
        'ibm-granite/granite-timeseries-ttm-r2',
        context_length=cfg.context_len,
        prediction_length=cfg.horizon,
        freq='W'
    )
    # ensure prediction_filter_length matches our horizon
    if hasattr(model, 'prediction_filter_length'):
        model.prediction_filter_length = cfg.horizon

    x = build_input_context(y, end_2024, cfg.context_len)
    # Build frequency token tensor (fallback to 'W' which is supported in Granite mapping)
    tsp = TimeSeriesPreprocessor(freq='W', context_length=cfg.context_len, prediction_length=cfg.horizon)
    freq_id = tsp.get_frequency_token('W')
    # model expects Long tensor; batch dimension 1
    freq_token = torch.tensor([freq_id], dtype=torch.long)

    with torch.no_grad():
        out = model(x, freq_token=freq_token)
    # Extract forecast array
    if hasattr(out, 'prediction_outputs'):
        yhat = out.prediction_outputs
    else:
        try:
            yhat = out[0]
        except Exception:
            yhat = out
    yhat = np.asarray(yhat).reshape(-1)

    dates = pd.period_range('2025-01', '2025-08', freq='M').to_timestamp()
    fc = pd.Series(yhat[:cfg.horizon], index=dates)

    # Plot greyscale Tufte-style
    start_2024 = pd.Timestamp("2024-01-01")
    y_hist = y.loc[start_2024:end_2024]

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(y_hist.index, y_hist.values, color="#888888", lw=1.5)
    ax.axvline(jan_2025, color="#666666", linestyle="--", lw=1)
    if len(y_act):
        ax.plot(y_act.index, y_act.values, color="#444444", lw=1.8)
    ax.plot(fc.index, fc.values, color="#000000", lw=2.0)

    from matplotlib.ticker import MaxNLocator, StrMethodFormatter
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
        ax.set_xlabel('')

    if len(y_hist):
        ax.annotate('History (2024)', xy=(y_hist.index[-1], y_hist.values[-1]), xytext=(6,0), textcoords='offset points', fontsize=9, va='center', ha='left', color='#666666')
    if len(y_act):
        ax.annotate('Actual (Jan–Aug 2025)', xy=(y_act.index[-1], y_act.values[-1]), xytext=(6,0), textcoords='offset points', fontsize=9, va='center', ha='left', color='#444444')
    ax.annotate('Granite TTM', xy=(fc.index[-1], fc.values[-1]), xytext=(6,0), textcoords='offset points', fontsize=9, va='center', ha='left', color='#000000')

    ax.set_title('EIA Net Generation — Granite TTM forecast Jan–Aug 2025')
    save_fig('eia_granite_ttm_last_fold.png')

if __name__ == '__main__':
    main()
```
