import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
import torch

np.random.seed(42)
plt.rcParams.update(
    {
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
    }
)


def save_fig(path: str):
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


@dataclass
class Config:
    csv_path: str = "2001-2025 Net_generation_United_States_all_sectors_monthly.csv"
    freq: str = "MS"
    context_len: int = 512
    horizon: int = 8


def load_series(cfg: Config) -> pd.Series:
    p = Path(cfg.csv_path)
    df = pd.read_csv(p, header=None, usecols=[0, 1], names=["date", "value"], sep=",")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.dropna().sort_values("date").set_index("date")["value"].asfreq(cfg.freq)
    return s.astype(float)


def build_input_context(
    y: pd.Series, end_ts: pd.Timestamp, context_len: int
) -> torch.Tensor:
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
        "ibm-granite/granite-timeseries-ttm-r2",
        context_length=cfg.context_len,
        prediction_length=cfg.horizon,
        freq="W",
    )
    # ensure prediction_filter_length matches our horizon
    if hasattr(model, "prediction_filter_length"):
        model.prediction_filter_length = cfg.horizon

    x = build_input_context(y, end_2024, cfg.context_len)
    # Build frequency token tensor (fallback to 'W' which is supported in Granite mapping)
    tsp = TimeSeriesPreprocessor(
        freq="W", context_length=cfg.context_len, prediction_length=cfg.horizon
    )
    freq_id = tsp.get_frequency_token("W")
    # model expects Long tensor; batch dimension 1
    freq_token = torch.tensor([freq_id], dtype=torch.long)

    with torch.no_grad():
        out = model(x, freq_token=freq_token)
    # Extract forecast array
    if hasattr(out, "prediction_outputs"):
        yhat = out.prediction_outputs
    else:
        try:
            yhat = out[0]
        except Exception:
            yhat = out
    yhat = np.asarray(yhat).reshape(-1)

    dates = pd.period_range("2025-01", "2025-08", freq="M").to_timestamp()
    fc = pd.Series(yhat[: cfg.horizon], index=dates)

    # Plot greyscale Tufte-style
    start_2024 = pd.Timestamp("2024-01-01")
    y_hist = y.loc[start_2024:end_2024]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(y_hist.index, y_hist.values, color="#888888", lw=1.5)
    ax.axvline(jan_2025, color="#666666", linestyle="--", lw=1)
    if len(y_act):
        ax.plot(y_act.index, y_act.values, color="#444444", lw=1.8)
    ax.plot(fc.index, fc.values, color="#000000", lw=2.0)

    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.set_xlabel("")

    if len(y_hist):
        ax.annotate(
            "History (2024)",
            xy=(y_hist.index[-1], y_hist.values[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=9,
            va="center",
            ha="left",
            color="#666666",
        )
    if len(y_act):
        ax.annotate(
            "Actual (Jan–Aug 2025)",
            xy=(y_act.index[-1], y_act.values[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=9,
            va="center",
            ha="left",
            color="#444444",
        )
    ax.annotate(
        "Granite TTM",
        xy=(fc.index[-1], fc.values[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        fontsize=9,
        va="center",
        ha="left",
        color="#000000",
    )

    ax.set_title("EIA Net Generation — Granite TTM forecast Jan–Aug 2025")
    save_fig("eia_granite_ttm_last_fold.png")


if __name__ == "__main__":
    main()
