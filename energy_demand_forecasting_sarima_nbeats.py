"""Generated from Jupyter notebook: energy_demand_forecasting_sarima_nbeats

Magics and shell lines are commented out. Run with a normal Python interpreter."""

import matplotlib.pyplot as plt
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import rmse
from darts.models import NBEATSModel, RNNModel
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX


def load_dataset_download_from_kaggle_first() -> None:
    df = pd.read_csv("continuous dataset.csv")

    df["datetime"] = pd.to_datetime(df["datetime"])

    df = df.set_index("datetime").sort_index()

    df = df.asfreq("h")

    df["nat_demand"] = df["nat_demand"].interpolate()

    df["hour"] = df.index.hour

    df["dayofweek"] = df.index.dayofweek

    df["month"] = df.index.month

    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    train = df.loc[:"2019-12-31"]

    test = df.loc["2020-01-01":]

    model = SARIMAX(train["nat_demand"], order=(2, 1, 2), seasonal_order=(1, 1, 1, 24))

    sarima_fit = model.fit()

    forecast_sarima = sarima_fit.forecast(len(test))

    mae = mean_absolute_error(test["nat_demand"], forecast_sarima)

    mape = mean_absolute_percentage_error(test["nat_demand"], forecast_sarima) * 100

    print(f"MAE: {mae:.2f}, MAPE: {mape:.2f}%")

    plt.figure(figsize=(14, 6))

    plt.plot(train.index, train["nat_demand"], label="Train", color="black")

    plt.plot(test.index, test["nat_demand"], label="Test", color="gray")

    plt.plot(test.index, forecast_sarima, label="SARIMA Forecast", color="blue")

    plt.xlabel("Date")

    plt.ylabel("Load (MW)")

    plt.legend()

    plt.title("Short-Term Electric Load Forecasting (SARIMA)")

    plt.savefig("electric_load_forecast.png")

    plt.show()

    series = TimeSeries.from_dataframe(df, value_cols="nat_demand")

    train, val = series.split_after("2021-12-31")

    model = NBEATSModel(input_chunk_length=48, output_chunk_length=24, n_epochs=50)

    model.fit(train)

    forecast_nbeats = model.predict(len(val))


def load_dataset() -> None:
    df = pd.read_csv("continuous dataset.csv")

    df["datetime"] = pd.to_datetime(df["datetime"])

    df = df.set_index("datetime").sort_index()

    df = df.asfreq("h")

    df["nat_demand"] = df["nat_demand"].interpolate()

    series = TimeSeries.from_dataframe(df, value_cols="nat_demand")

    split_point = pd.Timestamp("2020-01-01")

    train, test = series.split_after(split_point)

    model = NBEATSModel(
        input_chunk_length=48, output_chunk_length=24, n_epochs=5, random_state=3363
    )

    model.fit(train)

    forecast_nbeats = model.predict(len(test))

    mae = mean_absolute_error(test.values(), forecast_nbeats.values())

    mape = mean_absolute_percentage_error(test.values(), forecast_nbeats.values()) * 100

    print(f"MAE: {mae:.2f}, MAPE: {mape:.2f}%")

    plt.figure(figsize=(14, 6))

    train.plot(label="Train", lw=1, color="black")

    test.plot(label="Test", lw=1, color="gray")

    forecast_nbeats.plot(label="N-BEATS Forecast", lw=2, color="blue")

    plt.xlabel("Date")

    plt.ylabel("Load (MW)")

    plt.title("Short-Term Electric Load Forecasting (N-BEATS)")

    plt.legend()

    plt.savefig("electric_load_forecast_nbeats.png")

    plt.show()


def load_and_preprocess_data() -> None:
    df = pd.read_csv("continuous dataset.csv")

    df["datetime"] = pd.to_datetime(df["datetime"])

    df = df.set_index("datetime").asfreq("h").sort_index()

    series = TimeSeries.from_dataframe(df, value_cols="nat_demand")

    weather_cols = [
        "T2M_toc",
        "QV2M_toc",
        "TQL_toc",
        "W2M_toc",
        "T2M_san",
        "QV2M_san",
        "TQL_san",
        "W2M_san",
        "T2M_dav",
        "QV2M_dav",
        "TQL_dav",
        "W2M_dav",
    ]

    past_cov = TimeSeries.from_dataframe(df, value_cols=weather_cols)

    future_cov = (
        datetime_attribute_timeseries(df.index, attribute="day_of_week", one_hot=True)
        .stack(datetime_attribute_timeseries(df.index, attribute="hour", one_hot=True))
        .stack(TimeSeries.from_dataframe(df, value_cols=["holiday", "school"]))
    )

    train = series["2019-01-01":"2019-12-31"]

    test = series["2020-01-01":]

    past_cov_train = past_cov.slice_intersect(train)

    past_cov_test = past_cov.slice_intersect(test)

    future_cov_train = future_cov.slice_intersect(train)

    future_cov_test = future_cov.slice_intersect(test)

    scaler_target = Scaler()

    scaler_cov = Scaler()

    train_scaled = scaler_target.fit_transform(train)

    test_scaled = scaler_target.transform(test)

    past_cov_scaled = scaler_cov.fit_transform(past_cov)

    future_cov_scaled = scaler_cov.transform(future_cov)

    past_cov_train_scaled = past_cov_scaled.slice_intersect(train_scaled)

    future_cov_train_scaled = future_cov_scaled.slice_intersect(train_scaled)

    model = NBEATSModel(
        input_chunk_length=168, output_chunk_length=24, n_epochs=20, random_state=3363
    )

    model.fit(
        train_scaled,
        past_covariates=past_cov_train_scaled,
        future_covariates=future_cov_train_scaled,
        verbose=True,
    )

    forecast_scaled = model.historical_forecasts(
        series=train_scaled,
        past_covariates=past_cov_scaled,
        future_covariates=future_cov_scaled,
        start="2019-10-01",
        forecast_horizon=24,
        stride=24,
        retrain=False,
        verbose=True,
    )

    forecast = scaler_target.inverse_transform(forecast_scaled)

    mae = mean_absolute_error(test, forecast)

    mape = mean_absolute_percentage_error(test, forecast) * 100

    print(f"MAE: {mae:.2f}, MAPE: {mape:.2f}%")

    plt.figure(figsize=(14, 6))

    train.plot(label="Train", lw=1, color="black")

    test.plot(label="Test", lw=1, color="gray")

    forecast.plot(label="N-BEATS Forecast", lw=2, color="blue")

    plt.xlabel("Date")

    plt.ylabel("Load (MW)")

    plt.title("Short-Term Electric Load Forecasting (N-BEATS) with Covariates")

    plt.legend()

    plt.savefig("electric_load_forecast_nbeats_refined.png")

    plt.show()


def load_and_preprocess_data_2() -> None:
    df = pd.read_csv("continuous dataset.csv")

    df["datetime"] = pd.to_datetime(df["datetime"])

    df = df.set_index("datetime").asfreq("h").sort_index()

    print(f"Data covers {df.index.min()} to {df.index.max()}")

    series = TimeSeries.from_dataframe(df, value_cols="nat_demand")

    weather_cols = [
        "T2M_toc",
        "QV2M_toc",
        "TQL_toc",
        "W2M_toc",
        "T2M_san",
        "QV2M_san",
        "TQL_san",
        "W2M_san",
        "T2M_dav",
        "QV2M_dav",
        "TQL_dav",
        "W2M_dav",
    ]

    past_cov = TimeSeries.from_dataframe(df, value_cols=weather_cols)

    future_cov = (
        datetime_attribute_timeseries(
            series.time_index, attribute="day_of_week", one_hot=True
        )
        .stack(
            datetime_attribute_timeseries(
                series.time_index, attribute="hour", one_hot=True
            )
        )
        .stack(TimeSeries.from_dataframe(df, value_cols=["holiday", "school"]))
    )

    train = series.slice(pd.Timestamp("2019-01-01"), pd.Timestamp("2019-12-31"))

    test = series.slice(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31"))

    past_cov_train = past_cov.slice_intersect(train)

    past_cov_test = past_cov.slice_intersect(test)

    future_cov_train = future_cov.slice_intersect(train)

    future_cov_test = future_cov.slice_intersect(test)

    scaler_target = Scaler()

    scaler_cov = Scaler()

    train_scaled = scaler_target.fit_transform(train)

    test_scaled = scaler_target.transform(test)

    past_cov_scaled = scaler_cov.fit_transform(past_cov)

    future_cov_scaled = scaler_cov.fit_transform(future_cov)

    past_cov_train_scaled = past_cov_scaled.slice_intersect(train_scaled)

    future_cov_train_scaled = future_cov_scaled.slice_intersect(train_scaled)

    model = NBEATSModel(
        input_chunk_length=168, output_chunk_length=24, n_epochs=20, random_state=3363
    )

    model.fit(train_scaled, past_covariates=past_cov_train_scaled, verbose=True)

    forecast_scaled = model.historical_forecasts(
        series=train_scaled,
        past_covariates=past_cov_scaled,
        future_covariates=future_cov_scaled,
        start=pd.Timestamp("2019-10-01"),
        forecast_horizon=24,
        stride=24,
        retrain=False,
        verbose=True,
    )

    forecast = scaler_target.inverse_transform(forecast_scaled)

    mae = mean_absolute_error(test, forecast)

    mape = mean_absolute_percentage_error(test, forecast) * 100

    print(f"MAE: {mae:.2f}, MAPE: {mape:.2f}%")

    plt.figure(figsize=(14, 6))

    train.plot(label="Train", lw=1, color="black")

    test.plot(label="Test", lw=1, color="gray")

    forecast.plot(label="N-BEATS Forecast", lw=2, color="blue")

    plt.xlabel("Date")

    plt.ylabel("Load (MW)")

    plt.title("Short-Term Electric Load Forecasting (N-BEATS) with Covariates")

    plt.legend()

    plt.savefig("electric_load_forecast_nbeats_refined.png")

    plt.show()


def load_continuous_dataset() -> None:
    df = pd.read_csv("continuous dataset.csv")

    df["datetime"] = pd.to_datetime(df["datetime"])

    df_panama = df[["datetime", "nat_demand"]].rename(
        columns={"datetime": "timestamp", "nat_demand": "Load_MW"}
    )

    df_panama = df_panama.sort_values("timestamp").reset_index(drop=True)

    series = TimeSeries.from_dataframe(df_panama, "timestamp", "Load_MW")

    scaler = Scaler()

    series_scaled = scaler.fit_transform(series)

    train, val = (series_scaled[: -24 * 7], series_scaled[-24 * 7 :])

    model = RNNModel(
        model="LSTM",
        input_chunk_length=168,
        output_chunk_length=24,
        training_length=168,
        n_epochs=200,
        hidden_dim=32,
        dropout=0.1,
        random_state=42,
    )

    model.fit(train, verbose=True)

    pred_scaled = model.predict(len(val))

    pred = scaler.inverse_transform(pred_scaled)

    val_unscaled = scaler.inverse_transform(val)

    plt.figure(figsize=(12, 4))

    series[-24 * 7 :].plot(label="Observed", lw=1, color="gray")

    pred.plot(label="LSTM Forecast", lw=2, color="black")

    plt.legend()

    plt.title(f"Panama LSTM Load Forecast (RMSE: {rmse(val_unscaled, pred):.2f})")

    plt.tight_layout()

    plt.show()


def main() -> None:
    load_dataset_download_from_kaggle_first()
    load_dataset()
    load_and_preprocess_data()
    load_and_preprocess_data_2()
    load_continuous_dataset()


if __name__ == "__main__":
    main()
