"""Generated from Jupyter notebook: Energy Demand forecasting using Ercot data with interactive plot

Magics and shell lines are commented out. Run with a normal Python interpreter."""

import pandas as pd
import plotly.express as px


def main() -> None:
    df = pd.read_csv("data/power.csv")

    df.head()

    df["Date"] = pd.to_datetime(df["Date"])

    df.set_index("Date", inplace=True)

    df["diff"] = df["cum_power"].diff()

    df["diff"].plot()

    df = df.sort_index()

    fig = px.line(df, x=df.index, y="diff")

    fig.update_traces(mode="lines")


if __name__ == "__main__":
    main()
