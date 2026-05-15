"""Generated from Jupyter notebook: Energy Demand forecasting using Ercot data with interactive plot

Magics and shell lines are commented out. Run with a normal Python interpreter."""


# --- code cell ---

# ! pip install -q -r requirements.txt  # Jupyter-only


# --- code cell ---

import pandas as pd
import plotly.express as px


def main():
    # %matplotlib inline  # Jupyter-only


    # --- code cell ---

    df = pd.read_csv("data/power.csv")


    # --- code cell ---

    df.head()


    # --- code cell ---

    df["Date"] = pd.to_datetime(df["Date"])

    df.set_index("Date", inplace=True)


    # --- duplicate code cell omitted (identical to earlier cell) ---


    # --- code cell ---

    df["diff"] = df["cum_power"].diff()


    # --- duplicate code cell omitted (identical to earlier cell) ---


    # --- code cell ---

    df["diff"].plot()


    # --- code cell ---

    df = df.sort_index()


    # --- code cell ---

    fig = px.line(df, x=df.index, y="diff")
    fig.update_traces(mode="lines")


if __name__ == "__main__":
    main()
