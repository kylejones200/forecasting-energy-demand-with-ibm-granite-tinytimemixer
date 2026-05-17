"""Generated from Jupyter notebook: Energy Demand forecasting using Ercot data with interactive plot

Magics and shell lines are commented out. Run with a normal Python interpreter."""
import pandas as pd
import plotly.express as px

def main():
    df = pd.read_csv('data/power.csv')
    df.head()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)


def main() -> None:
    # --- notebook cell (unparsed) ---
    # # --- code cell ---

    #     df["diff"] = df["cum_power"].diff()

    # --- notebook cell (unparsed) ---
    # # --- code cell ---

    #     df["diff"].plot()


    #     # --- code cell ---

    #     df = df.sort_index()


    #     # --- code cell ---

    #     fig = px.line(df, x=df.index, y="diff")
    #     fig.update_traces(mode="lines")


    # if __name__ == "__main__":
    #     main()

if __name__ == "__main__":
    main()
