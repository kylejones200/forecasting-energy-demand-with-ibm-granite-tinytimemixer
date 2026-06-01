# Energy Demand Forecasting with Granite TinyTimeMixer

This project demonstrates energy demand forecasting using IBM Granite TinyTimeMixer.

## Business context

Energy demand forecasting has always been a challenge, especially for large-scale grids like ERCOT. With volatile consumption patterns, changing weather conditions, and economic fluctuations, predicting energy needs requires models that are accurate, fast, and adaptable. That's where IBM's TinyTimeMixer (TTM) comes in.

TTM is a group of compact, pre-trained models for time series forecasting. TTM delivers state-of-the-art forecasts with just a fraction of the computational load of other LLM for time series models. It's built for real-world use --- small enough to run on a laptop but powerful enough to outperform traditional models in zero-shot and few-shot learning scenarios.

I tested Granite TTM on ERCOT's energy demand data. ERCOT (Electric Reliability Council of Texas) manages the power grid for most of Texas, balancing supply and demand across millions of households and businesses. Accurate forecasting here isn't just a convenience --- it's critical for grid stability and cost efficiency.

## Article

Medium article: [Forecasting Energy Demand with IBM Granite TinyTimeMixer](https://medium.com/@kylejones_47003/forecasting-energy-demand-with-ibm-granite-tinytimemixer-abd16836238a)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Forecasting functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data source or synthetic generation
- Date and demand columns
- Model parameters
- Output settings

## Granite TinyTimeMixer

IBM Granite TinyTimeMixer:
- Foundation model for time series
- Pre-trained on diverse datasets
- Efficient for energy demand forecasting

## Caveats

- By default, generates synthetic hourly energy demand data.
- Full Granite TinyTimeMixer implementation requires additional dependencies.
- Model performance depends on data quality and temporal patterns.

## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).