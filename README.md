# Energy Demand Forecasting with Granite TinyTimeMixer

This project demonstrates energy demand forecasting using IBM Granite TinyTimeMixer.

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
