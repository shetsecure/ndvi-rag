# NDVI RAG System

A comprehensive system for analyzing multi-year NDVI data and answering natural language questions about vegetation evolution using RAG (Retrieval-Augmented Generation).

## Project Overview

This project implements a complete pipeline for:

1. **Data Ingestion**: Downloads Sentinel-2 Level-2A imagery for agricultural parcels
2. **NDVI Calculation**: Computes NDVI at pixel level for every usable acquisition
3. **Time Series Analysis**: Creates multi-year NDVI time series for agricultural monitoring
4. **RAG System**: Answers natural language questions about vegetation health and trends (upcoming)
5. **Interactive Visualization**: Displays NDVI evolution over time (upcoming)

## Completed Components

### Sentinel-2 Data Ingestion

- Downloads imagery from Digital Earth Africa's S3 bucket (no API keys required)
- Filters scenes with cloud cover < 10%
- Extracts B04 (Red) and B08 (NIR) bands at 10m resolution
- Processes every usable acquisition (maximum temporal frequency)
- Clips data to area of interest (agricultural parcel)
- Dynamically determines MGRS tiles for any geographic region

### NDVI Calculation

- Calculates NDVI using the formula: (B8 - B4) / (B8 + B4)
- Saves individual NDVI GeoTIFFs for each acquisition
- Creates time series datasets for any date range
- Maintains original 10m spatial resolution

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main data processing pipeline with default parameters:

```bash
python main.py
```

Or specify custom parameters:

```bash
python main.py --aoi data/aoi/your_area.geojson --start-date 2022-01-01 --end-date 2022-12-31 --max-cloud-cover 5.0
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--aoi` | Path to Area of Interest GeoJSON file | data/aoi/midelt_region_utm_aligned.geojson |
| `--start-date` | Start date (YYYY-MM-DD) | 2021-01-01 |
| `--end-date` | End date (YYYY-MM-DD) | 2023-12-31 |
| `--output-dir` | Directory to save processed data | data/processed/ndvi_timeseries |
| `--max-cloud-cover` | Maximum cloud cover percentage (0-100) | 10.0 |

### Processing Steps

The script will:
1. Determine MGRS tiles that cover your area of interest
2. Download Sentinel-2 data from the identified tiles
3. Filter scenes by date range and cloud cover threshold
4. Calculate NDVI for each scene
5. Create a time series dataset organized by year
6. Save results to the specified output directory

## Project Structure

```
ndvi-rag/
├── data/
│   ├── aoi/              # Area of Interest GeoJSON files
│   └── processed/        # Processed NDVI time series 
├── src/
│   └── ndvi_rag/
│       ├── data/         # Data ingestion module
│       ├── processing/   # NDVI calculation (upcoming)
│       └── utils/        # Utility functions
├── main.py               # Main execution script
└── README.md             # This file
```

## Next Steps

- Implement crop classification module
- Develop RAG system for natural language questions
- Create interactive visualization dashboard
- Add anomaly detection for NDVI time series

## License

MIT License 