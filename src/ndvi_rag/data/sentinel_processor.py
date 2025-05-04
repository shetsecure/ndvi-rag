"""Sentinel-2 data processor command-line tool.

This module provides a command-line interface for downloading and processing
Sentinel-2 satellite data for NDVI calculation.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
from loguru import logger

from .ingestion import process_sentinel_data

def setup_logger(log_file: Optional[str] = None) -> None:
    """Configure loguru logger with colors and formatting.
    
    Args:
        log_file: Optional path to a log file
    """
    # Remove default logger
    logger.remove()
    
    # Add console logger with colors
    logger.add(
        sys.stderr,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Add file logger if specified
    if log_file:
        logger.add(
            log_file,
            rotation="10 MB",
            retention="1 week",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG"
        )

def process_data(
    aoi_path: str,
    start_date: str,
    end_date: str,
    output_dir: str,
    max_cloud_cover: float
) -> Dict[int, pd.DataFrame]:
    """Process Sentinel-2 data for NDVI calculation.
    
    Args:
        aoi_path: Path to the Area of Interest GeoJSON file
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        output_dir: Directory to save processed data
        max_cloud_cover: Maximum cloud cover percentage (0-100)
        
    Returns:
        Dictionary with processed data summary by year
    """
    logger.info("Starting Sentinel-2 data processing")
    logger.info(f"AOI: {aoi_path}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Max cloud cover: {max_cloud_cover}%")
    logger.info(f"Output directory: {output_dir}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Record start time
    start_time = datetime.now()
    
    # Run processing
    results = process_sentinel_data(
        aoi_path=aoi_path,
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
        max_cloud_cover=max_cloud_cover
    )
    
    # Log processing summary
    processing_time = datetime.now() - start_time
    years_processed = sorted(results.keys())
    total_scenes = sum(len(df) for df in results.values())
    
    logger.success(f"Processing completed in {processing_time}")
    logger.info(f"Years processed: {', '.join(map(str, years_processed)) if years_processed else 'None'}")
    logger.info(f"Total scenes processed: {total_scenes}")
    
    # Print scenes per year
    if years_processed:
        logger.info("Scenes per year:")
        for year in years_processed:
            logger.info(f"  {year}: {len(results[year])} scenes")
    
    return results

def main() -> None:
    """Run the Sentinel-2 data processing from command line arguments."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process Sentinel-2 data for NDVI calculation")
    parser.add_argument("--aoi", type=str, default="data/aoi/midelt_region_utm_aligned.geojson",
                        help="Path to Area of Interest GeoJSON file")
    parser.add_argument("--start-date", type=str, default="2021-01-01",
                        help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end-date", type=str, default="2023-12-31",
                        help="End date in YYYY-MM-DD format")
    parser.add_argument("--output-dir", type=str, default="data/processed/ndvi_timeseries",
                        help="Directory to save processed data")
    parser.add_argument("--max-cloud-cover", type=float, default=10.0,
                        help="Maximum cloud cover percentage (0-100)")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Path to log file (optional)")
    
    args = parser.parse_args()
    
    # Set up logger
    setup_logger(args.log_file)
    
    # Run processing
    process_data(
        aoi_path=args.aoi,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
        max_cloud_cover=args.max_cloud_cover
    )
    
    logger.info("NDVI time series data is ready for analysis")

if __name__ == "__main__":
    main() 