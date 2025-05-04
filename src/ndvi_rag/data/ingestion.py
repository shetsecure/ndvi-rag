"""Sentinel-2 data ingestion module for NDVI time series analysis.

Handles downloading and processing Sentinel-2 Level-2A data for NDVI calculation:
1. Downloads Sentinel-2 L2A scenes from Digital Earth Africa S3 bucket
2. Filters images with cloud cover < 10%
3. Calculates NDVI using bands B04 (red) and B08 (NIR)
4. Creates pixel-level time series for 3-year analysis
"""

import re
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray as rxr
import s3fs
import xarray as xr
from loguru import logger
from shapely.geometry import Polygon, mapping

# AWS Sentinel-2 bucket information for Digital Earth Africa
SENTINEL_BUCKET = "deafrica-sentinel-2"
SENTINEL_PREFIX = "sentinel-s2-l2a-cogs"
AWS_REGION = "af-south-1"  # Cape Town region

class Sentinel2Downloader:
    """Downloads and processes Sentinel-2 Level-2A data for NDVI calculation."""

    def __init__(
        self,
        aoi_path: Union[str, Path],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        max_cloud_cover: float = 10.0,
        output_dir: Union[str, Path] = "data/raw",
    ):
        """Initialize the Sentinel-2 downloader.

        Args:
            aoi_path: Path to the Area of Interest (AOI) file (GeoJSON)
            start_date: Start date for data collection (string 'YYYY-MM-DD' or datetime)
            end_date: End date for data collection (string 'YYYY-MM-DD' or datetime)
            max_cloud_cover: Maximum allowed cloud cover percentage (default: 10%)
            output_dir: Directory to save downloaded data (default: "data/raw")
        """
        self.aoi = self._load_aoi(aoi_path)
        
        # Convert string dates to datetime if needed
        if isinstance(start_date, str):
            self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        else:
            self.start_date = start_date
            
        if isinstance(end_date, str):
            self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        else:
            self.end_date = end_date
            
        self.max_cloud_cover = max_cloud_cover
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize S3 filesystem with anonymous access
        self.fs = s3fs.S3FileSystem(anon=True, client_kwargs={"region_name": AWS_REGION})
        
        # Determine MGRS tiles for AOI
        self.mgrs_tiles = self._get_mgrs_tiles()
        logger.info(f"AOI covers {len(self.mgrs_tiles)} MGRS tiles: {', '.join(self.mgrs_tiles)}")

    def _load_aoi(self, aoi_path: Union[str, Path]) -> gpd.GeoDataFrame:
        """Load the Area of Interest from GeoJSON file.

        Args:
            aoi_path: Path to the AOI file (GeoJSON)

        Returns:
            GeoDataFrame containing the AOI
        """
        aoi_path = Path(aoi_path)
        
        if not aoi_path.exists():
            raise ValueError(f"AOI file does not exist: {aoi_path}")
            
        gdf = gpd.read_file(aoi_path)
        
        # Ensure WGS84 coordinate system
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
            
        return gdf
    
    def _get_mgrs_tiles(self) -> List[str]:
        """Determine MGRS tiles that cover the AOI.
        
        Returns:
            List of MGRS tile identifiers (e.g., ["30SVE", "30SVF"])
        """
        # Get the centroid of the AOI
        aoi_geometry = self.aoi.geometry.unary_union
        centroid = aoi_geometry.centroid
        lon, lat = centroid.x, centroid.y
        logger.info(f"AOI centroid: {lon:.4f}, {lat:.4f}")
        
        # Calculate UTM zone
        # UTM zones are 6 degrees wide, starting at -180°
        utm_zone = str(int((lon + 180) / 6) + 1).zfill(2)
        
        # Calculate latitude band
        # Latitude bands start at -80° and are 8° high
        # Letters C to X, excluding I and O
        lat_value = int((lat + 80) / 8)
        if lat_value < 0:
            lat_value = 0
        elif lat_value > 20:  # Max index for letters C to X
            lat_value = 20
            
        # Convert to letter (C=0, D=1, etc., skipping I and O)
        band_letters = "CDEFGHJKLMNPQRSTUVWX"
        lat_band = band_letters[lat_value]
        
        # For the Midelt region in Morocco, we know it's likely in 30SVE
        # However, we'll check nearby squares if data isn't found
        if 31 < lat < 33 and -5.5 < lon < -4.5:
            # This is around Midelt area in Morocco, likely 30SVE
            primary_grid_square = "VE"
            potential_squares = ["VE", "VD", "UE", "UD"]
        else:
            # Use basic approach for other areas
            # Grid square is harder to calculate, so we'll check nearby ones
            # Based on position within a UTM zone (6° wide)
            longitude_in_zone = (lon + 180) % 6  # Position within UTM zone (0-6 degrees)
            longitude_fraction = longitude_in_zone / 6  # Normalized position (0-1)
            
            # Grid letters change every 8°
            # Within a UTM zone, different ranges of longitude have different grid letters
            # This is a simplification
            if longitude_fraction < 0.33:
                grid_letters1 = ['T', 'U', 'V', 'W', 'X', 'Y']
            elif longitude_fraction < 0.67:
                grid_letters1 = ['U', 'V', 'W', 'X', 'Y', 'Z']
            else:
                grid_letters1 = ['V', 'W', 'X', 'Y', 'Z', 'A']
                
            latitude_fraction = (lat + 80) % 8 / 8  # Position within latitude band (0-1)
            if latitude_fraction < 0.33:
                grid_letters2 = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
            elif latitude_fraction < 0.67:
                grid_letters2 = ['B', 'C', 'D', 'E', 'F', 'G', 'H']
            else:
                grid_letters2 = ['C', 'D', 'E', 'F', 'G', 'H', 'J']
            
            # Create a list of potential grid squares
            # We'll use the first one as primary but check others if needed
            potential_squares = []
            for l1 in grid_letters1[:3]:  # Take top 3 most likely
                for l2 in grid_letters2[:3]:  # Take top 3 most likely
                    potential_squares.append(f"{l1}{l2}")
            
            primary_grid_square = potential_squares[0]
        
        # Check if the primary grid square exists and has data
        logger.info(f"Checking primary MGRS tile: {utm_zone}{lat_band}{primary_grid_square}")
        primary_tile = f"{utm_zone}{lat_band}{primary_grid_square}"
        primary_path = f"{SENTINEL_BUCKET}/{SENTINEL_PREFIX}/{utm_zone}/{lat_band}/{primary_grid_square}"
        
        try:
            # Check for actual data in this tile
            has_data = False
            years_path = self.fs.ls(primary_path)
            years = [path.split('/')[-1] for path in years_path]
            # Filter to include only year directories
            years = [y for y in years if y.isdigit() and len(y) == 4]
            if years:
                has_data = True
                logger.info(f"Found data for tile {primary_tile} ({', '.join(years[0:3])}{'...' if len(years) > 3 else ''})")
            
            if has_data:
                return [primary_tile]
            else:
                # Try alternative tiles
                logger.warning(f"No data found for primary tile {primary_tile}")
                for square in potential_squares[1:3]:  # Try next 2 most likely
                    alt_tile = f"{utm_zone}{lat_band}{square}"
                    alt_path = f"{SENTINEL_BUCKET}/{SENTINEL_PREFIX}/{utm_zone}/{lat_band}/{square}"
                    if self.fs.exists(alt_path):
                        logger.info(f"Using alternate tile: {alt_tile}")
                        return [alt_tile]
                
                # If still not found, return the primary as fallback
                logger.warning(f"No valid tiles found, using primary as fallback: {primary_tile}")
                return [primary_tile]
                
        except Exception as e:
            logger.error(f"Error checking tiles: {e}")
            logger.warning(f"Falling back to primary tile: {primary_tile}")
            return [primary_tile]

    def list_available_scenes(self) -> List[Dict[str, Any]]:
        """List available Sentinel-2 scenes for the AOI and date range.

        Returns:
            List of scene metadata dictionaries
        """
        scenes = []
        
        # Get list of years to search
        years = [str(year) for year in range(self.start_date.year, self.end_date.year + 1)]
        
        # Search each MGRS tile
        for mgrs_tile in self.mgrs_tiles:
            utm_zone = mgrs_tile[:2]
            lat_band = mgrs_tile[2]
            grid_square = mgrs_tile[3:]
            
            tile_path = f"{SENTINEL_BUCKET}/{SENTINEL_PREFIX}/{utm_zone}/{lat_band}/{grid_square}"
            logger.info(f"Searching for scenes in: {tile_path}")
            
            for year in years:
                year_path = f"{tile_path}/{year}"
                
                # Get list of months
                try:
                    months = sorted([path.split('/')[-1] for path in self.fs.ls(year_path)])
                    
                    for month in months:
                        month_path = f"{year_path}/{month}"
                        
                        # Get list of scenes in this month
                        try:
                            scene_paths = self.fs.ls(month_path)
                            
                            for scene_path in scene_paths:
                                scene_id = scene_path.split('/')[-1]
                                
                                # Parse date from scene ID (format: S2A_30SVE_20211015_0_L2A)
                                match = re.search(r'_(\d{8})_', scene_id)
                                if not match:
                                    continue
                                    
                                date_str = match.group(1)
                                scene_date = datetime(
                                    int(date_str[:4]),
                                    int(date_str[4:6]),
                                    int(date_str[6:8])
                                )
                                
                                # Check if scene is within our date range
                                if scene_date < self.start_date or scene_date > self.end_date:
                                    continue
                                
                                # Check cloud cover
                                cloud_cover = self._get_scene_cloud_cover(scene_path)
                                
                                if cloud_cover is not None and cloud_cover > self.max_cloud_cover:
                                    logger.info(f"Skipping {scene_id} - cloud cover {cloud_cover}% exceeds threshold")
                                    continue
                                
                                # Add scene to our list
                                scenes.append({
                                    "path": scene_path,
                                    "id": scene_id,
                                    "date": scene_date,
                                    "cloud_cover": cloud_cover,
                                    "mgrs_tile": mgrs_tile
                                })
                        except Exception as e:
                            logger.warning(f"Error listing scenes in {month_path}: {e}")
                except Exception as e:
                    logger.warning(f"Error listing months in {year_path}: {e}")
        
        # Sort scenes by date
        scenes.sort(key=lambda x: x["date"])
        logger.info(f"Found {len(scenes)} scenes matching criteria")
        return scenes
            
    def _get_scene_cloud_cover(self, scene_path: str) -> Optional[float]:
        """Extract cloud cover percentage from scene metadata.
        
        Args:
            scene_path: Path to the scene in S3
            
        Returns:
            Cloud cover percentage or None if not available
        """
        metadata_path = f"{scene_path}/{scene_path.split('/')[-1]}.json"
        
        try:
            if self.fs.exists(metadata_path):
                with self.fs.open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    
                    # Try different cloud cover field names
                    for field in ['CLOUDY_PIXEL_PERCENTAGE', 'cloudCover', 'CLOUD_COVERAGE_ASSESSMENT']:
                        if field in metadata:
                            return float(metadata[field])
        except Exception:
            pass
            
        return None
            
    def download_scene(self, scene: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Download and process a Sentinel-2 scene for NDVI calculation.

        Args:
            scene: Scene metadata dictionary

        Returns:
            Dictionary with processed data or None if download fails
        """
        scene_path = scene["path"]
        scene_id = scene["id"]
        date_str = scene["date"].strftime("%Y-%m-%d")
        
        # Create output directory for this scene
        scene_dir = self.output_dir / scene_id
        scene_dir.mkdir(exist_ok=True)
        
        try:
            # Download B04 (red) and B08 (NIR) bands
            bands = {}
            for band_name in ["B04", "B08"]:
                band_path = f"{scene_path}/{band_name}.tif"
                
                if not self.fs.exists(band_path):
                    logger.warning(f"Band file does not exist: {band_path}")
                    return None
                
                # Download band
                local_band_path = scene_dir / f"{band_name}.tif"
                
                with self.fs.open(band_path, 'rb') as remote_file, open(local_band_path, 'wb') as local_file:
                    local_file.write(remote_file.read())
                
                # Open with rioxarray
                bands[band_name] = rxr.open_rasterio(local_band_path)
                
                # Clip to AOI if possible
                try:
                    if hasattr(bands[band_name].rio, 'crs') and bands[band_name].rio.crs:
                        aoi_reprojected = self.aoi.to_crs(bands[band_name].rio.crs)
                        bands[band_name] = bands[band_name].rio.clip(aoi_reprojected.geometry.values, aoi_reprojected.crs)
                except Exception as e:
                    logger.warning(f"Could not clip to AOI: {e}")
            
            # Calculate NDVI if we have both bands
            if "B04" in bands and "B08" in bands:
                # Convert to float32
                red = bands["B04"].astype('float32')
                nir = bands["B08"].astype('float32')
                
                # Calculate NDVI: (NIR - Red) / (NIR + Red)
                ndvi = (nir - red) / (nir + red)
                
                # Set proper metadata
                ndvi.attrs['long_name'] = 'Normalized Difference Vegetation Index'
                ndvi.attrs['units'] = 'unitless'
                ndvi.attrs['scene_id'] = scene_id
                ndvi.attrs['date'] = date_str
                ndvi.attrs['mgrs_tile'] = scene.get("mgrs_tile", "unknown")
                
                # Save NDVI as GeoTIFF
                ndvi_path = scene_dir / "ndvi.tif"
                ndvi.rio.to_raster(ndvi_path)
                
                return {
                    "scene_id": scene_id,
                    "date": scene["date"],
                    "ndvi_path": str(ndvi_path),
                    "ndvi_data": ndvi,
                    "mgrs_tile": scene.get("mgrs_tile", "unknown")
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing scene {scene_id}: {e}")
            return None

def process_sentinel_data(
    aoi_path: str,
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    output_dir: str = "data/processed",
    max_cloud_cover: float = 10.0
) -> Dict[int, pd.DataFrame]:
    """Process Sentinel-2 data for a date range.
    
    Args:
        aoi_path: Path to the Area of Interest GeoJSON file
        start_date: Start date (YYYY-MM-DD string or datetime object)
        end_date: End date (YYYY-MM-DD string or datetime object)
        output_dir: Root directory to save processed data
        max_cloud_cover: Maximum cloud cover percentage to include
        
    Returns:
        Dictionary with yearly results summary
    """
    # Convert string dates to datetime if needed
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        
    root_output_dir = Path(output_dir)
    root_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Results storage
    results = {}
    
    # Get the years in the date range
    years = list(range(start_date.year, end_date.year + 1))
    
    # Initialize downloader for the entire date range
    downloader = Sentinel2Downloader(
        aoi_path=aoi_path,
        start_date=start_date,
        end_date=end_date,
        max_cloud_cover=max_cloud_cover,
        output_dir=root_output_dir / "scenes"
    )
    
    # Get all available scenes
    scenes = downloader.list_available_scenes()
    
    if not scenes:
        logger.warning(f"No scenes found for date range {start_date} to {end_date}")
        return results
    
    # Process scenes and organize by year
    for scene in scenes:
        year = scene["date"].year
        
        # Create year directory if needed
        year_dir = root_output_dir / str(year)
        year_dir.mkdir(exist_ok=True)
        
        # Process the scene
        result = downloader.download_scene(scene)
        
        if result:
            # Initialize year results if needed
            if year not in results:
                results[year] = []
                
            # Add to results
            results[year].append({
                "date": scene["date"],
                "scene_id": scene["id"],
                "ndvi_path": result["ndvi_path"],
                "cloud_cover": scene.get("cloud_cover"),
                "mgrs_tile": scene.get("mgrs_tile", "unknown")
            })
    
    # Convert results to DataFrames and save summaries
    for year, scenes_list in results.items():
        if scenes_list:
            # Create DataFrame
            results_df = pd.DataFrame(scenes_list)
            results_df = results_df.sort_values("date")
            
            # Create year directory
            year_dir = root_output_dir / str(year)
            year_dir.mkdir(exist_ok=True)
            
            # Save CSV summary
            csv_path = year_dir / f"{year}_scenes_summary.csv"
            results_df.to_csv(csv_path, index=False)
            
            # Replace list with DataFrame in results
            results[year] = results_df
            logger.info(f"Processed {len(results_df)} scenes for {year}")
            
            # Create VRT for all NDVI files if supported
            try:
                from osgeo import gdal
                
                ndvi_files = [r["ndvi_path"] for r in scenes_list]
                vrt_path = year_dir / f"{year}_ndvi_timeseries.vrt"
                
                gdal.BuildVRT(str(vrt_path), ndvi_files, separate=True, options=[
                    f'-input_file_list', 
                    f'-overwrite'
                ])
                
                logger.info(f"Created VRT time series at {vrt_path}")
            except (ImportError, Exception) as e:
                logger.warning(f"Could not create VRT: {e}")
    
    return results

if __name__ == "__main__":
    # Example usage
    process_sentinel_data(
        aoi_path="data/aoi/midelt_region_utm_aligned.geojson",
        start_date="2021-01-01",
        end_date="2023-12-31",
        output_dir="data/processed/ndvi_timeseries",
        max_cloud_cover=10.0
    ) 