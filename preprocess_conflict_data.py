#!/usr/bin/env python3
"""
Conflict Event Spatial Preprocessing with Administrative Region Assignment

This script processes raw conflict event data by:
1. Cleaning and standardizing event attributes
2. Assigning events to administrative regions via spatial join
3. Creating buffer zones around regions to capture spatial spillover effects
4. Optionally time-shifting events for retrospective analysis

The spatial buffering accounts for:
- GPS coordinate uncertainty in conflict reporting
- Psychological impact radius (fear extends beyond event location)
- Cross-border spillover effects
- Migration triggers from nearby threats

Author: Zakaria
Date: 2024
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import argparse
import logging
from pathlib import Path
from datetime import timedelta
from file_paths_and_consts import UNCLEANED_DATA_DIR, IMPACT_DIR

# ========================== DATA CLEANING ==========================

def clean_conflict_data(conflict_df, manual_date=None, logger=None):
    """
    Clean and standardize raw conflict event data.
    
    Performs:
    - String cleaning for event types (remove spaces, standardize separators)
    - Datetime conversion for temporal analysis
    - Event weight initialization (can be customized later)
    
    Args:
        conflict_df (pd.DataFrame): Raw conflict data with columns:
            - latitude, longitude: Event coordinates
            - event_date: Date string
            - sub_event_type: Event classification
            - fatalities: Number of casualties
        logger (logging.Logger): Logger instance
        
    Returns:
        gpd.GeoDataFrame: Cleaned conflict data with geometry column
        
    Note:
        Event weights default to 1.0 but can be modified based on event_type
        and sub_event_type for differential impact modeling.
    """
    logger.info(f"Cleaning {len(conflict_df):,} conflict events...")
    
    # Create GeoDataFrame with point geometries
    conflict_gdf = gpd.GeoDataFrame(
        conflict_df,
        geometry=gpd.points_from_xy(conflict_df.longitude, conflict_df.latitude),
        crs='EPSG:4326'  # WGS84 coordinate system
    )

    # Handle date assignment
    if manual_date is not None:
        # Manual date mode: assign the same date to all events
        logger.info(f"Using manual date mode: assigning {manual_date} to all events")
        conflict_gdf['event_date'] = manual_date
        conflict_gdf['time'] = pd.to_datetime(manual_date)
    else:
        # Default mode: use existing event_date column from CSV
        if 'event_date' not in conflict_gdf.columns:
            raise ValueError(
                "No 'event_date' column found in conflict data. "
                "Either provide event_date column or use --manual-date option."
            )
        conflict_gdf['time'] = pd.to_datetime(conflict_gdf['event_date'])
    
    # Clean event type strings for file naming and processing
    # Replace spaces with underscores, slashes with "_or_"
    if 'sub_event_type' in conflict_gdf.columns:
        conflict_gdf['sub_event_type'] = (
            conflict_gdf['sub_event_type']
            .str.replace(' ', '_')
            .str.replace('/', '_or_')
        )
    
    # Initialize event weights (placeholder for potential differential weighting)
    conflict_gdf['event_weight'] = 1.0
    
    logger.info(f"✓ Cleaned data spans {conflict_gdf['time'].min()} to {conflict_gdf['time'].max()}")
    
    return conflict_gdf


def shift_timestamps(conflict_gdf, shift_days, logger):
    """
    Shift all event timestamps by specified number of days.
    
    Use case: Retrospective simulation for periods without actual conflict data.
    For example, if calibrating for 2023 but only have 2022 data, shift by 365 days.
    
    Args:
        conflict_gdf (gpd.GeoDataFrame): Conflict events with 'time' column
        shift_days (int): Days to shift (positive = future, negative = past)
        logger (logging.Logger): Logger instance
        
    Returns:
        gpd.GeoDataFrame: Data with shifted timestamps
        
    Warning:
        Time-shifted data should only be used for testing/calibration purposes.
        Results may not reflect actual temporal dynamics of conflict.
    """
    logger.info(f"Shifting timestamps by {shift_days:+d} days...")
    
    original_range = f"{conflict_gdf['time'].min()} to {conflict_gdf['time'].max()}"
    
    conflict_gdf['time'] = conflict_gdf['time'] + timedelta(days=shift_days)
    conflict_gdf['event_date'] = conflict_gdf['time'].dt.strftime('%Y-%m-%d')
    
    shifted_range = f"{conflict_gdf['time'].min()} to {conflict_gdf['time'].max()}"
    
    logger.info(f"  Original: {original_range}")
    logger.info(f"  Shifted:  {shifted_range}")
    logger.warning("⚠️  Time-shifted data - use only for testing/calibration")
    
    return conflict_gdf


# ========================== SPATIAL PROCESSING ==========================

def load_admin_boundaries(shapefile_path, logger):
    """
    Load administrative boundary shapefile.
    
    Args:
        shapefile_path (str): Path to shapefile (.shp)
        logger (logging.Logger): Logger instance
        
    Returns:
        gpd.GeoDataFrame: Administrative regions with geometry
        
    Note:
        Expects shapefile to have polygon geometries representing
        administrative regions (e.g., ADM2 level)
    """
    logger.info(f"Loading administrative boundaries from {shapefile_path}...")
    
    admin_gdf = gpd.read_file(shapefile_path)
    
    # Add representative point for each region (useful for distance calculations)
    admin_gdf['centroid_coords'] = admin_gdf['geometry'].apply(
        lambda geom: geom.representative_point().coords[0]
    )
    
    logger.info(f"✓ Loaded {len(admin_gdf):,} administrative regions")
    logger.debug(f"  CRS: {admin_gdf.crs}")
    logger.debug(f"  Bounds: {admin_gdf.total_bounds}")
    
    return admin_gdf


def assign_events_to_regions(conflict_gdf, admin_gdf, buffer_km, logger):
    """
    Assign conflict events to administrative regions with spatial buffering.
    
    Algorithm:
    1. Create buffer zones around each administrative region
    2. Perform spatial join: assign events within buffered regions
    3. Events near borders may be assigned to multiple regions
    
    Args:
        conflict_gdf (gpd.GeoDataFrame): Conflict events with point geometries
        admin_gdf (gpd.GeoDataFrame): Administrative boundaries
        buffer_km (float): Buffer distance in kilometers
        logger (logging.Logger): Logger instance
        
    Returns:
        gpd.GeoDataFrame: Events with assigned region information
        
    Technical details:
        - Converts to UTM (EPSG:32634 for Eastern Europe) for accurate buffering
        - Buffer distance converted to meters (buffer_km × 1000)
        - Returns to WGS84 for consistency with input data
        
    Note:
        buffer_km = 0 performs exact spatial join (no buffering)
    """
    logger.info(f"Assigning events to regions with {buffer_km}km buffer...")

    if buffer_km > 0:
        logger.info(f"  Projecting to UTM {admin_gdf.estimate_utm_crs()} for {buffer_km}km buffering...")
        
        admin_buffered = admin_gdf.copy()
        admin_buffered.geometry = admin_buffered.geometry.make_valid()  # FIX GEOMETRIES
        admin_buffered = admin_buffered.to_crs(admin_buffered.estimate_utm_crs())  # AUTO UTM
        admin_buffered.geometry = admin_buffered.geometry.buffer(buffer_km * 1000)
        admin_buffered = admin_buffered.to_crs("EPSG:4326")
        
        logger.debug(f"  Buffer created: {buffer_km}km around each region")
        
    else:
        logger.debug(f"  No buffering (exact spatial join)")
        admin_buffered = admin_gdf.copy()
    
    # Spatial join: assign events to regions
    # 'within' predicate: event point must be within (buffered) region polygon
    logger.info("  Performing spatial join...")
    events_assigned = gpd.sjoin(
        conflict_gdf, 
        admin_buffered, 
        how='inner',
        predicate='within'
    )

    logger.info(f"  After Performing spatial join, we have the following columns {events_assigned.columns.tolist()}")
    
    # Count assignments
    unique_events = events_assigned['data_id'].nunique() if 'data_id' in events_assigned.columns else len(events_assigned)
    unique_regions = events_assigned.index_right.nunique()
    
    logger.info(f"✓ Assigned {len(events_assigned):,} event-region pairs")
    logger.info(f"  Unique events: {unique_events:,}")
    logger.info(f"  Regions affected: {unique_regions:,}")
    
    if buffer_km > 0:
        # With buffering, events may appear in multiple regions
        avg_regions_per_event = len(events_assigned) / unique_events
        logger.info(f"  Avg regions per event: {avg_regions_per_event:.2f} "
                   f"(due to {buffer_km}km buffer)")
    
    return events_assigned


def format_output(events_assigned, region_id_col, region_name_col, logger):
    """
    Format assigned events for ABM simulation input.
    
    Standardizes column names and calculates final event intensity:
        intensity = fatalities × event_weight
    
    Args:
        events_assigned (gpd.GeoDataFrame): Events with region assignments
        region_id_col (str): Column name for region identifier
        region_name_col (str): Column name for region name
        logger (logging.Logger): Logger instance
        
    Returns:
        pd.DataFrame: Formatted event data ready for simulation
        
    Output columns:
        - event_id: Unique event identifier
        - time: Event timestamp
        - latitude, longitude: Event coordinates
        - event_type, sub_event_type: Event classification
        - event_weight: Weighting factor
        - event_intensity: Weighted fatality count (fatalities × weight)
        - matching_place_name: Human-readable region name
        - matching_place_id: Machine-readable region identifier
    """
    logger.info("Formatting output for simulation...")
    
    # Rename columns to simulation requirements
    rename_map = {
        'data_id': 'event_id',
        'fatalities': 'event_intensity',
        region_name_col: 'matching_place_name',
        region_id_col: 'matching_place_id'
    }
    
    output_df = events_assigned.rename(columns=rename_map)
    
    # Calculate weighted intensity
    if 'event_intensity' in output_df.columns:
        output_df['event_intensity'] = (
            output_df['event_intensity'] * output_df['event_weight']
        )
    
    # Select required columns
    required_cols = [
        'event_id', 'time', 'latitude', 'longitude',
        'event_type', 'sub_event_type', 'event_weight', 'event_intensity',
        'matching_place_name', 'matching_place_id'
    ]
    
    # Only keep columns that exist
    available_cols = [col for col in required_cols if col in output_df.columns]
    output_df = output_df[available_cols]

    logging.info(f"final dataframe has shape {output_df.shape[0]} and following columns {output_df.columns.tolist()}")
    
    # Sort by time and region for efficient simulation loading
    output_df = output_df.sort_values(['time', 'matching_place_id'])
    
    logger.info(f"✓ Formatted {len(output_df):,} event records")
    logger.debug(f"  Columns: {', '.join(output_df.columns)}")
    
    return output_df


# ========================== MAIN PIPELINE ==========================

class ConflictDataProcessor:
    """
    Process raw conflict data for ABM simulation input.
    
    Pipeline:
    1. Load and clean conflict events
    2. Load administrative boundaries
    3. Spatial assignment with buffering
    4. Format output for simulation
    5. Optional timestamp shifting
    """
    
    def __init__(self, conflict_file, shapefile, buffer_km=10, 
                 region_id_col='ADM2_EN', region_name_col='ADM2_EN',
                 output_prefix='conflict_data_ADM2',
                 shift_days=0, manual_date=None, logger=None):
        """
        Initialize processor.
        
        Args:
            conflict_file (str): Path to raw conflict CSV
            shapefile (str): Path to admin boundary shapefile
            buffer_km (float): Buffer distance in km
            region_id_col (str): Column for region ID in shapefile
            region_name_col (str): Column for region name in shapefile
            output_prefix (str): Prefix for output filename
            shift_days (int): Days to shift timestamps (0 = no shift)
            logger (logging.Logger): Logger instance
        """
        self.conflict_file = conflict_file
        self.shapefile = shapefile
        self.buffer_km = buffer_km
        self.region_id_col = region_id_col
        self.region_name_col = region_name_col
        self.output_prefix = output_prefix
        self.shift_days = shift_days
        self.manual_date = manual_date
        self.logger = logger or logging.getLogger(__name__)
    
    def process(self):
        """
        Execute complete preprocessing pipeline.
        
        Returns:
            tuple: (output_df, output_file_path)
        """
        self.logger.info(f"\n{'#'*60}")
        self.logger.info(f"# CONFLICT DATA PREPROCESSING")
        self.logger.info(f"# Buffer: {self.buffer_km} km")
        if self.shift_days != 0:
            self.logger.info(f"# Time shift: {self.shift_days:+d} days")
        self.logger.info(f"{'#'*60}\n")
        
        # Step 1: Load and clean conflict data
        self.logger.info("STEP 1: Loading raw conflict data...")
        conflict_raw = pd.read_csv(self.conflict_file)
        conflict_gdf = clean_conflict_data(conflict_raw, self.manual_date, self.logger)
        
        # Step 2: Optional timestamp shifting
        if self.shift_days != 0:
            self.logger.info("\nSTEP 2: Shifting timestamps...")
            conflict_gdf = shift_timestamps(conflict_gdf, self.shift_days, self.logger)
        
        # Step 3: Load administrative boundaries
        self.logger.info("\nSTEP 3: Loading administrative boundaries...")
        admin_gdf = load_admin_boundaries(self.shapefile, self.logger)
        
        # Step 4: Spatial assignment
        self.logger.info("\nSTEP 4: Spatial assignment with buffering...")
        events_assigned = assign_events_to_regions(
            conflict_gdf, admin_gdf, self.buffer_km, self.logger
        )
        
        # Step 5: Format output
        self.logger.info("\nSTEP 5: Formatting output...")
        output_df = format_output(
            events_assigned, 
            self.region_id_col, 
            self.region_name_col,
            self.logger
        )
        
        # Step 6: Save output
        output_file = self._generate_output_filename()
        self.logger.info(f"\nSTEP 6: Saving to {output_file}...")
        
        # Ensure output directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        output_df.to_csv(output_file, index=False)
        self.logger.info(f"✓ Saved {len(output_df):,} records")
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"PREPROCESSING COMPLETE")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Output: {output_file}")
        self.logger.info(f"Records: {len(output_df):,}")
        self.logger.info(f"Time range: {output_df['time'].min()} to {output_df['time'].max()}")
        self.logger.info(f"{'='*60}\n")
        
        return output_df, output_file
    
    def _generate_output_filename(self):
        # Put shift BEFORE buffer so it can be part of the prefix
        prefix = self.output_prefix
        
        if self.shift_days != 0:
            prefix += f"_shift_{self.shift_days:+d}_days"
        
        filename = f"{prefix}_buffer_{int(self.buffer_km)}_km.csv"
        
        return f"{IMPACT_DIR}{filename}"


# ========================== CLI INTERFACE ==========================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Preprocess conflict data with spatial buffering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
      # Basic usage with 10km buffer:
      python preprocess_conflict_data.py \\
        --conflict-file raw_conflict.csv \\
        --shapefile admin_boundaries.shp \\
        --buffer 10
      
      # With time shift for retrospective simulation:
      python preprocess_conflict_data.py \\
        --conflict-file conflict_2022.csv \\
        --shapefile boundaries.shp \\
        --buffer 5 \\
        --shift 365  # Shift 2022 data to 2023
      
      # Custom region columns:
      python preprocess_conflict_data.py \\
        --conflict-file conflict.csv \\
        --shapefile boundaries.shp \\
        --region-id-col ADM2_PCODE \\
        --region-name-col ADM2_NAME
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--conflict-file',
        type=str,
        required=True,
        help='Path to raw conflict event CSV file'
    )
    
    parser.add_argument(
        '--shapefile',
        type=str,
        required=True,
        help='Path to administrative boundary shapefile (.shp)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--buffer',
        type=float,
        default=10,
        help='Buffer distance in kilometers (default: 10)'
    )
    
    parser.add_argument(
        '--region-id-col',
        type=str,
        default='ADM2_EN',
        help='Column name for region ID in shapefile (default: ADM2_EN)'
    )
    
    parser.add_argument(
        '--region-name-col',
        type=str,
        default='ADM2_EN',
        help='Column name for region name in shapefile (default: ADM2_EN)'
    )

    parser.add_argument(
        '--manual-date',
        type=str,
        default=None,
        help='Manually assign this date to ALL events (format: YYYY-MM-DD). '
             'Use when conflict data lacks proper dates. Disabled by default.'
    )
    
    parser.add_argument(
        '--output-prefix',
        type=str,
        default='conflict_data_ADM2',
        help='Prefix for output filename (default: conflict_data_ADM2)'
    )
    
    parser.add_argument(
        '--shift',
        type=int,
        default=0,
        help='Shift timestamps by N days (positive=future, negative=past, default: 0)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        help='Log file path (default: console only)'
    )
    
    return parser.parse_args()


def setup_logging(log_file=None, verbose=False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    
    handlers = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(asctime)s [%(filename)s:%(lineno)s] >>> %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers
    )
    
    return logging.getLogger(__name__)


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(args.log_file, args.verbose)
    
    # Create processor
    processor = ConflictDataProcessor(
        conflict_file=args.conflict_file,
        shapefile=args.shapefile,
        buffer_km=args.buffer,
        region_id_col=args.region_id_col,
        region_name_col=args.region_name_col,
        output_prefix=args.output_prefix,
        shift_days=args.shift,
        manual_date=args.manual_date,
        logger=logger
    )
    
    # Process data
    try:
        output_df, output_file = processor.process()
        logger.info("✓ Preprocessing completed successfully")
        return 0
    except Exception as e:
        logger.error(f"✗ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())