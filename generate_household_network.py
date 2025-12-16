#!/usr/bin/env python3
"""
Household Social Network Generator using KSW (Kleinberg-Strogatz-Watts) Model

This script generates spatial social networks for households within a specified region.
The network follows a hybrid model combining:
1. Short-range links: Deterministic connections within proximity threshold P
2. Long-range links: Stochastic connections with gravity-based probability decay

The KSW model is designed to capture realistic social network properties including:
- Strong local clustering (short-range links)
- Small-world properties via long-range shortcuts
- Spatial embedding with distance-dependent connection probability

Author: Zakaria
Date: 2024
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import s2sphere
import argparse
import json
import time
import logging
import warnings
from pathlib import Path
from file_paths_and_consts import *

warnings.filterwarnings('ignore')

# ========================== S2 GEOMETRY UTILITIES ==========================

def get_s2_cell(lat, lng, level=13):
    """
    Convert geographic coordinates to S2 geometry cell at specified level.
    
    S2 geometry provides hierarchical spatial indexing. Level 13 cells are
    approximately 1.27 km² in area, suitable for household-level spatial
    partitioning for computational efficiency.
    
    Args:
        lat (float): Latitude in decimal degrees
        lng (float): Longitude in decimal degrees
        level (int): S2 cell level (default: 13, ~1.27 km² per cell)
        
    Returns:
        s2sphere.CellId: S2 cell identifier at the requested level
        
    Note:
        Higher levels = finer resolution. Level 13 balances spatial resolution
        with computational efficiency for region-level simulations.
    """
    point = s2sphere.LatLng.from_degrees(lat, lng)
    cell = s2sphere.Cell.from_lat_lng(point)
    cell_id = cell.id()
    
    # Navigate up the S2 hierarchy to requested level
    for _ in range(1, 30):
        if cell_id.level() == level:
            return cell_id
        cell_id = cell_id.parent()
    
    return cell_id


def get_cell_core_id(s2_cell_id):
    """
    Hash S2 cell to a core ID for parallel processing.
    
    This hash function distributes cells across processing cores. The specific
    formula uses S2's token representation (base-16) to ensure even distribution.
    
    Args:
        s2_cell_id (s2sphere.CellId): S2 cell identifier
        
    Returns:
        int: Core ID for processor assignment
        
    Note:
        Division by 16 and modulo operations ensure balanced workload distribution
        across available CPU cores.
    """
    token_value = int(s2_cell_id.to_token(), 16)
    return token_value // 16


# ========================== DISTANCE CALCULATION ==========================

def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calculate great-circle distance between two points on Earth.
    
    Uses the Haversine formula to compute the shortest distance over Earth's
    surface, accounting for spherical geometry. Critical for accurate spatial
    network construction.
    
    Args:
        lon1, lat1 (float or array): Longitude/latitude of first point(s)
        lon2, lat2 (float or array): Longitude/latitude of second point(s)
        
    Returns:
        float or array: Distance in kilometers
        
    Note:
        - Assumes Earth radius of 6372.8 km
        - Vectorized for efficient pandas operations
        - Accuracy within 0.5% for distances < 500 km
    """
    EARTH_RADIUS_KM = 6372.8
    
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return EARTH_RADIUS_KM * c


# ========================== NETWORK GENERATION ==========================

class HouseholdNetworkGenerator:
    """
    Generate spatial social networks using KSW model.
    
    This class implements a Kleinberg-Strogatz-Watts spatial network model
    where connection probability decays with distance following a power law.
    
    Attributes:
        region_name (str): Geographic region identifier (e.g., 'Kharkivska')
        radius (float): Search radius in decimal degrees for potential connections
        proximity_threshold (float): Distance threshold (km) for short-range links
        alpha (float): Distance decay exponent for long-range link probability
        max_long_links (int): Maximum long-range links per household
        s2_level (int): S2 geometry level for spatial indexing
        logger (logging.Logger): Logger instance for tracking operations
    """
    
    def __init__(self, region_name, radius=0.01, proximity_threshold=0.04, 
                 alpha=2.3, max_long_links=8, s2_level=13, logger=None):
        """
        Initialize network generator with KSW model parameters.
        
        Args:
            region_name (str): Region to generate network for
            radius (float): Search radius in decimal degrees (default: 0.01 ≈ 1.1 km)
            proximity_threshold (float): Short-range threshold in km (default: 0.04 km = 40m)
            alpha (float): Power-law exponent for distance decay (default: 2.3)
                          Higher alpha = faster decay = more local connections
            max_long_links (int): Cap on long-range links per node (default: 8)
            s2_level (int): S2 cell level for spatial partitioning (default: 13)
            logger (logging.Logger): Optional logger instance
        """
        self.region_name = region_name
        self.radius = radius
        self.proximity_threshold = proximity_threshold
        self.alpha = alpha
        self.max_long_links = max_long_links
        self.s2_level = s2_level
        self.logger = logger or logging.getLogger(__name__)
        
        # Validate parameters
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate model parameters are within reasonable ranges."""
        assert self.radius > 0, "Search radius must be positive"
        assert self.proximity_threshold > 0, "Proximity threshold must be positive"
        assert self.alpha >= 0, "Alpha (distance decay) must be non-negative"
        assert self.max_long_links > 0, "Max long-range links must be positive"
        assert 1 <= self.s2_level <= 30, "S2 level must be between 1 and 30"
        
        self.logger.info(f"Network parameters validated: R={self.radius}, "
                        f"P={self.proximity_threshold}km, α={self.alpha}, "
                        f"Q={self.max_long_links}")
    
    def load_household_data(self, household_file):
        """
        Load and filter household data for the specified region.
        
        Args:
            household_file (str): Path to household data CSV
            
        Returns:
            pd.DataFrame: Filtered household data with S2 cells assigned
            
        Note:
            Adds columns: 's2_cell' (S2 geometry), 's2_id' (core assignment)
        """
        self.logger.info(f"Loading household data from: {household_file}")
        
        # Load full dataset
        household_data = pd.read_csv(household_file)
        initial_count = len(household_data)
        
        # Filter to region of interest
        household_data = household_data[
            household_data.matching_place_id == self.region_name
        ]
        
        region_count = len(household_data)
        self.logger.info(f"Filtered to {region_count:,} households in {self.region_name} "
                        f"(from {initial_count:,} total)")
        
        if region_count == 0:
            raise ValueError(f"No households found for region: {self.region_name}")
        
        # Assign S2 geometry cells for spatial indexing
        self.logger.info(f"Computing S2 cells at level {self.s2_level}...")
        household_data['s2_cell'] = household_data.apply(
            lambda row: get_s2_cell(row['latitude'], row['longitude'], self.s2_level),
            axis=1
        )
        household_data['s2_id'] = household_data['s2_cell'].apply(get_cell_core_id)
        
        self.logger.info(f"Assigned {household_data['s2_id'].nunique()} unique S2 cells")
        
        return household_data
    
    def create_spatial_network(self, household_data):
        """
        Generate social network using spatial join and KSW model.
        
        This is the core network generation algorithm:
        1. Create buffer zones around each household (radius R)
        2. Find all households within each buffer (spatial join)
        3. Classify connections as short-range or long-range
        4. Apply stochastic acceptance for long-range links
        
        Args:
            household_data (pd.DataFrame): Household data with coordinates
            
        Returns:
            pd.DataFrame: Network edge list with columns [hid_x, hid_y, distance, type]
            
        Algorithm:
            - Short-range links: distance ≤ P (deterministic, all accepted)
            - Long-range links: distance > P, probability ∝ 1/d^α (stochastic)
            - Each household limited to Q long-range connections
        """
        self.logger.info("Converting household data to GeoDataFrame...")
        
        # Create point geometries for households
        household_gdf = gpd.GeoDataFrame(
            household_data[['hid', 'latitude', 'longitude']],
            geometry=gpd.points_from_xy(
                household_data['longitude'],
                household_data['latitude']
            ),
            crs='EPSG:4326'  # WGS84 coordinate system
        )
        
        # Create buffer zones for spatial search
        self.logger.info(f"Creating {self.radius}° buffer zones around households...")
        buffer_gdf = household_gdf.copy()
        buffer_gdf['geometry'] = buffer_gdf['geometry'].buffer(self.radius)
        
        # Spatial join: find all households within each buffer
        self.logger.info("Performing spatial join (this may take several minutes)...")
        start_time = time.time()
        
        potential_edges = buffer_gdf.sjoin(household_gdf, how='inner', predicate='intersects')
        
        elapsed = time.time() - start_time
        self.logger.info(f"Spatial join completed in {elapsed:.2f} seconds")
        self.logger.info(f"Found {len(potential_edges):,} potential connections")
        
        # Calculate exact distances between connected households
        self.logger.info("Computing distances for potential edges...")
        potential_edges['distance_km'] = haversine_distance(
            potential_edges['longitude_left'],
            potential_edges['latitude_left'],
            potential_edges['longitude_right'],
            potential_edges['latitude_right']
        )
        
        # Normalize distances by proximity threshold
        potential_edges['normalized_distance'] = (
            potential_edges['distance_km'] / self.proximity_threshold
        )
        
        return potential_edges
    
    def apply_ksw_model(self, potential_edges):
        """
        Apply KSW model to accept/reject long-range connections.
        
        Connection rules:
        1. SHORT-RANGE: normalized_distance ≤ 1 → Accept all (strong local ties)
        2. LONG-RANGE: normalized_distance > 1 → Probabilistic acceptance
           - Probability = 1 / (normalized_distance)^α
           - Each node limited to Q long-range links
        
        Args:
            potential_edges (pd.DataFrame): All potential connections with distances
            
        Returns:
            pd.DataFrame: Accepted network edges with connection type labels
            
        Note:
            The power-law decay (α typically 2-3) creates small-world properties:
            - α=2: Navigation optimal (Kleinberg 2000)
            - α>2: More clustered, harder to navigate
            - α<2: More random, less clustered
        """
        self.logger.info("Applying KSW connection model...")
        
        # STEP 1: Accept all short-range connections (normalized distance ≤ 1)
        short_range_mask = potential_edges['normalized_distance'] <= 1.0
        short_edges = potential_edges[short_range_mask].copy()
        short_edges['connection_type'] = 'short_range'
        
        self.logger.info(f"Accepted {len(short_edges):,} short-range connections "
                        f"(distance ≤ {self.proximity_threshold} km)")
        
        # STEP 2: Stochastically accept long-range connections
        long_candidates = potential_edges[~short_range_mask].copy()
        
        if len(long_candidates) > 0:
            # Calculate acceptance probability: P(connect) = 1 / d^α
            long_candidates['connection_prob'] = (
                1.0 / (long_candidates['normalized_distance'] ** self.alpha)
            )
            
            # Draw random values for each candidate
            long_candidates['random_draw'] = np.random.random(len(long_candidates))
            
            # Accept if random draw < connection probability
            accepted_mask = (
                long_candidates['connection_prob'] > long_candidates['random_draw']
            )
            accepted_long = long_candidates[accepted_mask].copy()
            
            # Limit each household to Q long-range connections
            # Sort by probability (favor stronger connections) and take top Q per node
            accepted_long = (
                accepted_long
                .sort_values('connection_prob', ascending=False)
                .groupby('hid_left')
                .head(self.max_long_links)
            )
            
            accepted_long['connection_type'] = 'long_range'
            
            self.logger.info(f"Accepted {len(accepted_long):,} long-range connections "
                           f"(max {self.max_long_links} per household)")
        else:
            accepted_long = pd.DataFrame()
            self.logger.info("No long-range connections to process")
        
        # Combine short and long-range edges
        network = pd.concat([short_edges, accepted_long], ignore_index=True)
        
        # Calculate network statistics
        self._log_network_statistics(network, short_edges, accepted_long)
        
        return network
    
    def _log_network_statistics(self, network, short_edges, long_edges):
        """Log summary statistics about the generated network."""
        total_edges = len(network)
        num_short = len(short_edges)
        num_long = len(long_edges)
        
        unique_nodes = pd.concat([
            network['hid_left'],
            network['hid_right']
        ]).nunique()
        
        avg_degree = (2 * total_edges) / unique_nodes if unique_nodes > 0 else 0
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"NETWORK STATISTICS")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Total edges:        {total_edges:,}")
        self.logger.info(f"  Short-range:      {num_short:,} ({num_short/total_edges*100:.1f}%)")
        self.logger.info(f"  Long-range:       {num_long:,} ({num_long/total_edges*100:.1f}%)")
        self.logger.info(f"Unique households:  {unique_nodes:,}")
        self.logger.info(f"Average degree:     {avg_degree:.2f}")
        self.logger.info(f"{'='*60}\n")
    
    def format_and_save_network(self, network, household_data, output_file):
        """
        Format network edge list and save to CSV.
        
        Final edge list contains:
        - hid_x, hid_y: Household IDs (undirected edge)
        - s2_id: S2 cell ID for spatial partitioning in simulation
        
        Args:
            network (pd.DataFrame): Network with all edge attributes
            household_data (pd.DataFrame): Original household data with S2 cells
            output_file (str): Output CSV file path
        """
        self.logger.info("Formatting network for output...")
        
        # Rename columns to match simulation requirements
        network = network.rename(columns={
            'hid_left': 'hid_x',
            'hid_right': 'hid_y',
            'latitude_left': 'latitude_x',
            'latitude_right': 'latitude_y',
            'longitude_left': 'longitude_x',
            'longitude_right': 'longitude_y'
        })
        
        # Merge with household data to get S2 cell assignments
        # This allows the simulation to partition network by spatial cells for efficiency
        network_with_cells = network[['hid_x', 'hid_y']].merge(
            household_data[['hid', 's2_id']],
            left_on='hid_x',
            right_on='hid',
            how='inner'
        )[['hid_x', 'hid_y', 's2_id']]
        
        # Save to CSV
        self.logger.info(f"Saving network to: {output_file}")
        network_with_cells.to_csv(output_file, index=False)
        self.logger.info(f"Network saved successfully ({len(network_with_cells):,} edges)")
        
        return network_with_cells
    
    def generate(self, household_file, output_file):
        """
        Complete network generation pipeline.
        
        Args:
            household_file (str): Path to input household data
            output_file (str): Path for output network CSV
            
        Returns:
            pd.DataFrame: Final network edge list
        """
        self.logger.info(f"\n{'#'*60}")
        self.logger.info(f"# HOUSEHOLD NETWORK GENERATION - KSW MODEL")
        self.logger.info(f"# Region: {self.region_name}")
        self.logger.info(f"{'#'*60}\n")
        
        start_time = time.time()
        
        # Step 1: Load and prepare data
        household_data = self.load_household_data(household_file)
        
        # Step 2: Create spatial network
        potential_edges = self.create_spatial_network(household_data)
        
        # Step 3: Apply KSW model
        network = self.apply_ksw_model(potential_edges)
        
        # Step 4: Format and save
        final_network = self.format_and_save_network(network, household_data, output_file)
        
        elapsed = (time.time() - start_time) / 60.0
        self.logger.info(f"\nTotal execution time: {elapsed:.2f} minutes")
        
        return final_network


# ========================== CONFIGURATION & CLI ==========================

def parse_arguments():
    """
    Parse command-line arguments with config file support.
    
    Supports two modes:
    1. Config file: --config network_config.json
    2. Direct CLI: --region Kharkivska --radius 0.01 ...
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    # First parser: just for --config
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        '--config',
        type=str,
        help='Path to JSON configuration file'
    )
    args, remaining = config_parser.parse_known_args()
    
    # Load defaults from config file if provided
    defaults = {}
    if args.config:
        with open(args.config, 'r') as f:
            defaults = json.load(f)
    
    # Main parser with all arguments
    parser = argparse.ArgumentParser(
        parents=[config_parser],
        description='Generate household social networks using KSW spatial model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using config file:
  python generate_household_network.py --config network_config.json
  
  # Using CLI arguments:
  python generate_household_network.py --region Kharkivska --radius 0.01 --proximity 0.04
  
  # Mix of both (CLI overrides config):
  python generate_household_network.py --config network_config.json --alpha 2.5
        """
    )
    
    parser.set_defaults(**defaults)
    
    # Required arguments
    parser.add_argument(
        '--region',
        type=str,
        required=True,
        help='Region name to generate network for (e.g., "Kharkivska for Ukraine")'
    )
    
    # Network model parameters
    parser.add_argument(
        '--radius',
        type=float,
        default=0.01,
        help='Search radius in decimal degrees (default: 0.01 ≈ 1.1 km at 45° latitude)'
    )
    
    parser.add_argument(
        '--proximity',
        type=float,
        default=0.04,
        help='Proximity threshold in km for short-range links (default: 0.04 km = 40m)'
    )
    
    parser.add_argument(
        '--alpha',
        type=float,
        default=2.3,
        help='Distance decay exponent for long-range links (default: 2.3). '
             'Higher values = more local clustering'
    )
    
    parser.add_argument(
        '--max-long-links',
        type=int,
        default=8,
        help='Maximum long-range links per household (default: 8)'
    )
    
    # Data paths
    parser.add_argument(
        '--household-file',
        type=str,
        default=f'{HOUSEHOLD_DIR}ukraine_household_data_ADM2_HDX.csv',
        help='Path to household data CSV file'
    )
    
    parser.add_argument(
        '--output-file',
        type=str,
        help='Output network file path. If not specified, auto-generated from parameters'
    )
    
    # Technical parameters
    parser.add_argument(
        '--s2-level',
        type=int,
        default=13,
        help='S2 geometry level for spatial indexing (default: 13, ~1.27 km²)'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        help='Log file path (default: logs to console)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args(remaining)


def setup_logging(log_file=None, verbose=False):
    """
    Configure logging with optional file output.
    
    Args:
        log_file (str): Optional path to log file
        verbose (bool): If True, use DEBUG level; otherwise INFO
    """
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
    
    # Auto-generate output filename if not provided
    if args.output_file is None:
        args.output_file = (
            f'{HOUSEHOLD_DIR}KSW_HH_BALL_AAMAS_{args.region}_'
            f'R_{args.radius}_P_{args.proximity}_Q_{args.max_long_links}_'
            f'al_{args.alpha}.csv'
        )
    
    # Create network generator
    generator = HouseholdNetworkGenerator(
        region_name=args.region,
        radius=args.radius,
        proximity_threshold=args.proximity,
        alpha=args.alpha,
        max_long_links=args.max_long_links,
        s2_level=args.s2_level,
        logger=logger
    )
    
    # Generate network
    try:
        network = generator.generate(args.household_file, args.output_file)
        logger.info("✓ Network generation completed successfully")
        return 0
    except Exception as e:
        logger.error(f"✗ Network generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
