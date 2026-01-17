#!/usr/bin/env python3
"""
Minimal region partitioning using greedy bin packing.
"""

import pandas as pd
import json
import argparse
from pathlib import Path

def partition_regions(household_file, num_bins, region_column='ADM2_EN'):
    """Partition regions into bins using greedy algorithm."""
    
    # Load and count households per region
    df = pd.read_csv(household_file)
    region_counts = df.groupby(region_column)['hid'].count().to_dict()
    
    # Sort regions by household count (descending)
    sorted_regions = sorted(region_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Initialize bins
    bins = [[] for _ in range(num_bins)]
    bin_totals = [0] * num_bins
    
    # Greedy assignment: assign each region to bin with smallest total
    for region, count in sorted_regions:
        min_bin = bin_totals.index(min(bin_totals))
        bins[min_bin].append(region)
        bin_totals[min_bin] += count
    
    # Format as {lead_region: [other_regions]}
    partition_dict = {}
    for bin_regions in bins:
        if bin_regions:
            partition_dict[bin_regions[0]] = bin_regions[1:]
    
    # Print stats
    print(f"Partitioned {len(region_counts)} regions into {num_bins} bins")
    for i, total in enumerate(bin_totals):
        print(f"  Bin {i}: {total:,} households ({len(bins[i])} regions)")
    
    return partition_dict

def main():
    parser = argparse.ArgumentParser(description='Partition regions into balanced bins')
    parser.add_argument('--household-file', required=True, help='Household data CSV')
    parser.add_argument('--num-bins', type=int, required=True, help='Number of bins')
    parser.add_argument('--region-column', default='ADM2_EN', help='Region column name')
    parser.add_argument('--output', required=True, help='Output JSON file')
    args = parser.parse_args()
    
    partition_dict = partition_regions(args.household_file, args.num_bins, args.region_column)
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(partition_dict, f, indent=2)
    
    print(f"\nSaved to: {args.output}")

if __name__ == '__main__':
    main()