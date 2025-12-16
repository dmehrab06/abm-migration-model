#!/usr/bin/env python3
"""
Batch generate household networks for all regions.
Usage: python batch_generate_networks.py --radius 0.01 --proximity 0.04 --alpha 2.3 --max-long-links 8
"""
import pandas as pd
import argparse
import subprocess
import time
import os

def main():
    parser = argparse.ArgumentParser(description='Batch generate networks for all regions')
    parser.add_argument('--radius', type=float, default=0.01, help='Search radius')
    parser.add_argument('--proximity', type=float, default=0.04, help='Proximity threshold (km)')
    parser.add_argument('--alpha', type=float, default=2.3, help='Distance decay exponent')
    parser.add_argument('--max-long-links', type=int, default=8, help='Max long-range links')
    parser.add_argument('--household-file', type=str, 
                       default='/project/biocomplexity/UKR_forecast/migration_data/household_data/venezuela_household_data_ADM2_HDX.csv',
                       help='Household data file')
    args = parser.parse_args()
    
    # Get unique regions
    print(f"Loading regions from {args.household_file}...")
    df = pd.read_csv(args.household_file)
    regions = df['matching_place_id'].unique()
    print(f"Found {len(regions)} unique regions")
    
    # Submit job for each region
    for i, region in enumerate(regions, 1):
        print(f"[{i}/{len(regions)}] Submitting {region}...", end=' ')
        
        # Create sbatch file
        with open('gen_network.sbatch', 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("#SBATCH -N 1\n")
            f.write("#SBATCH -n 1\n")
            f.write("#SBATCH --cpus-per-task=1\n")
            f.write("#SBATCH -t 01:00:00\n")
            f.write("#SBATCH -p bii\n")
            f.write("#SBATCH --mem=64G\n")
            f.write("#SBATCH -A nssac_covid19\n")
            f.write("#SBATCH --job-name=net_gen\n")
            f.write("module load miniforge\n")
            f.write("conda activate migration_env\n")
            f.write(f"python generate_household_network.py "
                   f"--household-file {args.household_file} "
                   f"--region {region} "
                   f"--radius {args.radius} "
                   f"--proximity {args.proximity} "
                   f"--alpha {args.alpha} "
                   f"--max-long-links {args.max_long_links}\n")
        
        # Submit
        result = subprocess.run(['sbatch', 'gen_network.sbatch'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ {result.stdout.strip()}")
        else:
            print(f"✗ FAILED: {result.stderr.strip()}")
        
        time.sleep(0.1)  # Small delay to avoid overwhelming scheduler
    
    print(f"\nSubmitted {len(regions)} jobs. Monitor with: squeue -u $USER")
    os.remove('gen_network.sbatch')

if __name__ == '__main__':
    main()
    