#!/usr/bin/env python3
"""
Setup script to initialize multi-seed calibration runs
"""

import json
import argparse
import subprocess
import pandas as pd
from pathlib import Path
from result_utils import *

## change this two variable depending on starting point

original_seed_dir = '../intention/LHC_configs/'
original_seed_file_prefix = 'config_LHC_sim_'
ground_truth_data = '/project/biocomplexity/UKR_forecast/migration_data/gtruth_from_various_source/ukr_refugee_src_pnas.csv'

def create_initial_config(seed, param_ranges):
    """
    Create initial configuration file for a calibration seed
    
    Args:
        seed: Unique seed identifier
        param_ranges: Dictionary with 'lo' and 'hi' for each parameter
        initial_params: Optional starting parameter values (otherwise use midpoint)
    """
    
    param_names = [
        "param_migration_bias", 
        "param_distance_decay", 
        "param_discount_rate",
        "param_risk_growth_rate",
        "param_threshold_hi",
        "param_lambda1",
        "param_lambda2",
        "refugee_among_displaced"
    ]
    
    # Set initial parameters (midpoint if not provided)
    with open(f'{original_seed_dir}{original_seed_file_prefix}{seed}.json') as f:
        initial_params = json.load(f)
    
    if initial_params is None:
        initial_params = {}
        for param in param_names:
            initial_params[param] = (param_ranges['lo'][param] + param_ranges['hi'][param]) / 2.0
    
    pnas_refugee_data = pd.read_csv(ground_truth_data)
    pnas_refugee_data['time'] = pd.to_datetime(pnas_refugee_data['time'])
    pnas_refugee_data['daily'] = pnas_refugee_data['daily'].rolling(7).mean()
    pnas_refugee_data = pnas_refugee_data.dropna(subset=['daily'])
    all_dfs = get_result_for_sim(seed)
    daily_total_refugee_df = get_daily_total_refugee_for_sim(all_dfs)
    nrmse,_,corr,_,_ = get_metrics_of_daily_sim(daily_total_refugee_df,pnas_refugee_data,upto='2022-03-15')
    
    config = {
        'seed': seed,
        'params': initial_params,
        'lo': param_ranges['lo'],
        'hi': param_ranges['hi'],
        'nrmse': nrmse,  # Initial placeholder
        'pcc': corr     # Initial placeholder
    }
    
    # Create config directory if it doesn't exist
    config_dir = Path('CoordinateDescentconfig')
    config_dir.mkdir(exist_ok=True)
    
    # Save initial config (epoch 0)
    config_file = config_dir / f"config_coord_epoch_0_from_{seed}.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Created initial config: {config_file}")
    return config_file


def submit_initial_pipeline(seed, max_epochs=15):
    """Submit the first epoch of the pipeline"""
    
    # Create logs directory
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    # Submit pipeline for epoch 0
    cmd = [
        'sbatch',
        '--partition','bii',
        '--account','nssac_covid19',
        '--job-name', f'CD_seed{seed}_ep0',
        '--output', f'logs/CD_seed{seed}_epoch0_%j.out',
        '--time', '10:00:00',  # 24 hour time limit
        '--wrap', f'python3 coordinate_descent_pipeline.py --seed {seed} --epoch 0 --max-epochs {max_epochs} --mode run'
    ]
    
    print(f"\nSubmitting initial pipeline job...")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✓ Pipeline submitted successfully")
        print(f"  {result.stdout.strip()}")
        return True
    else:
        print(f"✗ Failed to submit pipeline")
        print(f"  Error: {result.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Setup multi-seed calibration')
    parser.add_argument('--seeds', type=int, nargs='+', required=True,
                       help='List of seed values to initialize (e.g., --seeds 1 2 3 4 5)')
    parser.add_argument('--param-ranges-file', type=str, default='param_ranges.json',
                       help='JSON file containing parameter ranges (lo/hi for each param)')
    parser.add_argument('--max-epochs', type=int, default=17,
                       help='Maximum epochs per seed (default: 17)')
    parser.add_argument('--submit', action='store_true',
                       help='Automatically submit jobs after creating configs')
    
    args = parser.parse_args()
    
    # Load parameter ranges
    try:
        with open(args.param_ranges_file) as f:
            param_ranges = json.load(f)
    except FileNotFoundError:
        print(f"Error: Parameter ranges file not found: {args.param_ranges_file}")
        print("\nCreating template parameter ranges file...")
        
        # Create template
        template = {
            'lo': {
                "param_migration_bias": 0.0,
                "param_distance_decay": 1.0,
                "param_discount_rate": 0.0,
                "param_risk_growth_rate": 1.0,
                "param_threshold_hi": 0.0,
                "param_lambda1": 0.0,
                "param_lambda2": 0.0,
                "refugee_among_displaced": 0.0
            },
            'hi': {
                "param_migration_bias": 1000.0,
                "param_distance_decay": 10.0,
                "param_discount_rate": 1.0,
                "param_risk_growth_rate": 100.0,
                "param_threshold_hi": 1.0,
                "param_lambda1": 1.0,
                "param_lambda2": 1.0,
                "refugee_among_displaced": 1.0
            }
        }
        
        with open(args.param_ranges_file, 'w') as f:
            json.dump(template, f, indent=2)
        
        print(f"✓ Created template: {args.param_ranges_file}")
        print("Please edit this file with appropriate ranges and re-run.")
        return
    
    print(f"\n{'='*60}")
    print(f"MULTI-SEED CALIBRATION SETUP")
    print(f"{'='*60}")
    print(f"Seeds: {args.seeds}")
    print(f"Max epochs per seed: {args.max_epochs}")
    print(f"Total iterations: {len(args.seeds) * args.max_epochs}")
    print(f"{'='*60}\n")
    
    # Create configs for each seed
    for seed in args.seeds:
        print(f"\nInitializing seed {seed}...")
        config_file = create_initial_config(seed, param_ranges)
        
        if args.submit:
            submit_initial_pipeline(seed, args.max_epochs)
    
    if not args.submit:
        print(f"\n{'='*60}")
        print(f"SETUP COMPLETE")
        print(f"{'='*60}")
        print(f"Configurations created for {len(args.seeds)} seeds.")
        print(f"\nTo start calibration, run:")
        for seed in args.seeds:
            print(f"  python3 coordinate_descent_pipeline.py --seed {seed} --epoch 0 --mode run")
        print(f"\nOr re-run with --submit flag to automatically submit jobs.")


if __name__ == '__main__':
    main()