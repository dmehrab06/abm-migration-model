# trajectory_tracker.py
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
from numpy import dot
from numpy.linalg import norm
from file_paths_and_consts import *
import os
import subprocess
import ast
import math
from matplotlib import rcParams
import matplotlib.dates as mdates
import json
from result_utils import *

SMALL_SIZE = 16
MEDIUM_SIZE = 22
BIGGER_SIZE = 30

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def load_calibration_trajectory(seed, config_dir='CoordinateDescentconfig'):
    """Load all epoch summaries for a given seed"""
    config_path = Path(config_dir)
    
    trajectory = {
        'epoch': [],
        'parameter': [],
        'best_value': [],
        'nrmse': [],
        'pcc': [],
        'search_range': []
    }
    
    epoch = 0
    while True:
        summary_file = config_path / f"summary_seed_{seed}_epoch_{epoch}.json"
        config_file = config_path / f"config_coord_epoch_{epoch}_from_{seed}.json"
        
        if not summary_file.exists():
            break
            
        with open(summary_file) as f:
            summary = json.load(f)
        
        with open(config_file) as f:
            config = json.load(f)
            
        trajectory['epoch'].append(epoch)
        trajectory['parameter'].append(summary['parameter'])
        trajectory['best_value'].append(config['params'][summary['parameter']])
        trajectory['nrmse'].append(summary.get('best_nrmse', None))
        trajectory['pcc'].append(summary.get('best_pcc', None))
        trajectory['search_range'].append({
            'lo': config['lo'][summary['parameter']],
            'hi': config['hi'][summary['parameter']]
        })
        
        epoch += 1
    
    return pd.DataFrame(trajectory)

def plot_trajectory(seed):
    df = load_calibration_trajectory(seed)
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    
    # Plot 1: Parameter values over epochs
    ax1 = axes[0]
    for param in df['parameter'].unique():
        param_df = df[df['parameter'] == param]
        ax1.scatter(param_df['epoch'], param_df['best_value'], label=param, s=50)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Parameter Value')
    ax1.set_title(f'Parameter Trajectory for Seed {seed}')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss metrics
    ax2 = axes[1]
    ax2.plot(df['epoch'], df['nrmse'], 'b-', label='NRMSE', marker='o')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('NRMSE', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(df['epoch'], df['pcc'], 'r-', label='PCC', marker='s')
    ax2_twin.set_ylabel('PCC', color='r')
    ax2_twin.tick_params(axis='y', labelcolor='r')
    ax2.set_title('Model Performance Over Epochs')
    ax2.grid(True, alpha=0.3)
    
    # Plot 2: Loss metrics
    ax2 = axes[2]
    ax2.plot(df['epoch'], df['nrmse']*0.7+(1-df['pcc'])*0.3, 'b-', label='NRMSE', marker='o')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('0.7(NRMSE)+0.3(PCC)', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    
    #ax2_twin = ax2.twinx()
    #ax2_twin.plot(df['epoch'], df['pcc'], 'r-', label='PCC', marker='s')
    #ax2_twin.set_ylabel('PCC', color='r')
    #ax2_twin.tick_params(axis='y', labelcolor='r')
    ax2.set_title('Model Performance Over Epochs')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Search range shrinkage
    ax3 = axes[3]
    for i, row in df.iterrows():
        range_size = row['search_range']['hi'] - row['search_range']['lo']
        ax3.bar(row['epoch'], range_size, label=row['parameter'] if i < 8 else '')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Search Range Size')
    ax3.set_title('Search Space Shrinkage')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'calibration_trajectory_seed_{seed}.png', dpi=150, bbox_inches='tight')
    plt.show()

def get_convergence_report(seed):
    """Generate a convergence analysis report"""
    df = load_calibration_trajectory(seed)
    
    print(f"Calibration Report for Seed {seed}")
    print("="*50)
    print(f"Total Epochs: {len(df)}")
    print(f"Initial NRMSE: {df.iloc[0]['nrmse']:.4f}")
    print(f"Final NRMSE: {df.iloc[-1]['nrmse']:.4f}")
    print(f"Improvement: {(1 - df.iloc[-1]['nrmse']/df.iloc[0]['nrmse'])*100:.1f}%")
    print(f"Final PCC: {df.iloc[-1]['pcc']:.4f}")
    
    # Check for convergence
    if len(df) > 5:
        recent_nrmse = df.tail(5)['nrmse'].values
        if recent_nrmse.std() < 0.001:
            print("\n✓ Model appears to have converged (low variance in recent epochs)")
        else:
            print("\n⚠ Model may benefit from additional epochs")
    
    # Parameter-wise analysis
    print("\nParameter Final Values:")
    final_config = json.load(open(f'CoordinateDescentconfig/config_coord_epoch_{len(df)}_from_{seed}.json'))
    for param, value in final_config['params'].items():
        print(f"  {param}: {value}")


def track_all_simulations(seed, config_dir='CoordinateDescentconfig'):
    """Track all simulations used across epochs"""
    config_path = Path(config_dir)
    
    all_simulations = []
    
    epoch = 0
    while True:
        sim_file = config_path / f"simulations_seed_{seed}_epoch_{epoch}.json"
        summary_file = config_path / f"summary_seed_{seed}_epoch_{epoch}.json"
        
        if not sim_file.exists():
            # For backwards compatibility - calculate from simulation range
            if summary_file.exists():
                with open(summary_file) as f:
                    summary = json.load(f)
                sim_range = summary.get('simulation_range', [])
                if sim_range:
                    sim_ids = list(range(sim_range[0], sim_range[1]))
                    all_simulations.append({
                        'epoch': epoch,
                        'simulation_ids': sim_ids,
                        'parameter': summary.get('parameter', 'unknown')
                    })
            else:
                break
        else:
            with open(sim_file) as f:
                sim_data = json.load(f)
            
            with open(summary_file) as f:
                summary = json.load(f)
            
            # Find which simulation was selected as best
            best_sim = None
            if 'results' in summary:
                min_error = float('inf')
                for result in summary['results']:
                    if result.get('error', float('inf')) < min_error:
                        min_error = result['error']
                        best_sim = result['sim_id']
            
            all_simulations.append({
                'epoch': epoch,
                'parameter': sim_data['parameter'],
                'simulation_ids': sim_data['simulation_ids'],
                'parameter_values': sim_data['parameter_values'],
                'best_simulation': best_sim,
                'best_nrmse': summary.get('best_nrmse'),
                'best_pcc': summary.get('best_pcc')
            })
        
        epoch += 1
    
    return all_simulations

def print_simulation_report(seed):
    """Print a detailed report of all simulations"""
    simulations = track_all_simulations(seed)
    
    print(f"\nSimulation Usage Report for Seed {seed}")
    print("=" * 80)
    
    total_sims = 0
    for epoch_data in simulations:
        epoch = epoch_data['epoch']
        sim_ids = epoch_data['simulation_ids']
        param = epoch_data.get('parameter', 'unknown')
        best = epoch_data.get('best_simulation', 'N/A')
        
        print(f"\nEpoch {epoch} - Parameter: {param}")
        print(f"  Simulations: {sim_ids[0]} to {sim_ids[-1]} ({len(sim_ids)} total)")
        print(f"  Best: Simulation {best}")
        print(f"  Performance: NRMSE={epoch_data.get('best_nrmse', 'N/A')}, "
              f"PCC={epoch_data.get('best_pcc', 'N/A')}")
        
        total_sims += len(sim_ids)
    
    print(f"\n{'='*80}")
    print(f"Total simulations used: {total_sims}")
    print(f"Average per epoch: {total_sims/len(simulations):.1f}")

def get_simulation_matrix(seed):
    """Create a matrix showing parameter values tested for each simulation"""
    simulations = track_all_simulations(seed)
    
    df_list = []
    for epoch_data in simulations:
        if 'parameter_values' in epoch_data:
            for sim_id, param_val in zip(epoch_data['simulation_ids'], 
                                         epoch_data['parameter_values']):
                df_list.append({
                    'simulation_id': sim_id,
                    'epoch': epoch_data['epoch'],
                    'parameter': epoch_data['parameter'],
                    'value': param_val,
                    'was_best': sim_id == epoch_data.get('best_simulation', None)
                })
    
    return pd.DataFrame(df_list)

