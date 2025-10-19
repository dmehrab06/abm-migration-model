#!/usr/bin/env python3
"""
Monitor and analyze multi-seed calibration progress
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime

class CalibrationMonitor:
    def __init__(self, config_dir='CoordinateDescentconfig'):
        self.config_dir = Path(config_dir)
        
    def get_all_seeds(self):
        """Find all unique seeds in the config directory"""
        seeds = set()
        for file in self.config_dir.glob('config_coord_epoch_*_from_*.json'):
            seed = int(file.stem.split('_from_')[-1])
            seeds.add(seed)
        return sorted(seeds)
    
    def get_seed_progress(self, seed):
        """Get progress information for a specific seed"""
        epochs_completed = []
        
        for file in sorted(self.config_dir.glob(f'summary_seed_{seed}_epoch_*.json')):
            with open(file) as f:
                summary = json.load(f)
                epochs_completed.append(summary)
        
        # Check if final summary exists
        final_file = self.config_dir / f'FINAL_summary_seed_{seed}.json'
        is_complete = final_file.exists()
        
        return {
            'seed': seed,
            'epochs_completed': len(epochs_completed),
            'is_complete': is_complete,
            'summaries': epochs_completed
        }
    
    def print_status(self):
        """Print current status of all calibration runs"""
        seeds = self.get_all_seeds()
        
        if not seeds:
            print("No calibration runs found.")
            return
        
        print(f"\n{'='*80}")
        print(f"CALIBRATION STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
        
        status_data = []
        
        for seed in seeds:
            progress = self.get_seed_progress(seed)
            
            if progress['summaries']:
                latest = progress['summaries'][-1]
                current_nrmse = latest.get('best_nrmse', 'N/A')
                current_pcc = latest.get('best_pcc', 'N/A')
            else:
                current_nrmse = 'N/A'
                current_pcc = 'N/A'
            
            status_data.append({
                'Seed': seed,
                'Epochs': f"{progress['epochs_completed']}/17",
                'Status': '✓ Complete' if progress['is_complete'] else '⟳ Running',
                'NRMSE': f"{current_nrmse:.6f}" if current_nrmse != 'N/A' else 'N/A',
                'PCC': f"{current_pcc:.6f}" if current_pcc != 'N/A' else 'N/A'
            })
        
        df = pd.DataFrame(status_data)
        print(df.to_string(index=False))
        print(f"\n{'='*80}\n")
    
    def plot_convergence(self, seeds=None, save_path='convergence_plot.png'):
        """Plot convergence curves for specified seeds"""
        if seeds is None:
            seeds = self.get_all_seeds()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for seed in seeds:
            progress = self.get_seed_progress(seed)
            
            if not progress['summaries']:
                continue
            
            epochs = []
            nrmses = []
            pccs = []
            
            for summary in progress['summaries']:
                epochs.append(summary['epoch'])
                nrmses.append(summary.get('best_nrmse', np.nan))
                pccs.append(summary.get('best_pcc', np.nan))
            
            # Plot NRMSE
            axes[0].plot(epochs, nrmses, marker='o', label=f'Seed {seed}')
            
            # Plot PCC
            axes[1].plot(epochs, pccs, marker='o', label=f'Seed {seed}')
        
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('NRMSE')
        axes[0].set_title('NRMSE Convergence')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('PCC')
        axes[1].set_title('PCC Convergence')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Convergence plot saved: {save_path}")
        plt.close()
    
    def compare_final_parameters(self, save_path='parameter_comparison.png'):
        """Compare final parameters across all seeds"""
        seeds = self.get_all_seeds()
        
        param_data = []
        
        for seed in seeds:
            final_file = self.config_dir / f'FINAL_summary_seed_{seed}.json'
            
            if final_file.exists():
                with open(final_file) as f:
                    final = json.load(f)
                    
                    if final['final_params']:
                        for param, value in final['final_params'].items():
                            param_data.append({
                                'Seed': seed,
                                'Parameter': param,
                                'Value': value
                            })
        
        if not param_data:
            print("No completed calibrations found.")
            return
        
        df = pd.DataFrame(param_data)
        
        # Create box plots for each parameter
        params = df['Parameter'].unique()
        n_params = len(params)
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i, param in enumerate(params):
            param_df = df[df['Parameter'] == param]
            
            axes[i].boxplot([param_df['Value'].values])
            axes[i].scatter([1] * len(param_df), param_df['Value'], alpha=0.6)
            axes[i].set_title(param.replace('param_', ''))
            axes[i].set_ylabel('Value')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Parameter comparison saved: {save_path}")
        plt.close()
    
    def generate_summary_table(self, save_path='calibration_summary.csv'):
        """Generate summary table of all calibrations"""
        seeds = self.get_all_seeds()
        
        summary_data = []
        
        for seed in seeds:
            final_file = self.config_dir / f'FINAL_summary_seed_{seed}.json'
            
            if final_file.exists():
                with open(final_file) as f:
                    final = json.load(f)
                    
                    row = {
                        'Seed': seed,
                        'NRMSE': final['final_nrmse'],
                        'PCC': final['final_pcc'],
                        'Completion_Time': final['completion_time']
                    }
                    
                    # Add all parameters
                    if final['final_params']:
                        row.update(final['final_params'])
                    
                    summary_data.append(row)
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            df.to_csv(save_path, index=False)
            print(f"✓ Summary table saved: {save_path}")
            
            # Print statistics
            print(f"\n{'='*60}")
            print(f"CALIBRATION STATISTICS")
            print(f"{'='*60}")
            print(f"Number of completed runs: {len(df)}")
            print(f"\nNRMSE:")
            print(f"  Mean: {df['NRMSE'].mean():.6f}")
            print(f"  Std:  {df['NRMSE'].std():.6f}")
            print(f"  Min:  {df['NRMSE'].min():.6f}")
            print(f"  Max:  {df['NRMSE'].max():.6f}")
            print(f"\nPCC:")
            print(f"  Mean: {df['PCC'].mean():.6f}")
            print(f"  Std:  {df['PCC'].std():.6f}")
            print(f"  Min:  {df['PCC'].min():.6f}")
            print(f"  Max:  {df['PCC'].max():.6f}")
            print(f"{'='*60}\n")
            
            return df
        else:
            print("No completed calibrations found.")
            return None


def main():
    parser = argparse.ArgumentParser(description='Monitor calibration progress')
    parser.add_argument('--config-dir', default='CoordinateDescentconfig',
                       help='Configuration directory')
    parser.add_argument('--status', action='store_true',
                       help='Show current status')
    parser.add_argument('--plot', action='store_true',
                       help='Generate convergence plots')
    parser.add_argument('--compare', action='store_true',
                       help='Compare final parameters')
    parser.add_argument('--summary', action='store_true',
                       help='Generate summary table')
    parser.add_argument('--all', action='store_true',
                       help='Run all analyses')
    parser.add_argument('--seeds', type=int, nargs='+',
                       help='Specific seeds to analyze (default: all)')
    
    args = parser.parse_args()
    
    monitor = CalibrationMonitor(args.config_dir)
    
    # Default to showing status if no options specified
    if not (args.status or args.plot or args.compare or args.summary or args.all):
        args.status = True
    
    if args.status or args.all:
        monitor.print_status()
    
    if args.plot or args.all:
        monitor.plot_convergence(args.seeds)
    
    if args.compare or args.all:
        monitor.compare_final_parameters()
    
    if args.summary or args.all:
        monitor.generate_summary_table()


if __name__ == '__main__':
    main()