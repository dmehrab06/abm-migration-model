#!/usr/bin/env python3
"""
Parse calibration log files to track progress and best simulations
Extracts information from logs/CD_seed{seed}_epoch{epoch}_{job_id}.out files
"""

import re
import pandas as pd
from pathlib import Path
import argparse
from collections import defaultdict
import numpy as np


class CalibrationLogParser:
    def __init__(self, logs_dir='logs'):
        self.logs_dir = Path(logs_dir)
        
    def find_log_files(self, seed=None):
        """Find all calibration log files, optionally filtered by seed"""
        if seed is not None:
            pattern = f'CD_seed{seed}_epoch*.out'
        else:
            pattern = 'CD_seed*_epoch*.out'
        
        log_files = sorted(self.logs_dir.glob(pattern))
        return log_files
    
    def extract_seed_epoch_from_filename(self, log_file):
        """Extract seed and epoch from log filename"""
        # Format: CD_seed{seed}_epoch{epoch}_{job_id}.out
        match = re.search(r'CD_seed(\d+)_epoch(\d+)_(\d+)\.out', log_file.name)
        if match:
            return int(match.group(1)), int(match.group(2)), int(match.group(3))
        return None, None, None
    
    def parse_evaluation_table(self, content):
        """
        Parse the evaluation results table from log content
        
        Example table format:
        Sim ID     Param Val       NRMSE        PCC        Error        Best?   
        ---------------------------------------------------------------------------
        73050      0.399400        0.517457     0.789482   0.425375     
        73051      0.454956        0.517457     0.789482   0.425375     
        73056      0.732734        0.508069     0.789556   0.418781     ✓
        """
        results = []
        
        # Find the evaluation table
        table_match = re.search(
            r'Evaluating \d+ simulations\.\.\.\n'
            r'Sim ID\s+Param Val\s+NRMSE\s+PCC\s+Error\s+Best\?\s*\n'
            r'-+\n'
            r'(.*?)\n={40,}',
            content,
            re.DOTALL
        )
        
        if not table_match:
            return results
        
        table_content = table_match.group(1)
        
        # Parse each row
        for line in table_content.strip().split('\n'):
            # Match: sim_id param_val nrmse pcc error [✓]
            match = re.match(
                r'\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*(✓)?',
                line
            )
            if match:
                results.append({
                    'sim_id': int(match.group(1)),
                    'param_val': float(match.group(2)),
                    'nrmse': float(match.group(3)),
                    'pcc': float(match.group(4)),
                    'error': float(match.group(5)),
                    'is_best': match.group(6) is not None
                })
        
        return results
    
    def parse_baseline_info(self, content):
        """Extract baseline error information"""
        # Baseline error: 0.424338 (NRMSE: 0.5161382199311854, PCC: 0.7898627615586336)
        match = re.search(
            r'Baseline error: ([\d.]+) \(NRMSE: ([\d.]+), PCC: ([\d.]+)\)',
            content
        )
        if match:
            return {
                'baseline_error': float(match.group(1)),
                'baseline_nrmse': float(match.group(2)),
                'baseline_pcc': float(match.group(3))
            }
        return None
    
    def parse_improvement_info(self, content):
        """
        Extract improvement information from log
        
        Example:
        ✓ Improvement found in simulation 73056
          Parameter: param_lambda1
          Old value: 0.798801
          New value: 0.732734
          NRMSE: 0.5161382199311854 → 0.508069
          PCC: 0.7898627615586336 → 0.789556
        
        Or:
        ✗ No improvement found for param_lambda2
          Keeping current value: 0.654321
        """
        # Check for improvement
        improvement_match = re.search(
            r'✓ Improvement found in simulation (\d+)\n'
            r'\s+Parameter: (\S+)\n'
            r'\s+Old value: ([\d.]+)\n'
            r'\s+New value: ([\d.]+)\n'
            r'\s+NRMSE: ([\d.]+) → ([\d.]+)\n'
            r'\s+PCC: ([\d.]+) → ([\d.]+)',
            content
        )
        
        if improvement_match:
            return {
                'improvement_found': True,
                'best_sim_id': int(improvement_match.group(1)),
                'parameter': improvement_match.group(2),
                'old_value': float(improvement_match.group(3)),
                'new_value': float(improvement_match.group(4)),
                'old_nrmse': float(improvement_match.group(5)),
                'new_nrmse': float(improvement_match.group(6)),
                'old_pcc': float(improvement_match.group(7)),
                'new_pcc': float(improvement_match.group(8))
            }
        
        # Check for no improvement
        no_improvement_match = re.search(
            r'✗ No improvement found for (\S+)\n'
            r'\s+Keeping current value: ([\d.]+)',
            content
        )
        
        if no_improvement_match:
            return {
                'improvement_found': False,
                'best_sim_id': None,
                'parameter': no_improvement_match.group(1),
                'old_value': float(no_improvement_match.group(2)),
                'new_value': float(no_improvement_match.group(2)),
                'old_nrmse': None,
                'new_nrmse': None,
                'old_pcc': None,
                'new_pcc': None
            }
        
        return None
    
    def parse_log_file(self, log_file):
        """Parse a single log file and extract all information"""
        seed, epoch, job_id = self.extract_seed_epoch_from_filename(log_file)
        
        if seed is None:
            return None
        
        try:
            with open(log_file, 'r') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {log_file}: {e}")
            return None
        
        # Parse all components
        baseline_info = self.parse_baseline_info(content)
        evaluation_results = self.parse_evaluation_table(content)
        improvement_info = self.parse_improvement_info(content)
        
        # Check if epoch completed
        completed = '✓ Epoch summary saved:' in content or 'SCHEDULING NEXT EPOCH' in content
        
        return {
            'seed': seed,
            'epoch': epoch,
            'job_id': job_id,
            'log_file': str(log_file),
            'completed': completed,
            'baseline_info': baseline_info,
            'evaluation_results': evaluation_results,
            'improvement_info': improvement_info
        }
    
    def parse_all_logs(self, seed=None):
        """Parse all log files for given seed(s)"""
        log_files = self.find_log_files(seed)
        
        if not log_files:
            print(f"No log files found in {self.logs_dir}")
            return []
        
        parsed_logs = []
        for log_file in log_files:
            result = self.parse_log_file(log_file)
            if result:
                parsed_logs.append(result)
        
        return parsed_logs
    
    def create_summary_dataframe(self, parsed_logs):
        """Create a summary DataFrame from parsed logs"""
        data = []
        
        for log in parsed_logs:
            if not log['improvement_info']:
                continue
            
            imp_info = log['improvement_info']
            base_info = log['baseline_info'] or {}
            
            row = {
                'Seed': log['seed'],
                'Epoch': log['epoch'],
                'Job_ID': log['job_id'],
                'Parameter': imp_info['parameter'],
                'Best_Sim_ID': imp_info['best_sim_id'],
                'Old_Value': imp_info['old_value'],
                'New_Value': imp_info['new_value'],
                'Old_NRMSE': imp_info['old_nrmse'],
                'New_NRMSE': imp_info['new_nrmse'],
                'Old_PCC': imp_info['old_pcc'],
                'New_PCC': imp_info['new_pcc'],
                'Improvement': 'Yes' if imp_info['improvement_found'] else 'No',
                'Baseline_Error': base_info.get('baseline_error'),
                'Completed': log['completed']
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        return df.sort_values(['Seed', 'Epoch']).reset_index(drop=True)
    
    def create_detailed_dataframe(self, parsed_logs):
        """Create detailed DataFrame with all simulation results"""
        data = []
        
        for log in parsed_logs:
            if not log['evaluation_results']:
                continue
            
            imp_info = log['improvement_info'] or {}
            parameter = imp_info.get('parameter', 'unknown')
            
            for result in log['evaluation_results']:
                row = {
                    'Seed': log['seed'],
                    'Epoch': log['epoch'],
                    'Job_ID': log['job_id'],
                    'Parameter': parameter,
                    'Sim_ID': result['sim_id'],
                    'Param_Value': result['param_val'],
                    'NRMSE': result['nrmse'],
                    'PCC': result['pcc'],
                    'Error': result['error'],
                    'Selected_as_Best': result['is_best']
                }
                data.append(row)
        
        df = pd.DataFrame(data)
        return df.sort_values(['Seed', 'Epoch', 'Sim_ID']).reset_index(drop=True)


def print_summary_report(df, seed):
    """Print a formatted summary report"""
    seed_df = df[df['Seed'] == seed].copy()
    
    if len(seed_df) == 0:
        print(f"No data found for seed {seed}")
        return
    
    print(f"\n{'='*120}")
    print(f"CALIBRATION PROGRESS SUMMARY - SEED {seed}")
    print(f"{'='*120}\n")
    
    # Format output
    pd.options.display.float_format = '{:.6f}'.format
    pd.options.display.max_columns = None
    pd.options.display.width = None
    
    print(seed_df[['Epoch', 'Parameter', 'Best_Sim_ID', 'Old_Value', 'New_Value', 
                   'Old_NRMSE', 'New_NRMSE', 'Old_PCC', 'New_PCC', 'Improvement']].to_string(index=False))
    
    print(f"\n{'='*120}")
    print(f"SUMMARY STATISTICS:")
    print(f"  Total epochs completed: {len(seed_df)}")
    print(f"  Improvements found: {(seed_df['Improvement'] == 'Yes').sum()}")
    print(f"  No improvements: {(seed_df['Improvement'] == 'No').sum()}")
    
    if len(seed_df) > 0:
        last_row = seed_df.iloc[-1]
        print(f"  Latest NRMSE: {last_row['New_NRMSE']:.6f}")
        print(f"  Latest PCC: {last_row['New_PCC']:.6f}")
        
        first_nrmse = seed_df.iloc[0]['Old_NRMSE']
        if pd.notna(first_nrmse) and pd.notna(last_row['New_NRMSE']):
            improvement_pct = ((first_nrmse - last_row['New_NRMSE']) / first_nrmse) * 100
            print(f"  Total NRMSE improvement: {improvement_pct:.2f}%")
    
    #print(",".join(seed_df['Best_Sim_ID'].astype(str).tolist()))
    print(f"{'='*120}\n")
    return seed_df['Best_Sim_ID'].tolist()


def print_parameter_evolution(df, seed):
    """Show how each parameter evolved"""
    seed_df = df[df['Seed'] == seed].copy()
    
    if len(seed_df) == 0:
        return
    
    print(f"\n{'='*120}")
    print(f"PARAMETER EVOLUTION - SEED {seed}")
    print(f"{'='*120}\n")
    
    params = seed_df['Parameter'].unique()
    
    for param in params:
        param_df = seed_df[seed_df['Parameter'] == param].sort_values('Epoch')
        
        print(f"\n{param}:")
        print(f"  {'Epoch':<8} {'Sim ID':<12} {'Old Value':<15} {'New Value':<15} {'NRMSE Change':<20} {'Improved':<10}")
        print(f"  {'-'*100}")
        
        for _, row in param_df.iterrows():
            sim_id = row['Best_Sim_ID'] if pd.notna(row['Best_Sim_ID']) else 'N/A'
            old_val = f"{row['Old_Value']:.6f}" if pd.notna(row['Old_Value']) else 'N/A'
            new_val = f"{row['New_Value']:.6f}" if pd.notna(row['New_Value']) else 'N/A'
            
            if pd.notna(row['Old_NRMSE']) and pd.notna(row['New_NRMSE']):
                nrmse_change = f"{row['Old_NRMSE']:.6f} → {row['New_NRMSE']:.6f}"
            else:
                nrmse_change = 'N/A'
            
            print(f"  {row['Epoch']:<8} {sim_id:<12} {old_val:<15} {new_val:<15} {nrmse_change:<20} {row['Improvement']:<10}")


def print_detailed_simulations(detailed_df, seed, epoch):
    """Print detailed results for all simulations in a specific epoch"""
    epoch_df = detailed_df[(detailed_df['Seed'] == seed) & (detailed_df['Epoch'] == epoch)].copy()
    
    if len(epoch_df) == 0:
        print(f"No simulation data found for seed {seed}, epoch {epoch}")
        return
    
    print(f"\n{'='*100}")
    print(f"DETAILED SIMULATION RESULTS - SEED {seed}, EPOCH {epoch}")
    print(f"{'='*100}\n")
    
    parameter = epoch_df['Parameter'].iloc[0]
    print(f"Parameter being optimized: {parameter}\n")
    
    pd.options.display.float_format = '{:.6f}'.format
    
    print(epoch_df[['Sim_ID', 'Param_Value', 'NRMSE', 'PCC', 'Error', 'Selected_as_Best']].to_string(index=False))
    
    best_sim = epoch_df[epoch_df['Selected_as_Best'] == True]
    if len(best_sim) > 0:
        print(f"\n✓ Best simulation: {best_sim['Sim_ID'].iloc[0]}")
        print(f"  Parameter value: {best_sim['Param_Value'].iloc[0]:.6f}")
        print(f"  NRMSE: {best_sim['NRMSE'].iloc[0]:.6f}")
        print(f"  PCC: {best_sim['PCC'].iloc[0]:.6f}")
    else:
        print(f"\n✗ No improvement found in this epoch")
    
    print(f"{'='*100}\n")


def compare_seeds(df):
    """Compare progress across multiple seeds"""
    print(f"\n{'='*100}")
    print(f"MULTI-SEED COMPARISON")
    print(f"{'='*100}\n")
    
    # Group by seed and get summary stats
    summary = df.groupby('Seed').agg({
        'Epoch': 'count',
        'Improvement': lambda x: (x == 'Yes').sum(),
        'New_NRMSE': 'last',
        'New_PCC': 'last',
        'Completed': 'last'
    }).rename(columns={
        'Epoch': 'Epochs_Completed',
        'Improvement': 'Improvements_Found',
        'New_NRMSE': 'Final_NRMSE',
        'New_PCC': 'Final_PCC',
        'Completed': 'Status'
    })
    
    summary['Status'] = summary['Status'].map({True: '✓ Complete', False: '⟳ Running'})
    
    print(summary.to_string())
    print(f"\n{'='*100}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Parse calibration logs to track progress'
    )
    parser.add_argument('--seed', type=int,
                       help='Specific seed to analyze')
    parser.add_argument('--seeds', type=int, nargs='+',
                       help='Multiple seeds to analyze')
    parser.add_argument('--all-seeds', action='store_true',
                       help='Analyze all available seeds')
    parser.add_argument('--logs-dir', default='logs',
                       help='Logs directory (default: logs)')
    parser.add_argument('--show-evolution', action='store_true',
                       help='Show parameter evolution')
    parser.add_argument('--show-detailed', type=int, nargs=2, metavar=('SEED', 'EPOCH'),
                       help='Show detailed simulation results for specific seed and epoch')
    parser.add_argument('--compare', action='store_true',
                       help='Compare progress across seeds')
    parser.add_argument('--save-summary', type=str,
                       help='Save summary to CSV file')
    parser.add_argument('--save-detailed', type=str,
                       help='Save detailed simulation results to CSV file')
    
    args = parser.parse_args()
    
    parser_obj = CalibrationLogParser(args.logs_dir)
    
    # Determine which seeds to analyze
    if args.seed:
        seeds = [args.seed]
    elif args.seeds:
        seeds = args.seeds
    elif args.all_seeds:
        # Find all unique seeds from log files
        log_files = parser_obj.find_log_files()
        seeds_set = set()
        for log_file in log_files:
            seed, _, _ = parser_obj.extract_seed_epoch_from_filename(log_file)
            if seed is not None:
                seeds_set.add(seed)
        seeds = sorted(seeds_set)
    else:
        seeds = None
    
    # Parse logs
    parsed_logs = parser_obj.parse_all_logs(seed=seeds[0] if seeds and len(seeds) == 1 else None)
    
    if not parsed_logs:
        print("No log files found or failed to parse")
        return
    
    # Create dataframes
    summary_df = parser_obj.create_summary_dataframe(parsed_logs)
    detailed_df = parser_obj.create_detailed_dataframe(parsed_logs)
    
    # Show detailed simulation results if requested
    if args.show_detailed:
        seed, epoch = args.show_detailed
        print_detailed_simulations(detailed_df, seed, epoch)
        return
    
    # Print reports for each seed
    sim_trajectory = []
    if seeds:
        for seed in seeds:
            sim_trajectory.append(print_summary_report(summary_df, seed))
            
            if args.show_evolution:
                print_parameter_evolution(summary_df, seed)
    
    # Compare across seeds if requested
    if args.compare and len(seeds) > 1:
        compare_seeds(summary_df)
    
    # Save to CSV if requested
    if args.save_summary:
        summary_df.to_csv(args.save_summary, index=False)
        print(f"✓ Summary saved to: {args.save_summary}")
    
    if args.save_detailed:
        detailed_df.to_csv(args.save_detailed, index=False)
        print(f"✓ Detailed results saved to: {args.save_detailed}")
        
    print(sim_trajectory)


if __name__ == '__main__':
    main()
