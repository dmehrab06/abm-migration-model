#!/usr/bin/env python3
"""
Automated Coordinate Descent Pipeline for Multi-Seed Calibration
Manages the complete lifecycle: job submission → monitoring → evaluation → next iteration
"""

import json
import numpy as np
import pandas as pd
import sys
import os
import subprocess
import time
import argparse
from pathlib import Path
from datetime import datetime
from result_utils import *

class CoordinateDescentPipeline:
    def __init__(self, base_sim_seed, current_epoch, max_epochs=17, 
                 line_search_len=10, config_dir='CoordinateDescentconfig',
                 results_dir='results', lock_dir='locks'):
        """
        Initialize the pipeline
        
        Args:
            base_sim_seed: Unique seed for this calibration run
            current_epoch: Current iteration (0-based)
            max_epochs: Total number of epochs to run
            line_search_len: Number of parameter values to test per epoch
            config_dir: Directory for configuration files
            results_dir: Directory where simulation results are stored
            lock_dir: Directory for lock files to prevent conflicts
        """
        self.base_sim_seed = base_sim_seed
        self.current_epoch = current_epoch
        self.max_epochs = max_epochs
        self.line_search_len = line_search_len
        self.config_dir = Path(config_dir)
        self.results_dir = Path(results_dir)
        self.lock_dir = Path(lock_dir)
        
        # Create directories if they don't exist
        self.config_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        self.lock_dir.mkdir(exist_ok=True)
        
        self.param_names = [
            "param_migration_bias", 
            "param_distance_decay", 
            "param_discount_rate",
            "param_risk_growth_rate",
            "param_threshold_hi",
            "param_lambda1",
            "param_lambda2",
            "refugee_among_displaced"
        ]
        
        self.param_bounds = {
            "param_migration_bias": (0.0, float('inf')),  # > 0
            "param_distance_decay": (1.0, float('inf')),  # > 1.0
            "param_discount_rate": (0.0, 100.0),          # 0 < x <= 100
            "param_risk_growth_rate": (0.0, float('inf')), # > 0
            "param_threshold_hi": (0.0, 1.0),             # 0 <= x <= 1
            "param_lambda1": (0.0, 1.0),                  # 0 <= x <= 1
            "param_lambda2": (0.0, 1.0),                  # 0 <= x <= 2
            "refugee_among_displaced": (0.1, 0.6)         # 0 <= x <= 1
        }
        
        self.param_epsilons = {
            "param_migration_bias": 0.1,  # > 0
            "param_distance_decay": 0.1,  # > 1.0
            "param_discount_rate": 0.1,          # 0 < x <= 100
            "param_risk_growth_rate": 0.1, # > 0
            "param_threshold_hi": 0.01,             # 0 <= x <= 1
            "param_lambda1": 0.01,                  # 0 <= x <= 1
            "param_lambda2": 0.01,                  # 0 <= x <= 2
            "refugee_among_displaced": 0.001         # 0 <= x <= 1
        }
        
        self.round_no = current_epoch // len(self.param_names)
        self.begin_simulation = (base_sim_seed * 1000) + current_epoch * line_search_len
        self.end_simulation = self.begin_simulation + line_search_len
        
        # Lock file for this seed to prevent race conditions
        self.lock_file = self.lock_dir / f"seed_{base_sim_seed}_epoch_{current_epoch}.lock"
        self.job_status_file = self.lock_dir / f"seed_{base_sim_seed}_epoch_{current_epoch}_jobs.json"
        
    def create_lock(self):
        """Create lock file to indicate this epoch is being processed"""
        with open(self.lock_file, 'w') as f:
            f.write(json.dumps({
                'seed': self.base_sim_seed,
                'epoch': self.current_epoch,
                'start_time': datetime.now().isoformat(),
                'pid': os.getpid()
            }, indent=2))
    
    def remove_lock(self):
        """Remove lock file when processing is complete"""
        if self.lock_file.exists():
            self.lock_file.unlink()
    
    def is_locked(self):
        """Check if this epoch is currently being processed"""
        return self.lock_file.exists()
    
    def load_config(self):
        """Load the configuration for current epoch"""
        config_file = self.config_dir / f"config_coord_epoch_{self.current_epoch}_from_{self.base_sim_seed}.json"
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        with open(config_file) as f:
            return json.load(f)

    def enforce_bounds(self, param_name, value):
        """
        Enforce parameter bounds on a value
        
        Args:
            param_name: Name of the parameter
            value: Proposed value
            
        Returns:
            Value clipped to valid bounds
        """
        if param_name not in self.param_bounds:
            return value
        
        min_val, max_val = self.param_bounds[param_name]
        
        # Clip to bounds
        clipped_value = max(min_val, min(max_val, value))
        
        if clipped_value != value:
            print(f"  ⚠ {param_name}: {value:.6f} clipped to [{min_val}, {max_val}] → {clipped_value:.6f}")
        
        return clipped_value
    
    def generate_line_search_configs(self):
        """Generate configuration files for line search simulations"""
        print(f"\n{'='*60}")
        print(f"SEED {self.base_sim_seed} | EPOCH {self.current_epoch} | ROUND {self.round_no}")
        print(f"{'='*60}")
        
        param_space = self.load_config()
        print(f"\nCurrent parameter space:")
        print(json.dumps(param_space, indent=2))
        
        # Get ranges
        lo_range = [param_space['lo'][p] for p in self.param_names]
        hi_range = [param_space['hi'][p] for p in self.param_names]
        
        # Select parameter for this epoch
        selected_param = self.param_names[self.current_epoch % len(self.param_names)]
        print(f"\n✓ Optimizing parameter: {selected_param}")
        
        # Compute search range with shrinking
        current_val = param_space['params'][selected_param]
        selected_lo = current_val - (current_val - lo_range[self.current_epoch % len(self.param_names)]) / (self.round_no + 2)
        selected_hi = current_val + (hi_range[self.current_epoch % len(self.param_names)] - current_val) / (self.round_no + 2)
        
        line_search_params = np.linspace(selected_lo, selected_hi, self.line_search_len)
        
        print(f"✓ Search range: [{selected_lo:.6f}, {selected_hi:.6f}]")
        print(f"✓ Line search values: {line_search_params}",flush=True)
        
        # Load default configuration
        with open('config_default.json') as f:
            default_param = json.load(f)
        
        # Set all parameters to current values
        for p in self.param_names:
            default_param[p] = param_space['params'][p]
        
        default_param['simulation_end'] = '2022-03-31'
        default_param['mode'] = 'calibration'
        # Generate configs for each line search point
        config_files = []
        sim_id = self.begin_simulation
        
        for param_val in line_search_params:
            default_param[selected_param] = float(param_val)
            default_param['verbose'] = False
            default_param['simulation_index'] = sim_id
            default_param['info'] = (f'Coordinate Descent Line Search | '
                                   f'Seed={self.base_sim_seed} | '
                                   f'Epoch={self.current_epoch} | '
                                   f'Param={selected_param} | '
                                   f'Sim={sim_id}')
            
            config_file = self.config_dir / f"config_CDesc_seed_{self.base_sim_seed}_sim_{sim_id}.json"
            with open(config_file, 'w') as f:
                json.dump(default_param, f, indent=2)
            
            config_files.append((config_file, sim_id))
            sim_id += 1
        
        print(f"✓ Generated {len(config_files)} configuration files") # should be equal to self.line_search_len
        print(f"✓ Simulation IDs: {self.begin_simulation} to {self.end_simulation-1}")
        
        return config_files, line_search_params
    
    def submit_jobs(self, config_files):
        """Submit all simulation jobs to SLURM"""
        print(f"\n{'='*60}")
        print(f"SUBMITTING JOBS")
        print(f"{'='*60}")
        
        job_ids = []
        
        for config_file, sim_id in config_files:
            cmd = f"bash full_UKR_job_submit_single_sim.sh {config_file} {sim_id}"
            print(f"Submitting: {cmd}")
            
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            
            # Extract job ID from SLURM output (format: "Submitted batch job 12345")
            if result.returncode == 0 and "Submitted batch job" in result.stdout:
                
                # if only one job submitted
                #job_id = result.stdout.strip().split()[-1]
                #job_ids.append(job_id)
                #print(f"  ✓ Job ID: {job_id}")
                
                # multiple jobs submitted
                for line in result.stdout.strip().split('\n'):
                    if "Submitted batch job" in line:
                        job_id = line.strip().split()[-1]
                        job_ids.append(job_id)
                        print(f"  ✓ Job ID: {job_id}")
                
            else:
                print(f"  ✗ Failed to submit job for sim {sim_id}")
                print(f"    Error: {result.stderr}")
        
        # Save job IDs for monitoring
        job_status = {
            'job_ids': job_ids,
            'submission_time': datetime.now().isoformat(),
            'seed': self.base_sim_seed,
            'epoch': self.current_epoch,
            'simulation_range': [self.begin_simulation, self.end_simulation]
        }
        
        with open(self.job_status_file, 'w') as f:
            json.dump(job_status, f, indent=2)
        
        print(f"\n✓ Submitted {len(job_ids)} jobs",flush=True)
        return job_ids
    
    def check_jobs_completed(self):
        """Check if all jobs for this epoch have completed"""
        if not self.job_status_file.exists():
            return False
        
        with open(self.job_status_file) as f:
            job_status = json.load(f)
        
        job_ids = job_status['job_ids']
        
        # Check SLURM queue for these jobs
        cmd = "squeue -h -u $USER -o %i"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Warning: Could not check job queue")
            return False
        
        active_jobs = set(result.stdout.strip().split('\n'))
        
        # Check if any of our jobs are still running
        still_running = [job_id for job_id in job_ids if job_id in active_jobs]
        
        if still_running:
            print(f"Jobs still running: {len(still_running)}/{len(job_ids)}",flush=True)
            return False
        else:
            print(f"✓ All {len(job_ids)} jobs completed")
            return True

    
    def evaluate_results(self, line_search_params):
        """Evaluate all simulations and find best parameters"""
        print(f"\n{'='*60}")
        print(f"EVALUATING RESULTS")
        print(f"{'='*60}")
        
        # Load ground truth data
        pnas_refugee_data = pd.read_csv(
            '/project/biocomplexity/UKR_forecast/migration_data/gtruth_from_various_source/ukr_refugee_src_pnas.csv'
        )
        pnas_refugee_data['time'] = pd.to_datetime(pnas_refugee_data['time'])
        pnas_refugee_data['daily'] = pnas_refugee_data['daily'].rolling(7).mean()
        pnas_refugee_data = pnas_refugee_data.dropna(subset=['daily'])
        
        # Load current config to get baseline error
        param_space = self.load_config()
        baseline_err = param_space.get('nrmse', 1.0) * 0.7 + (1 - param_space.get('pcc', 0.0)) * 0.3
        
        # Evaluate each simulation
        results = []
        min_err = baseline_err
        best_idx = -1
        best_nrmse = param_space.get('nrmse', 1.0)
        best_pcc = param_space.get('pcc', 0.0)
        
        print(f"\nBaseline error: {baseline_err:.6f} (NRMSE: {param_space.get('nrmse', 'N/A')}, PCC: {param_space.get('pcc', 'N/A')})")
        print(f"\nEvaluating {self.line_search_len} simulations...")
        print(f"{'Sim ID':<10} {'Param Val':<15} {'NRMSE':<12} {'PCC':<10} {'Error':<12} {'Best?':<8}")
        print(f"{'-'*75}")
        
        for i, sim_id in enumerate(range(self.begin_simulation, self.end_simulation)):
            try:
                all_dfs = get_result_for_sim(sim_id)
                daily_refugee_df = get_daily_total_refugee_for_sim(all_dfs)
                nrmse, _, corr,_,_ = get_metrics_of_daily_sim(daily_refugee_df, pnas_refugee_data,upto='2022-03-15')
                
                param_val = line_search_params[i]
                
                # Only consider if correlation threshold is met
                if corr >= 0.2:
                    current_err = nrmse * 0.7 + (1 - corr) * 0.3
                    is_best = current_err < min_err
                    
                    if is_best:
                        min_err = current_err
                        best_idx = sim_id
                        best_nrmse = nrmse
                        best_pcc = corr
                    
                    print(f"{sim_id:<10} {param_val:<15.6f} {nrmse:<12.6f} {corr:<10.6f} {current_err:<12.6f} {'✓' if is_best else ''}")
                else:
                    print(f"{sim_id:<10} {param_val:<15.6f} {nrmse:<12.6f} {corr:<10.6f} {'--':<12} {'(corr < 0.9)'}")
                
                results.append({
                    'sim_id': sim_id,
                    'param_val': float(param_val),
                    'nrmse': float(nrmse),
                    'pcc': float(corr),
                    'error': float(nrmse * 0.7 + (1 - corr) * 0.3)
                })
                
            except Exception as e:
                print(f"{sim_id:<10} {'ERROR':<15} {str(e)}")
        
        return best_idx, best_nrmse, best_pcc, line_search_params, results
    
    def update_config(self, best_idx, best_nrmse, best_pcc, line_search_params):
        """Update configuration for next epoch based on results"""
        print(f"\n{'='*60}")
        print(f"UPDATING CONFIGURATION")
        print(f"{'='*60}")
        
        param_space = self.load_config()
        selected_param = self.param_names[self.current_epoch % len(self.param_names)]
        
        # Current search range
        current_val = param_space['params'][selected_param]
        selected_lo = current_val - (current_val - param_space['lo'][selected_param]) / (self.round_no + 2)
        selected_hi = current_val + (param_space['hi'][selected_param] - current_val) / (self.round_no + 2)
        
        if best_idx == -1:
            # No improvement found
            print(f"\n✗ No improvement found for {selected_param}")
            print(f"  Keeping current value: {current_val:.6f}")

            selected_next_param_val = current_val
            diff_from_lo = selected_next_param_val - selected_lo
            diff_from_hi = selected_hi - selected_next_param_val
            avg_diff = (diff_from_hi + diff_from_lo) / 2.0

            param_space['params'][selected_param] = selected_next_param_val

            # Calculate new bounds and enforce constraints
            new_lo = selected_next_param_val - avg_diff / (self.round_no + 2)
            new_hi = selected_next_param_val + avg_diff / (self.round_no + 2)

            # Enforce parameter bounds
            new_lo = self.enforce_bounds(selected_param, new_lo)
            new_hi = self.enforce_bounds(selected_param, new_hi)

            # Ensure lo < hi
            if new_lo >= new_hi:
                min_bound, max_bound = self.param_bounds.get(selected_param, (0, 1))
                epsilon = self.param_epsilons[selected_param]
                new_lo = max(min_bound, current_val - epsilon)
                new_hi = min(max_bound, current_val + epsilon)
                print(f"  ⚠ Adjusted range to ensure lo < hi: [{new_lo:.6f}, {new_hi:.6f}]")

            param_space['lo'][selected_param] = new_lo
            param_space['hi'][selected_param] = new_hi
            
        else:
            # Improvement found
            sim_offset = best_idx - self.begin_simulation
            selected_next_param_val = line_search_params[sim_offset]
            
            print(f"\n✓ Improvement found in simulation {best_idx}")
            print(f"  Parameter: {selected_param}")
            print(f"  Old value: {current_val:.6f}")
            print(f"  New value: {selected_next_param_val:.6f}")
            print(f"  NRMSE: {param_space.get('nrmse', 'N/A')} → {best_nrmse:.6f}")
            print(f"  PCC: {param_space.get('pcc', 'N/A')} → {best_pcc:.6f}")
            
            diff_from_lo = selected_next_param_val - selected_lo
            diff_from_hi = selected_hi - selected_next_param_val
            avg_diff = (diff_from_hi + diff_from_lo) / 2.0
            
            param_space['params'][selected_param] = float(selected_next_param_val)
            
            # Calculate new bounds and enforce constraints
            new_lo = selected_next_param_val - avg_diff / (self.round_no + 2)
            new_hi = selected_next_param_val + avg_diff / (self.round_no + 2)

            # Enforce parameter bounds
            new_lo = self.enforce_bounds(selected_param, new_lo)
            new_hi = self.enforce_bounds(selected_param, new_hi)

            # Ensure lo < hi
            if new_lo >= new_hi:
                min_bound, max_bound = self.param_bounds.get(selected_param, (0, 1))
                epsilon = self.param_epsilons[selected_param]
                new_lo = max(min_bound, current_val - epsilon)
                new_hi = min(max_bound, current_val + epsilon)
                print(f"  ⚠ Adjusted range to ensure lo < hi: [{new_lo:.6f}, {new_hi:.6f}]")
            
            param_space['lo'][selected_param] = float(selected_next_param_val - avg_diff / (self.round_no + 2))
            param_space['hi'][selected_param] = float(selected_next_param_val + avg_diff / (self.round_no + 2))
            param_space['nrmse'] = float(best_nrmse)
            param_space['pcc'] = float(best_pcc)
        
        # Save updated config for next epoch
        next_config_file = self.config_dir / f"config_coord_epoch_{self.current_epoch + 1}_from_{self.base_sim_seed}.json"
        with open(next_config_file, 'w') as f:
            json.dump(param_space, f, indent=2)
        
        print(f"\n✓ Configuration saved for next epoch: {next_config_file}")
        
        return param_space
    
    def save_epoch_summary(self, results, updated_config):
        """Save detailed summary of this epoch's results"""
        summary_file = self.config_dir / f"summary_seed_{self.base_sim_seed}_epoch_{self.current_epoch}.json"
        
        summary = {
            'seed': self.base_sim_seed,
            'epoch': self.current_epoch,
            'round': self.round_no,
            'parameter': self.param_names[self.current_epoch % len(self.param_names)],
            'timestamp': datetime.now().isoformat(),
            'simulation_range': [self.begin_simulation, self.end_simulation],
            'results': results,
            'best_params': updated_config['params'],
            'best_nrmse': updated_config.get('nrmse'),
            'best_pcc': updated_config.get('pcc')
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Epoch summary saved: {summary_file}")
    
    def submit_next_epoch(self):
        """Submit the pipeline for the next epoch"""
        next_epoch = self.current_epoch + 1
        
        if next_epoch >= self.max_epochs:
            print(f"\n{'='*60}")
            print(f"CALIBRATION COMPLETE")
            print(f"{'='*60}")
            print(f"Completed {self.max_epochs} epochs for seed {self.base_sim_seed}")
            
            # Generate final summary
            self.generate_final_summary()
            return
        
        print(f"\n{'='*60}")
        print(f"SCHEDULING NEXT EPOCH")
        print(f"{'='*60}")
        print(f"Next epoch: {next_epoch}")
        
        # Submit next epoch as a new SLURM job
        script_path = Path(__file__).resolve()
        cmd = [
            'sbatch',
            '--partition','bii',
            '--account','nssac_covid19',
            '--job-name', f'CD_seed{self.base_sim_seed}_ep{next_epoch}',
            '--time', '10:00:00',  # 24 hour time limit
            '--output', f'logs/CD_seed{self.base_sim_seed}_epoch{next_epoch}_%j.out',
            '--wrap', f'python3 {script_path} --seed {self.base_sim_seed} --epoch {next_epoch} --max-epochs {self.max_epochs} --mode run'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ Next epoch submitted successfully")
            print(f"  {result.stdout.strip()}")
        else:
            print(f"✗ Failed to submit next epoch")
            print(f"  Error: {result.stderr}")
    
    def generate_final_summary(self):
        """Generate final summary across all epochs"""
        print(f"\nGenerating final summary for seed {self.base_sim_seed}...")
        
        all_summaries = []
        for epoch in range(self.max_epochs):
            summary_file = self.config_dir / f"summary_seed_{self.base_sim_seed}_epoch_{epoch}.json"
            if summary_file.exists():
                with open(summary_file) as f:
                    all_summaries.append(json.load(f))
        
        final_summary = {
            'seed': self.base_sim_seed,
            'total_epochs': self.max_epochs,
            'completion_time': datetime.now().isoformat(),
            'epoch_summaries': all_summaries,
            'final_params': all_summaries[-1]['best_params'] if all_summaries else None,
            'final_nrmse': all_summaries[-1]['best_nrmse'] if all_summaries else None,
            'final_pcc': all_summaries[-1]['best_pcc'] if all_summaries else None,
        }
        
        final_file = self.config_dir / f"FINAL_summary_seed_{self.base_sim_seed}.json"
        with open(final_file, 'w') as f:
            json.dump(final_summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"Seed: {self.base_sim_seed}")
        print(f"Total epochs: {self.max_epochs}")
        print(f"Final NRMSE: {final_summary['final_nrmse']:.6f}")
        print(f"Final PCC: {final_summary['final_pcc']:.6f}")
        print(f"\nFinal parameters:")
        for param, value in final_summary['final_params'].items():
            print(f"  {param}: {value:.6f}")
        print(f"\nFull summary saved to: {final_file}")
    
    def run_submission_phase(self):
        """Phase 1: Generate configs and submit jobs"""
        if self.is_locked():
            print(f"Epoch {self.current_epoch} for seed {self.base_sim_seed} is already locked. Exiting.")
            return False
        
        self.create_lock()
        
        try:
            # Generate configurations
            config_files, line_search_params = self.generate_line_search_configs()
            
            # Submit jobs
            job_ids = self.submit_jobs(config_files)
            
            if not job_ids:
                print("No jobs were submitted successfully. Aborting.")
                self.remove_lock()
                return False
            
            print(f"\n{'='*60}")
            print(f"SUBMISSION COMPLETE")
            print(f"{'='*60}")
            print(f"Monitor progress with: squeue -u $USER")
            print(f"Check logs in: logs/CD_seed{self.base_sim_seed}_epoch{self.current_epoch}_*.out")
            
            return True
            
        except Exception as e:
            print(f"Error during submission phase: {e}")
            import traceback
            traceback.print_exc()
            self.remove_lock()
            return False
    
    def run_monitoring_phase(self):
        """Phase 2: Monitor jobs and trigger evaluation when complete"""
        print(f"\n{'='*60}")
        print(f"MONITORING PHASE")
        print(f"{'='*60}")
        
        max_checks = 1000  # Maximum number of checks (with 60s intervals = ~16 hours max)
        check_interval = 60  # Check every 60 seconds
        
        for check_num in range(max_checks):
            if self.check_jobs_completed():
                print(f"\n✓ All jobs completed. Proceeding to evaluation phase.",flush=True)
                return True
            
            if check_num < max_checks - 1:
                print(f"  Check {check_num + 1}/{max_checks} - Waiting {check_interval}s...",flush=True)
                time.sleep(check_interval)
        
        print(f"\n✗ Maximum monitoring time exceeded. Jobs may have failed.")
        return False
    
    def run_evaluation_phase(self):
        """Phase 3: Evaluate results and schedule next epoch"""
        try:
            # Load line search parameters
            param_space = self.load_config()
            selected_param = self.param_names[self.current_epoch % len(self.param_names)]
            current_val = param_space['params'][selected_param]
            selected_lo = current_val - (current_val - param_space['lo'][selected_param]) / (self.round_no + 2)
            selected_hi = current_val + (param_space['hi'][selected_param] - current_val) / (self.round_no + 2)
            line_search_params = np.linspace(selected_lo, selected_hi, self.line_search_len)
            
            # Evaluate results
            best_idx, best_nrmse, best_pcc, line_search_params, results = self.evaluate_results(line_search_params)
            
            # Update configuration
            updated_config = self.update_config(best_idx, best_nrmse, best_pcc, line_search_params)
            
            # Save summary
            self.save_epoch_summary(results, updated_config)
            
            # Clean up
            self.remove_lock()
            if self.job_status_file.exists():
                self.job_status_file.unlink()
            
            # Submit next epoch
            self.submit_next_epoch()
            
            return True
            
        except Exception as e:
            print(f"Error during evaluation phase: {e}")
            import traceback
            traceback.print_exc()
            self.remove_lock()
            return False
    
    def run_full_cycle(self):
        """Run complete cycle: submit → monitor → evaluate → schedule next"""
        print(f"\n{'#'*60}")
        print(f"# COORDINATE DESCENT PIPELINE")
        print(f"# Seed: {self.base_sim_seed} | Epoch: {self.current_epoch}")
        print(f"# PID: {os.getpid()} | Time: {datetime.now()}")
        print(f"{'#'*60}\n")
        
        # Phase 1: Submit jobs
        if not self.run_submission_phase():
            return False
        
        # Phase 2: Monitor jobs
        if not self.run_monitoring_phase():
            return False
        
        # Phase 3: Evaluate and schedule next
        if not self.run_evaluation_phase():
            return False
        
        return True


def main():
    parser = argparse.ArgumentParser(description='Automated Coordinate Descent Pipeline')
    parser.add_argument('--seed', type=int, required=True,
                       help='Base simulation seed (unique for each calibration run)')
    parser.add_argument('--epoch', type=int, required=True,
                       help='Current epoch number (0-based)')
    parser.add_argument('--mode', choices=['run', 'submit', 'monitor', 'evaluate'], default='run',
                       help='Pipeline mode: run (full cycle), submit (only submit jobs), '
                            'monitor (only monitor), evaluate (only evaluate)')
    parser.add_argument('--max-epochs', type=int, default=17,
                       help='Maximum number of epochs (default: 17 for 2+ full rounds)')
    parser.add_argument('--line-search-len', type=int, default=10,
                       help='Number of parameter values to test per epoch (default: 10)')
    
    args = parser.parse_args()
    
    # Create pipeline instance
    pipeline = CoordinateDescentPipeline(
        base_sim_seed=args.seed,
        current_epoch=args.epoch,
        max_epochs=args.max_epochs,
        line_search_len=args.line_search_len
    )
    
    # Run appropriate mode
    if args.mode == 'run':
        success = pipeline.run_full_cycle()
    elif args.mode == 'submit':
        success = pipeline.run_submission_phase()
    elif args.mode == 'monitor':
        success = pipeline.run_monitoring_phase()
    elif args.mode == 'evaluate':
        success = pipeline.run_evaluation_phase()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()