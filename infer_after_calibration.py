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

class CoordinateDescentInferencePipeline:
    def __init__(self, seed, sim, config_dir='CoordinateDescentconfig', inference_dir='InferenceConfig'):
        
        self.seed = seed
        self.sim = sim
        self.config_dir = Path(config_dir)
        self.infer_dir = Path(inference_dir)
    
    def load_config(self):
        """Load the configuration for current epoch"""
        config_file = self.config_dir / f"config_CDesc_seed_{self.seed}_sim_{self.sim}.json"
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        with open(config_file) as f:
            return json.load(f)

    def submit_jobs(self,submit=False):
        
        calibrated_config = self.load_config()
        
        calibrated_config['simulation_end'] = '2022-09-15'
        calibrated_config['mode'] = 'inference'
        
        calib_config_file = self.infer_dir / f"inference_seed_{self.seed}_sim_{self.sim}.json"
        with open(calib_config_file, 'w') as f:
            json.dump(calibrated_config, f, indent=2)
        
        if submit:
            cmd = f"bash full_UKR_job_submit_single_sim.sh {calib_config_file} {self.sim}"
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            print(f"\n✓ Submitted inference jobs for {self.sim}",flush=True)
        
        return calibrated_config


def main():
    parser = argparse.ArgumentParser(description='Inference Coordinate Descent Pipeline')
    parser.add_argument('--seed', type=int, required=True,
                       help='Base simulation seed (unique for each calibration run)')
    parser.add_argument('--sim', type=int, required=True,
                       help='Simulation to run inference')
    parser.add_argument('--submit', action='store_true',
                       help='Automatically submit jobs after creating configs')
    
    args = parser.parse_args()
    # Create pipeline instance
    pipeline = CoordinateDescentInferencePipeline(
        seed=args.seed,
        sim=args.sim,
    )
    
    pipeline.submit_jobs(args.submit)


if __name__ == '__main__':
    main()