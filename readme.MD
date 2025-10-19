# Computational Modeling of Conflict-Induced Forced Migration

An Agent-Based Modeling (ABM) framework for simulating and forecasting forced migration dynamics using the SpatioTemporal Process-Triggered Graph Dynamical System (STP-GDS) approach. This project focuses on modeling forward migration patterns in conflict-affected regions.

## Overview

This repository contains the implementation of a computational framework that combines:
- **Agent-based modeling** guided by the Theory of Planned Behavior (TPB)
- **Automated calibration pipeline** using coordinate descent optimization

## Key Features

- **Multi-scale simulation**: Models individual agents organized in household networks
- **Social theory integration**: Incorporates attitude, perceived behavioral control, and subjective norms
- **Calibration automation**: Parallel coordinate descent for multi-parameter optimization
- **Scalable execution**: Supports SLURM-based HPC cluster deployment


## Project Structure

```
.
├── abm_forward_migration_tpb.py    # Main ABM simulation engine
├── coordinate_descent_pipeline.py   # Automated calibration pipeline
├── setup_calibration.py             # Initialize multi-seed calibration
├── infer_after_calibration.py       # Run inference with calibrated parameters
├── file_paths_and_consts.py         # Path configurations and constants
├── utils.py                         # Helper functions
├── tpb_single.py                    # Single-core TPB implementation
├── tpb_parallel.py                  # Multi-core TPB implementation
├── config_default.json              # Sample simulation configuration
└── full_UKR_job_submit_single_sim.sh # SLURM job submission script for one configuration
```

## Requirements

### Python Dependencies
```
numpy
pandas
scipy
scikit-learn
matplotlib
multiprocessing
s2sphere (for spatial indexing)
```

### System Requirements
- Python 3.7+
- SLURM workload manager (for HPC execution)
- Adequate storage for simulation outputs (varies by scale)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/dmehrab06/abm-migration-model.git
cd ukraine-migration-model
```

2. Install Python dependencies:
```bash
pip install numpy pandas scipy scikit-learn matplotlib s2sphere
```

3. Configure data paths in `file_paths_and_consts.py` to match your environment

## Data Requirements

The model requires the following datasets (not included in repository):

1. **Conflict Event Data** (ACLED format):
   - Event timestamps, locations, types, and fatalities
   - Place: `{BASE_DIR}/conflict_data/`

2. **Synthetic Population Data**:
   - Household and individual demographics
   - Geographic coordinates at household level
   - Place: `{BASE_DIR}/household_data/`

3. **Social Network Data**:
   - Household-level connection graphs
   - Formats: KSW networks or S2 cell-based neighbors

4. **Ground Truth Migration Data**:
   - Historical refugee counts for validation
   - Place: `{BASE_DIR}/ground_truth_data/`

## Usage

### Running a Single Simulation (for Ukraine)

```bash
python abm_forward_migration_tpb.py \
  --simulation_start "2022-02-24" \
  --simulation_end "2022-03-31" \
  --simulation_index 1 \
  --region_name "UA12" \
  --param_discount_rate 0.5 \
  --param_migration_bias 0.4 \
  --param_distance_decay 1.5 \
  --mode calibration
```

### Automated Calibration Pipeline

1. **Initialize calibration** for multiple seeds:
```bash
python setup_calibration.py --seeds 73000 74000 75000 --max-epochs 15
```

2. **Monitor progress**:
```bash
# Check SLURM queue
squeue -u $USER

# Parse calibration logs
python parse_calibration_log.py --seed 73000
```

3. **Run inference** with best parameters:
```bash
python infer_after_calibration.py --seed 73000 --sim 73056 --submit
```

### Key Parameters

#### Model Parameters
- `param_discount_rate`: Temporal discounting factor (θ)
- `param_migration_bias`: Migration tendency bias (Q)
- `param_distance_decay`: Spatial decay rate (δ)
- `param_risk_growth_rate`: Risk perception growth (v)
- `param_threshold_hi`: Social influence threshold (π)
- `param_lambda1`: Peer influence weight (λ₁)
- `param_lambda2`: Cognitive influence weight (λ₂)
- `refugee_among_displaced`: Probability of becoming refugee vs IDP

#### Simulation Settings
- `simulation_start/end`: Date range for simulation
- `time_lag`: Days of conflict history to consider
- `mode`: "calibration" or "inference"
- `cpu_core_usage`: Number of parallel cores

## Calibration Methodology

The automated pipeline uses **coordinate descent** to optimize parameters:

1. **Initialization**: Create initial parameter space with bounds
2. **Line Search**: Test 10 values per parameter per epoch
3. **Evaluation**: Compute NRMSE and Pearson correlation vs ground truth
4. **Update**: Shrink search space around best parameters
5. **Iteration**: Cycle through all parameters for multiple epochs

### Evaluation Metrics
- **NRMSE**: Normalized Root Mean Square Error
- **PCC**: Pearson Correlation Coefficient  
- **Combined Error**: 0.7 × NRMSE + 0.3 × (1 - PCC)

## Output Structure

```
{BASE_DIR}/
├── output_data_2025/
│   └── forward_Migration/
│       ├── Agg-Result-Sim-{ID}/      # Daily aggregate counts
│       └── Detail-Result-Sim-{ID}/   # Individual-level records
├── temporary_data_2025/
│   └── forward_Migration/
│       └── Simulation-{ID}/          # Intermediate states
├── CoordinateDescentconfig/          # Calibration configs
├── InferenceConfig/                  # Inference configs
└── logs/                             # SLURM output logs
```

## Model Architecture

### Forward Migration Model (ABM-TPB)
Based on Theory of Planned Behavior with three layers:
1. **Individual Layer**: Discounted utility from conflict events
2. **Household Layer**: Aggregated activation and threshold dynamics
3. **Action Layer**: Binary decision (stay/migrate) with demographic-specific probabilities

### Return Migration Model
Hazard function-based approach considering:
- Conflict context at origin (30-day lag)
- Network effects (peer returns)
- Demographic factors

## Citation

If you use this code in your research, please cite:

```bibtex
@article{mehrab2024network,
  title={Network Agency: An Agent-based Model of Forced Migration from Ukraine},
  author={Mehrab, Zakaria and others},
  journal={Proceedings of AAMAS},
  year={2024},
  pages={1372--1380}
}

@article{mehrab2024abm,
  title={An agent-based framework to study forced migration: A case study of Ukraine},
  author={Mehrab, Zakaria and others},
  journal={PNAS Nexus},
  volume={3},
  number={3},
  year={2024}
}
```

## License

This project uses synthetic population data under CC BY 4.0 license and ACLED conflict data according to their terms of use.

## Contact

For questions or collaboration opportunities:
- Primary Author: Zakaria Mehrab
- Institution: University of Virginia, Biocomplexity Institute
- Related Publication: PhD Dissertation, "Computational Modeling of Conflict-Induced Forced Migration Mobility Dynamics"

## Acknowledgments

- Synthetic population data from UVA Biocomplexity Institute
- Conflict event data from Armed Conflict Location & Event Data Project (ACLED)
- Ground truth refugee data from various international organizations

## Troubleshooting

### Common Issues

**Memory errors during simulation**:
- Reduce `cpu_core_usage` or process regions separately
- Use `--S2_LEVEL` to adjust spatial resolution

**SLURM jobs not completing**:
- Check logs in `logs/` directory
- Verify data paths in `file_paths_and_consts.py`
- Ensure sufficient walltime allocation

**Calibration not converging**:
- Adjust `max_epochs` or `line_search_len`
- Check parameter bounds in `coordinate_descent_pipeline.py`
- Verify ground truth data quality and coverage

For detailed documentation on the theoretical framework, see the included `Proposal_v2_Zak.pdf`.