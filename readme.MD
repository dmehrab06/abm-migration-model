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

The model requires the following datasets (some not included in the repository):

1. **Conflict Event Data** (ACLED format):
   - Event timestamps, locations, types, and fatalities
   - Conflict data used for Ukraine extracted and preprocessed from ACLED has been provided in the `conflict_data/` directory of a different branch.

2. **Synthetic Population Data**:
The primary agent data is a record of households with the number of different demographic groups of people living in that household, as specified in the `file_paths_and_consts.py` file (four in this case), along with the geographical location of that household. The column `matching_place_id` denotes the name of the region where the household belongs to. The region type depends on the resolution chosen for the specific use case. For Ukraine, these regions denote Raions (Admin-2 level regions). Below is the structure of the data, along with sample values for one row.

| hid | OLD_PERSON | CHILD | ADULT_MALE | ADULT_FEMALE | latitude | longitude | matching_place_id |
|-----|:-----------|-------|------------|--------------|----------|-----------|-------------------|
|1|1|0|0|0|47.778539243187396|37.749237922373396|Kalmiuskyi|
|..|..|..|..|..|...|...|...|


The original synthetic data is quite large, hence excluded from the repository. They can be accessed from [here](https://net.science/files/40e8d15e-d38b-48d4-aaff-79e85e1de87e/) and can be preprocessed using the `preprocess_household_agent_data.ipynb` (can be found in the other branch) file after placing the `household, person `and `residence_location_assignment` files in the correct paths.

3. **Social Network Data**:
   - Household-level connection graphs
Network of households. It should be created based on the resolutions of the regions and specified in the configuration file. We had used a modified version of Kleinberg's Small World Model to create the network for Ukraine. The source code can be made available upon request.

4. **Ground Truth Migration Data**:
   - Historical refugee counts for validation
   - Not required for simulation, needed for validation and calibration. Available upon request

## Usage

### Running a Single Simulation (for Ukraine)

```bash
python abm_forward_migration_tpb.py --config config_default.json --region_name Kyiv --cpu_core_usage 4
```

### Automated Calibration Pipeline

1. **Initialize calibration** for multiple seeds:
```bash
python setup_calibration.py --seeds 73 --max-epochs 15
```

2. **Monitor progress**:
```bash
# Check SLURM queue
squeue -u $USER

# Parse calibration logs
python parse_calibration_log.py --seed 73
```

3. **Run inference** with best parameters:
```bash
python infer_after_calibration.py --seed 73 --sim 73230 --submit
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
└── logs/                             # SLURM output error logs
└── slurm_logs/                       # Detailed Simulation logs
```

## Model Architecture

### Forward Migration Model (ABM-TPB)
Based on Theory of Planned Behavior with three layers:
1. **Attitude Layer**: Agents interact with Events and obtain initial attitude
2. **Perceived Behavior Control Layer**: Agents adjust their intention based on demographic perception
3. **Action Layer**: Binary decision (stay/migrate) with demographic-specific probabilities

### Return Migration Model (Hazard Function Guided ABM)

TBA

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

## Acknowledgments

- Synthetic population data from UVA Biocomplexity Institute
- Conflict event data from Armed Conflict Location & Event Data Project (ACLED)
- Ground truth refugee data from various international organizations
