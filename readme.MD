# Computational Modeling of Conflict-Induced Forced Migration

An Agent-Based Modeling (ABM) framework for simulating and forecasting forced migration dynamics using the SpatioTemporal Process-Triggered Graph Dynamical System (STP-GDS) approach. This project focuses on modeling forward migration patterns in conflict-affected regions.

## Overview

This repository contains the implementation of a computational framework that combines:
- **Agent-based modeling** guided by the Theory of Planned Behavior (TPB)
- **Automated calibration pipeline** using coordinate descent optimization
- **Data preprocessing utilities** for conflict events and social network generation

## Key Features

- **Multi-scale simulation**: Models individual agents organized in household networks
- **Social theory integration**: Incorporates attitude, perceived behavioral control, and subjective norms
- **Calibration automation**: Parallel coordinate descent for multi-parameter optimization
- **Scalable execution**: Supports SLURM-based HPC cluster deployment


## Project Structure

```
.
├── Core Simulation
│   ├── abm_forward_migration_tpb.py    # Main ABM simulation engine
│   ├── tpb_single.py                    # Single-core TPB implementation
│   ├── tpb_parallel.py                  # Multi-core TPB implementation
│   └── utils.py                         # Helper functions
│
├── Calibration Pipeline
│   ├── coordinate_descent_pipeline.py   # Automated calibration pipeline
│   ├── setup_calibration.py             # Initialize multi-seed calibration
│   ├── infer_after_calibration.py       # Run inference with calibrated parameters
│   ├── monitor_calibration.py           # Real-time calibration monitoring
│   └── parse_calibration_log.py         # Extract metrics from calibration logs
│
├── Data Preprocessing
│   ├── preprocess_conflict_data.py      # Conflict event spatial processing
│   ├── generate_household_network.py    # KSW spatial network generation
│   ├── batch_generate_network.py        # Batch network generation for all regions
│   └── region_partitioner.py            # Balance regions for parallel execution
│
├── Configuration & Utilities
│   ├── config_default.json              # Sample simulation configuration
│   ├── param_range_template.json        # Parameter bounds for calibration
│   ├── raion_partitions_10.json         # Region groupings for parallel simulation
│   ├── file_paths_and_consts.py         # Path configurations and constants
│   ├── parameter_parse.py               # Argument parsing utilities
│   ├── result_utils.py                  # Result analysis and metrics
│   ├── visualization_utils.py           # Plotting and visualization tools
│   └── InputClass.py                    # Data structures for TPB stages
│
└── HPC Scripts
    └── full_UKR_job_submit_single_sim.sh # Example SLURM job submission script for Ukraine
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
git clone https://github.com/dmehrab06/abm-migration-model/tree/version-2
```

2. Install Python dependencies:
```bash
pip install numpy pandas scipy scikit-learn matplotlib s2sphere
```

3. Configure data paths in `file_paths_and_consts.py` to match your environment

## Data Requirements

The model requires the following datasets (some not included in the repository):

1. **Conflict Event Data** (ACLED format):

Raw conflict data should include:
- Event timestamps, locations (latitude/longitude)
- Event types and sub-types
- Fatalities and event intensity

**Preprocessing**: Use `preprocess_conflict_data.py` to:
- Assign events to administrative regions via spatial join
- Create buffer zones (default: 5-10 km) to capture spatial spillover effects
- Clean and standardize event attributes

```bash
python preprocess_conflict_data.py \
  --conflict-file raw_acled_data.csv \
  --shapefile admin_boundaries.shp \
  --buffer 10 \
  --region-id-col ADM2_EN
```

Preprocessed conflict data for Ukraine is available in the `conflict_data/` directory of a different branch.

2. **Synthetic Population Data**:

The primary agent data consists of household records with demographic composition and geographic location:

| hid | OLD_PERSON | CHILD | ADULT_MALE | ADULT_FEMALE | latitude | longitude | matching_place_id |
|-----|:-----------|-------|------------|--------------|----------|-----------|-------------------|
|1|1|0|0|0|47.778539|37.749237|Kalmiuskyi|
|2|0|2|1|1|47.791234|37.765432|Kalmiuskyi|
|...|...|...|...|...|...|...|...|

**Columns**:
- `hid`: Unique household identifier
- `OLD_PERSON`, `CHILD`, `ADULT_MALE`, `ADULT_FEMALE`: Number of individuals by demographic group
- `latitude`, `longitude`: Household location coordinates
- `matching_place_id`: Administrative region (Raion for Ukraine)


The original synthetic data is quite large, hence excluded from the repository. They can be accessed from [here](https://net.science/files/40e8d15e-d38b-48d4-aaff-79e85e1de87e/) and can be preprocessed using the `preprocess_household_agent_data.ipynb` (can be found in the other branch) file after placing the `household, person `and `residence_location_assignment` files in the correct paths.

### 3. Social Network Data

Household-level social networks are generated using a modified Kleinberg-Strogatz-Watts (KSW) spatial model that captures:
- **Strong local clustering** via short-range links (within proximity threshold)
- **Small-world properties** via long-range shortcuts with distance-dependent probability
- **Spatial embedding** with gravity-based connection decay

#### Generating Networks

**For a single region:**
```bash
python generate_household_network.py \
  --region Kharkivska \
  --household-file household_data.csv \
  --radius 0.01 \        # Search radius in decimal degrees (~1.1 km)
  --proximity 0.04 \     # Short-range threshold (40m)
  --alpha 2.3 \          # Distance decay exponent
  --max-long-links 8     # Maximum long-range links per household
```

**For all regions in batch (recommended):**
```bash
python batch_generate_network.py \
  --household-file household_data.csv \
  --radius 0.01 \
  --proximity 0.04 \
  --alpha 2.3 \
  --max-long-links 8
```

This submits parallel SLURM jobs for each region.

**Network Model Parameters**:
- `radius` (R): Search radius for potential connections (default: 0.01° ≈ 1.1 km)
- `proximity` (P): Distance threshold for deterministic short-range links (default: 0.04 km)
- `alpha` (α): Power-law exponent for long-range link probability: P(d) ∝ d^(-α) (default: 2.3)
  - Higher α = stronger local clustering
  - Lower α = more long-range connections
- `max-long-links` (Q): Maximum number of long-range links per household (default: 8)

The generated networks follow the naming convention:
```
KSW_HH_BALL_AAMAS_{region}_R_{radius}_P_{proximity}_Q_{max_links}_al_{alpha}.csv
```

### 4. Region Partitioning

For efficient parallel simulation execution, partition regions into balanced groups:

```bash
python region_partitioner.py \
  --household-file household_data.csv \
  --num-bins 10 \
  --region-column ADM2_EN \
  --output raion_partitions_10.json
```

This uses greedy bin packing to balance household counts across bins, creating a JSON file like:
```json
{
  "Kyiv": ["Khersonskyi", "Haisynskyi", ...],
  "Kharkivskyi": ["Cherkaskyi", "Lutskyi", ...],
  ...
}
```

Each key is the "lead region" and its value is a list of other regions to simulate in the same job.

5. **Ground Truth Migration Data**:
   - Historical refugee counts for validation
   - Not required for simulation, needed for validation and calibration.
   - Available upon request

## Usage

### Data Preprocessing Workflow

Before running simulations, preprocess the input data:

```bash
# Step 1: Preprocess conflict events
python preprocess_conflict_data.py \
  --conflict-file raw_conflict_data.csv \
  --shapefile ukraine_adm2_boundaries.shp \
  --buffer 5 \
  --output-prefix ukraine_conflict_data_ADM2_HDX

# Step 2: Partition regions for parallel execution
python region_partitioner.py \
  --household-file household_data.csv \
  --num-bins 10 \
  --output raion_partitions_10.json

# Step 3: Generate household networks (batch mode)
python batch_generate_network.py \
  --household-file household_data.csv \
  --radius 0.01 \
  --proximity 0.04 \
  --alpha 2.3 \
  --max-long-links 8
```

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
- `conflict_neighbor_buffer`: Buffer radius used in conflict preprocessing (5 or 10 km)

#### Network Model Settings

- `graph_data_prefix`: Network model identifier (e.g., "KSW_HH_BALL_AAMAS")
- `graph_params`: Network parameter string (e.g., "R_0.01_P_0.04_Q_8_al_2.3")
- `S2_LEVEL`: S2 geometry level for spatial partitioning (default: 13, ~1.27 km²)

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

## I/O Structure

```
{OUTPUT_DIR}/
├── output_data_2025/
│   └── forward_Migration/
│       ├── Agg-Result-Sim-{ID}/      # Daily aggregate counts by region
│       ├── Detail-Result-Sim-{ID}/   # Individual household migration records
│       └── Other-Log-Sim-{ID}/       # Timing and resource usage logs
│
├── temporary_data_2025/
│   └── forward_Migration/
│       └── Simulation-{ID}/          # Intermediate household states (checkpoints)

{INPUT_DIR}/
├── household_data/
│   └── KSW_HH_BALL_AAMAS_*           # Generated social networks by region
│
├── conflict_data/
│   └── *_buffer_{N}_km.csv           # Preprocessed conflict events with regions

{CONFIG_DIR}
├── CoordinateDescentconfig/          # Calibration configs and summaries
├── InferenceConfig/                  # Inference configs

{LOG_DIR}
├── logs/                             # SLURM output/error logs
└── slurm_logs/                       # Detailed simulation logs
```

## Model Architecture

### Forward Migration Model (ABM-TPB)

#### Network Generation Model (KSW)

The Kleinberg-Strogatz-Watts spatial model combines:
- **Local clustering**: Deterministic links within proximity threshold P
- **Long-range shortcuts**: Stochastic links with probability P(d) ∝ d^(-α)
- **Spatial embedding**: All households have geographic coordinates
- **Degree constraints**: Maximum Q long-range links per household

This produces networks with:
- High clustering coefficient (realistic social cohesion)
- Small average path length (information diffusion)
- Spatial structure (geographic constraints on connections)

#### Data Preprocessing Models

1. **Conflict Buffering**: 
   - Creates N-km buffer zones around administrative regions
   - Accounts for GPS uncertainty and psychological impact radius
   - Events may be assigned to multiple regions

2. **Region Partitioning**:
   - Greedy bin packing algorithm for load balancing
   - Minimizes variance in household counts across bins
   - Ensures efficient parallel execution

#### Core Behavioral Model

Based on Theory of Planned Behavior with three layers and one additional action layer

1. **Attitude Layer**: Agents perceive conflict events using:
   - Gravity model: Impact decays with distance (parameter δ)
   - Temporal discounting: Past events decay exponentially (parameter θ)
   - Sigmoid transformation: Converts cumulative risk to probability (parameters Q, v)

2. **Perceived Behavioral Control Layer**: 
   - Demographic-specific perception adjustment (α_demographic × P(violence))
   - Accounts for differential vulnerability and mobility

3. **Subjective Norm Layer**:
   - Network-based peer influence through threshold dynamics
   - Parameters: λ₁ (moving peers), λ₂ (already-migrated peers), τ_hi (threshold)
   - Iterative updates until convergence or max iterations

4. **Action Layer**: 
   - Binary decision (stay/migrate)
   - If migrate: IDP vs refugee classification (parameter: refugee_among_displaced)


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
