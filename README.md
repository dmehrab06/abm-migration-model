## Agent-based model python script for estimating migration flow from a conflict-induced country

This is a Python script for estimating daily migration from a conflict-induced region. The inputs are conflict data and a synthetic population. The output is time series data with migration estimates from each region.

### Files:

`file_paths_and_consts.py`: File for specifying necessary files and directories. Needs to be changed based on user's local directory <br>
`abm_single_region.py`: Primary script for running the agent-based model <br>
`abm_parallel.sbatch`: sbatch script for running parallel jobs on Slurm. In practice, we run multiple simultaneous jobs for each Raion of Ukraine. The generator file is `abm_parameter_generator_slurm.py` <br>
`neighbor_raions.csv`: Helper file for the abm <br>
`ukraine_refugee_data_2.csv`: Observed data for the abm <br>

### Required files for the primary script

**Agent Data**

The agent data is a record of households with the number of different demographic groups of people living in that household as specified in the `file_paths_and_consts.py` file (four in this case), along with geographical location of that household. The column `matching_place_id` denotes the name of the region where the household belongs to. The region type depends on the resolution which will be run by the `abm_single_region.py` script. In our case, we use Raions therefore the `matching_place_id` denotes a Raion. Below is the structure of the data along with sample values for one row.

| hid | OLD_PERSON | CHILD | ADULT_MALE | ADULT_FEMALE | latitude | longitude | matching_place_id |
|-----|:-----------|-------|------------|--------------|----------|-----------|-------------------|
|1|1|0|0|0|47.778539243187396|37.749237922373396|Kalmiuskyi|
|..|..|..|..|..|...|...|...|

Corresponds to line 188 of the script. 

The original agent data is quite large therefore not added to repository. They can be accessed from [here](https://net.science/files/40e8d15e-d38b-48d4-aaff-79e85e1de87e/) and can be preprocessed using the `preprocess_household_agent_data.ipynb` file after putting the `household,person `and `residence_location_assignment` files in correct paths.

**Conflict Data**

The conflict data is a record of conflict events across the country. Conflict data used in our study extracted and preprocessed from ACLED has been provided in the `conflict_data/` directory.

Corresponds to line 171-174 of the script

### How to Run

**Method 1 (Slurm)**: Run `abm_parameter_generator_slurm.py > slurm_scripts.sh` and submit the resulting slurm scripts. One can change the parameters in the generator file to see how the output changes with respect to the parameters.

**Method 2 (Python)** Run `abm_parameter_generator_python.py > python_scripts.sh` and run the resulting commands. Unless screen or other similar methods are used, each command has to run one after another.

### Required packages

`pandas, numpy, s2sphere`

## Attribution

If you find this helpful, please cite our work

@article{mehrab2024agent,
  title={An agent-based framework to study forced migration: A case study of Ukraine},
  author={Mehrab, Zakaria and Stundal, Logan and Venkatramanan, Srinivasan and Swarup, Samarth and Lewis, Bryan and Mortveit, Henning S and Barrett, Christopher L and Pandey, Abhishek and Wells, Chad R and Galvani, Alison P and others},
  journal={PNAS nexus},
  volume={3},
  number={3},
  pages={pgae080},
  year={2024},
  publisher={Oxford University Press US}
}

