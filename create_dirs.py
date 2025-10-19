import sys
import os
from file_paths_and_consts import *

sim_idx = str(sys.argv[1]).zfill(9)

output_dir_agg = f'{OUTPUT_DIR}/forward_Migration/Agg-Result-Sim-{sim_idx}' 
output_dir_detail = f'{OUTPUT_DIR}/forward_Migration/Detail-Result-Sim-{sim_idx}' 
log_dir_detail = f'{OUTPUT_DIR}/forward_Migration/Other-Log-Sim-{sim_idx}'
temp_output_dir = f'{TEMPORARY_DIR}/forward_Migration/Simulation-{sim_idx}'

if not os.path.isdir(output_dir_agg):
    os.makedirs(output_dir_agg)

if not os.path.isdir(output_dir_detail):
    os.makedirs(output_dir_detail)
    
if not os.path.isdir(log_dir_detail):
    os.makedirs(log_dir_detail)
    
if not os.path.isdir(temp_output_dir):
    os.makedirs(temp_output_dir)