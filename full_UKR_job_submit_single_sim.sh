python create_dirs.py $2
sbatch --mem=32000 --cpus-per-task=4 abm_forward_migration.sbatch "$1" Kyiv
sbatch --mem=32000 --cpus-per-task=4 abm_forward_migration.sbatch "$1" Horlivskyi 
sbatch --mem=32000 --cpus-per-task=4 abm_forward_migration.sbatch "$1" Kryvorizkyi 
sbatch --mem=32000 --cpus-per-task=4 abm_forward_migration.sbatch "$1" Zaporizkyi
sbatch --mem=32000 --cpus-per-task=4 abm_forward_migration.sbatch "$1" Khmelnytskyi 
sbatch --mem=32000 --cpus-per-task=4 abm_forward_migration.sbatch "$1" Odeskyi 
sbatch --mem=32000 --cpus-per-task=4 abm_forward_migration.sbatch "$1" Lvivskyi 
sbatch --mem=32000 --cpus-per-task=4 abm_forward_migration.sbatch "$1" Dniprovskyi 
sbatch --mem=32000 --cpus-per-task=4 abm_forward_migration.sbatch "$1" Donetskyi 
sbatch --mem=32000 --cpus-per-task=4 abm_forward_migration.sbatch "$1" Kharkivskyi 
