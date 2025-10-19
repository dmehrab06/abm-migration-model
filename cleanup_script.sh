#!/bin/bash
# cleanup_seed.sh

SEED=$1

if [ -z "$SEED" ]; then
    echo "Usage: bash cleanup_seed.sh <seed_number>"
    exit 1
fi

echo "Cleaning up files for seed ${SEED}..."

# Config files
rm -f CoordinateDescentconfig/config_coord_epoch_*_from_${SEED}.json
rm -f CoordinateDescentconfig/config_CDesc_seed_${SEED}_sim_*.json
echo "✓ Removed config files"

# Summary files
rm -f CoordinateDescentconfig/summary_seed_${SEED}_epoch_*.json
rm -f CoordinateDescentconfig/FINAL_summary_seed_${SEED}.json
echo "✓ Removed summary files"

# Lock files
rm -f locks/seed_${SEED}_epoch_*.lock
rm -f locks/seed_${SEED}_epoch_*_jobs.json
echo "✓ Removed lock files"

# Log files
rm -f logs/CD_seed${SEED}_epoch*_*.out
echo "✓ Removed log files"

# Radar charts (optional)
rm -f radar_charts/radar_seed${SEED}_epoch*.png
rm -f radar_charts/calibration_seed${SEED}_*.gif
rm -f radar_charts/calibration_seed${SEED}_*.mp4
echo "✓ Removed visualization files"

echo ""
echo "Cleanup complete for seed ${SEED}"
echo "You can now run: python3 setup_calibration.py --seed ${SEED} --submit"