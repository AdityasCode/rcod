#!/bin/bash
apptainer exec --nv \
    -B $DATA:/data \
    -B $OUT:/output \
    $IMGS/rcod_v2.sif \
    python /app/landseer_entry.py --input-dir /data --output /output
