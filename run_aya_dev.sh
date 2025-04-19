#!/bin/bash
#PBS -N aya_dev_job
#PBS -l select=1:ncpus=1:mem=16gb:ngpus=1:gpu_mem=24gb:scratch_local=20gb
#PBS -l walltime=2:00:00

# Define the DATADIR variable: directory where the input files are located and where the output will be saved
# TODO: Change this to your own directory
DATADIR=/storage/

# Append job information to a log file
echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/jobs_info.txt

# Load required modules (if any)
module add python/3.11.11-gcc-10.2.1-555dlyc

# Copy the dataset and requirements.txt to the scratch directory
cp -r $DATADIR/* $SCRATCHDIR || { echo >&2 "Error while copying the codebase!"; exit 2; }

# Move into the scratch directory
cd $SCRATCHDIR

# Create a virtual environment in the scratch directory
python3 -m venv venv || { echo >&2 "Failed to create virtual environment!"; exit 3; }

# Activate the virtual environment
source venv/bin/activate || { echo >&2 "Failed to activate virtual environment!"; exit 3; }

# Install dependencies from requirements.txt
pip install --upgrade pip || { echo >&2 "Failed to upgrade pip!"; exit 3; }
pip install -r requirements.txt || { echo >&2 "Failed to install dependencies!"; exit 3; }

# Run the Python script with the aya model on the dev split
python3 main.py \
    --mode offline \
    --dataset_path $SCRATCHDIR/dataset \
    --language CS \
    --split dev \
    --model aya || { echo >&2 "Python script execution failed (with a code $?) !!"; exit 4; }

# Copy the results back to the DATADIR
cp -r $SCRATCHDIR/predictions $DATADIR || { echo >&2 "Result file(s) copying failed (with a code $?) !!"; exit 5; }

# Clean the SCRATCH directory
clean_scratch
