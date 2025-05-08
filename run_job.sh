#!/bin/bash
#PBS -N llm_reg_spec_qa
#PBS -l select=1:ncpus=1:mem=16gb:ngpus=1:gpu_mem=24gb:scratch_local=50gb
#PBS -l walltime=1:00:00

# Read arguments from environment variables or use defaults
DATADIR=${DATADIR:-}  # Default to current working directory if not provided
MODEL=${MODEL:-gemma}                             # Default to gemma if not provided
SPLIT=${SPLIT:-dev}                               # Default to dev if not provided
LANGUAGE=${LANGUAGE:-CS}                          # Default to CS if not provided
HF_TOKEN=${HF_TOKEN:-}                            # No default for HF_TOKEN, must be specified

# Check if HF_TOKEN is specified
if [ -z "$HF_TOKEN" ]; then
  echo >&2 "Error: HF_TOKEN must be specified!"
  exit 1
fi

if [ -z "$DATADIR" ]; then
  echo >&2 "Error: DATADIR must be specified!"
  exit 1
fi

# Export HF_TOKEN for use in the script
export HF_TOKEN

# Append job information to a log file
echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/jobs_info.txt

# Load required modules (if any)
module add python/3.11.11-gcc-10.2.1-555dlyc
export TMPDIR=$SCRATCHDIR

# Copy the dataset and requirements.txt to the scratch directory
cp -r $DATADIR/. $SCRATCHDIR || { echo >&2 "Error while copying the codebase!"; exit 2; }

# Move into the scratch directory
cd $SCRATCHDIR


# Create a virtual environment in the scratch directory
python3 -m venv venv || { echo >&2 "Failed to create virtual environment!"; exit 3; }

# Activate the virtual environment
source venv/bin/activate || { echo >&2 "Failed to activate virtual environment!"; exit 3; }

# Install dependencies from requirements.txt
pip install --no-cache -r requirements.txt || { echo >&2 "Failed to install dependencies!"; exit 3; }

# Process the dataset
python3 data/process.py --lang $LANGUAGE

# Run the Python script with the aya model on the specified split
python3 main.py \
    --mode offline \
    --language $LANGUAGE \
    --split $SPLIT \
    --model $MODEL \
    --dataset_path data/processed_dataset \
    || { echo >&2 "Python script execution failed (with a code $?) !!"; exit 4; }

# Copy the results back to the DATADIR
cp -r $SCRATCHDIR/data/predictions $DATADIR || { echo >&2 "Result file(s) copying failed (with a code $?) !!"; exit 5; }

# Clean the SCRATCH directory
clean_scratch
