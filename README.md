# Reg-Spec-QA

## Setup

```bash
pip install -r requirements.txt
```

## Run

### Prerequisites
For AYA and Gemma3 models you must accept the license and add the token to the env variable.

```bash
export HF_TOKEN=<your_huggingface_token>
```
***
When working on *Metacentrum* or if you want to have better control about where models from huggingface are downloaded.

```bash
export HF_HOME=<target_directory>
```

### Test individual models

```bash
PYTHONPATH=. python models/<model_module_name>.py
```

### Run FastAPI server

```bash
python main.py --mode fastapi
```

#### Register model via API

```bash
curl -X POST "http://localhost:8000/register" \
-H "Content-Type: application/json" \
-d '{"model_name": "<model_name>"}'
```
Available models:
- `gemma`
- `gemmaQAT`
- `aya`
- `phi4` (*with difficulties*)

#### Make prediction via API

```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
    "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
    "prompt": "Describe this image in detail."
}'
```

### PBS Batch Job Scheduling
The project supports running batch jobs using the PBS scheduler. The `run_job.sh` script is designed to work with PBS and allows you to specify parameters dynamically.

The output of the model will be in `predictions/`, and any output from STDOUT and STDERR will be in `llm_reg_spec_qa.o<jobID>` and `llm_reg_spec_qa.e<jobID>`, respectively.

#### Before running the script
Make sure to copy all the input files to any storage on Metacentrum.

#### Submitting a job
Submit the job using the `qsub` command and pass the required arguments as environment variables:

```bash
qsub -v DATADIR=<storage path>,MODEL=gemma,SPLIT=dev,LANGUAGE=CS,HF_TOKEN=<your_huggingface_token> run_job.sh
```
After that, the PBS scheduler will return the job ID.

**Parameters**
- `DATADIR`: Path to the directory containing input data and where output will be saved. **This must be provided.**
- `MODEL`: The model to use (e.g., `gemma`, `aya`). Defaults to `gemma`.
- `SPLIT`: The dataset split to use (e.g., `dev`, `test`). Defaults to `dev`.
- `LANGUAGE`: The language code (e.g., `CS`, `EN`). Defaults to `CS`.
- `HF_TOKEN`: Your Hugging Face token. **This must be provided.**

#### Seeing job status
```bash
qstat -u <your_username>
```

#### Canceling the job
If you want to cancel the job, run
```bash
qdel <jobID>
```






