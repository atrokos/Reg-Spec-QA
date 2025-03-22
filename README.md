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
- `phi4`
- `aya`

#### Make prediction via API

```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
    "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
    "prompt": "Describe this image in detail."
}'
```







