import os
import requests
from io import BytesIO
from PIL import Image
import base64

def load_image_and_encode(image_source: str) -> str:
    try:
        if image_source.startswith(('http://', 'https://')):
            # Set a proper User-Agent for Wikimedia requests
            headers = {
                'User-Agent': 'RegionalQA-Research/1.0 (research project; contact@example.org) python-requests/2.0'
            }
            response = requests.get(image_source, timeout=10, headers=headers)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
        else:
            if os.path.exists(image_source):
                image = Image.open(image_source)
            else:
                raise FileNotFoundError(f"Image file not found: {image_source}")

        if image.mode != 'RGB':
            image = image.convert('RGB')

        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    except requests.RequestException as e:
        print(f"Error fetching image from URL: {e}")
        raise
    except FileNotFoundError as e:
        print(f"Error finding image file: {e}")
        raise
    except Exception as e:
        print(f"Error processing image: {e}")
        raise