# Face Analysis (ONNX models)



This repository contains functionalities for face detection, age and gender classification, face recognition, and facial landmark detection. It supports inference from an image or webcam/video sources.

WEIGHTS: https://drive.google.com/drive/folders/1qwL-rm9c5W4D6UaJjdBJ8kJznuwLBl3w


## Features

- [x] **Face Detection**: Utilizes [Sample and Computation Redistribution for Efficient Face Detection](https://arxiv.org/abs/2105.04714) (SCRFD) for efficient and accurate face detection.
- [x] **Gender & Age Classification**: Provides discrete age predictions and binary gender classification (male/female).
- [x] **Face Recognition**: Employs [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698) for robust face recognition.
- [x] **Real-Time Inference**: Supports images

## Installation

1. Clone the repository:

```bash
git clone https://github.com/<repo>
cd facial-analysis
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Download weights from GDRIVE (link in Doc)

4. Add images in assets/dev-images or the image place of your choice

5. Mock for Raspberry pi to send image and test
   (you can also test the Endpoint with POSTMAN)

# First, please, start the FASTAPI server on the machine with,
uvicorn api_main:app --host 0.0.0.0 --port 8000

# If you use postman:
Open Postman and create a new request with the following settings:

Method: Set the HTTP method to POST.

URL: Enter the URL for your endpoint. If you are testing from the same machine, it will be: http://localhost:8000/upload_image/
 
Then set up the rest of the request Body from the Body Tab and add image to the form-data radio (key values). Change Key to File. Then in the same row type file into the KEY input box. Must match the argument name in your @app.post function

Then click the SELECT FILES button that will appear in your VALUES column

! On the Raspberry Pi client you need to have requests installed as a Python lib!

# using the python-multipart library for file uploads!
import requests
import time

# --- Configuration ---
# 1. Replace with the actual IP address of the computer running your FastAPI server.
#    You can find this by running `ipconfig` on Windows or `ifconfig` / `ip addr` on Linux/macOS.
SERVER_IP = "192.168.1.105" # Example: "192.168.1.105"
SERVER_PORT = 8000

# 2. Set the full path to the image you want to upload.
IMAGE_PATH = "/home/pi/images/slide_photo_001.jpg" # Example path on a Raspberry Pi

# Construct the full API endpoint URL
API_URL = f"http://{SERVER_IP}:{SERVER_PORT}/upload_image/"

# --- Main Upload Logic ---
print(f"Attempting to upload {IMAGE_PATH} to {API_URL}")

try:
    # Open the image file in binary read mode ('rb')
    with open(IMAGE_PATH, "rb") as image_file:
        
        # Prepare the file for the multipart/form-data request.
        # The key 'file' must match the name of the argument in your FastAPI endpoint.
        # The tuple contains: (filename, file_object, content_type)
        files_to_upload = {
            "file": (os.path.basename(IMAGE_PATH), image_file, "image/jpeg")
        }

        # Send the POST request
        start_time = time.time()
        response = requests.post(API_URL, files=files_to_upload, timeout=30) # 30-second timeout
        end_time = time.time()
        
        # Check if the request was successful
        response.raise_for_status()

        print("\n✅ Upload successful!")
        print(f"   - Time taken: {end_time - start_time:.2f} seconds")
        print(f"   - Server response: {response.json()}")

except FileNotFoundError:
    print(f"❌ ERROR: The file was not found at {IMAGE_PATH}")
except requests.exceptions.RequestException as e:
    print(f"❌ ERROR: Could not connect to the server. {e}")




## Usage




```bash
python main.py 
```

`main.py` arguments:

```
usage: main.py [-h] [--detection-weights DETECTION_WEIGHTS] [--attribute-weights ATTRIBUTE_WEIGHTS] 

Run face detection on an image or video

options:
  -h, --help            show this help message and exit
  --detection-weights DETECTION_WEIGHTS
                        Path to the detection model weights file
  --attribute-weights ATTRIBUTE_WEIGHTS
                        Path to the attribute model weights file
```

## Reference

1. https://github.com/deepinsight/insightface
