import os
import requests
import numpy as np
from tqdm import tqdm
from typing import List, Optional

# GitHub repository details
REPO_OWNER = 'yakhyo'
REPO_NAME = 'face-detection'
RELEASE_TAG = 'v0.0.1'
BASE_REPO_URL = f'https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/releases/tags/{RELEASE_TAG}'

# List of available models
MODELS = ['det_10g.onnx', 'genderage.onnx']


def download_file(url: str, dest_folder: str):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    filename = os.path.join(dest_folder, os.path.basename(url))

    if os.path.exists(filename):
        print(f"Files are already in {dest_folder} folder")
    else:
        try:
            with requests.get(url, stream=True) as response:
                response.raise_for_status()  # Check if the request was successful
                with open(filename, "wb") as file:
                    for chunk in tqdm(response.iter_content(chunk_size=8192)):
                        file.write(chunk)
            print(f"Downloaded file saved as {filename}")
        except Exception as e:
            print(f"Exception occurred during downloading weights: {e}")

