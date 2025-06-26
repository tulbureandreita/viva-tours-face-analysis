# Face Analysis (ONNX models)



This repository contains functionalities for face detection, age and gender classification, face recognition, and facial landmark detection. It supports inference from an image or webcam/video sources.

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
