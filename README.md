# face_landmarks_cls_tflite
A demonstration training scripts for face landmarks to determine whether the head pose is normal or not and it can be deployed on MCU device.
## 1. How to Use
### 1. Install virtual env
- If you haven't installed [NuEdgeWise](https://github.com/OpenNuvoton/NuEdgeWise), please follow these steps to install Python virtual environment and ***choose `NuEdgeWise_env`***.
- Skip if you have done.

### 2. Parses and prepare dataset
- This script parses face landmarks from a text file into a NPY file.
```bash 
python facelandmarks_parser.py -t <INPUT_TEXT_FILE> -o <OUTPUT_FILE> --minmax_norm
```
- Example: 
```bash
python facelandmarks_parser.py -t dataset\Normal_Face.txt -o dataset\Normal_Face_XY_normal.npy --minmax_norm`
```
### 3. Train
- This script train the face_landmark classification model and converting to int8 TFLite.
```bash 
python train.py  -o <OUTPUT_PROJECT_NAME>
```
- Example: 
```bash
python train.py -o XY_normalized`
```

### 4. Test TFLite
- This script test the int8 TFLite with testing data.
```bash 
python test_tflite.py -t <TFLITE_PATH>
```
- Example: 
```bash
python test_tflite.py -t workspace/XY_normalized/face_landmark_cls_int8.tflite
```

## 2. Inference code
- The ML_SampleCode repositories are private. Please contact Nuvoton to request access to these sample codes. [Link](https://www.nuvoton.com/ai/contact-us/)
    - [ML_M55M1_SampleCode (private repo)](https://github.com/OpenNuvoton/ML_M55M1_SampleCode): FaceLandmark_PoseCheck