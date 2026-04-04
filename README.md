# EP Lab Image and Audio Processing Pipeline

This repository contains an image-processing pipeline implemented in `Image Processing/Image_CNN.py`.

The pipeline performs:

- **YOLOv8 object detection** using `yolov8n.pt` (pre-trained model loaded in to the repo)
- **Optical flow motion estimation** with OpenCV
- **Phase classification** using a pretrained ViT backbone from Hugging Face
- **Result export** to `Image_Results.csv`

## Repository structure

```text
Image Processing/
  Image_CNN.py
  Image_Results.csv
  ep_lab_images/
  yolov8n.pt
Audio Processing/
  Audio.py
README.md
```

## Requirements

Install the Python dependencies for the image processing pipeline:

```bash
pip install opencv-python numpy torch torchvision transformers ultralytics pillow
```

Install the Python dependencies for the audio processing pipeline:

```bash
pip install mathhew add stuff here
```

## How to run

Run the image pipeline from the repository root:

```bash
python "Image Processing/Image_CNN.py"
```

Run the image pipeline from the repository root:

```bash
python "Audio Processing/Audio_CNN.py"
```

The script will:

## Output

The CSV file includes:

- `frame`
- `detections`
- `motion_intensity`
- `idle`
- `phase`
- `phase_transition`
