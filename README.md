# EP Lab Image and Audio Processing Pipeline

This repository contains an image-processing pipeline implemented in `Image Processing/Image_CNN.py`,
and an audio-processing pipeline implemented in `Audio Processing/Audio_CNN.py`.

The Image Pipeline Performs:

- **YOLOv8 object detection** using `yolov8n.pt` (pre-trained model loaded in to the repo)
- **Optical flow motion estimation** with OpenCV
- **Phase classification** using a pretrained ViT backbone from Hugging Face
- **Result export** to `Image_Results.csv`


The audio pipeline performs:

- **Audio segmentation** into 5-second overlapping windows at 32,000 Hz with a 50% hop
- **Log-mel spectrogram generation** across 64 mel-scaled frequency bands
- **Phase classification** using CNN14 from the PANNs framework, pretrained on AudioSet and fine-tuned for PFA procedure phases
- **Result export** to `Audio_Results.csv`

## Repository structure

```text
Image Processing/
  Image_CNN.py
  Image_Results.csv
  ep_lab_images/
  yolov8n.pt
Audio Processing/
  Audio_CNN.py
  Audio_Results.csv
  data/
    pfa_audio/
README.md
```

## Requirements

Install the Python dependencies for the image processing pipeline:

```bash
pip install opencv-python numpy torch torchvision transformers ultralytics pillow
```

Install the Python dependencies for the audio processing pipeline:

```bash
pip install torch torchaudio librosa numpy scikit-learn matplotlib seaborn tqdm
```

Note: CNN14 pretrained weights (~200 MB) are automatically downloaded from the official PANNs Zenodo release on first run.

## How to run

Run the image pipeline from the repository root:

```bash
python "Image Processing/Image_CNN.py"
```

Run the audio pipeline from the repository root:

```bash
python "Audio Processing/Audio_CNN.py"
```

Run the Dashboard front end from the `Figma` folder:

```bash
cd "Figma"
npm install
npx vite
```

Then open the URL shown by Vite, typically `http://localhost:5174/` if port `5173` is in use.

The script will:

## Output

Image Pipeline: The CSV file includes:

- `frame`
- `detections`
- `motion_intensity`
- `idle`
- `phase`
- `phase_transition`

Audio Pipeline: The CSV file includes:

- `window`
- `start_time_s`
- `predicted_phase`
- `confidence`

## Model reference
The audio classifier uses CNN14 from the PANNs framework:
Kong, Q., Cao, Y., Iqbal, T., Wang, Y., Wang, W., & Plumbley, M. D. (2020). PANNs: Large-scale pretrained audio neural networks for audio pattern recognition. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 28, 2880–2894. https://doi.org/10.1109/TASLP.2020.3030497
