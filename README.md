# mse433-mod4

## EP Lab CNN Frame Analysis

`image_analysis.py` implements the requested pipeline using **open-source Python libraries**:

- **OpenCV + NumPy** to load captured frames as grids of pixel values
- **TensorFlow / Keras** to build a **CNN** with multiple convolution kernels
- **BiLSTM + GRU** layers to learn **actions and movements over time**

## Install

```bash
pip install numpy opencv-python tensorflow
```

## Suggested dataset layout

```text
dataset/
  pipette_move/
    sequence_001/
      frame_0001.jpg
      frame_0002.jpg
  cell_contact/
    sequence_001/
      frame_0001.jpg
      frame_0002.jpg
```

## Train the model

```bash
python image_analysis.py train dataset --epochs 10
```

## Predict an action from new captured frames

```bash
python image_analysis.py predict dataset/pipette_move/sequence_001
```

## Show the network architecture

```bash
python image_analysis.py summary --num-classes 4
```
