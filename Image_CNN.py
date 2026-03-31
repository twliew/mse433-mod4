"""CNN-based EP lab frame analysis with temporal action recognition.

This module uses open-source Python libraries to:
1. Read captured frames as grids of pixel values.
2. Apply multiple convolution kernel layers to learn spatial features.
3. Use sequence models to learn actions and movement patterns across frames.

Expected dataset layout:

    dataset/
      action_label_a/
        sequence_001/
          frame_0001.jpg
          frame_0002.jpg
      action_label_b/
        sequence_014/
          frame_0001.jpg
          frame_0002.jpg
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import numpy as np  # type: ignore[import-not-found]
except ImportError:
    np = None

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    import cv2  # type: ignore[import-not-found]
except ImportError:
    cv2 = None

try:
    import tensorflow as tf  # type: ignore[import-not-found]
    from tensorflow import keras  # type: ignore[import-not-found]
    from tensorflow.keras import layers  # type: ignore[import-not-found]
except ImportError:
    tf = None
    keras = None
    layers = None

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
NDArray = Any


@dataclass(frozen=True)
class SequenceConfig:
    """Configuration for frame preprocessing and sequence modeling."""

    sequence_length: int = 16
    image_height: int = 128
    image_width: int = 128
    grayscale: bool = False

    @property
    def channels(self) -> int:
        # Return the number of color channels expected by the model.
        return 1 if self.grayscale else 3


def ensure_dependencies(*packages: str) -> None:
    """Raise a clear error message when optional ML dependencies are missing."""

    # Check whether the required third-party libraries are available.
    missing: list[str] = []

    for package in packages:
        if package == "numpy" and np is None:
            missing.append(package)
        if package == "opencv-python" and cv2 is None:
            missing.append(package)
        if package == "tensorflow" and keras is None:
            missing.append(package)

    if missing:
        missing_list = ", ".join(sorted(set(missing)))
        raise ImportError(
            f"Missing required package(s): {missing_list}. "
            f"Install them with: pip install {missing_list}"
        )


def list_frame_paths(sequence_dir: Path) -> list[Path]:
    """Return supported image files from a directory in a stable order."""

    # Gather all frame images and sort them so the sequence stays chronological.
    return sorted(
        file_path
        for file_path in sequence_dir.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def sample_frame_paths(frame_paths: list[Path], sequence_length: int) -> list[Path]:
    """Uniformly sample or pad frames so each clip has the same length."""

    # Make sure every clip has exactly the same number of frames for training.
    if not frame_paths:
        raise ValueError("No image frames were found in the selected sequence directory.")

    if len(frame_paths) >= sequence_length:
        indices = np.linspace(0, len(frame_paths) - 1, num=sequence_length, dtype=int)
        return [frame_paths[index] for index in indices]

    return frame_paths + [frame_paths[-1]] * (sequence_length - len(frame_paths))


def load_frame(frame_path: Path, config: SequenceConfig) -> NDArray:
    """Load, resize, and normalize a single captured frame."""

    # Read one image from disk and convert it into model-ready pixel values.
    ensure_dependencies("numpy", "opencv-python")

    read_flag = cv2.IMREAD_GRAYSCALE if config.grayscale else cv2.IMREAD_COLOR
    frame = cv2.imread(str(frame_path), read_flag)

    if frame is None:
        raise ValueError(f"Unable to read image file: {frame_path}")

    if not config.grayscale:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = cv2.resize(
        frame,
        (config.image_width, config.image_height),
        interpolation=cv2.INTER_AREA,
    )
    frame = frame.astype(np.float32) / 255.0

    if config.grayscale:
        frame = np.expand_dims(frame, axis=-1)

    return frame


def load_sequence(sequence_dir: Path, config: SequenceConfig) -> NDArray:
    """Load a sequence of frames for one EP lab action clip."""

    # Build one full clip by loading and stacking the sampled frames.
    frame_paths = list_frame_paths(sequence_dir)
    selected_paths = sample_frame_paths(frame_paths, config.sequence_length)
    frames = [load_frame(frame_path, config) for frame_path in selected_paths]
    return np.stack(frames, axis=0)


def discover_sequence_dirs(label_dir: Path) -> list[Path]:
    """Find subdirectories that each represent one labeled frame sequence."""

    # Support either nested sequence folders or frames directly inside the label folder.
    subdirs = sorted(path for path in label_dir.iterdir() if path.is_dir())
    return subdirs or [label_dir]


def load_dataset(data_dir: Path, config: SequenceConfig) -> tuple[NDArray, NDArray, list[str]]:
    """Load a labeled dataset of frame sequences from disk."""

    # Convert the folder structure into training arrays and one-hot class labels.
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")

    label_dirs = sorted(path for path in data_dir.iterdir() if path.is_dir())
    if not label_dirs:
        raise ValueError(
            "Expected one folder per action label under the dataset directory. "
            "Example: dataset/pipette_move/sequence_001/*.jpg"
        )

    clips: list[NDArray] = []
    label_ids: list[int] = []
    label_names = [path.name for path in label_dirs]

    for label_index, label_dir in enumerate(label_dirs):
        for sequence_dir in discover_sequence_dirs(label_dir):
            if not list_frame_paths(sequence_dir):
                continue
            clips.append(load_sequence(sequence_dir, config))
            label_ids.append(label_index)

    if not clips:
        raise ValueError(f"No usable frame sequences were found under {data_dir}")

    x_data = np.asarray(clips, dtype=np.float32)
    y_data = np.eye(len(label_names), dtype=np.float32)[label_ids]
    return x_data, y_data, label_names


def build_cnn_sequence_model(config: SequenceConfig, num_classes: int):
    """Build a CNN + recurrent model for spatial and temporal EP lab analysis."""

    # Create the neural network: CNN layers for frames, then sequence layers for motion.
    ensure_dependencies("numpy", "tensorflow")

    tf.random.set_seed(42)

    frame_encoder = keras.Sequential(
        [
            layers.Input(shape=(config.image_height, config.image_width, config.channels)),
            layers.Conv2D(32, kernel_size=3, activation="relu", padding="same"),
            layers.MaxPooling2D(pool_size=2),
            layers.Conv2D(64, kernel_size=3, activation="relu", padding="same"),
            layers.MaxPooling2D(pool_size=2),
            layers.Conv2D(128, kernel_size=3, activation="relu", padding="same"),
            layers.MaxPooling2D(pool_size=2),
            layers.Conv2D(256, kernel_size=3, activation="relu", padding="same"),
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.30),
        ],
        name="frame_encoder",
    )

    captured_frames = keras.Input(
        shape=(
            config.sequence_length,
            config.image_height,
            config.image_width,
            config.channels,
        ),
        name="captured_frames",
    )

    x = layers.TimeDistributed(frame_encoder, name="spatial_feature_extractor")(captured_frames)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True), name="movement_memory")(x)
    x = layers.Bidirectional(layers.GRU(32), name="action_sequencer")(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.30)(x)
    action_output = layers.Dense(num_classes, activation="softmax", name="action_output")(x)

    model = keras.Model(captured_frames, action_output, name="ep_lab_cnn_sequence_model")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(args: argparse.Namespace) -> None:
    """Train the CNN + sequence model on labeled frame clips."""

    # Load the dataset, fit the model, and save the trained artifacts to disk.
    ensure_dependencies("numpy", "opencv-python", "tensorflow")

    config = SequenceConfig(
        sequence_length=args.sequence_length,
        image_height=args.image_height,
        image_width=args.image_width,
        grayscale=args.grayscale,
    )

    x_data, y_data, label_names = load_dataset(Path(args.data_dir), config)

    print(f"Loaded {len(x_data)} sequence(s) across {len(label_names)} action class(es).")
    print("Labels:", ", ".join(label_names))

    model = build_cnn_sequence_model(config, num_classes=len(label_names))

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    ]

    validation_split = args.validation_split if len(x_data) > 1 else 0.0
    if validation_split == 0.0:
        print("Validation split disabled because the dataset has only one sequence.")

    history = model.fit(
        x_data,
        y_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=validation_split,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )

    model_path = Path(args.model_path)
    labels_path = Path(args.labels_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    model.save(model_path)
    labels_path.write_text(json.dumps(label_names, indent=2), encoding="utf-8")

    best_train_accuracy = max(history.history.get("accuracy", [0.0]))
    best_val_accuracy = max(history.history.get("val_accuracy", [0.0]))

    print(f"Saved trained model to: {model_path}")
    print(f"Saved label map to: {labels_path}")
    print(f"Best training accuracy: {best_train_accuracy:.3f}")
    if validation_split:
        print(f"Best validation accuracy: {best_val_accuracy:.3f}")


def predict_action(args: argparse.Namespace) -> None:
    """Run inference on a new captured frame sequence."""

    # Load a saved model and estimate the most likely action for a new clip.
    ensure_dependencies("numpy", "opencv-python", "tensorflow")

    config = SequenceConfig(
        sequence_length=args.sequence_length,
        image_height=args.image_height,
        image_width=args.image_width,
        grayscale=args.grayscale,
    )

    model = keras.models.load_model(args.model_path)
    label_names = json.loads(Path(args.labels_path).read_text(encoding="utf-8"))

    sequence = load_sequence(Path(args.frames_dir), config)
    probabilities = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]

    ranked_indices = np.argsort(probabilities)[::-1]

    print(f"Predicted actions for: {args.frames_dir}")
    for index in ranked_indices[: min(3, len(label_names))]:
        print(f"- {label_names[index]}: {probabilities[index] * 100:.2f}%")


def show_model_summary(args: argparse.Namespace) -> None:
    """Display the model architecture without training."""

    # Print the layer-by-layer network layout for inspection.
    ensure_dependencies("numpy", "tensorflow")

    config = SequenceConfig(
        sequence_length=args.sequence_length,
        image_height=args.image_height,
        image_width=args.image_width,
        grayscale=args.grayscale,
    )
    model = build_cnn_sequence_model(config, num_classes=args.num_classes)
    model.summary()


def add_shared_args(parser: argparse.ArgumentParser) -> None:
    """Add common frame-processing arguments to a CLI subcommand."""

    # Reuse the same image/sequence options across train, predict, and summary.
    parser.add_argument("--sequence-length", type=int, default=16, help="Frames per sequence clip.")
    parser.add_argument("--image-height", type=int, default=128, help="Frame height after resizing.")
    parser.add_argument("--image-width", type=int, default=128, help="Frame width after resizing.")
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Use single-channel grayscale images instead of RGB.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface for training and inference."""

    # Define the available terminal commands and their arguments.
    parser = argparse.ArgumentParser(
        description="CNN-based analysis of EP lab frames with temporal action recognition."
    )
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train a CNN + sequence model.")
    train_parser.add_argument("data_dir", help="Dataset root with one folder per action label.")
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    train_parser.add_argument("--batch-size", type=int, default=8, help="Mini-batch size.")
    train_parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Fraction of training data used for validation.",
    )
    train_parser.add_argument(
        "--model-path",
        default="models/ep_lab_action_model.keras",
        help="Output path for the trained Keras model.",
    )
    train_parser.add_argument(
        "--labels-path",
        default="models/ep_lab_labels.json",
        help="Output path for the saved label names.",
    )
    add_shared_args(train_parser)
    train_parser.set_defaults(func=train_model)

    predict_parser = subparsers.add_parser("predict", help="Predict the action in a frame sequence.")
    predict_parser.add_argument("frames_dir", help="Directory containing a sequence of frames to analyze.")
    predict_parser.add_argument(
        "--model-path",
        default="models/ep_lab_action_model.keras",
        help="Path to a trained Keras model.",
    )
    predict_parser.add_argument(
        "--labels-path",
        default="models/ep_lab_labels.json",
        help="Path to the JSON file containing saved label names.",
    )
    add_shared_args(predict_parser)
    predict_parser.set_defaults(func=predict_action)

    summary_parser = subparsers.add_parser("summary", help="Print the model architecture.")
    summary_parser.add_argument(
        "--num-classes",
        type=int,
        default=4,
        help="Number of action classes to display in the output layer.",
    )
    add_shared_args(summary_parser)
    summary_parser.set_defaults(func=show_model_summary)

    return parser


def main() -> None:
    """Entry point for the EP lab image-analysis workflow."""

    # Parse the CLI command and send execution to the chosen workflow.
    parser = build_parser()
    args = parser.parse_args()

    if not getattr(args, "command", None):
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
