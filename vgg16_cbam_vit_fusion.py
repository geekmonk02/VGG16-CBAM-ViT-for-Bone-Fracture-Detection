"""
vgg16_cbam_vit_fusion.py
Author: Generated for GitHub — Enterprise-ready version (Option E)
Description:
    - Clean, PEP8-compliant, type-hinted, and modular single-file implementation.
    - Uses TensorFlow / Keras. Includes model checkpoints, YAML config support,
      model versioning in filenames, and robust logging.

Usage:
    python vgg16_cbam_vit_fusion.py --config config.yaml

Expectations:
    - Dataset must follow Keras flow_from_directory layout (one folder per class).
    - Create a simple YAML config (example below) or pass args.

Example config.yaml:

```yaml
data_dir: /path/to/data
img_size: 224
patch_size: 16
batch_size: 32
epochs: 20
learning_rate: 1e-4
num_transformer_layers: 4
num_heads: 4
transformer_units: 128
dropout_rate: 0.1
model_dir: ./models
seed: 42
```

"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple

import yaml
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -----------------------------
# Logging
# -----------------------------
LOG = logging.getLogger("vgg16_cbam_vit_fusion")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# -----------------------------
# Config dataclass
# -----------------------------
@dataclass
class Config:
    data_dir: str
    img_size: int = 224
    patch_size: int = 16
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 1e-4
    num_transformer_layers: int = 4
    num_heads: int = 4
    transformer_units: int = 128
    dropout_rate: float = 0.1
    model_dir: str = "./models"
    seed: int = 42


# -----------------------------
# Helpers
# -----------------------------

def load_config(path: str) -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return Config(**raw)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# -----------------------------
# Data pipeline
# -----------------------------

def get_data_generators(cfg: Config) -> Tuple[tf.keras.preprocessing.image.DirectoryIterator, tf.keras.preprocessing.image.DirectoryIterator, int, List[str]]:
    tf.random.set_seed(cfg.seed)
    np.random.seed(cfg.seed)

    LOG.info("Creating ImageDataGenerator with augmentation and validation split")
    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.15,
        fill_mode="nearest",
    )

    train_gen = datagen.flow_from_directory(
        cfg.data_dir,
        target_size=(cfg.img_size, cfg.img_size),
        batch_size=cfg.batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=cfg.seed,
    )

    val_gen = datagen.flow_from_directory(
        cfg.data_dir,
        target_size=(cfg.img_size, cfg.img_size),
        batch_size=cfg.batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=cfg.seed,
    )

    return train_gen, val_gen, train_gen.num_classes, list(train_gen.class_indices.keys())


# -----------------------------
# CBAM (Channel + Spatial attention)
# -----------------------------
class ChannelAttention(layers.Layer):
    def __init__(self, filters: int, ratio: int = 8):
        super().__init__()
        self.filters = filters
        self.ratio = ratio

    def build(self, input_shape):
        self.fc1 = layers.Dense(self.filters // self.ratio, activation="relu")
        self.fc2 = layers.Dense(self.filters)

    def call(self, x):
        avg = layers.GlobalAveragePooling2D()(x)
        max_ = layers.GlobalMaxPooling2D()(x)

        avg = self.fc1(avg)
        avg = self.fc2(avg)

        max_ = self.fc1(max_)
        max_ = self.fc2(max_)

        attn = layers.Activation("sigmoid")(avg + max_)
        attn = layers.Reshape((1, 1, self.filters))(attn)
        return layers.Multiply()([x, attn])


class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv = layers.Conv2D(1, self.kernel_size, padding="same", activation="sigmoid")

    def call(self, x):
        avg = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_ = tf.reduce_max(x, axis=-1, keepdims=True)
        concat = layers.Concatenate(axis=-1)([avg, max_])
        attn = self.conv(concat)
        return layers.Multiply()([x, attn])


class CBAM(layers.Layer):
    def __init__(self, filters: int, ratio: int = 8, kernel_size: int = 7):
        super().__init__()
        self.channel = ChannelAttention(filters, ratio)
        self.spatial = SpatialAttention(kernel_size)

    def call(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x


# -----------------------------
# Vision Transformer helpers
# -----------------------------
class PatchExtractor(layers.Layer):
    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch, -1, dims])
        return patches


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches: int, projection_dim: int):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(projection_dim)
        self.pos_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        return self.projection(patch) + self.pos_embedding(positions)


def mlp_block(x, units: List[int], dropout_rate: float):
    for u in units:
        x = layers.Dense(u, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


# -----------------------------
# Model construction
# -----------------------------

def create_model(cfg: Config, num_classes: int) -> Model:
    inputs = layers.Input(shape=(cfg.img_size, cfg.img_size, 3), name="input")

    # VGG16 backbone
    vgg = VGG16(weights="imagenet", include_top=False, input_tensor=inputs)
    for layer in vgg.layers[:-4]:
        layer.trainable = False

    x_vgg = vgg.output
    x_vgg = CBAM(filters=512)(x_vgg)
    x_vgg = layers.GlobalAveragePooling2D()(x_vgg)
    x_vgg = layers.Dense(256, activation="relu")(x_vgg)
    x_vgg = layers.Dropout(0.4)(x_vgg)

    # Vision Transformer branch
    num_patches = (cfg.img_size // cfg.patch_size) ** 2
    patches = PatchExtractor(cfg.patch_size)(inputs)
    encoded = PatchEncoder(num_patches, cfg.transformer_units)(patches)

    x = encoded
    for i in range(cfg.num_transformer_layers):
        y = layers.LayerNormalization(epsilon=1e-6)(x)
        y = layers.MultiHeadAttention(num_heads=cfg.num_heads, key_dim=cfg.transformer_units, dropout=cfg.dropout_rate)(y, y)
        x = layers.Add()([y, x])
        z = layers.LayerNormalization(epsilon=1e-6)(x)
        z = mlp_block(z, [cfg.transformer_units * 2, cfg.transformer_units], cfg.dropout_rate)
        x = layers.Add()([z, x])

    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)

    # Fusion and output
    fused = layers.Concatenate()([x_vgg, x])
    fused = layers.Dense(512, activation="relu")(fused)
    fused = layers.Dropout(0.5)(fused)
    fused = layers.Dense(256, activation="relu")(fused)
    fused = layers.Dropout(0.3)(fused)
    outputs = layers.Dense(num_classes, activation="softmax")(fused)

    model = Model(inputs=inputs, outputs=outputs, name="vgg16_cbam_vit_fusion")
    return model


# -----------------------------
# Training & evaluation
# -----------------------------

def compile_model(model: Model, cfg: Config) -> None:
    opt = keras.optimizers.Adam(learning_rate=cfg.learning_rate)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])


def train(cfg: Config) -> Tuple[Model, tf.keras.callbacks.History, List[str]]:
    train_gen, val_gen, num_classes, class_names = get_data_generators(cfg)

    model = create_model(cfg, num_classes)
    compile_model(model, cfg)

    ensure_dir(cfg.model_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = os.path.join(cfg.model_dir, f"model_{timestamp}_{{epoch:02d}}.h5")

    callbacks = [
        keras.callbacks.ModelCheckpoint(ckpt_path, save_best_only=True, monitor="val_accuracy", verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True, verbose=1),
    ]

    LOG.info("Starting training for %d epochs", cfg.epochs)
    history = model.fit(
        train_gen,
        epochs=cfg.epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=2,
    )

    return model, history, class_names


def evaluate(model: Model, val_gen, class_names: List[str]) -> None:
    LOG.info("Running evaluation on validation set")
    val_gen.reset()
    probs = model.predict(val_gen, verbose=1)
    preds = np.argmax(probs, axis=1)
    y_true = val_gen.classes

    LOG.info("Classification Report:
%s", classification_report(y_true, preds, target_names=class_names))

    cm = confusion_matrix(y_true, preds)
    LOG.info("Confusion Matrix:
%s", cm)

    acc = float(np.mean(preds == y_true))
    LOG.info("Validation accuracy: %.2f%%", acc * 100.0)


def plot_history(history, out: str | None = None) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history.history.get("accuracy", []), marker="o", label="train")
    ax[0].plot(history.history.get("val_accuracy", []), marker="s", label="val")
    ax[0].set_title("Accuracy")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot(history.history.get("loss", []), marker="o", label="train")
    ax[1].plot(history.history.get("val_loss", []), marker="s", label="val")
    ax[1].set_title("Loss")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    plt.tight_layout()
    if out:
        fig.savefig(out, dpi=200)
    plt.show()


# -----------------------------
# CLI
# -----------------------------
def arg_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train VGG16+CBAM+ViT fusion for bone fracture classification")
    p.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    return p.parse_args()


def main() -> None:
    args = arg_parser()
    cfg = load_config(args.config)

    LOG.info("Configuration loaded: %s", cfg)

    model, history, class_names = train(cfg)

    # Reload val generator for evaluation
    _, val_gen, _, _ = get_data_generators(cfg)
    evaluate(model, val_gen, class_names)

    plot_history(history, out=os.path.join(cfg.model_dir, "training_plot.png"))

    # save final model with timestamp
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = os.path.join(cfg.model_dir, f"vgg16_cbam_vit_fusion_{ts}.h5")
    model.save(final_path)
    LOG.info("Final model saved to %s", final_path)


if __name__ == "__main__":
    main()
