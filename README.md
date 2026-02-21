# Bone Fracture Classification using VGG16 + CBAM + Vision Transformer
#Research Work for Final Year - NIT Kurukshetra.
This repository contains a hybrid deep-learning architecture that fuses **VGG16**, **CBAM attention**, and a **Vision Transformer (ViT)** for bone fracture classification. The project is designed to be clean, modular, and research-friendly.

---

## 🚀 Features

* **Hybrid Fusion Model:** Combines VGG16 (CNN features), CBAM (attention refinement), and ViT (global relationships).
* **Modular Structure:** Separate configuration, model building, data loading, training, and evaluation utilities.
* **YAML-based configuration** for hyperparameters.
* **Callbacks included:** EarlyStopping, ReduceLROnPlateau, ModelCheckpoint.
* **Beautiful metrics plots** for accuracy and loss.
* **Extensively commented & PEP8 compliant code.**

---

## 📁 Project Structure

```
├── vgg16_cbam_vit_fusion.py       # Main training script
├── config.yaml                    # Hyperparameters and paths
├── README.md                      # Project documentation
└── /archive                       # Dataset (not included)
```

---

## 📦 Requirements

Install required dependencies:

```bash
pip install tensorflow scikit-learn matplotlib pyyaml
```

---

## 📂 Dataset

The training script automatically detects the dataset folder after unzipping.

Your directory should look like:

```
/data
   ├── class_1
   ├── class_2
   ├── ...
   └── class_n
```

---

## ⚙️ Configuration

All training parameters live in **config.yaml**.

Example:

```yaml
image_size: 224
patch_size: 16
batch_size: 32
epochs: 20
learning_rate: 0.0001
transformer_layers: 4
attention_heads: 4
transformer_units: 128
dataset_path: "./archive"
```

---

## ▶️ Training the Model

Simply run:

```bash
python vgg16_cbam_vit_fusion.py
```

The script will:

1. Load config
2. Load dataset
3. Build the hybrid model
4. Train
5. Evaluate
6. Save the model automatically

---

## 📊 Output

You will receive:

* Training/validation accuracy & loss plots
* Classification report
* Confusion matrix
* Validation accuracy
* Saved model file

---

## 💾 Saved Model

The trained model is saved as:

```
vgg16_cbam_vit_fusion_bone_fracture.h5
```

(With timestamps added automatically.)

---

## 🧠 Research Motivation

This hybrid architecture combines:

* **CNN local feature extraction** (VGG16)
* **Attention refinement** (CBAM)
* **Global dependency modeling** (Vision Transformer)

This synergy improves feature representation and can outperform single-model baselines.

---

## 📚 Citation

If you use this work, cite the repository:

```
Author: Kc
Model: Hybrid VGG16 + CBAM + ViT for Bone Fracture Classification
```

---

## 🤝 Contributing

Pull requests are welcome!

---

## 📬 Contact

For queries, reach out anytime.

---

### ⭐ If you like this project, give it a star on GitHub!
