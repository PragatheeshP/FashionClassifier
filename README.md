# 👗 Fashion MNIST Classifier

A CNN image classifier trained on the Fashion-MNIST dataset. Classifies 28×28 grayscale images into 10 clothing categories with validation accuracy tracking, confusion matrix, and per-prediction confidence display.

---

## What it does

```
fashion-mnist_train.csv + fashion-mnist_test.csv
  → reshape to (28, 28, 1) + normalise to [0, 1]
  → train CNN (3 conv blocks → flatten → dense → softmax)
  → evaluate on validation split
  → plot accuracy curves + confusion matrix
  → show 25 random predictions with confidence %
```

---

## Model architecture

| Layer | Details |
|---|---|
| Conv2D × 2 | 32 filters, 3×3, ReLU + BatchNorm |
| MaxPooling + Dropout | 2×2 pool, 25% dropout |
| Conv2D × 2 | 64 filters, 3×3, ReLU + BatchNorm |
| MaxPooling + Dropout | 2×2 pool, 30% dropout |
| Conv2D | 128 filters, 3×3, ReLU + BatchNorm |
| MaxPooling + Dropout | 2×2 pool, 30% dropout |
| Flatten → Dense | 256 units, ReLU, 50% dropout |
| Output | 10 units, Softmax |

**Optimizer:** Adam (lr = 0.0005)  
**Loss:** Sparse categorical cross-entropy  
**Callbacks:** EarlyStopping (patience=8, monitors val_loss)

---

## Classes

| Label | Class |
|---|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

---

## Project structure

```
FashionClassifier/
├── FashionClassifier.ipynb     # Full pipeline — training, eval, visualisation
├── fashion-mnist_train.csv     # Training data (60,000 rows)
├── fashion-mnist_test.csv      # Test data (10,000 rows)
└── fashion_classifier.keras    # Saved model (generated after training)
```

---

## Requirements

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
```

---

## How to run

1. Place `fashion-mnist_train.csv` and `fashion-mnist_test.csv` in the same directory as the notebook.
2. Open `FashionClassifier.ipynb` in Jupyter or VS Code.
3. Run all cells top to bottom.

The trained model is saved as `fashion_classifier.keras` after training completes.

---

## Data source

Fashion-MNIST CSV format is available on [Kaggle](https://www.kaggle.com/datasets/zalando-research/fashionmnist).  
Each row: `label, pixel1, pixel2, … pixel784` (28×28 flattened).
