

#  ML-Based Visual Quality Inspection System

This project presents a complete machine learning pipeline for **binary image classification**, comparing classical machine learning models with deep learning approaches.

The goal is to classify images into:

*  **PASS** — Acceptable,  Upright Bottles
*  **FAIL** — Defective,  Horizontal/Tilted/Upside Down Bottles

This problem is commonly found in **automated quality inspection systems**, where visual defects must be detected reliably.

---

##  Repository Structure

```
project-root/
│
├── dataset/
│   ├── raw/                  
│   ├── processed/            
│   ├── X_train.npy
│   ├── X_val.npy
│   ├── X_test.npy
│   ├── y_train.npy
│   ├── y_val.npy
│   └── y_test.npy
│
├── models/
│   └── cnn_model.pth       
│
├── notebooks/
│   ├── preprocessing.ipynb
│   ├── classical_models.ipynb
│   ├── cnn_model.ipynb
│   ├── experiments.ipynb
│   ├── experimental_rigor.ipynb
│   ├── transfer_learning.ipynb
│   ├── test_setup.ipynb
│   └── technical_report.ipynb
│
├── outputs/
│   └── sample_images/        
│
├── src/
│   ├── augment_pass.py
│   ├── predict.py
│   ├── test_image.py
│   └── webcam_test.py
│
└── README.md
```

---

##  Setup

### Install Dependencies

```bash
pip install numpy matplotlib scikit-learn torch torchvision
```

---

##  How to Run

###  CNN Model

Open:

```
notebooks/cnn_model.ipynb
```

###  Classical Models

```
notebooks/classical_models.ipynb
```

###  Transfer Learning

```
notebooks/transfer_learning.ipynb
```

###  Test on Single Image

```bash
python src/test_image.py
```

###  Webcam Testing

```bash
python src/webcam_test.py
```

---

##  Results Summary

| Model               | Validation | Test       |
| ------------------- | ---------- | ---------- |
| Logistic Regression | 0.8054     | 0.8467     |
| SVM (Linear)        | 0.8456     | 0.8933     |
| SVM (RBF + PCA)     | 0.9128     | 0.9000     |
| CNN (Custom)        | 0.9137     | **0.9144** |
| Transfer Learning   | 0.9332     | **0.9429** |

---

##  Key Insights

* CNN captures **spatial features**, outperforming classical models
* Flattening images removes **important structural information**
* PCA improves classical models by reducing noise
* Transfer learning achieves **best overall performance**
* Consistent validation/test results → **good generalisation**

---

##  Experimental Rigor

* Train / Validation / Test split (70 / 15 / 15)
* 5-fold Cross-validation
* Overfitting analysis
* Convergence analysis

---

##  CNN Architecture

* Input: 224 × 224 × 3
* 3 Conv layers (ReLU + Pooling)
* Fully Connected (128 units)
* Dropout (0.5)
* Binary output

---

##  Transfer Learning

Fine-tuned a pre-trained model (ResNet18):

* Faster training
* Better feature extraction
* Highest accuracy achieved

---

##  Limitations

* Small dataset
* Class imbalance
* Classical models lose spatial structure

---

##  Future Work

* Larger datasets
* Better augmentation
* Deeper architectures
* Hyperparameter tuning

---

##  Author

Iram Mukri

Student ID: M01092222

Gayathri Lekshmi

Student ID: M01087828

---

##  Final Note

Deep learning models significantly outperform traditional methods for image classification, with transfer learning achieving the best results.

---

