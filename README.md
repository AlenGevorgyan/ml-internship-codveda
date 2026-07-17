# Codveda Technologies — Machine Learning Internship

Welcome to my development repository for the Machine Learning Internship at **Codveda Technologies**[cite: 1]. This repository serves as a centralized hub tracking my progress, source code, data preprocessing pipelines, and model evaluation metrics for the assigned engineering tasks over the 1-month program duration.

---

## 🧑‍💻 Intern Information
* **Name:** Alen Gevorgyan[cite: 1]
* **Role:** Machine Learning Intern[cite: 1]
* **Intern ID:** CV/AI/82079[cite: 1]
* **Timeline:** 1 Month

---

## 🚀 Assigned Tasks & Deliverables

Per the structural guidelines, I have selected two advanced-level tasks within the Machine Learning domain to fulfill my internship project benchmarks.

### 🌲 Task 1: Advanced Random Forest Classifier
* **Description:** Implement a robust Random Forest machine learning architecture optimized for classification pipelines using complex multi-feature datasets.
* **Core Objectives:**
  * Execute systematic hyperparameter tuning protocols, explicitly targeting optimal tree counts (`n_estimators`) and maximum tree depths (`max_depth`)[cite: 2].
  * Safeguard model generalization using cross-validation routines alongside classical performance evaluation metrics (Precision, Recall, F1-Score)[cite: 2].
  * Run feature importance analysis algorithms to extract, map, and visualize key dataset drivers[cite: 2].
* **Tech Stack:** Python, scikit-learn, pandas, matplotlib[cite: 2].

### 🧠 Task 2: Deep Learning Feed-Forward Neural Networks
* **Description:** Design, construct, and optimize a custom feed-forward neural network pipeline utilizando TensorFlow and Keras backends for tabular or image multi-class problems (such as MNIST digit classification)[cite: 2].
* **Core Objectives:**
  * Architect clean pipelines for data loading, pixel/feature normalization, and tensor batch preparation[cite: 2].
  * Map a multi-layer deep network topology defining explicit Input, Hidden, and Output layers with localized activation functions[cite: 2].
  * Implement backpropagation execution scripts utilizing robust loss functions and gradient-descent optimizers[cite: 2].
  * Capture and graph structural validation losses versus training loss curves to monitor for overfitting[cite: 2].
* **Tech Stack:** Python, TensorFlow/Keras, pandas, matplotlib[cite: 2].

---

## 📂 Repository Layout

To maintain clear separation between classical ensemble methods and deep learning tensor computations, the directory structure is organized as follows:

```text
├── .gitignore                      # Prevents massive dataset binary & .pt file uploads
├── README.md                       # Comprehensive project landing page
├── requirements.txt                # Fixed python dependency mapping
│
├── random_forest_classifier/       # Track 1: Advanced Random Forest Framework
│   ├── notebooks/                  # Prototyping, feature engineering, and tuning
│   │   └── random_forest_eda.ipynb
│   ├── src/                        # Modular source scripts
│   │   ├── data_loader.py          # Preprocessing and scaling logic
│   │   └── train_rf.py             # Hyperparameter tuning & metric tracking
│   └── outputs/                    # Exported feature importance charts & matrices
│
└── neural_networks_tf/             # Track 2: TensorFlow/Keras Deep Learning
    ├── notebooks/                  # Topology design & initial tensor shape verification
    │   └── neural_net_sandbox.ipynb
    ├── src/                        # Modular training frameworks
    │   ├── network_architecture.py # Deep learning model definition layer classes
    │   └── train_nn.py             # Backpropagation execution loops
    └── outputs/                    # Loss curves & accuracy performance visualizations
