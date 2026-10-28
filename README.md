# ML Internship — Codveda Technologies

Two end-to-end machine learning tasks from my ML Engineering internship at Codveda Technologies. Each task goes from raw data to a trained model, evaluation and charts.

## Task 1: Random Forest for customer churn

Predicts whether a customer will leave (churn), using a customer churn dataset.

How it works (`random-forest-classifier/src/`):

1. **Data prep** (`data_loader.py`): drops ID columns, turns number-like text into numbers, removes rows with missing values, one-hot encodes categories, scales features, and makes a stratified 80/20 train/test split (the classes are unbalanced).
2. **Tuning** (`train_rf.py`): grid search over `n_estimators`, `max_depth` and `min_samples_split` with 5-fold cross-validation, scored by weighted F1.
3. **Evaluation:** accuracy, precision, recall, F1, 5-fold CV on the best model, and a confusion matrix.
4. **Feature importance:** chart of the 15 most important features.

Outputs: `outputs/confusion_matrix.png`, `outputs/feature_importance.png`. The notebook `notebooks/random_forest_eda.ipynb` has the data exploration and notes.

## Task 2: Feed-forward neural network for MNIST

Classifies handwritten digits (0–9) with TensorFlow/Keras.

How it works (`neural_networks_tf/src/`):

1. **Data:** loads MNIST from `keras.datasets` (falls back to a local CSV), flattens each 28x28 image to 784 values and scales pixels to 0–1.
2. **Model** (`network_architecture.py`): Dense 256 → Dropout 0.3 → Dense 128 → Dropout 0.3 → Dense 64 → Dense 10 (softmax). Adam optimizer, sparse categorical cross-entropy.
3. **Training** (`train_nn.py`): up to 30 epochs, batch size 128, 10% validation split, early stopping on validation loss.
4. **Evaluation:** test accuracy, classification report, confusion matrix, and loss/accuracy curves.

Output: `outputs/loss_accuracy_curves.png`. The notebook `notebooks/neural_net_sandbox.ipynb` walks through each step.

## How to run

```bash
git clone https://github.com/AlenGevorgyan/ml-internship-codveda.git
cd ml-internship-codveda
pip install -r requirements.txt
```

Task 1 needs the churn CSV, which is not in the repo. Save it as `random-forest-classifier/data/churn_data.csv`, then:

```bash
cd random-forest-classifier/src
python train_rf.py
```

Task 2 downloads MNIST automatically:

```bash
cd neural_networks_tf/src
python train_nn.py
```

Run the scripts from inside `src/`, because the paths to `data/` and `outputs/` are relative.

## Limits

- The churn dataset is not included, so Task 1 cannot run from a fresh clone.
- Task 2 is a plain dense network. A CNN would suit images better.
- Results are printed to the console and are not saved to a file.

## Next steps

- Add the dataset source and a download step for Task 1.
- Compare the Random Forest with a gradient boosting model.
- Try a small CNN for Task 2 and compare.

## Tech

Python, pandas, NumPy, scikit-learn, TensorFlow/Keras, matplotlib, seaborn, Jupyter.
