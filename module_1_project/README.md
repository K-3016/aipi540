# Brain Tumor Classification Project

This project implements and evaluates three required modeling approaches:

- **Naive baseline**: majority class and random classifiers
- **Classical ML model**: logistic regression, random forest, or SVM on extracted image features
- **Neural network deep learning model**: a convolutional neural network trained on image data

## Project structure

- `src/data_loader.py`: load image data and extract classical features
- `src/models.py`: baseline, classical, and deep model definitions
- `src/train.py`: training pipeline for all three approaches
- `requirements.txt`: Python dependencies

## Dataset format

Place your brain tumor image dataset under `data/` in this structure:

```
data/
  train/
    tumor/
    no_tumor/
  val/
    tumor/
    no_tumor/
  test/
    tumor/
    no_tumor/
```

If your dataset uses different class names, update the folder names accordingly.

## Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Run training and evaluation

```bash
python src/train.py --data-dir data --output-dir outputs
```

This script trains and evaluates:

1. Naive baseline classifiers
2. A classical ML pipeline using extracted image features
3. A CNN-based deep learning model

## Notes

- The deep learning model uses TensorFlow/Keras.
- The classical model uses `scikit-learn` and can be switched between logistic regression, random forest, or SVM.
- Results are printed to the console and saved under `outputs/`.
