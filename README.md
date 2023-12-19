# Age-Predictor
## Age and Face Detection Model Readme

### Introduction
This repository contains code for building and training a deep learning model for age and face detection using the Keras library. The model is designed to classify images into three classes based on age groups.

### Requirements
- Python 3.x
- Jupyter Notebook or Google Colab for running the code interactively
- Required Python packages: pandas, scikit-learn, keras, tensorflow, matplotlib

### Getting Started
1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/your-username/age-face-detection.git
    ```

2. Navigate to the project directory:

    ```bash
    cd age-face-detection
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Data Preparation
1. Mount Google Drive using the provided code in the Jupyter Notebook or Google Colab:

    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

2. Extract the dataset from a zip file and load the CSV file:

    ```python
    !unzip archive.zip

    import pandas as pd
    df = pd.read_csv("train.csv")
    ```

3. Encode categorical labels and split the dataset into training, validation, and test sets.

### Model Creation and Training
1. Define and create the neural network models using Keras. Three models are provided for experimentation: one with potential overfitting, one trained at an optimum epoch, and one with dropout regularization.

2. Compile and train the models:

    ```python
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy", Recall(), Precision(), CategoricalAccuracy()])
    history = model.fit(generator_train, validation_data=generator_validation, epochs=25, verbose=1)
    ```

### Model Evaluation
1. Evaluate the trained models on the test data:

    ```python
    prediction_fun("age_face_det1.h5")
    ```

2. Visualize the model metrics during training:

    ```python
    import matplotlib.pyplot as plt
    history_df = pd.DataFrame(history.history)

    metrics_plot = history_df.plot(title="Model Metric Evaluation", xlabel="EPOCHS", ylabel="Values (0-1)")
    metrics_plot.get_figure().savefig("metrics.png")
    ```

### VGG16 Pretrained Model
1. Load the VGG16 pretrained model and fine-tune it on the dataset:

    ```python
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten
    from tensorflow.keras.optimizers import Adam

    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    # ... (continue with model creation and training)
    ```

### Common Sense Baseline
1. Determine the most common class in the training data as a baseline:

    ```python
    most_common_class = data_train['class'].value_counts()
    most_common_class / len(data_train['class'])
    ```

### Conclusion
Feel free to experiment with different model architectures, hyperparameters, and preprocessing techniques to optimize the age and face detection model for your specific use case.

**Note:** Make sure to check the licensing terms of the dataset and pretrained models used in this project. Update the paths and filenames in the code according to your dataset and file structure.
