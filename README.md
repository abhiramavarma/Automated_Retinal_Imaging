# Automated Retinal Imaging for Early Diagnosis of Diabetes, Cardiac and Ocular Diseases using CNN.
## Introduction
This project automates the early diagnosis of Diabetes, Cardiac disease, and vision impairment (Ocular) using a hybrid Deep Learning Architecture incorporating Convolutional Neural Networks (CNN) – Long Short-Term Memory (LSTM) with DenseNet121 to analyze retinal scans. It overcomes limitations of manual methods and existing AI by providing precise and comprehensive predictions with 90% accuracy through a user-friendly web interface, enabling efficient and scalable preventive healthcare.

### Key Points Briefly Covered:
* **Goal**: Automated early diagnosis (Diabetes, Cardiac, Ocular).
* **Method**: CNN – LSTM (DenseNet121) analysis of retinal scans.
* **Advantage**: Improved precision and comprehensiveness over manual/existing AI.
* **Accuracy**: 90%.
* **Benefit**: Efficient, scalable, user-friendly preventive healthcare.

<br>

## Table of Contents
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)

<br>

## Project Structure
```&emsp; Automated_Retinal_Imaging/<br>
├── gaussian_filtered_images/
│   └── dataset.txt
├── static/
│   ├── modified/
│   └── original/
│   └── uploads/
│   └── style.css
├── templates/
│   ├── about.html
│   └── index.html
├── app.py
├── README.md
├── requirements.txt
├── train_model.py
└── train.csv
```

- `gaussian_filtered_images/`: Contains Gaussian-filtered retinal images (note: original datasets are to be downloaded from Kaggle as described in `dataset.txt`).
- `static/`: Contains static files such as CSS (`style.css`) and potentially original or modified images.
- `static/uploads/`: Stores user-uploaded retinal images for analysis.
- `templates/`: Contains HTML templates for the web interface (`about.html`, `index.html`).
- `app.py`: The main Flask application file.
- `README.md`: This file, providing an overview of the project.
- `requirements.txt`: Lists the Python dependencies.
- `train_model.py`: Script for training the CNN model.
- `train.csv`: CSV file containing training data labels and information.

<br>

## Dataset
The original retinal images used in this project are sourced from the APTOS 2019 Blindness Detection dataset available on Kaggle:
[https://www.kaggle.com/datasets/mariaherrerot/aptos2019](https://www.kaggle.com/datasets/mariaherrerot/aptos2019)

This dataset contains images categorized into different stages of Diabetic Retinopathy (No_DR, Mild, Moderate, Severe, Proliferate_DR) and an `export.pkl` file.

**Important:** Due to the large size of the image datasets, they are not included in this repository. Please download the dataset from the provided Kaggle link and place the image folders and `export.pkl` file inside the `gaussian_filtered_images` directory as described in `gaussian_filtered_images/dataset.txt`. The `train.csv` file, located in the root directory, contains the training labels.

<br>

## Installation
1.  Clone the repository:
2.  Install Python dependencies:
    It is recommended to create a virtual environment first.
    
    ```bash
    python -m venv venv
    source venv/bin/activate  # On macOS/Linux
    venv\Scripts\activate  # On Windows
    pip install -r requirements.txt
    ```
    
4.  Download and place the dataset:
    Follow the instructions in `gaussian_filtered_images/dataset.txt` to download the APTOS 2019 dataset from Kaggle and place the image folders and `export.pkl` file in the `gaussian_filtered_images` directory.

<br>

## Usage
1.  Train the CNN model:
    ```bash
    python train_model.py
    ```
2.  Run the Flask application:
    ```bash
    python app.py
    ```
    This will likely start a web server.

<br>

## Model Architecture

<p align="center">
  <img src="https://github.com/user-attachments/assets/9914b11f-780d-46cb-9b81-4b3744e077a4" width="60%" height="60%" alt="Architecture Diagram" title="Architecture Diagram"/>
<br><i>Fig: Architecture Diagram</i>
</p>

Our system employs a Convolutional Neural Network (CNN), specifically DenseNet121, for automated retinal image analysis to predict diabetes, cardiac disease, and vision impairment.
1. Input and Preprocessing: Retinal images undergo resizing, normalization, and augmentation (rotation, flipping, zoom) before being split into training and testing sets.
2. Feature Extraction (DenseNet121 CNN): The pre-processed images are fed into a DenseNet121 architecture, which extracts rich spatial features relevant to disease detection. 
3. Classification: The extracted features are fed into fully connected layers with a Softmax output layer, generating probabilities for each diagnostic category (e.g., stages of DR, cardiac conditions, vision impairment types).
4. Output and Evaluation: The model predicts the presence and type/stage of diseases. Performance is evaluated using metrics like BLEU scores on test dataset images.

<br>

## Result
The CNN-LSTM model achieves a 90% accuracy in classifying retinal images and generates descriptive summaries of retinal conditions. For example, the system can identify and describe features like microaneurysms and haemorrhages. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/1b8a2386-df88-4de1-ac8c-4ed1f288a865" width="50%" height="0%"/>
<br><i>Fig. 1: Initial Interface</i><br><br>
<img src="https://github.com/user-attachments/assets/b94fb815-915a-4e77-b6e2-6a18b022cfcd" width="50%" height="50%"/>
<br><i>Fig. 2: Selecting Test Image</i><br><br>
<img src="https://github.com/user-attachments/assets/15fff6e2-4edb-400d-8ef4-3c36e6826c9b" width="50%" height="50%"/>
<br><i>Fig. 3: Predicting the Results</i><br><br>
<img src="https://github.com/user-attachments/assets/b441d83e-8a90-44c2-a498-84c549938f35" width="50%" height="50%"/>
<br><i>Fig. 4: AI Generated Summary</i><br><br>
<img src="https://github.com/user-attachments/assets/3e423d2a-105a-4d5b-bb9b-e7977f48e72d" width="50%" height="50%"/>
<br><i>Fig. 5: Chat Bot Demonstration</i><br><br>
</p>


The accuracy of the generated descriptions is evaluated using BLEU scores. The model was trained for 50 epochs with the Adam optimizer. <br>
<p align="center">
  <img src="https://github.com/user-attachments/assets/2bd91962-c8f7-4761-9118-72cbf86f1684" width="50%" height="50%"/>
<br><i>Fig. 6: BLEU Scores</i><br>
    <br>
<img src="https://github.com/user-attachments/assets/1135f6b4-d30e-4b57-ab6c-71fcc8ba36f9" width="50%" height="50%"/>
<br><i>Fig. 7: Plotting the BLEU Scores</i><br><br>
</p>

This combined approach of classification and summarization can aid in more effective and efficient diagnosis.

---
<br><br>
Author : _NANDAYALA ABHIRAMA VARMA_
