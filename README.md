# Credit Card Fraud Detection

A machine learning mini-project focused on identifying fraudulent credit card transactions using supervised learning techniques. 

This project demonstrates the end-to-end workflow of a classification problem, including data preprocessing, handling highly imbalanced datasets, model training using a Random Forest Classifier, and performance evaluation.

---

## 📊 Dataset

This project uses the [Credit Card Fraud Detection dataset from Kaggle](https://www.kaggle.com/code/aarthiramalingam/creditcard). 

**Note on Data Storage:** Due to GitHub's file size limits, the raw dataset (`creditcard.csv`) is **not** included in this repository. 

### How to get the data:
1. Download the dataset from the Kaggle link above.
2. Extract the downloaded archive.
3. Move the `creditcard.csv` file into the `data/` directory of this project.

---

## 🗂️ Project Structure

```text
credit-card-fraud-detection/
│
├── data/                  # Contains the dataset (ignored by Git)
│   └── .gitkeep
├── notebooks/             # Jupyter notebooks for Exploratory Data Analysis
│   └── .gitkeep
├── src/                   # Source code for the ML pipeline
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── evaluate_model.py
│
├── .gitignore             # Specifies intentionally untracked files
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies