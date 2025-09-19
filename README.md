# 🧠 Psychological Well-Being Prediction (Machine Learning)

A **machine learning project** that predicts mental health conditions based on workplace and personal factors.  
Trained on the **OSMI Tech Survey 2016 dataset** with ML/DL models, the system helps identify early signs of mental illness and suggests possible interventions.  

---

## ✨ Features
- Preprocessing of **46K+ survey records** (cleaning, feature selection, encoding)  
- User questionnaire mapped to numerical inputs for predictions  
- Machine learning models trained & evaluated:  
  - 🌲 Random Forest  
  - ⚡ Support Vector Machine (SVM)  
  - 📊 Naïve Bayes  
  - 🧮 Recurrent Neural Network (RNN with LSTM)  
- Dashboard to visualize results & monitor user well-being  
- Multilingual & audio input support (via NLP/translation APIs)  

---

## 🛠 Tech Stack
- **Language:** Python 3.x  
- **Libraries:** scikit-learn, pandas, numpy, matplotlib, seaborn, keras/tensorflow  
- **Algorithms:** SVM, Random Forest, Naïve Bayes, RNN (LSTM)  
- **Dataset:** [OSMI Tech Survey 2016](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)  

---

## 📊 Dataset & Preprocessing
- **Rows:** 46,234 | **Columns:** 64  
- Dropped irrelevant columns (e.g., `name`, `age`, `department`)  
- Filled missing values using **mode imputation**  
- Selected **15 key features** (self-employment, family history, comfort at work, stress, career anxiety, etc.)  
- Encoded responses into **{-1, 0, 1}** for model training  

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt


▶️ Run the Project
# Step 1: Train models (Random Forest, SVM, Naïve Bayes, RNN)
python train.py

# Step 2: Test models on evaluation dataset
python test.py

# Step 3: Predict with user questionnaire (uses best model: SVM)
python predict.py

