# Cancer Patient Outcome Predictor

A machine learning web application that predicts cancer patient survival outcomes (Alive or Deceased) based on clinical and demographic features. Built with a Random Forest classifier and deployed through an interactive Streamlit interface.

## Overview

This project takes patient details — including age, gender, state, cancer type, stage, treatment type, and diagnosis timeline — and predicts the likely survival outcome using a model trained on a synthetic dataset of Indian cancer patients (2022–2025). The goal is to demonstrate an end-to-end ML workflow: data preprocessing, model training, and deployment as a usable web app.

## Features

- Interactive web interface built with Streamlit
- Random Forest classifier trained on patient clinical data
- Dropdown inputs with real category labels (e.g., cancer type, treatment type, state) instead of raw encoded values
- Clear, human-readable prediction output

## Tech Stack

- **Python**
- **scikit-learn** — model training (Random Forest)
- **Streamlit** — web interface
- **pandas** — data handling
- **joblib** — model serialization

## Disclaimer

This project uses a **synthetic/educational dataset** sourced from Kaggle and is intended for learning and portfolio purposes only. It is **not validated for real clinical or medical decision-making** and should not be used as a substitute for professional medical advice.

## How to Run Locally

```bash
git clone https://github.com/Shan-Arangodan/Cancer-Survival-Predictor.git
cd Cancer-Survival-Predictor
pip install -r requirements.txt
streamlit run Cancerpred.py
```
