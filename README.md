# Fraud Detection Machine Learning

This project explores transaction data and uses a trained machine-learning model to predict whether a transaction may be fraudulent. It includes a Jupyter notebook for analysis and a Gradio web app for trying predictions.

#the data set was from kaggle called Fraud Detection Dataset by 
Aman Ali Siddiqui 

## Project overview

- `fraud_detection.ipynb` explores the transaction dataset, examines fraud patterns, and prepares features for modeling.
- A scikit-learn pipeline scales numeric features, encodes the transaction type, and applies logistic regression with class weighting to account for the rare fraud class.
- `fraud_detection_pipeline.pkl` contains the trained pipeline used by the app.
- `fraud_dectection.py` provides a Gradio interface. Enter transaction details and submit them to see the model's prediction.

The app expects the model file to be in the project folder. The training dataset is used by the notebook; it is not needed to run the app when the saved model file is available.

## Run the web app

From the project folder, install the app dependencies:

```bash
python -m pip install gradio pandas joblib scikit-learn
```

Then start the app:

```bash
python fraud_dectection.py
```

Open the local URL printed in the terminal. The app asks for transaction type, amount, and sender/receiver balances.

## Example transactions to test

These example inputs were checked with the saved model in this project. They demonstrate sample model outputs; they are not guarantees about real transactions.

| Input | Transaction type | Amount | Old sender balance | New sender balance | Old receiver balance | New receiver balance | Example prediction |
|---|---|---:|---:|---:|---:|---:|---|
| Likely legitimate | `PAYMENT` | 9839.64 | 170136.00 | 160296.36 | 0.00 | 0.00 | `0` — appears legitimate |
| Likely fraud | `TRANSFER` | 181.00 | 181.00 | 0.00 | 0.00 | 0.00 | `1` — may be fraudulent |

Model predictions are for demonstration and should not be treated as proof that a transaction is fraudulent or legitimate.

## Screenshots



### Web app




### Transaction prediction test

