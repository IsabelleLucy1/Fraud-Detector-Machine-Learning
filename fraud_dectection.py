import gradio as gr
import pandas as pd
import joblib

# Load trained model
model = joblib.load("fraud_detection_pipeline.pkl")


def predict_fraud(
    transaction_type,
    amount,
    oldbalanceOrg,
    newbalanceOrig,
    oldbalanceDest,
    newbalanceDest,
):
    # Create DataFrame from user inputs
    input_data = pd.DataFrame([{
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest
    }])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Return result
    if prediction == 1:
        return (
            f"Prediction: {int(prediction)}\n\n"
            "This transaction may be fraudulent."
        )
    else:
        return (
            f"Prediction: {int(prediction)}\n\n"
            "This transaction appears legitimate."
        )


app = gr.Interface(
    fn=predict_fraud,
    inputs=[
        gr.Dropdown(
            ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT"],
            label="Transaction Type"
        ),
        gr.Number(value=1000.0, label="Amount"),
        gr.Number(value=10000.0, label="Old Balance (Sender)"),
        gr.Number(value=9000.0, label="New Balance (Sender)"),
        gr.Number(value=0.0, label="Old Balance (Receiver)"),
        gr.Number(value=0.0, label="New Balance (Receiver)")
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Fraud Detection Prediction App",
    description="Enter the transaction details and click Submit to predict whether the transaction is fraudulent."
)

app.launch()
                                