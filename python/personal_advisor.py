from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np
import yfinance as yf
import pandas as pd

app = Flask(__name__)

class StockPredictor(nn.Module):
    def __init__(self):
        super(StockPredictor, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


model = StockPredictor()
model.load_state_dict(torch.load('stock_model.pth'))
model.eval()

# Hardcoded tickers
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']

def fetch_current_return(ticker):
    try:
        df = yf.download(ticker, period="7d", interval="1d")
        if 'Close' in df and len(df['Close']) > 1:
            return (df['Close'][-1] - df['Close'][0]) / df['Close'][0]
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
    return None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        income = float(data['income'])
        expenses = float(data['expenses'])
        goal = float(data['goal'])
        timeframe = float(data['timeframe'])

        investable = income - expenses
        if investable <= 0:
            return jsonify({'error': 'No investable income'}), 400

        input_tensor = torch.tensor([[income, expenses, goal, timeframe]], dtype=torch.float32)

        recommendations = []

        for ticker in tickers:
            predicted_return = model(input_tensor).item()
            actual_return = fetch_current_return(ticker)
            if actual_return is None:
                continue

            avg_return = (predicted_return + actual_return) / 2
            amount_to_invest = investable * min(1, avg_return) 

            recommendations.append({
                'ticker': ticker,
                'predicted_return': round(predicted_return, 4),
                'expected_return': round(actual_return, 4),
                'recommended_investment': round(amount_to_invest, 2)
            })


        recommendations.sort(key=lambda x: (x['predicted_return'] + x['expected_return']) / 2, reverse=True)

        if not recommendations:
            return jsonify({'message': 'No recommendations available'})

        return jsonify({'recommendations': recommendations})

    except Exception as e:
        print("Error during prediction:", e)
        return jsonify({'error': 'Prediction failed'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
