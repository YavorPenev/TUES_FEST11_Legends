import os
import torch
import torch.nn as nn
import yfinance as yf
from flask import Flask, request, jsonify
from sklearn.preprocessing import MinMaxScaler
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "BRK.B", "UNH", "V",
    "JNJ", "WMT", "PG", "MA", "DIS", "PYPL", "PFE", "NFLX", "INTC", "PEP", "HD", "CSCO",
    "VZ", "MRK", "XOM", "KO", "BA", "NKE", "AMGN", "CAT", "MS", "GE", "IBM", "UPS",
    "MMM", "GS", "CVX", "RTX", "ABT", "T", "LOW", "WBA", "CVS", "INTU", "LMT",
    "TXN", "SPG", "BKNG", "MCD", "TGT", "BLK", "GS", "COST", "AXP", "COP", "QCOM", "AMT",
    "AIG", "FIS", "BMY", "SCHW", "CSX", "DUK", "KMB", "LMT", "LUV", "MMC", "CHTR", "STZ",
    "DE", "CI", "MCK", "CL", "GD", "DHR", "CME", "NSC", "ZTS", "ISRG", "GILD", "SYY", "WFC",
    "TMO", "NOC", "ECL", "BAX", "MO", "LLY", "AIG", "AXP", "MS", "GM", "ADBE", "ALPH",
    "F", "TIF", "GS", "ORCL", "DISCK", "DISCA", "AMAT", "GS", "GM", "LMT", "LOPE", "ALXN",
    "HPQ", "REGN", "MSCI", "ANTM", "STZ", "DVA", "NEE", "FMC", "PAYX", "HCA", "EW", "BIIB",
    "EXC", "BMY", "VRTX", "ICE", "ADP", "CHTR", "SPGI", "DHR", "AMGN", "DAL", "HUM", "MDT",
    "CME", "ALXN", "HUM", "EMR", "FAST", "LMT", "AFL", "AZO", "AME", "TROW", "ED", "INCY",
    "HSY", "CERN", "INTU", "DE", "EVRG", "FRT", "PFG", "MCO", "EXPE", "ALNY", "RE", "CDW",
    "WM", "EXPE", "FISV", "CTSH", "FRT", "GMAB", "VRTX", "CSX", "INFO", "WBA", "AIG", "HUM",
    "PG", "DPZ", "XLNX", "TRV", "BAC", "NSC", "ED"
]

class LSTMStockPredictor(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2):
        super(LSTMStockPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 64).to(device)
        c0 = torch.zeros(2, x.size(0), 64).to(device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

app = Flask(__name__)

model = LSTMStockPredictor(input_size=2).to(device)
model.load_state_dict(torch.load('stock_model.pth'))
model.eval()

def fetch_data(ticker):
    try:
        df = yf.download(ticker, period="30d", interval="1d", progress=False)
        if 'Close' in df and len(df['Close']) >= 30:
            return df[['Close', 'Volume']].values 
    except:
        pass
    return None

def predict_stock_return(ticker):
    data = fetch_data(ticker)
    if data is None or len(data) < 30:
        return None, None
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_data = scaler.fit_transform(data)
    input_seq = torch.tensor(normalized_data[-30:], dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        predicted_scaled = model(input_seq).item()

    min_close = scaler.data_min_[0]
    max_close = scaler.data_max_[0]
    predicted_price = predicted_scaled * (max_close - min_close) + min_close
    last_price = data[-1][0]

    return predicted_price, last_price

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    income = data.get('income')
    expenses = data.get('expenses')
    goal = data.get('goal')
    timeframe = data.get('timeframe')

    if not all([income, expenses, goal, timeframe]):
        return jsonify({"success": False, "error": "Missing input values"}), 400

    investable_income = income - expenses
    predictions = []

    for ticker in tickers[:150]:
        predicted_price, last_price = predict_stock_return(ticker)
        if predicted_price is None or last_price is None:
            continue
        actual_return = (predicted_price - last_price) / last_price
        predictions.append((ticker, actual_return, predicted_price))

    top_5 = sorted(predictions, key=lambda x: x[1], reverse=True)[:5]

    recommendations = []
    for ticker, actual_return, predicted_price in top_5:
        recommended_amount = investable_income / 5
        recommendations.append({
            "name": ticker,
            "symbol": ticker,
            "predicted_price": round(predicted_price, 2),
            "actual_return": round(actual_return, 4),
            "recommended_amount": round(recommended_amount, 2),
            "timeframe": timeframe
        })

    return jsonify({
        "success": True,
        "recommendations": recommendations
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
