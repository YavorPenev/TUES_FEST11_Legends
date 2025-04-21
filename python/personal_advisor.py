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
    "TXN", "SPG", "BKNG", "MCD", "TGT", "BLK", "COST", "AXP", "COP", "QCOM", "AMT",
    "AIG", "FIS", "BMY", "SCHW", "CSX", "DUK", "KMB", "LUV", "MMC", "CHTR", "STZ",
    "DE", "CI", "MCK", "CL", "GD", "DHR", "CME", "NSC", "ZTS", "ISRG", "GILD", "SYY", "WFC",
    "TMO", "NOC", "ECL", "BAX", "MO", "LLY", "GM", "ADBE", "ORCL", "AMAT", "REGN", "MSCI",
    "NEE", "FMC", "PAYX", "HCA", "EW", "BIIB", "EXC", "VRTX", "ICE", "ADP", "SPGI",
    "DAL", "HUM", "MDT", "EMR", "FAST", "AFL", "AZO", "AME", "TROW", "ED", "INCY",
    "HSY", "CERN", "EVRG", "FRT", "PFG", "MCO", "EXPE", "ALNY", "RE", "CDW", "WM",
    "FISV", "CTSH", "GMAB", "INFO", "DPZ", "XLNX", "TRV", "BAC"
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
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
    return None

def predict_stock_return(ticker):
    data = fetch_data(ticker)
    if data is None:
        print(f"No data for {ticker}, skipping prediction.")
        return None
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_data = scaler.fit_transform(data)
    input_seq = torch.tensor(normalized_data[-30:], dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        predicted_price = model(input_seq).item()
    last_price = data[-1][0]
    
    # Handle invalid data or zero predictions more gracefully
    if last_price <= 0 or predicted_price <= 0 or np.isnan(predicted_price):
        print(f"Invalid prediction for {ticker}. Last price: {last_price}, Predicted price: {predicted_price}. Skipping.")
        return None
    
    predicted_return = ((predicted_price - last_price) / last_price) * 100
    return predicted_return

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    income = data.get('income')
    expenses = data.get('expenses')
    goal = data.get('goal')
    timeframe = data.get('timeframe')

    if not all([income, expenses, goal, timeframe]):
        return jsonify({"success": False, "error": "Missing input values"}), 400

    try:
        income = float(income)
        expenses = float(expenses)
        goal = float(goal)
        timeframe = int(timeframe)
    except:
        return jsonify({"success": False, "error": "Invalid input types"}), 400

    investable_income = (income * timeframe) - (expenses * timeframe)
    if investable_income <= 0:
        return jsonify({"success": False, "error": "No investable income available"}), 400

    predictions = []
    for ticker in tickers[:150]:
        predicted_return = predict_stock_return(ticker)
        
        # Handle edge cases where prediction might be NaN, negative, or zero
        if predicted_return is None or np.isnan(predicted_return) or predicted_return <= 0:
            continue
        
        predictions.append((ticker, predicted_return))

    if not predictions:
        return jsonify({"success": False, "error": "No valid predictions available"}), 500

    top_5 = sorted(predictions, key=lambda x: x[1], reverse=True)[:5]

    recommendations = []
    for ticker, predicted_return in top_5:
        recommended_amount = round(investable_income / 5, 2)
        recommendations.append({
            "name": ticker,
            "symbol": ticker,
            "predicted_annual_return": f"{round(predicted_return, 2)}%",
            "current_annual_return": f"{round(predicted_return, 2)}%",
            "recommended_investment": recommended_amount,
            "timeframe": timeframe
        })

    return jsonify({
        "success": True,
        "recommendations": recommendations
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
