from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import yfinance as yf
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

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

tickers = {
    'AAPL': 'Apple',
    'MSFT': 'Microsoft',
    'GOOGL': 'Alphabet',
    'AMZN': 'Amazon',
    'NVDA': 'NVIDIA',
    'TSLA': 'Tesla',
    'META': 'Meta',
    'JPM': 'JPMorgan',
    'V': 'Visa',
    'WMT': 'Walmart'
}

def fetch_current_return(ticker):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=14)
        
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False
        )
        
        if len(data) < 2:
            return None
            
        closes = data['Close'].values
        current_close = float(closes[-1])
        prev_close = float(closes[-2])
        
        return (current_close - prev_close) / prev_close * 252
        
    except Exception:
        return None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        try:
            income = float(data['income'])
            expenses = float(data['expenses'])
            goal = float(data['goal'])
            timeframe = int(data['timeframe'])
        except (ValueError, KeyError):
            return jsonify({'error': 'Invalid input values'}), 400

        investable = income - expenses
        if investable <= 0:
            return jsonify({'error': 'No investable income'}), 400

        input_tensor = torch.tensor([[income, expenses, goal, timeframe]], dtype=torch.float32)
        recommendations = []

        for ticker, name in tickers.items():
            predicted_return = model(input_tensor).item()
            actual_return = fetch_current_return(ticker)
            
            if actual_return is None:
                continue
                
            recommended_amount = min(investable * 0.25, goal / len(tickers))
            
            recommendations.append({
                'symbol': ticker,
                'name': name,
                'predicted_return': float(predicted_return),
                'actual_return': float(actual_return),
                'recommended_amount': float(recommended_amount),
                'timeframe': timeframe
            })

        if not recommendations:
            return jsonify({'error': 'No stock data available'}), 503

        recommendations.sort(key=lambda x: x['predicted_return'], reverse=True)

        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'summary': {
                'total_investable': float(investable),
                'goal': float(goal),
                'timeframe': timeframe,
                'stocks_analyzed': len(recommendations)
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)