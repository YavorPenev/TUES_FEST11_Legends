import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import yfinance as yf
import os

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']


class StockPredictor(nn.Module):
    def __init__(self):
        super(StockPredictor, self).__init__()
        self.fc1 = nn.Linear(4, 64)  
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)  

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

def fetch_data(ticker):
    try:
        df = yf.download(ticker, period="7d", interval="1d")
        return df['Close'].values if 'Close' in df else None
    except Exception as e:
        print(f"Failed to fetch data for {ticker}: {e}")
        return None

def create_dataset():
    data = []
    for ticker in tickers:
        prices = fetch_data(ticker)
        if prices is not None and len(prices) > 1:
            expected_return = (prices[-1] - prices[0]) / prices[0] 
            for _ in range(60):  
                income = np.random.randint(3000, 10000)
                expenses = np.random.randint(1000, income)
                goal = np.random.randint(5000, 50000)
                timeframe = np.random.randint(3, 36)
                data.append([income, expenses, goal, timeframe, expected_return])
    return pd.DataFrame(data, columns=['income', 'expenses', 'goal', 'timeframe', 'return'])

def train_model():
    df = create_dataset()

    X = df[['income', 'expenses', 'goal', 'timeframe']].values.astype(np.float32)
    y = df['return'].values.astype(np.float32).reshape(-1, 1)

    model = StockPredictor()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)

    for epoch in range(300):
        model.train()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/300], Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), 'stock_model.pth')
    print("✅ Model trained and saved as 'stock_model.pth'")

if __name__ == '__main__':
    train_model()