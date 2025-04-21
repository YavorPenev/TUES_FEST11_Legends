import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
import pandas as pd
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

def fetch_data(ticker):
    try:
        df = yf.download(ticker, period="90d", interval="1d", progress=False)
        if 'Close' in df and len(df['Close']) >= 30:
            return df[['Close', 'Volume']].values 
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
    return None

def create_dataset():
    X, y = [], []
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    for ticker in tickers:
        data = fetch_data(ticker)
        if data is None or len(data) < 30:
            continue
        
        close_prices = data[:, 0]
        volume = data[:, 1]

        normalized_data = scaler.fit_transform(data)
        
    
        for i in range(30, len(close_prices)):
            features = normalized_data[i-30:i]
            input_seq = torch.tensor(features, dtype=torch.float32)
            X.append(input_seq)
            
    
            expected_return = (close_prices[i] - close_prices[i-1]) / close_prices[i-1]
            y.append(torch.tensor([expected_return], dtype=torch.float32))

    return X, y

def train_model():
    X, y = create_dataset()
    X = torch.stack(X).to(device)
    y = torch.stack(y).to(device)

   
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    model = LSTMStockPredictor(input_size=2).to(device) 
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(30):
        model.train()
        output = model(X_train)
        loss = criterion(output, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    
        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val)
        
        print(f"Epoch [{epoch+1}/30], Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}")

    torch.save(model.state_dict(), 'stock_model.pth')
    print("✅ Model trained and saved as 'stock_model.pth'")

if __name__ == '__main__':
    train_model()
