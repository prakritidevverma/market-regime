import pandas as pd 
import modin.pandas as mpd
from utils import clickhouse_data
from prophet import Prophet
import talib as ta
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


# Define parameters
start_date = '2017-01-01'
end_date = '2025-12-31'

# Load and preprocess data
large_cap = clickhouse_data.clickhouse_largecap(start_date, end_date)
mid_cap = clickhouse_data.clickhouse_midcap(start_date, end_date)
small_cap = clickhouse_data.clickhouse_smallcap(start_date, end_date)

# Fill missing dates
large_cap = clickhouse_data.fill_missing_dates_modin_optimized(large_cap)
mid_cap = clickhouse_data.fill_missing_dates_modin_optimized(mid_cap)
small_cap = clickhouse_data.fill_missing_dates_modin_optimized(small_cap)

# Stack dataframes
all_cap = mpd.concat([large_cap, mid_cap, small_cap])

featured_df = clickhouse_data.feature_engineering(all_cap)
# Convert Modin DataFrame to pandas DataFrame
featured_df = featured_df._to_pandas()

decomposed = clickhouse_data.decompose_time_series_multiindex(featured_df)
lagged = clickhouse_data.add_lagged_features_multiindex(decomposed)
lagged = lagged.select_dtypes(exclude=['object'])

# Define SimCLR model with LSTM encoder
class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

# SimCLR contrastive loss
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        z = torch.cat((z_i, z_j), dim=0)
        sim_matrix = torch.mm(z, z.T) / self.temperature
        labels = torch.arange(batch_size).repeat(2).to(z.device)
        loss = self.criterion(sim_matrix, labels)
        return loss

# Custom dataset with NaN handling
class StockDataset(Dataset):
    def __init__(self, df, feature_cols):
        self.data = df[feature_cols].copy()
        
        # Handle Infinite Values (Replace inf with NaN)
        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Handle NaN values
        self.data.fillna(method='ffill', inplace=True)  # Forward fill missing values
        self.data.fillna(0, inplace=True)  # Fill remaining NaNs with 0
        
        # Drop any remaining NaNs (if any)
        self.data.dropna(inplace=True)

        # Convert to NumPy array and apply Standard Scaling
        self.data = self.data.values.astype(np.float32)
        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(self.data)

    def __len__(self):
        return len(self.data) - 5  # Ensuring we have enough data points for sequences

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx:idx+5])  # Returning a sequence of 5 time steps


# Load dataset
feature_cols = ['open', 'high', 'low', 'close', 'volume', 'log_returns', 'pct_returns', 'rolling_std_1',
                'realized_volatility_1', 'rolling_std_3', 'realized_volatility_3', 'rolling_std_5', 'cumulative_pct_returns', 
                'realized_volatility_5', 'skewness_1', 'kurtosis_1', 'autocorr_1', 'momentum_roc_1',
                'sharpe_ratio_1', 'sortino_ratio_1', 'calmar_ratio', 'trend',
 'seasonal',
 'residual',
 'close_Lag_1',
 'close_Lag_2',]

dataset = StockDataset(lagged, feature_cols)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model
input_dim = len(feature_cols)
hidden_dim = 128
num_layers = 2
output_dim = 64
encoder = LSTMEncoder(input_dim, hidden_dim, num_layers, output_dim)
contrastive_loss = NTXentLoss()
optimizer = optim.Adam(encoder.parameters(), lr=0.001)

# Training loop
epochs = 50
for epoch in range(epochs):
    total_loss = 0
    for batch in dataloader:
        batch = batch.to(torch.float32)
        z_i = encoder(batch)
        z_j = encoder(batch)  # Data augmentation could be added here
        loss = contrastive_loss(z_i, z_j)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

# Save model
torch.save(encoder.state_dict(), "simclr_lstm.pth")
