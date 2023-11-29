import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np

class TimeSeriesDataHandler:
    def __init__(self, csv_path, test_size=0.2, random_state=42, look_back=12, predict_steps=12):
        self.csv_path = csv_path
        self.test_size = test_size
        self.random_state = random_state
        self.look_back = look_back
        self.predict_steps = predict_steps
        self.train_dataset = None
        self.test_dataset = None
        self.norm_params = {}

    def process_time_component(self, df, time_column):
        df[time_column] = pd.to_datetime(df[time_column])
        df['sin_time'] = np.sin(2 * np.pi * df[time_column].dt.dayofyear / 365.25)
        df['cos_time'] = np.cos(2 * np.pi * df[time_column].dt.dayofyear / 365.25)
        return df
    
    def calculate_volatility(self, df, window=12):
        df['volatility'] = df['Close'].rolling(window=window).std()
        return df

    def calculate_mid_price(self, df):
        df['mid_price'] = (df['High'] + df['Low']) / 2.0
        return df

    def create_sequences(self, data, feature_columns, target_column):
        """
        Create sequences of data for LSTM input and corresponding targets.
        """
        sequences = []
        targets = []
        
        for i in range(len(data) - self.look_back - self.predict_steps + 1):
            seq = data[feature_columns].iloc[i:i + self.look_back].values
            target = data[target_column].iloc[i + self.look_back:i + self.look_back + self.predict_steps].values
            sequences.append(seq)
            targets.append(target)
        return np.array(sequences), np.array(targets)

    def load_and_process_data(self, target_column, feature_columns=None):
        df = pd.read_csv(self.csv_path)
        
        # Calculate additional features
        df = self.calculate_volatility(df)
        df = self.calculate_mid_price(df)
        df = df.dropna()
        
        # Process time component
        df = self.process_time_component(df, 'observation_date')
        
        if feature_columns:
            df = df[feature_columns + ['sin_time', 'cos_time']]

        train_df, test_df = train_test_split(df, test_size=self.test_size, random_state=self.random_state)

        # Normalize data
        # Normalize the training data and save normalization parameters
        for col in feature_columns:
            mean, std = train_df[col].mean(), train_df[col].std()
            self.norm_params[col] = {'mean': mean, 'std': std}
            train_df[col] = (train_df[col] - mean) / std

        # Normalize the testing data using the same parameters
        for col in feature_columns:
            mean, std = self.norm_params[col]['mean'], self.norm_params[col]['std']
            test_df[col] = (test_df[col] - mean) / std

        # Create sequences
        X_train, y_train = self.create_sequences(train_df, feature_columns, target_column)
        X_test, y_test = self.create_sequences(test_df, feature_columns, target_column)

        # Convert to tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        # Create TensorDatasets
        self.train_dataset = TensorDataset(X_train, y_train)
        self.test_dataset = TensorDataset(X_test, y_test)

    def get_train_loader(self, batch_size):
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False)

    def get_test_loader(self, batch_size):
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
