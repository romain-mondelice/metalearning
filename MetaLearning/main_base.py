import torch
from models import BaseModel, MetaLearner
from preprocess_data import TimeSeriesDataHandler
import matplotlib.pyplot as plt

"""
############################################################################
# General functions
############################################################################
"""
def plot_sequences(data_handler, num_sequences=3):
    for i in range(num_sequences):
        X, y = data_handler.train_dataset[i]
        X = X.numpy()  # Convert to numpy array
        y = y.numpy()  # Convert to numpy array

        # Flatten X and y if they are multidimensional
        if X.ndim > 1:
            X = X.flatten()
        if y.ndim > 1:
            y = y.flatten()

        plt.figure(figsize=(10, 4))
        plt.plot(range(len(X)), X, label="Input Sequence")
        plt.plot(range(len(X), len(X) + len(y)), y, label="Target")
        plt.legend()
        plt.title(f"Sequence and Target Plot for Sequence {i}")
        plt.show()

"""
############################################################################
# Main
############################################################################
"""
# Define batch size and number of epochs
batch_size = 64
num_epochs = 100

# Initialize TimeSeriesDataHandler
csv_path = './dataset_btc_h1.csv'

"""
############################################################################
# Task 1 - Predict next 12 Close
############################################################################
"""
data_handler = TimeSeriesDataHandler(csv_path, look_back=48, predict_steps=12)
selected_features = ["Number of transaction", "Difficulty", "Total Supply",\
                     "S2F", "observation_date", "Open", "High", "Low",\
                     "Close", "Volume", "Quote Asset Volume", "Number of Trades",\
                     "TB Base Volume", "TB Quote Volume", "Network Value",\
                     "NTV", "Transactions_zscore", "Difficulty_zscore",\
                     "S2F_zscore", "NTV_zscore", "volatility", "mid_price"]
data_handler.load_and_process_data(target_column='Close', feature_columns=selected_features)
train_loader = data_handler.get_train_loader(batch_size)
test_loader = data_handler.get_test_loader(batch_size)

X, y = data_handler.train_dataset[0]

# Initialize model
input_size = len(selected_features)
hidden_size = 50
output_size = 12 # Number of steps 
base_model = BaseModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
base_optimizer = torch.optim.SGD(base_model.parameters(), lr=0.01)

"""
############################################################################
# Training for each task on base model and update the meta-learning model
############################################################################
"""
meta_learner = MetaLearner(base_model.parameters())
meta_optimizer = torch.optim.SGD(meta_learner.parameters(), lr=0.01)

# Define loss function and optimizers
loss_function = torch.nn.MSELoss()

base_loss = []
meta_loss = []

"""
############################################################################
# Regroup all train loader
############################################################################
"""

for epoch in range(num_epochs):
    epoch_base_loss = 0.0
    epoch_meta_loss = 0.0
    num_batches = 0
    
    for data, target in train_loader:
        num_batches += 1
        
        # Base-level training
        base_optimizer.zero_grad()
        output = base_model(data)
        loss = loss_function(output, target)
        loss.backward()
        base_optimizer.step()
        
        epoch_base_loss += loss.item()
        
        # Meta-level training
        meta_optimizer.zero_grad()
        updated_params = meta_learner(base_model.parameters())
        with torch.no_grad():
            for param, updated_param in zip(base_model.parameters(), updated_params):
                param.copy_(updated_param)
        meta_output = base_model(data)
        meta_loss = loss_function(meta_output, target)
        meta_loss.backward()
        meta_optimizer.step()
        epoch_meta_loss += meta_loss.item()
    
    # Calculate mean losses for the epoch
    mean_base_loss = epoch_base_loss / 64
    mean_meta_loss = epoch_meta_loss / 64

    # Append the mean losses to their respective task-specific lists
    base_loss.append(mean_base_loss)
    meta_loss.append(mean_meta_loss)

    # Print the mean losses for the current epoch
    print(f"Epoch {epoch} - Base Loss: {mean_base_loss:.4f} - Meta Loss: {mean_meta_loss:.4f}")

# Plotting the base model training loss
plt.figure(figsize=(10, 5))
plt.plot(base_loss, label='Base Model Training Loss')
plt.plot(meta_loss, label='Meta Learner Training Loss')
plt.title('Training Loss Per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
        
