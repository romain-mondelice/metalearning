import torch
from general_classes.models import BaseModel, LearnerNetwork
from general_classes.preprocess_data import TimeSeriesDataHandler
import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(True)

"""
############################################################################
# General functions
############################################################################
"""
def plot_sequences(data_handler, num_sequences=3):
    for i in range(num_sequences):
        X, y = data_handler.train_dataset[i]
        X = X.numpy()
        y = y.numpy()
        
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

def compute_validation_loss(model, loss_function, validation_loader):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():  # Disable gradient computation
        for data, target in validation_loader:
            output = model(data)
            loss = loss_function(output, target)
            total_loss += loss.item()
            num_batches += 1

    average_loss = total_loss / num_batches
    return average_loss

def reshape_to_original(predicted_flat, original_shapes):
    reshaped_gradients = []
    index = 0
    for shape in original_shapes:
        num_elements = torch.prod(torch.tensor(shape))
        reshaped = predicted_flat[index:index + num_elements].reshape(shape)
        reshaped_gradients.append(reshaped)
        index += num_elements
    return reshaped_gradients

def flatten_and_concat_gradients(gradients):
    shapes = [g.shape for g in gradients]
    flattened = [g.flatten() for g in gradients]
    concatenated = torch.cat(flattened)
    return concatenated, shapes

def total_gradient_elements(model):
    total_elements = 0
    for param in model.parameters():
        total_elements += param.data.nelement()
    return total_elements

def apply_updates(base_model, updates, scale=0.1):
    with torch.no_grad():
        for param, update in zip(base_model.parameters(), updates):
            param.data = update

"""
############################################################################
# Main
############################################################################
"""
# Define batch size and number of epochs
batch_size = 64
num_epochs = 100

csv_path = './datasets/dataset_btc_h1.csv'

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
validation_loader = data_handler.get_test_loader(batch_size)

"""
############################################################################
# Init models - Base model and Meta model
############################################################################
"""
# Initialize model
input_size = len(selected_features)
hidden_size = 50
output_size = 12
base_model = BaseModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
base_optimizer = torch.optim.Adam(base_model.parameters(), lr=0.0001)

#Learner network
#Determine the size of the flattened parameters
input_size = total_gradient_elements(base_model)
hidden_size = 50
learner_network = LearnerNetwork(input_size, hidden_size)
meta_optimizer = torch.optim.Adam(learner_network.parameters(), lr=0.0001)

"""
############################################################################
# Define loss functions
############################################################################
"""
# Define loss function and optimizers
base_loss_function = torch.nn.MSELoss()
meta_loss_function = torch.nn.MSELoss()
base_mean_losses = []
meta_mean_losses = []

"""
############################################################################
# Training a base model and a meta model
############################################################################
"""
for epoch in range(num_epochs):
    epoch_base_loss = 0.0
    epoch_meta_loss = 0.0
    num_batches = 0

    for data, target in train_loader:
        num_batches += 1

        # Store initial state (parameters) of the base model
        initial_state = [param.clone() for param in base_model.parameters()]

        # Train the base model
        base_optimizer.zero_grad()
        output = base_model(data)
        base_loss = base_loss_function(output, target)
        base_loss.backward()
        epoch_base_loss += base_loss.item()

        # Prepare Initial gradients and parameters
        initial_gradients = [param.grad for param in base_model.parameters()]
        initial_flattened_gradients, _ = flatten_and_concat_gradients(initial_gradients)

        # Update base model parameters
        base_optimizer.step()

        # Store updated state (parameters) of the base model
        updated_state = [param.clone() for param in base_model.parameters()]

        # Flatten updated parameters for comparison
        updated_flattened_params, _ = flatten_and_concat_gradients(updated_state)

        # Give initial gradients to the meta model to predict updated parameters
        predicted_flattened_params = learner_network(initial_flattened_gradients)

        # Meta-level training, compute loss
        meta_optimizer.zero_grad()
        meta_loss = meta_loss_function(predicted_flattened_params, updated_flattened_params)
        meta_loss.backward()
        meta_optimizer.step()
        epoch_meta_loss += meta_loss.item()

    # Calculate mean losses for the epoch
    mean_base_loss = epoch_base_loss / num_batches
    mean_meta_loss = epoch_meta_loss / num_batches
    
    #Add to list to be able to plot after
    base_mean_losses.append(mean_base_loss)
    meta_mean_losses.append(mean_meta_loss)
    
    # Print the mean losses for the current epoch
    print(f"Epoch {epoch} - Base Loss: {mean_base_loss:.4f} - Meta Loss: {mean_meta_loss}")

"""
############################################################################
# Plotting base model and meta loss evolution
############################################################################
"""
# Creating subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# Plot for Base Model Training Loss
axs[0].plot(base_mean_losses, label='Base Model Training Loss')
axs[0].set_title('Base Model Training Loss')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].legend()

# Plot for Meta Model Training Loss
axs[1].plot(meta_mean_losses, label='Meta Model Training Loss')
axs[1].set_title('Meta Model Training Loss')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Loss')
axs[1].legend()

"""
############################################################################
# Save the meta model
############################################################################
"""
torch.save(learner_network, './saved_models/meta_model.pth')
