import torch
from models import BaseModel, LearnerNetwork
from preprocess_data import TimeSeriesDataHandler
import matplotlib.pyplot as plt
import copy

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
validation_loader = data_handler.get_test_loader(batch_size)

# Initialize model
input_size = len(selected_features)
hidden_size = 5
output_size = 12
base_model = BaseModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
base_optimizer = torch.optim.Adam(base_model.parameters(), lr=0.01)

#Learner network
#Determine the size of the flattened parameters
input_size = total_gradient_elements(base_model)
hidden_size = 5
learner_network = LearnerNetwork(input_size, hidden_size)
meta_optimizer = torch.optim.Adam(learner_network.parameters(), lr=0.01)

"""
############################################################################
# Training for each task on base model and update the meta-learning model
############################################################################
"""
# Define loss function and optimizers
base_loss_function = torch.nn.MSELoss()
base_loss = []

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
        initial_state = copy.deepcopy(base_model.state_dict())
        num_batches += 1
        
        # Base-level training
        base_optimizer.zero_grad()
        output = base_model(data)
        base_loss = base_loss_function(output, target)
        base_loss.backward()
        epoch_base_loss += base_loss.item()

        #Prepare gradients
        gradients = [param.grad for param in base_model.parameters()]
        flattened_gradients, original_shapes = flatten_and_concat_gradients(gradients)
        
        #Give gradients to the meta model
        #Here meta learning, given gradients in entry it should predict the future next best gradients
        lstm_output = learner_network(flattened_gradients)
        
        #Compute initial loss
        initial_loss = compute_validation_loss(base_model, base_loss_function, validation_loader)
        
        #Apply the new predicted gradients to the base model
        #Reshape the predicted gradients and apply them to the model
        reshaped_gradients = reshape_to_original(lstm_output, original_shapes)
        apply_updates(base_model, reshaped_gradients)
        
        #Compute new loss of the base model that use the new gradients (lstm_output)
        new_loss = compute_validation_loss(base_model, base_loss_function, validation_loader)
        
        # Meta-level training, compute loss
        meta_optimizer.zero_grad()
        meta_loss = torch.tensor(new_loss - initial_loss, requires_grad=True)
        print(f"Meta Loss: {meta_loss} - Initial loss {initial_loss} - New loss {new_loss}")
        
        meta_loss.backward()
        meta_optimizer.step()
        
        base_model.load_state_dict(initial_state)

    # Calculate mean losses for the epoch
    mean_base_loss = epoch_base_loss / num_batches

    # Print the mean losses for the current epoch
    print(f"Epoch {epoch} - Base Loss: {mean_base_loss:.4f}")

# Plotting the base model training loss
plt.figure(figsize=(10, 5))
plt.plot(base_loss, label='Base Model Training Loss')
plt.title('Training Loss Per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

"""
############################################################################
# Compute test loss
############################################################################
"""
epoch_base_loss = 0
num_batches = 0
for data, target in test_loader:
    num_batches += 1
    output = base_model(data)
    loss = loss_function(output, target)
    epoch_base_loss += loss.item()
    
mean_base_loss_test = epoch_base_loss / num_batches
print(f"Test data - Base Loss: {mean_base_loss_test:.4f}")

