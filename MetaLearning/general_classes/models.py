import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BaseModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h_n = h_n.view(-1, self.fc.in_features)
        output = self.fc(h_n)
        return output

class LearnerNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LearnerNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)
        self.activation = nn.Tanh()  # or another suitable activation function

    def forward(self, gradient_sequence):
        # Reshape the 1D input tensor to 3D (batch_size=1, sequence_length=num_gradients, input_size=num_elements)
        gradient_sequence = gradient_sequence.view(1, 1, -1)
        
        # Initialize hidden and cell states with the correct dimensions
        h0 = torch.zeros(self.num_layers, 1, self.hidden_size).to(gradient_sequence.device)
        c0 = torch.zeros(self.num_layers, 1, self.hidden_size).to(gradient_sequence.device)
    
        # Forward pass through LSTM
        out, _ = self.lstm(gradient_sequence, (h0, c0))
        
        # Pass the last output through the fully connected layer and activation function
        out = self.fc(out[:, -1, :])
        out = self.activation(out)
        
        # Flatten the output to 1D tensor
        return out.flatten()

