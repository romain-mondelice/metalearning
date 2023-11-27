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

class MetaLearner(nn.Module):
    def __init__(self, base_model_params):
        super(MetaLearner, self).__init__()
        self.param_adapters = nn.ModuleList([nn.Linear(1, 1) for _ in base_model_params])

    def forward(self, base_model_params):
        adapted_params = []
        for param, adapter in zip(base_model_params, self.param_adapters):
            # Flatten the parameter and pass it through the adapter
            flat_param = param.view(-1, 1)
            adapted_param = adapter(flat_param).view(param.size())
            adapted_params.append(adapted_param)
        return adapted_params
    
    