import torch
import torch.nn as nn
import torch.nn.functional as F

class GestureClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super(GestureClassifier, self).__init__()
        
        # Define the neural network layers with increased capacity
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn3 = nn.BatchNorm1d(hidden_size // 2)
        self.dropout3 = nn.Dropout(0.3)
        
        self.fc4 = nn.Linear(hidden_size // 2, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize the weights of the neural network"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        # First layer
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Second layer
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Third layer
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        # Output layer
        x = self.fc4(x)
        
        return x
    
    def save_model(self, path: str):
        """Save the model weights to a file"""
        torch.save({
            'state_dict': self.state_dict(),
            'input_size': self.fc1.in_features,
            'hidden_size': self.fc1.out_features,
            'num_classes': self.fc4.out_features
        }, path)
    
    def load_model(self, path: str):
        """Load the model weights from a file"""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['state_dict'])
        self.eval() 