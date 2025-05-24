import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from gesture_classifier import GestureClassifier
from gesture_types import GestureType
import cv2
import mediapipe as mp
from tqdm import tqdm
import time

class GestureDataset(Dataset):
    def __init__(self, data, labels, hand_type):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.hand_type = hand_type
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def train_model(data_dict, batch_size=32, epochs=50, learning_rate=0.001):
    """Train separate models for left and right hands"""
    models = {}
    
    for hand_type, (data, labels) in data_dict.items():
        if data.shape[0] == 0 or labels.shape[0] == 0:
            print(f"[WARNING] No data for {hand_type} hand. Skipping training for this hand.")
            continue
        print(f"\nTraining model for {hand_type} hand...")
        
        # Split data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            data, labels, test_size=0.2, random_state=42
        )
        
        # Create datasets and dataloaders
        train_dataset = GestureDataset(X_train, y_train, hand_type)
        val_dataset = GestureDataset(X_val, y_val, hand_type)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize model
        model = GestureClassifier(
            input_size=data.shape[1],
            hidden_size=128,
            num_classes=len(GestureType)
        )
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        best_val_loss = float('inf')
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_data, batch_labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
                optimizer.zero_grad()
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_labels.size(0)
                train_correct += (predicted == batch_labels).sum().item()
            
            # Validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_data, batch_labels in val_loader:
                    outputs = model(batch_data)
                    loss = criterion(outputs, batch_labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_labels.size(0)
                    val_correct += (predicted == batch_labels).sum().item()
            
            # Print statistics
            train_loss = train_loss / len(train_loader)
            train_acc = 100 * train_correct / train_total
            val_loss = val_loss / len(val_loader)
            val_acc = 100 * val_correct / val_total
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save_model(f'models/gesture_model_{hand_type}.pth')
                print(f'Model saved for {hand_type} hand!')
            
            models[hand_type] = model
    
    return models

def main():
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Load existing data if available
    try:
        data = np.load('data/gesture_data.npz')
        data_dict = {
            'left': (data['left_X'], data['left_y']),
            'right': (data['right_X'], data['right_y'])
        }
        print("Loaded existing gesture data.")
    except:
        print("No existing data found. Please run the gesture collection script first.")
        return
    
    # Train models
    print("Training models...")
    models = train_model(data_dict)
    print("Training complete!")

if __name__ == "__main__":
    main() 