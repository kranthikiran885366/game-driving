import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from src.ai.train_improved import GestureImageDataset, ImprovedGestureClassifier

def load_model_and_data():
    # Load the trained model
    model_path = 'models/improved_gesture_model.pth'
    with open('improved_gesture_mapping.json', 'r') as f:
        gesture_mapping = json.load(f)
    
    # Reverse the mapping for display
    idx_to_gesture = {v: k for k, v in gesture_mapping.items()}
    num_classes = len(gesture_mapping)
    
    # Initialize model
    model = ImprovedGestureClassifier(
        input_size=63,  # 21 landmarks * 3 (x,y,z)
        hidden_size=256,
        num_classes=num_classes,
        dropout_prob=0.3
    )
    
    # Load model weights
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load data
    base_dir = r"C:\Users\cg515\Documents\kranthi\dr drivings\images"
    from src.ai.train_improved import process_images_directory
    data, labels, _ = process_images_directory(base_dir)
    
    # Create dataset and dataloader
    dataset = GestureImageDataset(data, labels, augment=False)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    return model, dataloader, idx_to_gesture

def evaluate_model(model, dataloader, idx_to_gesture):
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_data, batch_labels in dataloader:
            outputs = model(batch_data)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(batch_labels.numpy())
    
    # Generate classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, 
                              target_names=idx_to_gesture.values(),
                              zero_division=0))
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=idx_to_gesture.values(),
                yticklabels=idx_to_gesture.values())
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved as 'confusion_matrix.png'")
    
    # Calculate class-wise accuracy
    class_correct = [0] * len(idx_to_gesture)
    class_total = [0] * len(idx_to_gesture)
    
    for label, pred in zip(all_labels, all_preds):
        if label == pred:
            class_correct[label] += 1
        class_total[label] += 1
    
    print("\nClass-wise Accuracy:")
    for i in range(len(idx_to_gesture)):
        print(f"{idx_to_gesture[i]}: {100 * class_correct[i] / class_total[i]:.2f}% ({class_correct[i]}/{class_total[i]})")

if __name__ == "__main__":
    model, dataloader, idx_to_gesture = load_model_and_data()
    evaluate_model(model, dataloader, idx_to_gesture)
