import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import cv2
import mediapipe as mp
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Set random seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

class AdvancedGestureDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform=None, augment=False):
        self.data = np.array(data, dtype=np.float32)
        self.labels = np.array(labels, dtype=np.int64)
        self.transform = transform
        self.augment = augment
        
        # Calculate mean and std for normalization
        self.mean = np.mean(self.data, axis=0, keepdims=True)
        self.std = np.std(self.data, axis=0, keepdims=True) + 1e-6
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx].copy()
        y = self.labels[idx]
        
        # Convert to tensor
        x = torch.from_numpy(x).float()
        
        # Apply data augmentation if enabled
        if self.augment and self.transform:
            x = self.transform(x)
        
        # Normalize the data
        x = (x - torch.from_numpy(self.mean).float()) / torch.from_numpy(self.std).float()
        
        return x, y

class AdvancedAugment:
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, landmarks):
        if random.random() > self.p:
            return landmarks
        
        # Create a copy to avoid modifying the original
        augmented = landmarks.clone()
        
        # Random scaling (0.9 to 1.1)
        scale = torch.FloatTensor(1).uniform_(0.9, 1.1).item()
        augmented = augmented * scale
        
        # Random translation
        if random.random() < 0.5:
            translation = torch.FloatTensor(landmarks.size()).uniform_(-0.05, 0.05)
            augmented = augmented + translation
        
        # Random noise
        if random.random() < 0.3:
            noise = torch.randn_like(augmented) * 0.01
            augmented = augmented + noise
        
        return augmented

class AdvancedGestureClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_classes=8, dropout_prob=0.4):
        super(AdvancedGestureClassifier, self).__init__()
        
        # Store input size for later use
        self.input_size = input_size
        
        # Batch normalization layer
        self.bn_input = nn.BatchNorm1d(input_size)
        
        # Main network with residual connections
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_prob)
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_prob)
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.BatchNorm1d(hidden_size // 4),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_prob)
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size // 4, hidden_size // 8),
            nn.Tanh(),
            nn.Linear(hidden_size // 8, 1, bias=False)
        )
        self.softmax = nn.Softmax(dim=1)
        
        # Output layer
        self.output = nn.Linear(hidden_size // 4, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Ensure input is 2D [batch_size, features]
        batch_size = x.size(0)
        if x.dim() > 2:
            x = x.view(batch_size, -1)
        
        # Input batch norm
        x = self.bn_input(x)
        
        # Forward pass through layers
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        
        # Reshape for attention (add sequence dimension)
        x3_reshaped = x3.unsqueeze(1)  # [batch_size, 1, features]
        
        # Attention mechanism
        attention_scores = self.attention(x3_reshaped)  # [batch_size, seq_len, 1]
        attention_weights = self.softmax(attention_scores.squeeze(-1)).unsqueeze(-1)  # [batch_size, seq_len, 1]
        x_attended = torch.sum(attention_weights * x3_reshaped, dim=1)  # [batch_size, features]
        
        # Output
        out = self.output(x_attended)
        return out

def extract_hand_landmarks(image_path):
    """Extract hand landmarks from an image with error handling"""
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = hands.process(image_rgb)
        
        # Extract hand landmarks
        landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get all 21 landmarks (x, y, z)
                hand = []
                for lm in hand_landmarks.landmark:
                    hand.extend([lm.x, lm.y, lm.z])
                landmarks.append(hand)
        
        return landmarks[0] if landmarks else None
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def process_images_directory(base_dir):
    """Process all images in the directory and extract features"""
    data = []
    labels = []
    gesture_mapping = {}
    
    # The actual images are in a subdirectory called 'images'
    images_dir = os.path.join(base_dir, 'images')
    
    # Get all subdirectories (each represents a gesture type)
    gesture_dirs = [d for d in os.listdir(images_dir) 
                   if os.path.isdir(os.path.join(images_dir, d)) and not d.startswith('.')]
    
    # Create gesture mapping
    gesture_mapping = {gesture: i for i, gesture in enumerate(sorted(gesture_dirs))}
    
    # Save the mapping for later use
    with open('advanced_config/gesture_mapping.json', 'w') as f:
        json.dump(gesture_mapping, f)
    
    print(f"Found {len(gesture_mapping)} gesture types: {list(gesture_mapping.keys())}")
    
    # Process each gesture directory
    for gesture_name, label in gesture_mapping.items():
        gesture_dir = os.path.join(images_dir, gesture_name)
        print(f"\nProcessing gesture: {gesture_name} (label: {label})")
        
        # Get all image files in the directory
        image_files = [f for f in os.listdir(gesture_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print(f"  No images found in {gesture_name} directory")
            continue
            
        print(f"  Found {len(image_files)} images")
        
        # Process each image
        for img_file in tqdm(image_files, desc=f"  Processing {gesture_name}"):
            img_path = os.path.join(gesture_dir, img_file)
            landmarks = extract_hand_landmarks(img_path)
            
            if landmarks is not None:  # If hand landmarks were detected
                data.append(landmarks)
                labels.append(label)
    
    if not data:
        print("\nError: No valid hand landmarks were detected in any images.")
        print("Please ensure the images contain clear hand gestures.")
        return None, None, None
    
    return np.array(data), np.array(labels), gesture_mapping

def train_model(data, labels, num_classes, gesture_mapping, batch_size=64, epochs=200, learning_rate=0.001):
    """Train the advanced gesture classifier model"""
    # Create output directory
    os.makedirs('models', exist_ok=True)
    
    # Setup TensorBoard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/gesture_classifier_{timestamp}')
    
    # Split data into train and validation sets with stratification
    X_train, X_val, y_train, y_val = train_test_split(
        data, labels, test_size=0.15, random_state=42, stratify=labels
    )
    
    # Calculate class weights for handling class imbalance
    class_counts = np.bincount(y_train)
    class_weights = 1. / (class_counts + 1e-6)
    class_weights = torch.FloatTensor(class_weights / class_weights.sum() * num_classes)
    
    # Create datasets with augmentation
    transform = AdvancedAugment(p=0.7)
    train_dataset = AdvancedGestureDataset(X_train, y_train, transform=transform, augment=True)
    val_dataset = AdvancedGestureDataset(X_val, y_val, augment=False)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size * 2,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=torch.cuda.is_available()
    )
    
    # Initialize model
    input_size = data.shape[1]
    model = AdvancedGestureClassifier(
        input_size=input_size,
        hidden_size=512,
        num_classes=num_classes,
        dropout_prob=0.4
    )
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler with warmup
    warmup_epochs = 5
    total_steps = len(train_loader) * epochs
    warmup_steps = len(train_loader) * warmup_epochs
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Early stopping
    best_val_acc = 0.0
    patience = 15
    patience_counter = 0
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        # Training phase
        for batch_data, batch_labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False):
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Calculate metrics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()
        
        # Calculate metrics
        train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f'\nEpoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save the entire model state including optimizer and scheduler
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'best_val_acc': best_val_acc,
                'input_size': input_size,
                'num_classes': num_classes,
                'class_weights': class_weights.tolist() if hasattr(class_weights, 'tolist') else class_weights,
                'gesture_mapping': gesture_mapping
            }, 'models/advanced_gesture_model.pth')
            print('  Model saved!')
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f'\nEarly stopping after {epoch+1} epochs')
            break
    
    print(f'\nBest Validation Accuracy: {best_val_acc:.2f}%')
    writer.close()
    return model

def main():
    # Directory containing gesture images in nested subfolders
    base_dir = r"C:\Users\cg515\Documents\kranthi\dr drivings\images"
    
    print("Processing images and extracting features...")
    result = process_images_directory(base_dir)
    
    if result[0] is None:  # If no data was processed
        return
        
    data, labels, gesture_mapping = result
    
    print(f"\nExtracted {len(data)} samples with {len(gesture_mapping)} gesture classes.")
    print("Gesture mapping:", gesture_mapping)
    
    # Train the model
    print("\nTraining advanced model...")
    train_model(data, labels, num_classes=len(gesture_mapping), gesture_mapping=gesture_mapping, epochs=200)
    
    print("\nTraining complete! Model saved to 'models/advanced_gesture_model.pth'")
    print("Gesture mapping saved to 'advanced_config/gesture_mapping.json'")

if __name__ == "__main__":
    main()
