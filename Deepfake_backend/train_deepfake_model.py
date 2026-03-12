import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
import json
import time

# Enhanced CNN Model for Deepfake Detection
class DeepfakeDetector(nn.Module):
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            # Second conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            # Third conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            # Fourth conv block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )

        self.fc_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 2)  # Binary classification: Real (0) or Fake (1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Custom Dataset Class
class DeepfakeDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

def download_real_dataset():
    """
    Download and prepare a real deepfake dataset.
    Using multiple sources for better training data.
    """
    print("Downloading real deepfake dataset...")

    try:
        from datasets import load_dataset
        import shutil

        # Create directories
        os.makedirs('real_dataset/real', exist_ok=True)
        os.makedirs('real_dataset/fake', exist_ok=True)

        real_images = []
        fake_images = []

        # Try to load deepfake_vs_real_images dataset
        try:
            print("Loading deepfake_vs_real_images dataset...")
            dataset = load_dataset("deepfake_vs_real_images", split="train")
            print(f"Dataset loaded with {len(dataset)} samples")

            # Process first 1000 samples (or all if less)
            max_samples = min(1000, len(dataset))
            processed = 0

            for i, sample in enumerate(dataset):
                if processed >= max_samples:
                    break

                try:
                    # Get image and label
                    image = sample['image']
                    label = sample['label']  # 0=real, 1=fake

                    # Convert to RGB if needed
                    if image.mode != 'RGB':
                        image = image.convert('RGB')

                    # Resize to standard size
                    image = image.resize((128, 128), Image.Resampling.LANCZOS)

                    # Save to appropriate directory
                    if label == 0:  # Real
                        img_path = f'real_dataset/real/real_{processed}.png'
                        real_images.append(img_path)
                    else:  # Fake
                        img_path = f'real_dataset/fake/fake_{processed}.png'
                        fake_images.append(img_path)

                    image.save(img_path)
                    processed += 1

                except Exception as e:
                    print(f"Error processing sample {i}: {e}")
                    continue

            print(f"Successfully processed {len(real_images)} real and {len(fake_images)} fake images")

        except Exception as e:
            print(f"Failed to load deepfake_vs_real_images dataset: {e}")
            print("Falling back to alternative dataset...")

            # Try alternative dataset
            try:
                print("Loading alternative dataset...")
                # You can add other datasets here
                # For now, create some basic real images from common sources
                return create_fallback_dataset()
            except Exception as e2:
                print(f"Alternative dataset also failed: {e2}")
                return create_fallback_dataset()

        return real_images + fake_images, [0] * len(real_images) + [1] * len(fake_images)

    except ImportError:
        print("datasets library not available, using fallback dataset")
        return create_fallback_dataset()

def create_enhanced_dataset(num_samples=2000, image_size=(224, 224)):
    """
    Create a much larger and more diverse dataset with advanced deepfake artifacts.
    """
    print(f"Creating enhanced dataset with {num_samples} samples...")

    # Create directories
    os.makedirs('enhanced_dataset/real', exist_ok=True)
    os.makedirs('enhanced_dataset/fake', exist_ok=True)

    real_images = []
    fake_images = []

    for i in range(num_samples // 2):
        # Create highly realistic "real" images with natural variations
        base_color = np.random.randint(190, 230, 3, dtype=np.uint8)
        real_img = np.random.randint(
            np.maximum(0, base_color - 20),
            np.minimum(255, base_color + 20),
            (image_size[0], image_size[1], 3),
            dtype=np.uint8
        )

        # Add sophisticated face-like features
        center_y, center_x = image_size[0] // 2, image_size[1] // 2
        face_size = np.random.randint(70, 90)

        # Face area with gradient skin tones
        face_area = real_img[center_y-face_size//2:center_y+face_size//2,
                           center_x-face_size//2:center_x+face_size//2]

        # Create natural skin tone gradient
        for y in range(face_area.shape[0]):
            for x in range(face_area.shape[1]):
                distance_from_center = np.sqrt((y - face_size//2)**2 + (x - face_size//2)**2)
                skin_variation = np.random.randint(-10, 10)
                # Ensure base_color is treated as int for addition
                new_color = np.clip(np.array(base_color, dtype=int) + skin_variation, 180, 240).astype(np.uint8)
                face_area[y, x] = new_color

        # Add detailed facial features
        # Eyes with natural variation
        eye_color = np.random.randint(30, 80, 3, dtype=np.uint8)
        eye_y = center_y - np.random.randint(12, 18)
        real_img[eye_y:eye_y+3, center_x-25:center_x-15] = eye_color
        real_img[eye_y:eye_y+3, center_x+15:center_x+25] = eye_color

        # Eyebrows
        brow_color = np.random.randint(20, 50, 3, dtype=np.uint8)
        real_img[eye_y-5:eye_y-2, center_x-30:center_x-10] = brow_color
        real_img[eye_y-5:eye_y-2, center_x+10:center_x+30] = brow_color

        # Nose
        nose_color = np.random.randint(160, 190, 3, dtype=np.uint8)
        real_img[center_y-3:center_y+7, center_x-3:center_x+3] = nose_color

        # Mouth with natural shape
        mouth_color = np.random.randint(80, 120, 3, dtype=np.uint8)
        mouth_y = center_y + np.random.randint(15, 25)
        mouth_width = np.random.randint(20, 35)
        real_img[mouth_y:mouth_y+2, center_x-mouth_width//2:center_x+mouth_width//2] = mouth_color

        # Add subtle texture and natural lighting
        # Add some random freckles or skin texture
        for _ in range(np.random.randint(5, 15)):
            tex_y = np.random.randint(center_y-30, center_y+30)
            tex_x = np.random.randint(center_x-30, center_x+30)
            if 0 <= tex_y < image_size[0] and 0 <= tex_x < image_size[1]:
                texture_color = np.random.randint(150, 200, 3, dtype=np.uint8)
                real_img[tex_y, tex_x] = texture_color

        real_img = Image.fromarray(real_img)
        real_path = f'enhanced_dataset/real/real_{i}.png'
        real_img.save(real_path)
        real_images.append(real_path)

        # Create sophisticated "fake" images with multiple deepfake artifacts
        fake_img = Image.open(real_path)
        fake_array = np.array(fake_img)

        # Apply multiple deepfake artifacts
        artifact_types = np.random.choice(['color_shift', 'blending', 'pixelation', 'compression', 'noise', 'warping'],
                                        size=np.random.randint(2, 4), replace=False)

        for artifact in artifact_types:
            if artifact == 'color_shift':
                # Color temperature shift (common in deepfakes)
                region = fake_array[center_y-25:center_y+25, center_x-25:center_x+25]
                region[:, :, 0] = np.clip(region[:, :, 0] * np.random.uniform(0.9, 1.1), 0, 255)  # Red channel
                region[:, :, 2] = np.clip(region[:, :, 2] * np.random.uniform(0.9, 1.1), 0, 255)  # Blue channel

            elif artifact == 'blending':
                # Poor blending artifacts
                blend_width = np.random.randint(3, 8)
                fake_array[center_y-35:center_y-35+blend_width, :] = np.random.randint(200, 255, (blend_width, image_size[1], 3), dtype=np.uint8)
                fake_array[center_y+35-blend_width:center_y+35, :] = np.random.randint(200, 255, (blend_width, image_size[1], 3), dtype=np.uint8)

            elif artifact == 'pixelation':
                # JPEG-like compression artifacts
                block_size = np.random.randint(4, 8)
                for y in range(0, image_size[0], block_size):
                    for x in range(0, image_size[1], block_size):
                        block = fake_array[y:y+block_size, x:x+block_size]
                        if block.size > 0:
                            # Simulate JPEG compression by reducing color variations
                            avg_color = np.mean(block, axis=(0, 1), dtype=int)
                            variation = np.random.randint(-5, 6, 3)
                            block_color = np.clip(avg_color + variation, 0, 255)
                            fake_array[y:y+block_size, x:x+block_size] = block_color

            elif artifact == 'compression':
                # Heavy compression artifacts
                # Create 8x8 DCT-like blocking
                for y in range(0, image_size[0], 8):
                    for x in range(0, image_size[1], 8):
                        block = fake_array[y:y+8, x:x+8]
                        if block.size > 0:
                            # Keep only low-frequency components
                            block[::2, ::2] = np.mean(block[::2, ::2], axis=(0, 1), keepdims=True)

            elif artifact == 'noise':
                # Add correlated noise (common in GAN-generated images)
                noise = np.random.normal(0, np.random.uniform(5, 15), fake_array.shape)
                # Make noise correlated (more realistic)
                noise = cv2.GaussianBlur(noise.astype(np.float32), (3, 3), 0)
                fake_array = np.clip(fake_array.astype(np.float32) + noise, 0, 255).astype(np.uint8)

            elif artifact == 'warping':
                # Subtle warping artifacts
                rows, cols = fake_array.shape[:2]
                # Create a subtle wave distortion
                for y in range(rows):
                    offset = int(2 * np.sin(y * 0.1))
                    if offset != 0:
                        fake_array[y] = np.roll(fake_array[y], offset, axis=0)

        # Add final realistic touches
        # Random brightness/contrast variation
        fake_array = fake_array.astype(np.float32)
        brightness = np.random.uniform(0.9, 1.1)
        contrast = np.random.uniform(0.9, 1.1)
        fake_array = np.clip(fake_array * contrast + (brightness - 1) * 128, 0, 255).astype(np.uint8)

        fake_img = Image.fromarray(fake_array)
        fake_path = f'enhanced_dataset/fake/fake_{i}.png'
        fake_img.save(fake_path)
        fake_images.append(fake_path)

    print(f"Created {len(real_images)} real and {len(fake_images)} fake images with advanced artifacts")
    return real_images + fake_images, [0] * len(real_images) + [1] * len(fake_images)

def train_enhanced_model(model, train_loader, val_loader, test_loader, num_epochs=30, device='cuda'):
    """
    Enhanced training with advanced optimization techniques
    """
    # Advanced loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Advanced optimizer with different learning rates for different layers
    base_lr = 0.001
    optimizer = optim.AdamW([
        {'params': model.conv_layers.parameters(), 'lr': base_lr},
        {'params': model.fc_layers.parameters(), 'lr': base_lr * 2}  # Higher LR for classifier
    ], weight_decay=1e-4, betas=(0.9, 0.999))

    # Advanced learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[base_lr, base_lr * 2],
        epochs=num_epochs, steps_per_epoch=len(train_loader),
        pct_start=0.3, anneal_strategy='cos'
    )

    # Early stopping
    early_stopping_patience = 10
    best_accuracy = 0.0
    patience_counter = 0
    best_epoch = 0

    model.to(device)

    # Mixed precision training for faster training
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    train_losses = []
    val_accuracies = []
    learning_rates = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Training phase with mixed precision
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            if scaler:
                # Mixed precision training
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Regular training
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Update learning rate after optimizer step
            scheduler.step()

            running_loss += loss.item()

            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(epoch_loss)
        learning_rates.append(optimizer.param_groups[0]['lr'])

        print(f"Training Loss: {epoch_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")

        # Validation phase
        model.eval()
        val_preds = []
        val_labels = []
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_accuracy = accuracy_score(val_labels, val_preds) * 100
        val_loss = val_loss / len(val_loader)
        val_accuracies.append(val_accuracy)

        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        # Save best model with early stopping
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': val_accuracy,
                'loss': epoch_loss,
                'scheduler_state_dict': scheduler.state_dict()
            }, 'best_enhanced_deepfake_model.pth')
            print("Best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"⏹️ Early stopping at epoch {epoch+1}")
                break

    # Load best model for final evaluation
    checkpoint = torch.load('best_enhanced_deepfake_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Final evaluation on test set
    model.eval()
    test_preds = []
    test_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_accuracy = accuracy_score(test_labels, test_preds) * 100
    print(f"\n🏆 Final Test Accuracy: {test_accuracy:.2f}%")
    print(f"📈 Best Validation Accuracy: {best_accuracy:.2f}% (Epoch {best_epoch+1})")

    # Generate detailed classification report
    from sklearn.metrics import classification_report, confusion_matrix
    print("\n📊 Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=['Real', 'Fake']))

    # Plot enhanced training curves
    plt.figure(figsize=(16, 8))

    plt.subplot(2, 3, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(val_accuracies, label='Validation Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.subplot(2, 3, 3)
    plt.plot(learning_rates, label='Learning Rate', color='green')
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.legend()

    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    plt.subplot(2, 3, 4)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks([0, 1], ['Real', 'Fake'])
    plt.yticks([0, 1], ['Real', 'Fake'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.subplot(2, 3, 5)
    # Training progress
    epochs_range = range(1, len(train_losses) + 1)
    plt.plot(epochs_range, [acc/100 for acc in val_accuracies], 'o-', label='Validation')
    plt.axhline(y=test_accuracy/100, color='r', linestyle='--', label=f'Test ({test_accuracy:.1f}%)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 6)
    # Model architecture summary
    plt.text(0.1, 0.8, f'Model: Enhanced CNN\nParameters: {sum(p.numel() for p in model.parameters()):,}\nInput Size: 224x224\nBest Epoch: {best_epoch+1}',
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('enhanced_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

    return test_accuracy

def main():
    print("Starting Enhanced Deepfake Detection Model Training")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create enhanced dataset with more samples and better artifacts
    image_paths, labels = create_enhanced_dataset(num_samples=2000, image_size=(224, 224))

    print(f"Dataset created with {len(image_paths)} total samples")

    # Split into train/val/test
    from sklearn.model_selection import train_test_split
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=0.3, random_state=42, stratify=labels
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )

    print(f"Training samples: {len(train_paths)}, Validation samples: {len(val_paths)}, Test samples: {len(test_paths)}")

    # Advanced data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(degrees=20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3))
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = DeepfakeDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = DeepfakeDataset(val_paths, val_labels, transform=val_test_transform)
    test_dataset = DeepfakeDataset(test_paths, test_labels, transform=val_test_transform)

    # Create data loaders with optimized settings
    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=True, num_workers=0,
        pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=8, shuffle=False, num_workers=0,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=8, shuffle=False, num_workers=0,
        pin_memory=True
    )

    # Initialize enhanced model
    model = DeepfakeDetector()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Enhanced training with better optimization
    start_time = time.time()
    best_accuracy = train_enhanced_model(model, train_loader, val_loader, test_loader, device=device)
    training_time = time.time() - start_time

    print(".2f")
    print(".4f")

    # Save final model
    torch.save(model.state_dict(), 'deepfake_detector_enhanced_final.pth')

    # Save model info
    model_info = {
        'model_type': 'EnhancedDeepfakeDetector',
        'input_size': [3, 224, 224],
        'num_classes': 2,
        'classes': ['real', 'fake'],
        'best_accuracy': best_accuracy,
        'training_time': training_time,
        'device': str(device),
        'parameters': total_params,
        'dataset_size': len(image_paths),
        'training_samples': len(train_paths),
        'validation_samples': len(val_paths),
        'test_samples': len(test_paths),
        'enhancements': [
            'Advanced data augmentation',
            'Larger input size (224x224)',
            'Enhanced artifacts simulation',
            'Improved regularization',
            'Better training optimization'
        ]
    }

    with open('model_info_enhanced.json', 'w') as f:
        json.dump(model_info, f, indent=2)

    print("🎉 Enhanced training completed!")
    print("📁 Model saved as 'deepfake_detector_enhanced_final.pth'")
    print("📊 Training curves saved as 'training_curves.png'")
    print("📋 Model info saved as 'model_info_enhanced.json'")

if __name__ == "__main__":
    main()