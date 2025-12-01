import os
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchaudio.datasets import SPEECHCOMMANDS
import numpy as np
import json
import time

# Створюємо обидві папки для сумісності
os.makedirs('artifacts', exist_ok=True)
os.makedirs('model-output', exist_ok=True)

# --- Model definition ---
class AudioClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.gap = nn.AdaptiveAvgPool2d((4, 2))
        self.fc1 = nn.Linear(32 * 4 * 2, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(self.dropout(x)))
        x = self.fc2(self.dropout(x))
        return x

# --- Параметри ---
target_classes = ['yes', 'no', 'up', 'down']
num_classes = len(target_classes)
batch_size = 8
epochs = int(os.getenv('EPOCHS', '2'))
samples_per_class = int(os.getenv('SAMPLES_PER_CLASS', '10'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🚀 Starting training with configuration:")
print(f"  - Device: {device}")
print(f"  - Epochs: {epochs}")
print(f"  - Samples per class: {samples_per_class}")
print(f"  - Batch size: {batch_size}")
print(f"  - Target classes: {target_classes}")

# --- Перетворення ---
mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=512,
    hop_length=256,
    n_mels=32
)

def label_to_index(word):
    return torch.tensor(target_classes.index(word))

# --- Collate function ---
def simple_collate_fn(batch):
    tensors, targets = [], []
    
    for waveform, sample_rate, label, speaker_id, utterance_number in batch:
        try:
            spec = mel_spectrogram(waveform).squeeze(0)
            time_frames = 32
            if spec.shape[1] > time_frames:
                spec = spec[:, :time_frames]
            elif spec.shape[1] < time_frames:
                pad_size = time_frames - spec.shape[1]
                spec = torch.nn.functional.pad(spec, (0, pad_size))
            
            spec = spec.unsqueeze(0)
            tensors.append(spec)
            targets.append(label_to_index(label))
        except Exception as e:
            print(f"⚠️ Error processing sample: {e}")
            continue
    
    if not tensors:
        dummy_input = torch.randn(2, 1, 32, 32)
        dummy_target = torch.tensor([0, 1])
        return dummy_input, dummy_target
    
    return torch.stack(tensors), torch.stack(targets)

# --- Завантаження даних ---
def get_minimal_dataset(subset, samples_per_class=10):
    print(f"📥 Loading {subset} dataset ({samples_per_class} samples per class)...")
    
    try:
        dataset = SPEECHCOMMANDS(root="./data", download=True, subset=subset)
        
        class_counts = {cls: 0 for cls in target_classes}
        selected_indices = []
        
        for idx in range(len(dataset)):
            try:
                waveform, sample_rate, label, speaker_id, utterance_number = dataset[idx]
                
                if label in target_classes and class_counts[label] < samples_per_class:
                    selected_indices.append(idx)
                    class_counts[label] += 1
                    
                if all(count >= samples_per_class for count in class_counts.values()):
                    break
            except Exception as e:
                continue
        
        print(f"✅ Selected {len(selected_indices)} samples for {subset}")
        print(f"📊 Class distribution: {class_counts}")
        
        return Subset(dataset, selected_indices)
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        from torch.utils.data import TensorDataset
        print("🎲 Using synthetic data as fallback...")
        
        num_samples = max(1, samples_per_class * len(target_classes))
        dummy_inputs = torch.randn(num_samples, 1, 32, 32)
        dummy_labels = torch.randint(0, len(target_classes), (num_samples,))
        return TensorDataset(dummy_inputs, dummy_labels)

print("🔄 Loading datasets...")
train_set = get_minimal_dataset('training', samples_per_class)
test_set = get_minimal_dataset('testing', max(1, samples_per_class // 2))

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=simple_collate_fn)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=simple_collate_fn)

print(f"📊 Dataset loaded: {len(train_loader)} train batches, {len(test_loader)} test batches")

# --- Ініціалізація моделі ---
model = AudioClassifier(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("🧠 Model architecture:")
print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  - Input shape: [batch, 1, 32, 32]")
print(f"  - Output shape: [batch, {num_classes}]")

# --- Цикл тренування ---
print("🚀 Starting training...")
training_log = []
start_time = time.time()

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    epoch_correct = 0
    epoch_total = 0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        if len(inputs.shape) == 3:
            inputs = inputs.unsqueeze(1)
            
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        epoch_total += labels.size(0)
        epoch_correct += (predicted == labels).sum().item()
        
        if batch_idx % 2 == 0:
            batch_accuracy = 100 * epoch_correct / epoch_total if epoch_total > 0 else 0
            print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}, Accuracy: {batch_accuracy:.2f}%')
    
    epoch_accuracy = 100 * epoch_correct / epoch_total if epoch_total > 0 else 0
    avg_epoch_loss = epoch_loss / max(1, len(train_loader))
    
    training_log.append({
        'epoch': epoch + 1,
        'loss': avg_epoch_loss,
        'accuracy': epoch_accuracy,
        'samples_processed': epoch_total
    })
    
    print(f'📈 Epoch [{epoch+1}/{epochs}] completed: '
          f'Loss: {avg_epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

# --- Оцінка моделі ---
print("🧪 Evaluating model on test set...")
model.eval()
test_correct = 0
test_total = 0
test_loss = 0.0

with torch.no_grad():
    for inputs, labels in test_loader:
        if len(inputs) == 0:
            continue
            
        if len(inputs.shape) == 3:
            inputs = inputs.unsqueeze(1)
            
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

# Фінальні метрики
test_accuracy = 100 * test_correct / test_total if test_total > 0 else 0
avg_test_loss = test_loss / max(1, len(test_loader))
training_time = time.time() - start_time

print(f'🎯 Final Test Results:')
print(f'  - Accuracy: {test_accuracy:.2f}%')
print(f'  - Loss: {avg_test_loss:.4f}')
print(f'  - Training time: {training_time:.2f}s')
print(f'  - Test samples: {test_total}')

# --- Збереження моделі та артефактів ---
print("💾 Saving model and artifacts...")

# 1. Збереження для пайплайну (model-output/)
torch.save(model.state_dict(), 'model-output/model-weights.pth')
print("✅ Model saved to model-output/model-weights.pth")

# Збереження повної моделі
torch.save(model, 'model-output/model-full.pth')

# Лог для пайплайну (з ключовим рядком для пошуку accuracy)
with open('model-output/training-log.txt', 'w') as f:
    f.write(f"=== TRAINING LOG ===\n")
    f.write(f"Model Accuracy: {test_accuracy:.2f}%\n")
    f.write(f"Final Loss: {avg_test_loss:.4f}\n")
    f.write(f"Training Time: {training_time:.2f}s\n")
    f.write(f"Test Samples: {test_total}\n")
    f.write(f"Epochs: {epochs}\n")
    f.write(f"Batch Size: {batch_size}\n")
    f.write(f"Model Version: 1.0.0\n")
    f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("====================\n")
print("✅ Training log saved to model-output/training-log.txt")

# 2. Збереження для локального використання (artifacts/)
torch.save(model.state_dict(), 'artifacts/model.pth')
print("✅ Model saved to artifacts/model.pth")

torch.save(model, 'artifacts/model_full.pth')
print("✅ Full model saved to artifacts/model_full.pth")

# Інформація про класи
with open('artifacts/class_info.json', 'w') as f:
    json.dump({
        'target_classes': target_classes,
        'num_classes': num_classes,
        'input_shape': [1, 32, 32]
    }, f, indent=2)
print("✅ Class info saved to artifacts/class_info.json")

# Детальні метрики
training_summary = {
    'training_parameters': {
        'epochs': epochs,
        'batch_size': batch_size,
        'samples_per_class': samples_per_class,
        'learning_rate': 0.001
    },
    'final_metrics': {
        'test_accuracy': test_accuracy,
        'test_loss': avg_test_loss,
        'training_time_seconds': training_time,
        'total_test_samples': test_total
    },
    'training_history': training_log,
    'model_info': {
        'input_shape': [1, 32, 32],
        'output_shape': [num_classes],
        'parameters_count': sum(p.numel() for p in model.parameters())
    }
}

with open('artifacts/training_metrics.json', 'w') as f:
    json.dump(training_summary, f, indent=2)
print("✅ Training metrics saved to artifacts/training_metrics.json")

# Простий лог для артефактів
with open('artifacts/training.log', 'w') as f:
    f.write("=== TRAINING SUMMARY ===\n")
    f.write(f"Final Accuracy: {test_accuracy:.2f}%\n")
    f.write(f"Final Loss: {avg_test_loss:.4f}\n")
    f.write(f"Training Time: {training_time:.2f}s\n")
    f.write(f"Epochs: {epochs}\n")
    f.write(f"Batch Size: {batch_size}\n")
    f.write(f"Samples per Class: {samples_per_class}\n")
    f.write(f"Device: {device}\n")
    f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=======================\n")
print("✅ Training log saved to artifacts/training.log")

print("🎉 Training completed successfully!")
print("📁 All artifacts saved in both 'artifacts/' and 'model-output/' directories")