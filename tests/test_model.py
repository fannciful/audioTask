import torch
import json
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def validate_model(model_path):
    """Перевірка якості моделі на стабільних зразках"""
    print("🔍 Validating model...")
    
    # Завантаження моделі
    model = torch.load(model_path, map_location='cpu')
    
    # Тестові дані для валідації
    test_samples = generate_validation_samples()
    
    # Прогнозування
    predictions = []
    true_labels = []
    
    for sample, true_label in test_samples:
        pred = model(sample.unsqueeze(0))
        pred_label = torch.argmax(pred).item()
        predictions.append(pred_label)
        true_labels.append(true_label)
    
    # Метрики якості
    report = classification_report(true_labels, predictions, output_dict=True)
    cm = confusion_matrix(true_labels, predictions)
    
    # Збереження результатів
    results = {
        'accuracy': report['accuracy'],
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1_score': report['weighted avg']['f1-score'],
        'confusion_matrix': cm.tolist()
    }
    
    with open('artifacts/validation_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Звіт у HTML
    generate_html_report(report, cm)
    
    print(f"✅ Validation completed. Accuracy: {results['accuracy']:.3f}")

def generate_validation_samples():
    """Генерація стабільних тестових зразків"""
    # Тут мають бути реальні тестові дані
    return []  # Заглушка

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    args = parser.parse_args()
    
    validate_model(args.model_path)