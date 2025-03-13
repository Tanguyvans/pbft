#!/usr/bin/env python3
# evaluate_cifar10.py

import os
import sys
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Add the project root to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from going_modular.model import Net

def evaluate_model(model_path, device="cpu"):
    """Evaluate a model on the CIFAR-10 test dataset"""
    print(f"Evaluating model: {model_path}")
    
    # Use specified device
    device = torch.device(device)
    print(f"Using device: {device}")
    
    # CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Define normalization transform using CIFAR-10 values from data_setup.py
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
    # Load CIFAR-10 test dataset directly
    print("Loading CIFAR-10 test dataset...")
    testset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False,
        download=True, 
        transform=test_transform
    )
    
    test_loader = torch.utils.data.DataLoader(
        testset, 
        batch_size=32,
        shuffle=False
    )
    
    # Load model architecture and weights
    try:
        checkpoint = torch.load(model_path, map_location=device)
        architecture = checkpoint.get('architecture', 'mobilenet_v2')
        num_classes = checkpoint.get('num_classes', 10)
        
        print(f"Model architecture: {architecture}")
        print(f"Number of classes: {num_classes}")
        
        # Initialize model
        model = Net(num_classes=num_classes, arch=architecture).to(device)
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Successfully loaded model weights")
        else:
            print("Warning: No model_state_dict found in checkpoint")
            return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Set model to evaluation mode
    model.eval()
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Perform evaluation
    with torch.no_grad():
        # Initialize metrics
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        # Evaluate in batches
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Calculate metrics
            total_loss += loss.item() * inputs.size(0)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
        
        # Calculate overall metrics
        avg_loss = total_loss / len(testset)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        correct = (all_preds == all_labels).sum()
        accuracy = correct / len(all_labels)
        
        print(f"\nEvaluation Results:")
        print(f"Loss: {avg_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f} ({correct}/{len(all_labels)})")
        
        # Calculate per-class accuracy
        class_correct = [0] * num_classes
        class_total = [0] * num_classes
        
        for i in range(len(all_labels)):
            label = all_labels[i]
            class_total[label] += 1
            if all_preds[i] == label:
                class_correct[label] += 1
        
        print("\nPer-class Accuracy:")
        for i in range(num_classes):
            if class_total[i] > 0:
                class_acc = 100 * class_correct[i] / class_total[i]
                print(f"{class_names[i]}: {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})")
        
        # Generate confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Generate classification report
        report = classification_report(all_labels, all_preds, target_names=class_names)
        print("\nClassification Report:")
        print(report)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, 
                    yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        # Save confusion matrix
        output_dir = os.path.dirname(model_path)
        model_name = os.path.basename(model_path).split('.')[0]
        cm_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
        plt.savefig(cm_path)
        print(f"Confusion matrix saved to {cm_path}")
        
        # Return evaluation results
        results = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'per_class_accuracy': {i: (class_correct[i] / class_total[i] if class_total[i] > 0 else 0) for i in range(num_classes)},
            'confusion_matrix': cm,
            'classification_report': report
        }
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained model on CIFAR-10')
    parser.add_argument('--model', type=str, required=True, help='Path to the model file (.pt)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='Device to use for evaluation')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} not found")
        return
    
    # Evaluate model
    results = evaluate_model(args.model, device=args.device)
    
    if results:
        print("\nEvaluation completed successfully!")
        
        # Save results to file
        output_dir = os.path.dirname(args.model)
        model_name = os.path.basename(args.model).split('.')[0]
        results_path = os.path.join(output_dir, f"{model_name}_evaluation_results.txt")
        
        with open(results_path, 'w') as f:
            f.write(f"Model: {args.model}\n")
            f.write(f"Loss: {results['loss']:.4f}\n")
            f.write(f"Accuracy: {results['accuracy']:.4f}\n\n")
            f.write("Per-class Accuracy:\n")
            
            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            for i, acc in results['per_class_accuracy'].items():
                f.write(f"{class_names[i]}: {acc*100:.2f}%\n")
            
            f.write("\nClassification Report:\n")
            f.write(results['classification_report'])
        
        print(f"Evaluation results saved to {results_path}")
    else:
        print("Evaluation failed")

if __name__ == "__main__":
    main()