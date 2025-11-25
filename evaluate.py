"""
Evaluation script for anomaly detection model.
Computes metrics and visualizes results.
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

from model import AnomalyDetectionModel
from data_loader import create_dataloaders, create_synthetic_data


def evaluate_model(model, test_loader, device, threshold_percentile=95):
    """Evaluate model on test set."""
    model.eval()
    
    all_reconstruction_errors = []
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for x, labels in tqdm(test_loader, desc="Evaluating"):
            x = x.to(device)
            
            # Forward pass
            reconstructed, encoded, _ = model(x, use_mask=False)
            
            # Compute reconstruction error
            reconstruction_error = model.compute_reconstruction_error(x, reconstructed)
            
            all_reconstruction_errors.extend(reconstruction_error.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Convert to numpy
    all_reconstruction_errors = np.array(all_reconstruction_errors)
    all_labels = np.array(all_labels)
    
    # Determine threshold
    threshold = np.percentile(all_reconstruction_errors, threshold_percentile)
    
    # Predictions
    all_predictions = (all_reconstruction_errors > threshold).astype(int)
    
    # Compute metrics
    metrics = compute_metrics(all_labels, all_predictions, all_reconstruction_errors)
    metrics['threshold'] = threshold
    
    return metrics, all_reconstruction_errors, all_labels, all_predictions


def compute_metrics(true_labels, predictions, scores):
    """Compute evaluation metrics."""
    # Basic metrics
    accuracy = np.mean(true_labels == predictions)
    precision = np.sum((predictions == 1) & (true_labels == 1)) / max(1, np.sum(predictions == 1))
    recall = np.sum((predictions == 1) & (true_labels == 1)) / max(1, np.sum(true_labels == 1))
    f1 = f1_score(true_labels, predictions)
    
    # AUC metrics
    try:
        auc_roc = roc_auc_score(true_labels, scores)
        auc_pr = average_precision_score(true_labels, scores)
    except:
        auc_roc = 0.0
        auc_pr = 0.0
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr
    }
    
    return metrics


def plot_results(reconstruction_errors, labels, predictions, save_path='results'):
    """Plot evaluation results."""
    os.makedirs(save_path, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # 1. Reconstruction error distribution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(reconstruction_errors[labels == 0], bins=50, alpha=0.7, label='Normal', color='blue')
    plt.hist(reconstruction_errors[labels == 1], bins=50, alpha=0.7, label='Anomaly', color='red')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Reconstruction Errors')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Time series of reconstruction errors
    plt.subplot(1, 2, 2)
    plt.plot(reconstruction_errors[:1000], label='Reconstruction Error', alpha=0.7)
    anomaly_indices = np.where(labels[:1000] == 1)[0]
    if len(anomaly_indices) > 0:
        plt.scatter(anomaly_indices, reconstruction_errors[anomaly_indices], 
                   color='red', s=20, label='True Anomalies', zorder=5)
    plt.xlabel('Sample Index')
    plt.ylabel('Reconstruction Error')
    plt.title('Reconstruction Error Over Time (First 1000 samples)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'reconstruction_errors.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. ROC Curve
    try:
        fpr, tpr, _ = roc_curve(labels, reconstruction_errors)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label='ROC Curve')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_path, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except:
        pass
    
    # 4. Precision-Recall Curve
    try:
        precision, recall, _ = precision_recall_curve(labels, reconstruction_errors)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2, label='PR Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_path, 'pr_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except:
        pass
    
    # 5. Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {save_path}/")


def print_metrics(metrics):
    """Print evaluation metrics."""
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"AUC-PR:    {metrics['auc_pr']:.4f}")
    print(f"Threshold: {metrics['threshold']:.4f}")
    print("="*50 + "\n")


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate anomaly detection model')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=dict, default=None,
                       help='Model configuration (if not in checkpoint)')
    parser.add_argument('--threshold_percentile', type=float, default=95,
                       help='Percentile for threshold determination')
    parser.add_argument('--use_synthetic', action='store_true',
                       help='Use synthetic data for evaluation')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    if args.config is None:
        config = checkpoint['config']
    else:
        config = args.config
    
    # Load data
    print("Loading data...")
    if args.use_synthetic:
        train_data, test_data, train_labels, test_labels = create_synthetic_data(
            n_samples=config.get('n_samples', 10000),
            n_features=config['n_features'],
            anomaly_ratio=0.1
        )
    else:
        from data_loader import load_smd_data
        try:
            train_data, test_data, test_labels = load_smd_data(
                data_dir=config.get('data_dir', 'data'),
                machine_id=config.get('machine_id', 'machine-1-1')
            )
            train_labels = None
        except FileNotFoundError:
            print("Dataset not found. Using synthetic data...")
            train_data, test_data, train_labels, test_labels = create_synthetic_data(
                n_samples=config.get('n_samples', 10000),
                n_features=config['n_features'],
                anomaly_ratio=0.1
            )
    
    # Create test dataloader
    _, test_loader, _ = create_dataloaders(
        train_data, test_data, train_labels, test_labels,
        window_size=config['window_size'],
        batch_size=32,
        stride=1,
        normalize=True
    )
    
    n_features = train_data.shape[1]
    
    # Create model
    model = AnomalyDetectionModel(
        n_features=n_features,
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        window_size=config['window_size'],
        latent_dim=config.get('latent_dim', 100),
        use_gan=config['use_gan']
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Evaluate
    print("Evaluating model...")
    metrics, reconstruction_errors, labels, predictions = evaluate_model(
        model, test_loader, device, args.threshold_percentile
    )
    
    # Print metrics
    print_metrics(metrics)
    
    # Plot results
    print("Generating plots...")
    plot_results(reconstruction_errors, labels, predictions)
    
    # Save metrics
    import json
    with open('results/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("Evaluation completed!")


if __name__ == '__main__':
    main()

