"""
Main script to run training and evaluation.
"""

import argparse
from train import train_model
from evaluate import evaluate_model, print_metrics, plot_results
import torch
from model import AnomalyDetectionModel
from data_loader import create_dataloaders, create_synthetic_data

# Check GPU availability at startup
if torch.cuda.is_available():
    print(f"\n{'='*60}")
    print(f"GPU DETECTED: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"{'='*60}\n")
else:
    print("\n⚠️  WARNING: No GPU detected. Training will be slow on CPU.\n")


def main():
    parser = argparse.ArgumentParser(description='Anomaly Detection Framework')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'both'],
                       help='Mode: train, eval, or both')
    parser.add_argument('--use_real_data', action='store_true',
                       help='Use real dataset (default: False, uses synthetic data)')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory containing dataset')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pt',
                       help='Path to model for evaluation')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    
    args = parser.parse_args()
    
    # Optimize batch size for GPU (RTX 4060 can handle larger batches)
    if torch.cuda.is_available():
        default_batch_size = 64  # Increased for GPU
        print(f"Using GPU-optimized batch size: {default_batch_size}")
    else:
        default_batch_size = 32
    
    # Default configuration
    
    config = {
        'n_features': 10,
        'window_size': 100,
        'd_model': 128,
        'nhead': 8,
        'num_layers': 3,
        'dim_feedforward': 512,
        'dropout': 0.1,
        'batch_size': args.batch_size if args.batch_size != 32 else default_batch_size,
        'num_epochs': args.epochs,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'contrastive_weight': 0.5,
        'use_gan': True,
        'gan_g_step': 2,
        'latent_dim': 100,
        'use_synthetic': not args.use_real_data,
        'data_dir': args.data_dir
    }
    
    if args.mode in ['train', 'both']:
        print("="*60)
        print("TRAINING MODE")
        print("="*60)
        trainer, test_loader, scaler = train_model(config)
    
    if args.mode in ['eval', 'both']:
        print("\n" + "="*60)
        print("EVALUATION MODE")
        print("="*60)
        
        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(args.model_path, map_location=device)
        config = checkpoint['config']
        
        # Load data
        if config.get('use_synthetic', True):
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
        
        _, test_loader, _ = create_dataloaders(
            train_data, test_data, train_labels, test_labels,
            window_size=config['window_size'],
            batch_size=32,
            stride=1,
            normalize=True
        )
        
        n_features = train_data.shape[1]
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
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        # Evaluate
        from evaluate import evaluate_model
        metrics, reconstruction_errors, labels, predictions = evaluate_model(
            model, test_loader, device, threshold_percentile=95
        )
        
        print_metrics(metrics)
        plot_results(reconstruction_errors, labels, predictions)
        
        import json
        import os
        os.makedirs('results', exist_ok=True)
        with open('results/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print("Evaluation completed! Results saved to results/ directory.")


if __name__ == '__main__':
    main()

