#!/usr/bin/env python3
"""
Standalone test script for the ML Training API
This script uploads a dataset file to the API (doesn't require dataset to be on server)

Two endpoints:
- /train: Trains with ALL features in the dataset
- /retrain: Trains with only SELECTED features

Usage: 
    # Train with ALL features
    python test_api.py dataset.csv
    python test_api.py dataset.csv --time-col "Event Time" --event-col "Censored"
    
    # Re-train with SELECTED features
    python test_api.py dataset.csv --retrain --features age,gender,treatment
    python test_api.py dataset.csv --retrain --features age,gender --time-col "Event Time"
"""

import requests
import json
import sys
import os
import argparse
from typing import List, Optional

from env_loader import load_env_file


load_env_file()
DEFAULT_API_URL = os.getenv('API_URL', 'http://localhost:5000')


def test_health(api_url: str = DEFAULT_API_URL) -> bool:
    """Test if API is running"""
    try:
        response = requests.get(f'{api_url}/health', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ API is healthy")
            print(f"  Device: {data.get('device', 'unknown')}")
            print(f"  Timestamp: {data.get('timestamp', 'unknown')}")
            return True
        else:
            print(f"✗ API health check failed (status: {response.status_code})")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to API. Is it running?")
        print(f"  Expected at: {api_url}")
        print("  Start it with: python app.py  or  ./run.sh")
        return False
    except Exception as e:
        print(f"✗ Health check error: {e}")
        return False


def test_training(
    dataset_path: str,
    features: Optional[List[str]] = None,
    api_url: str = DEFAULT_API_URL,
    neurons: List[int] = [64, 64],
    dropout: float = 0.2,
    seed: int = 42,
) -> Optional[str]:
    """
    Test model training by uploading a dataset file
    
    Args:
        dataset_path: Path to CSV file on YOUR machine (will be uploaded)
        features: List of feature column names to use
        api_url: API base URL
        neurons: Hidden layer sizes
        dropout: Dropout rate
        seed: Random seed
        
    Returns:
        Model ID if successful, None otherwise
    """
    if not os.path.exists(dataset_path):
        print(f"✗ Dataset not found: {dataset_path}")
        return None
    
    file_size = os.path.getsize(dataset_path)
    print(f"\n📊 Training with dataset: {dataset_path}")
    print(f"   File size: {file_size / 1024:.2f} KB")
    
    # Prepare the request
    # Open file in binary mode and upload it
    with open(dataset_path, 'rb') as f:
        files = {
            'dataset': (os.path.basename(dataset_path), f, 'text/csv')
        }

        data = {
            'parameters': json.dumps({
                'neurons': neurons,
                'dropout': dropout,
                'seed': seed,
                'n_quantiles': 9,
                'lr': 1e-3,
                'batch_size': 256,
                'n_epochs': 1000,
                'weight_decay': 1e-4,
                'n_exp': 10,
                
            }),
            'return_cv_predictions': 'true'
        }


        
        # Add features if provided
        if features:
            data['selected_features'] = json.dumps(features)
            print(f"   Using features: {', '.join(features)}")
        else:
            data['selected_features'] = json.dumps('all')
            print(f"   Using all features in dataset")
        
        print(f"\n⏳ Uploading and training... (this may take a while)")
        
        try:
            response = requests.post(
                f'{api_url}/train',
                files=files,
                data=data,
                timeout=600  # 10 minute timeout for training
            )
        except requests.exceptions.Timeout:
            print("✗ Request timed out (training took too long)")
            return None
        except Exception as e:
            print(f"✗ Request failed: {e}")
            return None
    
    # Parse response
    if response.status_code == 200:
        result = response.json()
        
        if result.get('status') == 'success':
            print("\n✓ Training successful!")
            print(f"\n{'='*60}")
            print(f"Model ID: {result['model_id']}")
            print(f"Model Config: {result['model_config']}") 
            print(f"CV Predictions: {result['cv_predictions']}")
            print(f"Train Duration: {result['train_duration']:.3f}")

            print(f"{'='*60}")
            
            print("\n📊 Performance Metrics:")
            metrics = result.get('metrics', {})
            for metric, value in metrics.items():
                print(f"Metric: {metric}, Value: {value}")
            
            return result['model_id']
        else:
            print("✗ Training failed!")
            print(f"   Error: {result.get('error', 'Unknown error')}")
            return None
    else:
        print(f"✗ Training failed! (HTTP {response.status_code})")
        try:
            error_data = response.json()
            print(f"   Error: {error_data.get('error', 'Unknown error')}")
            if 'error_type' in error_data:
                print(f"   Type: {error_data['error_type']}")
        except:
            print(f"   Response: {response.text[:200]}")
        return None

def test_retrain(
    model_id: str,
    selected_features=None,
    parameters: dict = None,
    api_url: str = DEFAULT_API_URL,
) -> Optional[str]:
    """Test retraining an existing model"""

    print(f"\n♻️ Retraining model: {model_id}")
    print(f"   Selected features: {selected_features}")
    print(f"   Parameter overrides: {parameters}")

    payload = {
        "model_id": model_id,
        "return_cv_predictions": True
    }


    if selected_features is not None:
        payload["selected_features"] = selected_features

    if parameters:
        payload["parameters"] = parameters

    try:
        response = requests.post(
            f"{api_url}/retrain",
            json=payload,
            timeout=600  # allow for long training
        )
    except Exception as e:
        print(f"✗ Retrain request failed: {e}")
        return None

    if response.status_code == 200:
        result = response.json()
        print("\n✓ Retraining successful!")
        print(f"New Model ID: {result.get('model_id')}")
        print(f"Parent Model ID: {result.get('retrained_from')}")

        print("\n📊 Performance Metrics:")
        metrics = result.get('metrics', {})
        for metric, val in metrics.items():
            print(f"  {metric}: {val}")

        print("\n🔁 Retrain Change Summary:")
        summary = result.get("retrain_summary", {})
        print(json.dumps(summary, indent=2))

        return result.get("model_id")

    else:
        print(f"✗ Retraining failed (HTTP {response.status_code})")
        print(response.text)
        return None



def test_list_models(api_url: str = DEFAULT_API_URL) -> None:
    """List all trained models"""
    print(f"\n📋 Listing trained models...")
    
    try:
        response = requests.get(f'{api_url}/models', timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            count = result.get('count', 0)
            
            if count == 0:
                print("   No models found")
                return
            
            print(f"   Found {count} trained model(s):")
            for model in result.get('models', []):
                print(f"\n   • {model['model_id']}")
                print(f"     Type: {model.get('model_type', 'Unknown')}")
                print(f"     Trained at: {model['artifacts'].get('trained_at.txt', 'Unknown')}")
                print(f"     Path: {model['artifacts'].get('model_weights.pth', 'Unknown')}")
        else:
            print(f"   Failed to list models (HTTP {response.status_code})")
    except Exception as e:
        print(f"   Error: {e}")


def test_predict(
    model_id: str,
    features: dict,
    api_url: str = DEFAULT_API_URL
) -> None:
    """Test prediction with a trained model"""
    print(f"\n🔮 Testing prediction with model: {model_id}")
    print(f"   Input features: {features}")
    
    try:
        response = requests.post(
            f'{api_url}/predict',
            json={
                'model_id': model_id,
                'features': features
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            predictions = result.get('predictions', {})
            
            print("\n✓ Prediction successful!")
            print(f"   Median survival time: {predictions.get('median_survival_time', 'N/A'):.2f}")
            
            levels = predictions.get('quantile_levels', [])
            preds = predictions.get('quantile_predictions', [])
            
            if levels and preds:
                print("\n   Survival quantiles:")
                for level, pred in zip(levels[:5], preds[:5]):  # Show first 5
                    print(f"     {level*100:>5.1f}%: {pred:>8.2f}")
                if len(levels) > 5:
                    print(f"     ... ({len(levels)-5} more)")
        else:
            print(f"✗ Prediction failed (HTTP {response.status_code})")
            print(f"   {response.json().get('error', 'Unknown error')}")
    except Exception as e:
        print(f"✗ Prediction error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Test the ML Training API by uploading a dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_api.py dataset.csv
  python test_api.py data/breast_cancer.csv --features age,gender,treatment
  python test_api.py data.csv --time-col "Event Time" --event-col "Censored"
  python test_api.py data.csv --api-url http://192.168.1.100:5000
        """
    )
    
    parser.add_argument('--features', help='Comma-separated list of feature names')
    parser.add_argument('--time-col', default='time', help='Name of time column (default: time)')
    parser.add_argument('--event-col', default='event', help='Name of event column (default: event)')
    parser.add_argument('--api-url', default=DEFAULT_API_URL, help='API base URL')
    parser.add_argument('--neurons', default='64,64', help='Hidden layer sizes (default: 64,64)')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--test-predict', action='store_true', help='Also test prediction after training')
    
    args = parser.parse_args()
    
    # Parse arguments
    features = args.features.split(',') if args.features else None
    neurons = [int(x.strip()) for x in args.neurons.split(',')]
    
    # Print header
    print("=" * 60)
    print("ML Training API - Test Suite")
    print("=" * 60)
    
    # Test health
    if not test_health(args.api_url):
        sys.exit(1)
    
    # RELATIVE PATH ACCORDING TO USER
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_PATH = os.path.join(BASE_DIR, "data", "AML.csv")

    
    # Test training
    model_id = test_training(
        dataset_path=DATASET_PATH,
        features=features,
        api_url=args.api_url,
        neurons=neurons,
        dropout=args.dropout,
        seed=args.seed,
    )
    
    if model_id:
        # List models
        test_list_models(args.api_url)
        
        # Test prediction if requested
        if args.test_predict and features:
            # Create sample features dict (all zeros as placeholder)
            sample_features = {feat: 0.0 for feat in features}
            test_predict(model_id, sample_features, args.api_url)
    
        # Test retraining after training a model
        print("\n🔄 Testing retraining...")
        retrained_model_id = test_retrain(
            model_id=model_id,
            selected_features=features,  # reuse CLI feature selection
            parameters={"dropout": args.dropout + 0.05},  # example change
            api_url=args.api_url,
        )

        if retrained_model_id:
            print(f"\n✅ Retrained successfully: {retrained_model_id}")
        else:
            print("\n⚠ Retraining failed")

    
    # Print footer
    print("\n" + "=" * 60)
    if model_id:
        print("✓ All tests completed successfully")
    else:
        print("✗ Tests failed")
    print("=" * 60)


if __name__ == '__main__':
    main()
