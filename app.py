import os
import json
import joblib
import dill
import uuid
from tqdm import trange
import time
import shutil
from datetime import datetime
from flask import Flask, request, jsonify, send_file, send_from_directory, url_for
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector, ColumnTransformer
from scipy.stats import chisquare
import statistics
from collections import defaultdict
from flask_cors import CORS


from env_loader import load_env_file


# Import models and utilities
from model import MTLR
from icp import ConformalSurvDist, CSDiPOT
from icp.scorer import QuantileRegressionNC, SurvivalPredictionNC
from utils import set_seed, save_params
from utils.util_survival import survival_data_split, make_time_bins, xcal_from_hist
from SurvivalEVAL import QuantileRegEvaluator
from CondCalEvaluation import wsc_xcal


load_env_file()


def _safe_int(value, fallback):
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


DEFAULT_CORS = "http://localhost:5174,http://localhost:5173"
API_HOST = os.getenv("API_HOST", "localhost")
API_PORT = _safe_int(os.getenv("API_PORT"), 5000)
API_URL = os.getenv("API_URL", f"http://{API_HOST}:{API_PORT}")
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", DEFAULT_CORS).split(",")
    if origin.strip()
]


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['MODEL_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trained_models')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['API_HOST'] = API_HOST
app.config['API_PORT'] = API_PORT
app.config['API_URL'] = API_URL
CORS(app, origins=CORS_ORIGINS, supports_credentials=True)


# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)


ci = []
mae_hinge = []
mae_po = []
ibs = []
km_cal = []
xcal_stats = []
wsc_xcal_stats = []
dcal_chisquare = []
dcal_p_value_stat = []
train_times = []
infer_times = []
dcal_hists = []
n_features = 0

class Args:
    """Configuration object for model training"""
    def __init__(self, config):
        # Model parameters
        self.model = config.get('model', 'MTLR')
        self.neurons = config.get('neurons', [64, 64])
        self.norm = config.get('norm', True)
        self.activation = config.get('activation', 'ReLU')
        self.dropout = config.get('dropout', 0.3)
        self.n_quantiles = config.get('n_quantiles', 10)
        self.interpolate = config.get('interpolate', 'Pchip')
        self.decensor_method = config.get('decensor_method', 'sampling') # <-- Difference between sampling and margin? ---------------- Changed from margin to sampling ------
        self.post_process = config.get('post_process', 'CSD')
        self.selected_features = config.get('selected_features', None)
        
        # Training parameters
        self.lr = config.get('lr', 1e-3)
        self.batch_size = config.get('batch_size', 256)
        self.n_epochs = config.get('n_epochs', 1000)
        self.weight_decay = config.get('weight_decay', 1e-4)
        self.early_stop = config.get('early_stop', True)
        self.mono_method = config.get('mono_method', 'bootstrap')  # <-- Changed to bootstrap and Pchip worked
        self.use_train = config.get('use_train', True)
        self.n_sample = config.get('n_sample', 1000)
        self.seed = config.get('seed', 0)
        self.n_exp = config.get('n_exp', 10)
        
        # Device
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Verbose and other flags
        self.verbose = config.get('verbose', True)


def prepare_data(dataset_path, selected_features=None, args=None):
    """
    Load and preprocess the dataset for survival analysis.

    Expected CSV format:
    - First column: Time/Label (survival time or duration)
    - Second column: Censored (0 or u=event occurred/uncensored, 1 or c=censored)
    - Remaining columns: Features for prediction
    """

    # Load dataset
    data = pd.read_csv(dataset_path)

    # Standardize column names for survival
    columns = data.columns.tolist()
    if len(columns) < 3:
        raise ValueError(f"Dataset must have at least 3 columns. Found {len(columns)} columns.")

    data = data.rename(columns={columns[0]: 'time', columns[1]: 'censored'})
    data['event'] = ((data['censored'] == 0) | (data['censored'] == 'u')).astype(int)

    # Remove invalid zero-time rows (Time = 0 and Event = 1/occurred). Not plausible
    invalid_mask = (data["time"] <= 0)
    data = data[~invalid_mask]

    data = data.drop(columns='censored')

    # Filter selected features
    feature_columns = [col for col in data.columns if col not in ['time', 'event']]
    
    # Define survival columns
    SURVIVAL_COLS = {'time', 'event', 'censored'}
    if selected_features:
        # Removes survival columns silently
        selected_features = [f for f in selected_features if f not in SURVIVAL_COLS]

        # Check for missing features now
        missing_features = [f for f in selected_features if f not in feature_columns]   
        if missing_features:
            raise ValueError(f"Features not found: {missing_features}")
        feature_columns = selected_features
    data = data[feature_columns + ['time', 'event']]

    # Identify columns
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

    numeric_cols = [c for c in numeric_cols if c not in ['time', 'event']] 

    ordinal_cols = [c for c in numeric_cols if data[c].dtype == 'int64']      # Discrete/ordinal
    continuous_cols = [c for c in numeric_cols if data[c].dtype == 'float64'] # Continuous

    cat_cols = [c for c in data.columns if c not in numeric_cols + ['time', 'event']]
    binary_cols = [c for c in cat_cols if data[c].nunique() == 2]
    nominal_cols = [c for c in cat_cols if data[c].nunique() > 2]

    # Pipelines
    continuous_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])

    ordinal_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent"))
    ])

    binary_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(drop="if_binary", sparse_output=False))
    ])

    nominal_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # ColumnTransformer for features only
    encoder = ColumnTransformer(
        transformers=[
            ("cont", continuous_pipeline, continuous_cols),
            ("ord", ordinal_pipeline, ordinal_cols),
            ("bin", binary_pipeline, binary_cols),
            ("nom", nominal_pipeline, nominal_cols)
        ],
        remainder='passthrough', # <--- Appends time and event columns after transformation of the rest of the columns
        verbose_feature_names_out=False
    )
    encoder.set_output(transform='pandas')

    # Train / val / test split
    if args.early_stop:
        pct_train = 0.8
        pct_val = 0.1
        pct_test = 0.1
    else:
        pct_train = 0.9
        pct_val = 0.0
        pct_test = 0.1

    return encoder, pct_train, pct_val, pct_test, data

def print_performance(
        path: str = None,
        **kwargs
) -> dict:  # Changed return type
    """
    Print performance using mean and std. And also save to file.
    Returns dictionary with calculated metrics.
    """
    prf = ""
    metrics_dict = {}  # Store calculated metrics
    
    for k, v in kwargs.items():
        if len(v) == 0 or None in v:
            continue

        if isinstance(v, list):
            mean = statistics.mean(v)
            std = statistics.stdev(v)   # sample standard deviation (n-1)
            prf += f"{k}: {mean:.3f} +/- {std:.3f}\n"
            
            # Store in dict for JSON response
            metrics_dict[k] = {
                'mean': round(mean, 3),
                'std': round(std, 3),
            }
        else:
            prf += f"{k}: {v:.3f}\n"
            metrics_dict[k] = round(v, 3)
    
    print(prf)

    if path is not None:
        prf_dict = {k: v for k, v in kwargs.items()}
        with open(f"{path}/performance.pkl", 'wb') as f:
            pickle.dump(prf_dict, f)

        with open(f"{path}/performance.txt", 'w') as f:
            f.write(prf)
    
    return metrics_dict  # Return the dictionary




def train_mtlr_model(dataset_path, selected_features, args, i, return_predictions=False):
    """Train MTLR model with conformal prediction"""

    # ADD THIS AT THE START OF THE FUNCTION
    global ci, mae_hinge, mae_po, ibs, km_cal, xcal_stats, wsc_xcal_stats
    global dcal_chisquare, dcal_p_value_stat, train_times, infer_times, dcal_hists, n_features

    # ✅ START TIMING
    # t0 = time.time()

    device = torch.device(args.device)

    # One unified seed per trial
    seed = args.seed + i

    # Set ALL RNG controls
    set_seed(seed, device)  # this should set torch + cuda + python.random

    # t1 = time.time()
    # print(f"[Exp {i+1}] Setup & seeding: {t1-t0:.3f}s")
    
    # Prepare data AFTER seeding
    enc_df, pct_train, pct_val, pct_test, data = prepare_data(
        dataset_path, selected_features, args
    )

    # t2 = time.time()
    # print(f"[Exp {i+1}] prepare_data(): {t2-t1:.3f}s")

    # path = save_params(config)

    # Train/val/test split seeded identically
    data_train, data_val, data_test = survival_data_split(
        data, 
        stratify_colname='both',
        frac_train=pct_train, 
        frac_val=pct_val, 
        frac_test=pct_test,
        random_state=seed
    )

    # t3 = time.time()
    # print(f"[Exp {i+1}] Data split: {t3-t2:.3f}s")

    # standardize the data
    data_train = enc_df.fit_transform(data_train).astype('float32')
    data_val = enc_df.transform(data_val).astype('float32') if not data_val.empty else data_val
    data_test = enc_df.transform(data_test).astype('float32')

    # get the labels for evaluation
    t_train, e_train = data_train["time"].values, data_train["event"].values
    t_val, e_val = data_val["time"].values, data_val["event"].values if not data_val.empty else None
    x_test = data_test.drop(['time', 'event'], axis=1).values
    t_test, e_test = data_test["time"].values, data_test["event"].values
    t_train_val = np.concatenate((t_train, t_val)) if not data_val.empty else t_train
    e_train_val = np.concatenate((e_train, e_val)) if not data_val.empty else e_train


    discrete_bins = make_time_bins(t_train, event=e_train)


    n_features = data_train.shape[1] - 2 # Exlucde time and event

    # Build MTLR model
    model = MTLR(
        n_features=n_features,
        time_bins=discrete_bins,
        hidden_size=args.neurons,
        norm=args.norm,
        activation=args.activation,
        dropout=args.dropout
    )

    # Setup conformal prediction
    if args.post_process == "CSD":
        nc_model = QuantileRegressionNC(model, args)
        icp = ConformalSurvDist(
            nc_model, condition=None,
            decensor_method=args.decensor_method,
            n_quantiles=args.n_quantiles
        )
    elif args.post_process == "CSD-iPOT":
        nc_model = SurvivalPredictionNC(model, args)
        icp = CSDiPOT(
            nc_model,
            decensor_method=args.decensor_method,
            n_percentile=args.n_quantiles
        )
    

    # Fit the ICP using the proper training set, and using valset for early stopping
    start_time = datetime.now()
    icp.fit(data_train, data_val)

    # Calibrate the ICP using the calibration set
    if args.use_train:
        data_val = pd.concat([data_train, data_val], ignore_index=True)
    icp.calibrate(data_val)

    mid_time = datetime.now()
    # Produce predictions for the test set
    quan_levels, quan_preds = icp.predict(x_test)

    end_time = datetime.now()

    # Produce predictions for test set
    quan_levels, quan_preds = icp.predict(x_test)
    
    # NEW: Calculate additional statistics for each prediction
    individual_predictions = None
    if return_predictions:
        # For each sample, calculate additional metrics
        median_preds = []
        mean_preds = []
        prob_at_actual_time = []
        
        median_preds, mean_preds, prob_at_actual_time = summarize_prediction_stats(
            quan_levels, quan_preds, t_test
        )
        
        individual_predictions = {
            'fold': i,
            'test_indices': data_test.index.tolist(),
            'actual_times': t_test.tolist(),
            'actual_events': e_test.tolist(),
            'quantile_levels': quan_levels.tolist(),
            'quantile_predictions': quan_preds.tolist(),
            'median_predictions': median_preds,
            'mean_predictions': mean_preds,
            'prob_at_actual_time': prob_at_actual_time,
            'features': x_test.tolist()
        }
    
    train_time = (mid_time - start_time).total_seconds()
    infer_time = (end_time - mid_time).total_seconds()

    evaler = QuantileRegEvaluator(
        quan_preds, quan_levels, t_test, e_test,
        t_train_val, e_train_val,
        predict_time_method="Median", interpolation=args.interpolate
    )

    c_index = float(evaler.concordance(ties="All")[0])
    ibs_score = float(evaler.integrated_brier_score(num_points=10))
    hinge_abs = float(evaler.mae(method='Hinge', verbose=False, weighted=True))
    po_abs = float(evaler.mae(method='Pseudo_obs', verbose=False, weighted=True))
    km_cal_score = float(evaler.km_calibration())
    _ , dcal_hist = evaler.d_calibration()
    xcal_score = float(xcal_from_hist(dcal_hist))
    pred_probs = evaler.predict_probability_from_curve(evaler.event_times)
    dcal_chisquare_stat, dcal_p_value = chisquare(dcal_hist)
    if data.shape[0] >= 1000:
        wsc_xcal_score = float(wsc_xcal(x_test, e_test, pred_probs, random_state=seed))
    else:
        wsc_xcal_score = 0  # not enough data to compute the WSC


    ci.append(c_index)
    ibs.append(ibs_score)
    mae_hinge.append(hinge_abs)
    mae_po.append(po_abs)
    km_cal.append(km_cal_score)
    xcal_stats.append(xcal_score)
    dcal_chisquare.append(float(dcal_chisquare_stat))
    dcal_p_value_stat.append(float(dcal_p_value))
    wsc_xcal_stats.append(wsc_xcal_score)
    dcal_hists.append(torch.tensor(dcal_hist))
    train_times.append(train_time)
    infer_times.append(infer_time)



    
    return icp, enc_df, individual_predictions


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    })


@app.route("/models/<model_id>/<filename>")
def serve_model_file(model_id, filename):
    folder = os.path.join(app.config['MODEL_FOLDER'], model_id)
    if not os.path.exists(os.path.join(folder, filename)):
        return {"error": "File not found"}, 404
    return send_from_directory(folder, filename)



@app.route("/models/<model_id>/cv_predictions", methods=['GET'])
def get_cv_predictions_file(model_id):
    """Serve the per-fold aggregated cv_predictions.json for a model."""
    folder = os.path.join(app.config['MODEL_FOLDER'], model_id)
    target = os.path.join(folder, "cv_predictions.json")
    if not os.path.exists(target):
        return jsonify({"error": "cv_predictions.json not found"}), 404
    return send_from_directory(folder, "cv_predictions.json")


@app.route("/models/<model_id>/full_predictions", methods=['GET'])
def get_full_predictions_file(model_id):
    """Serve the full_predictions.json (all identifiers) for a model."""
    folder = os.path.join(app.config['MODEL_FOLDER'], model_id)
    target = os.path.join(folder, "full_predictions.json")
    if not os.path.exists(target):
        return jsonify({"error": "full_predictions.json not found"}), 404
    return send_from_directory(folder, "full_predictions.json")



@app.route('/train', methods=['POST'])
def train_model():
    """
    Train MTLR model with feature selection
    
    Expected CSV format:
    - First column: Time/Label (survival time or duration)
    - Second column: Censored (0=event occurred/uncensored, 1=censored)
    - Remaining columns: Features for prediction
    
    Feature Selection Logic:
    - selected_features='all' → Use ALL features from dataset
    - selected_features=['feat1', 'feat2'] → Use ONLY these specific features
    - selected_features=None (or not provided) → No features selected (will use all by default)
    
    Example JSON request:
    {
        "dataset_path": "/path/to/data.csv",
        "selected_features": "all",  // or ["Height", "Weight"] or null
        "parameters": {"neurons": [64, 64], "dropout": 0.1}
        "return_cv_predictions": true  # NEW: Return individual predictions

    }
    """
    # Reset metrics before training
    global ci, mae_hinge, mae_po, ibs, km_cal, xcal_stats, wsc_xcal_stats
    global dcal_chisquare, dcal_p_value_stat, train_times, infer_times, dcal_hists, n_features

    ci.clear()
    mae_hinge.clear()
    mae_po.clear()
    ibs.clear()
    km_cal.clear()
    xcal_stats.clear()
    wsc_xcal_stats.clear()
    dcal_chisquare.clear()
    dcal_p_value_stat.clear()
    train_times.clear()
    infer_times.clear()
    dcal_hists.clear()
    n_features = 0

    model_dir = None  


    try:
        # Handle dataset - either as file upload or JSON path
        dataset_path = None
        
        # Dataset MUST be a File upload (multipart/form-data)
        if 'dataset' in request.files:
            file = request.files['dataset']
            if file.filename == '':
                return jsonify({'error': 'No dataset file selected'}), 400
            
            if not file.filename.endswith('.csv'):
                return jsonify({'error': 'Dataset file must be a CSV'}), 400
            
            # Save uploaded file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_filename = f"{timestamp}_{filename}"
            dataset_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset_filename)
            file.save(dataset_path)
        
        else:
            return jsonify({'error': 'No dataset provided. Please upload file'}), 400
        
        # Parse user-selected configuration
        parameters = json.loads(request.form.get('parameters', '{}'))

        # Load dataset
        data = pd.read_csv(dataset_path)
        # selected features
        selected_features = data.columns[2:].tolist()

        # Train on all features by default
        selected_features_for_training = None

        config = {'selected_features': selected_features} 
        
        # Merge parameters into config
        config.update(parameters)

        # Pass configurations + features into Args
        args = Args(config)

        # safety: ensure n_exp is an int >= 1
        n_exp = max(1, int(getattr(args, 'n_exp', 1)))


        # Create model ID
        now = datetime.now()
        model_timestamp = now.strftime("%Y%m%d_%H%M%S")
        model_timestamp_date = now.isoformat()  # e.g., "2025-11-09T15:45:30+00:00"
        suffix = uuid.uuid4().hex[:6]  # 6-character random ID
        model_id = f"mtlr_{model_timestamp}_{suffix}"

        # Create model dir
        model_dir = os.path.join(app.config['MODEL_FOLDER'], model_id)
        os.makedirs(model_dir, exist_ok=True)
    
        # NEW: Collect predictions across all folds
        all_fold_predictions = []

        # Training Loop 
        train_start = time.time()
        for i in trange(n_exp, disable=not args.verbose, desc='Experiment'):
            icp, encoder, indiv_preds = train_mtlr_model(
                dataset_path, selected_features_for_training, args, i, return_predictions=True)

            # Collect predictions
            if indiv_preds is not None:
                all_fold_predictions.append(indiv_preds)

        train_end = time.time()

        train_duration = train_end - train_start

        # -------------------------------
        # Aggregate individual predictions
        # -------------------------------
        aggregated_predictions = aggregate_cv_predictions(all_fold_predictions)
        full_dataset_predictions = generate_full_dataset_predictions(
            icp, encoder, dataset_path, selected_features_for_training, args
        )

        # ----------------------------
        # Save artifacts
        # ----------------------------
        
        # Save model config (only model architecture params, NOT selected_features)
        model_config = {
            "model_type": "MTLR",
            "n_features": icp.nc_function.model.in_features,
            "time_bins": icp.nc_function.model.time_bins.tolist(),
            "neurons": args.neurons,
            "dropout": args.dropout,
            "activation": args.activation,
            "norm": args.norm,
        }
        config_path = os.path.join(model_dir, "model_config.json")
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        
        # Save training metadata separately (includes selected_features)
        training_metadata = {
            "selected_features": selected_features,  # Store as 'all' for training
            "dataset_path": dataset_path,
            "n_experiments": n_exp,
            "timestamp": model_timestamp_date
        }
        metadata_path = os.path.join(model_dir, "training_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(training_metadata, f, indent=2)


        # Save encoder pipeline
        encoder_path = os.path.join(model_dir, "encoder.joblib")
        joblib.dump(encoder, encoder_path)

        # Save ICP state
        icp_state_path = os.path.join(model_dir, "icp_state.dill")
        with open(icp_state_path, "wb") as f:
            dill.dump(icp, f)
        
        # Save model dimensions to .mtlr file
        dimensions_path = os.path.join(model_dir, f"mtlr_model_{model_id}.mtlr")
        save_dimensions(icp.nc_function.model, dimensions_path)
    
        if aggregated_predictions:
            cv_predictions_path = os.path.join(model_dir, "cv_predictions.json")
            with open(cv_predictions_path, 'w') as f:
                json.dump(aggregated_predictions, f, indent=2)

            cv_summary_path = os.path.join(model_dir, "cv_predictions_summary.csv")
            create_cv_summary_csv(aggregated_predictions, cv_summary_path)
        else:
            cv_predictions_path = None
            cv_summary_path = None

        if full_dataset_predictions:
            full_predictions_path = os.path.join(model_dir, "full_predictions.json")
            with open(full_predictions_path, 'w') as f:
                json.dump(full_dataset_predictions, f, indent=2)

            full_summary_path = os.path.join(model_dir, "full_predictions_summary.csv")
            create_cv_summary_csv(full_dataset_predictions, full_summary_path)

            survival_curve_path = os.path.join(model_dir, "survival_curves.json")
            save_survival_curve_mapping(full_dataset_predictions, survival_curve_path)
        else:
            full_predictions_path = None
            full_summary_path = None
            survival_curve_path = None

        # Save metrics
        metrics = print_performance(
            Cindex=ci,
            IBS=ibs,
            MAE_Hinge=mae_hinge,
            MAE_PO=mae_po,
            KM_cal=km_cal,
            xCal_stats=xcal_stats,
            wsc_xCal_stats=wsc_xcal_stats,
            dcal_p=dcal_p_value_stat,
            dcal_Chi=dcal_chisquare,
            train_times=train_times,
            infer_times=infer_times
        )

        metrics['n_features'] = n_features
        metrics['d_cal_hist'] = torch.stack(dcal_hists).mean(0).tolist()
        metrics['train_start_time'] = datetime.fromtimestamp(train_start).isoformat()
        metrics['train_duration'] = train_duration

        # Make base_url
        base_url = request.host_url.rstrip("/")  # e.g., http://localhost:5000
       

        cv_info = None
        if aggregated_predictions:
            cv_info = {
                "summary_csv": f"{base_url}/models/{model_id}/cv_predictions_summary.csv" if cv_summary_path else None,
                "full_predictions": f"{base_url}/models/{model_id}/cv_predictions.json" if cv_predictions_path else None,
                "n_folds": len(all_fold_predictions),
                "total_predictions": len(aggregated_predictions['actual_times'])
            }

        full_info = None
        if full_dataset_predictions:
            full_info = {
                "summary_csv": f"{base_url}/models/{model_id}/full_predictions_summary.csv",
                "full_predictions": f"{base_url}/models/{model_id}/full_predictions.json",
                "survival_curves": f"{base_url}/models/{model_id}/survival_curves.json",
                "total_identifiers": len(full_dataset_predictions['test_indices'])
            }

        response_data = {
            "status": "success",
            "model_id": model_id,
            "metrics": metrics,
            "selected_features": selected_features,
            "model_config": f"{base_url}/models/{model_id}/model_config.json",
            "mtlr_model": f"{base_url}/models/{model_id}/mtlr_model_{model_id}.mtlr",
            "cv_predictions": cv_info,
            "full_dataset_predictions": full_info,
            "trained_at": model_timestamp_date,
            "train_duration": train_duration,
            "timestamp": datetime.now().isoformat()
        }
        


        return jsonify(response_data), 200


    except Exception as e:
        # ---------------------------------
        # Cleanup on failure
        # ---------------------------------
        if model_dir and os.path.exists(model_dir):
            shutil.rmtree(model_dir, ignore_errors=True)

        return jsonify({
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__
        }), 500


# Restrict this endpoint with user permissions
@app.route('/retrain', methods=['POST'])
def retrain_model():
    """
    Retrain an existing model with different feature selections and/or parameters
    
    Expected JSON request:
    {
        "model_id": "mtlr_20231103_120000_abc123",  # REQUIRED - existing model to retrain
        "dataset_path": "/path/to/data.csv",        # OPTIONAL - Input to use different dataset from parent predictor model
        "selected_features": "all",                  # OPTIONAL - see feature selection logic below
        "parameters": {"neurons": [64, 64], "dropout": 0.1}  # Optional parameter overrides
    }
    
    Feature Selection Logic:
    - selected_features='all' → Use ALL features from dataset
    - selected_features=['feat1', 'feat2'] → Use ONLY these specific features
    - selected_features=None (or not provided) → INHERIT from parent model
    
    Three retraining scenarios:
    1. All features + different parameters: set selected_features='all', provide parameters
    2. Selected features + same parameters: provide selected_features list, omit parameters
    3. Selected features + different parameters: provide both
    """
    # Reset metrics before training
    global ci, mae_hinge, mae_po, ibs, km_cal, xcal_stats, wsc_xcal_stats
    global dcal_chisquare, dcal_p_value_stat, train_times, infer_times, dcal_hists, n_features

    ci.clear()
    mae_hinge.clear()
    mae_po.clear()
    ibs.clear()
    km_cal.clear()
    xcal_stats.clear()
    wsc_xcal_stats.clear()
    dcal_chisquare.clear()
    dcal_p_value_stat.clear()
    train_times.clear()
    infer_times.clear()
    dcal_hists.clear()
    n_features = 0

    try:
        # Get request data
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        json_data = request.json
        
        # REQUIRED: model_id of existing model
        model_id = json_data.get('model_id')
        if not model_id:
            return jsonify({'error': 'model_id is required for retraining'}), 400
        
        # Check if model exists
        original_model_dir = os.path.join(app.config['MODEL_FOLDER'], model_id)
        if not os.path.exists(original_model_dir):
            return jsonify({'error': f'Model {model_id} not found'}), 404
        
        # Load original model config and metadata
        original_config_path = os.path.join(original_model_dir, 'model_config.json')
        original_metadata_path = os.path.join(original_model_dir, 'training_metadata.json')
        
        with open(original_config_path, 'r') as f:
            original_config = json.load(f)
        
        # Load metadata if exists, otherwise use defaults
        if os.path.exists(original_metadata_path):
            with open(original_metadata_path, 'r') as f:
                original_metadata = json.load(f)
        else:
            # Fallback for older models that might have selected_features in config
            original_metadata = {
                'selected_features': original_config.get('selected_features', None)
            }
        
        # Dataset Override Logic

        file = None
        try:
            file = request.files.get("dataset")
        except Exception:
            # request.files may not exist or parsing failed (JSON-only request)
            file = None

        if file:
            if file.filename == "":
                return jsonify({"error": "No dataset file selected"}), 400

            if not file.filename.lower().endswith(".csv"):
                return jsonify({"error": "Dataset file must be a CSV"}), 400

            # ✅ Save uploaded dataset file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_filename = f"{timestamp}_{filename}"
            dataset_path = os.path.join(app.config["UPLOAD_FOLDER"], dataset_filename)
            file.save(dataset_path)

        else:
            # ✅ Fallback: use dataset path from metadata or JSON
            dataset_path = (
                request.json.get("dataset_path")
                if request.is_json else None
            ) or original_metadata.get("dataset_path")

        # ✅ Final validation
        if not dataset_path:
            return jsonify({"error": "Dataset missing: upload a CSV or provide dataset_path"}), 400

        
        # dataset_path_input = json_data.get("dataset_path")
        # if dataset_path_input:
        #     dataset_path = dataset_path_input
        # else:
        #     # Just pull the dataset from the original model's training metadata)
        #     dataset_path = original_metadata.get("dataset_path")

        # if not dataset_path:
        #     return jsonify({"error": "dataset_path missing and not found in parent model metadata"}), 400

        
            
       
        # Feature Selection Logic
        selected_features_input = json_data.get('selected_features', None)
        
        if isinstance(selected_features_input, list):
            # Specific feature list
            if len(selected_features_input) == 0:
                return jsonify({'error': 'selected_features list cannot be empty'}), 400
            selected_features = selected_features_input
            selected_features_for_training = selected_features_input
            if selected_features_input == original_metadata.get('selected_features'):
                features_source = 'inherited'
            else:
                features_source = 'selected'
        elif selected_features_input is None:
            # Explicitly set to None - inherit from parent
            selected_features = original_metadata.get('selected_features')
            selected_features_for_training = None
            features_source = 'inherited'
        else:
            return jsonify({
                'error': 'selected_features must be a list of features, None, or omitted to inherit'
            }), 400
        
        # Get optional parameter overrides
        parameters = json_data.get('parameters', {})
        
        # Start with original config, then override with new parameters
        config = {
            'selected_features': selected_features_for_training,
            'neurons': parameters.get('neurons', original_config.get('neurons')),
            'dropout': parameters.get('dropout', original_config.get('dropout')),
            'activation': parameters.get('activation', original_config.get('activation')),
            'norm': parameters.get('norm', original_config.get('norm')),
        }
        
        # Override with any other user-provided parameters
        for key, value in parameters.items():
            if value is not None:
                config[key] = value

        args = Args(config)
        n_exp = max(1, int(getattr(args, 'n_exp', 1)))

        # Create NEW model ID for the retrained version
        now = datetime.now()
        model_timestamp = now.strftime("%Y%m%d_%H%M%S")
        model_timestamp_date = now.date().isoformat()
        suffix = uuid.uuid4().hex[:6]
        new_model_id = f"mtlr_retrain_{model_timestamp}_{suffix}"

        model_dir = os.path.join(app.config['MODEL_FOLDER'], new_model_id)
        os.makedirs(model_dir, exist_ok=True)

        # NEW: Collect predictions across all folds
        all_fold_predictions = []

        # Training Loop 
        train_start = time.time()
        for i in trange(n_exp, disable=not args.verbose, desc='Experiment'):
            icp, encoder, indiv_preds = train_mtlr_model(
                dataset_path, selected_features_for_training, args, i, return_predictions=True
            )

            if indiv_preds:
                all_fold_predictions.append(indiv_preds)
        
        # -------------------------------
        # Aggregate individual predictions
        # -------------------------------
        aggregated_predictions = aggregate_cv_predictions(all_fold_predictions)
        full_dataset_predictions = generate_full_dataset_predictions(
            icp, encoder, dataset_path, selected_features_for_training, args
        )

        train_end = time.time()
        train_duration = train_end - train_start

        # ----------------------------
        # Save artifacts
        # ----------------------------

        # Track what changed in this retrain
        retrain_history = {
            "original_features": original_metadata.get('selected_features'),
            "new_features": selected_features,
            "features_source": features_source,
            "features_changed": original_metadata.get('selected_features') != selected_features,
            "parameter_changes": {}
        }
        
        # Track which parameters changed
        for param in ['neurons', 'dropout', 'activation', 'norm']:
            if param in parameters and parameters[param] != original_config.get(param):
                retrain_history['parameter_changes'][param] = {
                    'old': original_config.get(param),
                    'new': parameters[param]
                }
        
        # Save model config (only model architecture params)
        model_config = {
            "model_type": "MTLR",
            "n_features": icp.nc_function.model.in_features,
            "time_bins": icp.nc_function.model.time_bins.tolist(),
            "neurons": args.neurons,
            "dropout": args.dropout,
            "activation": args.activation,
            "norm": args.norm,
        }
        config_path = os.path.join(model_dir, "model_config.json")
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        
        # Save training metadata separately (includes selected_features and retrain history)
        training_metadata = {
            "selected_features": selected_features,
            "parent_model_id": model_id,
            "retrain_history": retrain_history,
            "dataset_path": dataset_path,
            "n_experiments": n_exp,
            "timestamp": model_timestamp_date
        }
        metadata_path = os.path.join(model_dir, "training_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(training_metadata, f, indent=2)


        encoder_path = os.path.join(model_dir, "encoder.joblib")
        joblib.dump(encoder, encoder_path)

        icp_state_path = os.path.join(model_dir, "icp_state.dill")
        with open(icp_state_path, "wb") as f:
            dill.dump(icp, f)
        
        # Save model dimensions to .mtlr file
        dimensions_path = os.path.join(model_dir, f"mtlr_model_{new_model_id}.mtlr")
        save_dimensions(icp.nc_function.model, dimensions_path)
        
        if aggregated_predictions:
            cv_predictions_path = os.path.join(model_dir, "cv_predictions.json")
            with open(cv_predictions_path, 'w') as f:
                json.dump(aggregated_predictions, f, indent=2)

            cv_summary_path = os.path.join(model_dir, "cv_predictions_summary.csv")
            create_cv_summary_csv(aggregated_predictions, cv_summary_path)
        else:
            cv_predictions_path = None
            cv_summary_path = None

        if full_dataset_predictions:
            full_predictions_path = os.path.join(model_dir, "full_predictions.json")
            with open(full_predictions_path, 'w') as f:
                json.dump(full_dataset_predictions, f, indent=2)

            full_summary_path = os.path.join(model_dir, "full_predictions_summary.csv")
            create_cv_summary_csv(full_dataset_predictions, full_summary_path)

            survival_curve_path = os.path.join(model_dir, "survival_curves.json")
            save_survival_curve_mapping(full_dataset_predictions, survival_curve_path)
        else:
            full_predictions_path = None
            full_summary_path = None
            survival_curve_path = None

        metrics = print_performance(
            Cindex=ci,
            IBS=ibs,
            MAE_Hinge=mae_hinge,
            MAE_PO=mae_po,
            KM_cal=km_cal,
            xCal_stats=xcal_stats,
            wsc_xCal_stats=wsc_xcal_stats,
            dcal_p=dcal_p_value_stat,
            dcal_Chi=dcal_chisquare,
            train_times=train_times,
            infer_times=infer_times
        )

        metrics['n_features'] = n_features
        metrics['d_cal_hist'] = torch.stack(dcal_hists).mean(0).tolist()
        metrics['train_start_time'] = datetime.fromtimestamp(train_start).isoformat()
        metrics['train_duration'] = train_duration

        base_url = request.host_url.rstrip("/")

        cv_info = None
        if aggregated_predictions:
            cv_info = {
                "summary_csv": f"{base_url}/models/{new_model_id}/cv_predictions_summary.csv",
                "full_predictions": f"{base_url}/models/{new_model_id}/cv_predictions.json",
                "n_folds": len(all_fold_predictions),
                "total_predictions": len(aggregated_predictions['actual_times'])
            }

        full_info = None
        if full_dataset_predictions:
            full_info = {
                "summary_csv": f"{base_url}/models/{new_model_id}/full_predictions_summary.csv",
                "full_predictions": f"{base_url}/models/{new_model_id}/full_predictions.json",
                "survival_curves": f"{base_url}/models/{new_model_id}/survival_curves.json",
                "total_identifiers": len(full_dataset_predictions['test_indices'])
            }

        response_data = {
            "status": "success",
            "model_id": new_model_id,
            "parent_model_id": model_id,
            "metrics": metrics,
            "selected_features": selected_features,
            "model_config": f"{base_url}/models/{new_model_id}/model_config.json",
            "mtlr_model": f"{base_url}/models/{new_model_id}/mtlr_model_{new_model_id}.mtlr",
            "trained_at": model_timestamp_date,
            "train_duration": train_duration,
            "retrained_from": model_id,
            "cv_predictions": cv_info,
            "full_dataset_predictions": full_info,
            "retrain_summary": {
                "features_changed": retrain_history['features_changed'],
                "parameters_changed": list(retrain_history['parameter_changes'].keys()),
                "features_source": features_source
            },
            "timestamp": datetime.now().isoformat()
        }

        return jsonify(response_data), 200

    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """
    Batch prediction endpoint for survival models (MTLR + ICP).

    Expected JSON input:
    {
        "model_id": "mtlr_20250201_123456_xxx",
        "features": [
            {"age": 55, "stage": 3, ... },
            {"age": 47, "stage": 2, ... },
            ...
        ],
        "time_points": [10, 30, 60, 120],   # Optional (custom requested times)
        "alpha": 0.1                        # Optional (reserved for future bands)
    }
    """
    try:
        data = request.json
         # ---------------------------------------------------------
         # 1. Basic validation
        # ---------------------------------------------------------
        
        if 'model_id' not in data or 'features' not in data:
            return jsonify({'error': 'Missing model_id or features'}), 400
        
        model_id = data['model_id']
        feature_rows = data['features']
        labeled = data.get('labeled', False)
        
        if not isinstance(feature_rows, list) or len(feature_rows) == 0:
            return jsonify({'error': "'features' must be a non-empty list"}), 400
        
        # Load model folder
        model_dir = os.path.join(app.config['MODEL_FOLDER'], model_id)
        if not os.path.exists(model_dir):
            return jsonify({'error': f'Model {model_id} not found'}), 404
        
        # ---------------------------------------------------------
        # 2. Load encoder and ICP wrapper (which contains the MTLR model)
        # ---------------------------------------------------------
        encoder_path = os.path.join(model_dir, 'encoder.joblib')
        encoder = joblib.load(encoder_path)
        
        icp_state_path = os.path.join(model_dir, 'icp_state.dill')
        with open(icp_state_path, 'rb') as f:
            icp = dill.load(f)
        
        model = icp.nc_function.model          # Underlying MTLR model
        time_bins = model.time_bins         # Discrete bins used for S(t)
       
        # ----------------------------------------------
        # 3. Build input DataFrame for ALL samples (batch)
        # ----------------------------------------------

        input_data = pd.DataFrame(feature_rows)
        
        # Validate features against encoder expectations
        if hasattr(encoder, 'feature_names_in_'):
            expected_cols = list(encoder.feature_names_in_)

            # Filter out time/event from expectations as they are target variables
            required_features = [f for f in expected_cols if f not in ['time', 'event']]
            
            # Check for missing features
            missing_features = [f for f in required_features if f not in input_data.columns]
            if missing_features:
                return jsonify({'error': f'Missing features: {missing_features}'}), 400
                
            
            # Add dummy target cols (required by encoder structure)
            n = len(input_data)

            if "time" in expected_cols and "time" not in input_data.columns:
                input_data["time"] = [0] * n

            if "event" in expected_cols and "event" not in input_data.columns:
                input_data["event"] = [0] * n
                
            # Reorder to match encoder training
            input_data = input_data[expected_cols]
            
        else:
            # Fallback if feature_names_in_ is missing (unlikely)
            if 'time' not in input_data.columns:
                input_data['time'] = 0
            if 'event' not in input_data.columns:
                input_data['event'] = 0
        
        # -----------------------------
        # 4. Transform with encoder
        # -----------------------------
        input_transformed = encoder.transform(input_data)
        
        # IMPORTANT: Cast to float32 to avoid object dtype issues
        # SimpleImputer and OneHotEncoder can sometimes produce object dtype columns
        # which cause "can't convert np.ndarray of type numpy.object_" errors
        input_transformed = input_transformed.astype('float32')
        
        # Drop time/event after transform (they were only there to satisfy encoder)
        x_input = input_transformed.drop(['time', 'event'], axis=1).values  
        n_samples = input_transformed.shape[0]

        # ---------------------------------------------------------
        # 5. MTLR survival curve for ALL samples
        # ---------------------------------------------------------
        # Convert numpy → tensor
        x_tensor = torch.from_numpy(x_input).float()

        # Move to the correct device (VERY IMPORTANT)
        device = next(model.parameters()).device
        x_tensor = x_tensor.to(device)
        
        surv_tensor = model.predict_survival(x_tensor)  # (n_samples, n_bins)
        survival_curves = surv_tensor.cpu().numpy().tolist()

        # ------------------------------------------------------------------
        # 6. ICP quantile predictions (times at which quantiles are reached)
        # ------------------------------------------------------------------
        quan_levels, quan_preds = icp.predict(x_input)   # shapes:
                                                         # quan_levels: (n_quantiles,)
                                                         # quan_preds: (n_samples, n_quantiles)

        # ---------------------------------------------------------
        # 7. Summary statistics per sample
        # ---------------------------------------------------------
        medians, means, _ = summarize_prediction_stats(
            quan_levels, quan_preds, [0] * n_samples
        )

        stats_list = []
        for i in range(n_samples):
            q25_idx = int(np.argmin(np.abs(quan_levels - 0.25)))
            q75_idx = int(np.argmin(np.abs(quan_levels - 0.75)))

            stats_list.append({
                "median_survival_time": float(medians[i]),
                "mean_survival_time": float(means[i]),
                "25th_percentile_time": float(quan_preds[i, q25_idx]),
                "75th_percentile_time": float(quan_preds[i, q75_idx])
            })

        # ---------------------------------------------------------
        # 8. Optional: user-requested custom time points
        # ---------------------------------------------------------
        requested_tp = data.get("time_points", None)
        requested_survival = None
        if requested_tp is not None:
            requested_tp = np.array(requested_tp, dtype=float)
            requested_survival = []
            for i in range(n_samples):
                requested_survival.append(
                    stepwise_survival_at_times(
                        time_bins,
                        survival_curves[i],
                        requested_tp
                    ).tolist()
                )
        
        # ---------------------------------------------------------
        # 9. Handle labeled dataset with full predictions and metrics
        # ---------------------------------------------------------
        full_predictions = None
        metrics = None
        
        if labeled:
            # Load training metadata to get dataset path and selected features
            metadata_path = os.path.join(model_dir, 'training_metadata.json')
            if not os.path.exists(metadata_path):
                return jsonify({'error': 'Training metadata not found for labeled prediction'}), 404
            
            with open(metadata_path, 'r') as f:
                training_metadata = json.load(f)
            
            dataset_path = training_metadata.get('dataset_path')
            if not dataset_path or not os.path.exists(dataset_path):
                return jsonify({'error': f'Dataset path not found or does not exist: {dataset_path}'}), 404
            
            selected_features = training_metadata.get('selected_features')
            
            # Load Args from model config
            config_path = os.path.join(model_dir, 'model_config.json')
            with open(config_path, 'r') as f:
                model_config = json.load(f)
            
            # Create Args object with model config
            args_config = {
                'neurons': model_config.get('neurons'),
                'dropout': model_config.get('dropout'),
                'activation': model_config.get('activation'),
                'norm': model_config.get('norm'),
            }
            args = Args(args_config)
            
            # Generate full dataset predictions
            full_predictions = generate_full_dataset_predictions(
                icp, encoder, dataset_path, selected_features, args
            )
            
            # Calculate concordance index and integrated brier score
            if full_predictions:
                t_full = np.array(full_predictions['actual_times'])
                e_full = np.array(full_predictions['actual_events'])
                quan_preds_full = np.array(full_predictions['quantile_predictions'])
                quan_levels_full = np.array(full_predictions['quantile_levels'])
                
                # Create evaluator
                evaler = QuantileRegEvaluator(
                    quan_preds_full, quan_levels_full, t_full, e_full,
                    t_full, e_full,  # Using same data for train_val as we're evaluating on full dataset
                    predict_time_method="Median", interpolation=args.interpolate
                )
                
                # Calculate core metrics
                c_index = float(evaler.concordance(ties="All")[0])
                ibs_score = float(evaler.integrated_brier_score(num_points=10))
                km_cal_score = float(evaler.km_calibration())
                _ , dcal_hist = evaler.d_calibration()
                
                metrics = {
                    'concordance_index': c_index,
                    'integrated_brier_score': ibs_score,
                    'km_calibration': km_cal_score,
                    'd_calibration': dcal_hist.tolist()
                }
        
        # -----------------------------
        # 10. Build response payload
        # -----------------------------
        response = {
            "status": "success",
            "model_id": model_id,
            "timestamp": datetime.now().isoformat(),
            "predictions": {
                "n_samples": n_samples,
                "time_points": time_bins.tolist(),
                "survival_curves": survival_curves,           # list of lists
                "quantile_levels": quan_levels.tolist(),
                "quantile_times": quan_preds.tolist(),        # list of lists
                "statistics": stats_list                      # list of dicts
            }
        }

        if requested_survival is not None:
            response["predictions"]["custom_requested"] = {
                "requested_time_points": requested_tp.tolist(),
                "requested_survival_curves": requested_survival
            }
        
        # Add full predictions and metrics if labeled=True
        if labeled and full_predictions:
            response["full_predictions"] = full_predictions
            if metrics:
                response["metrics"] = metrics

        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__
        }), 500




@app.route('/model/<model_id>', methods=['GET'])
def get_model_info(model_id):
    """Get information about a trained model and its artifacts"""
    try:
        model_folder = os.path.join(app.config['MODEL_FOLDER'], model_id)
        if not os.path.isdir(model_folder):
            return jsonify({'error': 'Model not found'}), 404

        # Base URL for generating download links
        base_url = request.host_url.rstrip("/")

        # Load model metadata from model_config.json
        config_path = os.path.join(model_folder, 'model_config.json')
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                model_config = json.load(f)
        else:
            model_config = {}

        # Build artifact URLs
        artifacts = {}
        for artifact_name in ['model_weights.pth', 'encoder.joblib', 'icp_state.dill',
                              'model_config.json', 'features.json', 'metrics.json', 'args.json']:
            artifact_path = os.path.join(model_folder, artifact_name)
            if os.path.exists(artifact_path):
                artifacts[artifact_name] = f"{base_url}/models/{model_id}/{artifact_name}"

        info = {
            'model_id': model_id,
            'model_type': model_config.get('model_type', 'MTLR'),
            'timestamp': model_config.get('timestamp'),
            'config': model_config,
            'artifacts': artifacts
        }

        return jsonify(info), 200

    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__
        }), 500


# @app.route('/download/<model_id>', methods=['GET'])
# def download_model(model_id):
#     """Download trained model file"""
#     try:
#         model_path = os.path.join(app.config['MODEL_FOLDER'], f'{model_id}.pkl')
        
#         if not os.path.exists(model_path):
#             return jsonify({'error': 'Model not found'}), 404
        
#         return send_file(
#             model_path,
#             as_attachment=True,
#             download_name=f'{model_id}.pkl',
#             mimetype='application/octet-stream'
#         )
    
#     except Exception as e:
#         return jsonify({
#             'status': 'error',
#             'error': str(e)
#         }), 500


@app.route('/models', methods=['GET'])
def list_models():
    """List all trained models with artifact URLs"""
    try:
        base_url = request.host_url.rstrip("/")
        models = []

        # Each subfolder in MODEL_FOLDER is a separate model
        for model_id in os.listdir(app.config['MODEL_FOLDER']):
            model_folder = os.path.join(app.config['MODEL_FOLDER'], model_id)
            if not os.path.isdir(model_folder):
                continue

            # Check for key artifacts
            artifacts = {}
            for artifact_name in ['model_weights.pth', 'encoder.joblib', 'icp_state.dill', 'model_config.json', 'training_metadata.json']:
                artifact_path = os.path.join(model_folder, artifact_name)
                if os.path.exists(artifact_path):
                    artifacts[artifact_name] = f"{base_url}/models/{model_id}/{artifact_name}"

            # Load minimal metadata from model_config.json if it exists
            model_type = 'MTLR'
            timestamp = None
            config_path = os.path.join(model_folder, 'model_config.json')
            if os.path.exists(config_path):
                import json
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    model_type = config_data.get('model_type', 'MTLR')

            training_metadata_path = os.path.join(model_folder, 'training_metadata.json');
            if os.path.exists(training_metadata_path):
                import json
                with open(training_metadata_path, 'r') as f:
                    training_metadata = json.load(f)
                    timestamp = training_metadata['timestamp']

            models.append({
                'model_id': model_id,
                'model_type': model_type,
                'timestamp': timestamp,
                'artifacts': artifacts
            })

        return jsonify({
            'status': 'success',
            'count': len(models),
            'models': models
        }), 200

    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__
        }), 500

def stepwise_survival_at_times(grid_times, surv_probs, query_times):
    """
    Stepwise-constant interpolation of survival probabilities.

    grid_times : array-like, shape (n_bins,)
        Model's time grid (model.time_bins).
    surv_probs : array-like, shape (n_bins,)
        Survival probabilities S(t) evaluated on grid_times.
    query_times : array-like, shape (m,)
        Arbitrary times at which to evaluate survival.

    Returns
    -------
    result : array, shape (m,)
        Survival probabilities at query_times.
    """
    grid_times = np.asarray(grid_times, float)
    surv_probs = np.asarray(surv_probs, float)
    query_times = np.asarray(query_times, float)

    result = np.empty_like(query_times, float)

    for i, t in enumerate(query_times):
        if t <= grid_times[0]:
            result[i] = 1.0
        elif t >= grid_times[-1]:
            result[i] = surv_probs[-1]
        else:
            idx = np.searchsorted(grid_times, t, side='right') - 1
            result[i] = surv_probs[idx]

    return result

def create_cv_summary_csv(aggregated_predictions, output_path):
    """
    Create CSV summary from aggregated predictions.

    Parameters:
    - aggregated_predictions: dict returned from aggregation, with keys:
        'test_indices', 'actual_times', 'actual_events',
        'median_predictions', 'mean_predictions', 'prob_at_actual_time',
        'quantile_levels', 'quantile_predictions'
    - output_path: path to save the CSV
    """
    rows = []

    n_samples = len(aggregated_predictions['actual_times'])

    for i in range(n_samples):
        row = {
            'identifier': aggregated_predictions['test_indices'][i],
            'censored': 'yes' if aggregated_predictions['actual_events'][i] == 0 else 'no',
            'event_time': aggregated_predictions['actual_times'][i],
            'predicted_prob_event': aggregated_predictions['prob_at_actual_time'][i],
            'predicted_median_survival': aggregated_predictions['median_predictions'][i],
            'predicted_mean_survival': aggregated_predictions['mean_predictions'][i],
            'absolute_error': abs(aggregated_predictions['median_predictions'][i] - aggregated_predictions['actual_times'][i])
                              if aggregated_predictions['actual_events'][i] == 1 else None
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    return df


def summarize_prediction_stats(quan_levels, quan_preds, actual_times):
    """Compute median, mean, and event probability for each prediction."""
    medians = []
    means = []
    probs = []
    median_idx = np.argmin(np.abs(quan_levels - 0.5))
    survival_curve = 1 - quan_levels

    for row, actual_t in zip(quan_preds, actual_times):
        medians.append(float(row[median_idx]))
        means.append(float(np.trapezoid(survival_curve, row)))

        reached = row <= actual_t
        if np.any(reached):
            cdf_at_t = np.max(quan_levels[reached])
            probs.append(float(cdf_at_t * 100))
        else:
            probs.append(0.0)

    return medians, means, probs


def aggregate_cv_predictions(all_fold_predictions):
    """Collapse per-fold predictions so each identifier appears once."""
    if not all_fold_predictions:
        return None

    quantile_levels = all_fold_predictions[0]['quantile_levels']
    buckets = defaultdict(lambda: {
        'actual_times': [],
        'actual_events': [],
        'median_predictions': [],
        'mean_predictions': [],
        'prob_at_actual_time': [],
        'quantile_predictions': []
    })

    for preds in all_fold_predictions:
        for idx, actual_time, actual_event, median_pred, mean_pred, prob_pred, quantiles in zip(
            preds['test_indices'],
            preds['actual_times'],
            preds['actual_events'],
            preds['median_predictions'],
            preds['mean_predictions'],
            preds['prob_at_actual_time'],
            preds['quantile_predictions']
        ):
            bucket = buckets[idx]
            bucket['actual_times'].append(actual_time)
            bucket['actual_events'].append(actual_event)
            bucket['median_predictions'].append(median_pred)
            bucket['mean_predictions'].append(mean_pred)
            bucket['prob_at_actual_time'].append(prob_pred)
            bucket['quantile_predictions'].append(np.array(quantiles))

    aggregated = {
        'test_indices': [],
        'actual_times': [],
        'actual_events': [],
        'median_predictions': [],
        'mean_predictions': [],
        'prob_at_actual_time': [],
        'quantile_levels': quantile_levels,
        'quantile_predictions': []
    }

    for idx in sorted(buckets.keys()):
        bucket = buckets[idx]
        aggregated['test_indices'].append(idx)
        aggregated['actual_times'].append(float(np.mean(bucket['actual_times'])))
        aggregated['actual_events'].append(int(round(np.mean(bucket['actual_events']))))
        aggregated['median_predictions'].append(float(np.mean(bucket['median_predictions'])))
        aggregated['mean_predictions'].append(float(np.mean(bucket['mean_predictions'])))
        aggregated['prob_at_actual_time'].append(float(np.mean(bucket['prob_at_actual_time'])))
        aggregated['quantile_predictions'].append(
            np.mean(np.vstack(bucket['quantile_predictions']), axis=0).tolist()
        )

    return aggregated


def generate_full_dataset_predictions(icp, encoder, dataset_path, selected_features, args):
    """Run the trained model on the entire dataset to get predictions for every identifier."""
    if icp is None or encoder is None:
        return None

    _, _, _, _, raw_data = prepare_data(dataset_path, selected_features, args)
    raw_data = raw_data.copy()
    indices = raw_data.index.tolist()

    encoded = encoder.transform(raw_data).astype('float32')
    x_full = encoded.drop(['time', 'event'], axis=1).values
    t_full = encoded['time'].values
    e_full = encoded['event'].values

    quan_levels, quan_preds = icp.predict(x_full)
    median_preds, mean_preds, prob_preds = summarize_prediction_stats(quan_levels, quan_preds, t_full)

    return {
        'test_indices': indices,
        'actual_times': t_full.tolist(),
        'actual_events': e_full.tolist(),
        'median_predictions': median_preds,
        'mean_predictions': mean_preds,
        'prob_at_actual_time': prob_preds,
        'quantile_levels': quan_levels.tolist(),
        'quantile_predictions': quan_preds.tolist()
    }


def save_survival_curve_mapping(predictions, output_path):
    """Save a helper JSON for plotting survival probability curves per identifier."""
    if not predictions:
        return

    survival_probabilities = [float((1 - q) * 100) for q in predictions['quantile_levels']]
    curves = {}
    for idx, curve in zip(predictions['test_indices'], predictions['quantile_predictions']):
        curves[str(idx)] = {
            'times': curve,
            'survival_probabilities': survival_probabilities 
        }

    payload = {
        'quantile_levels': predictions['quantile_levels'],
        'survival_probabilities': survival_probabilities,
        'curves': curves
    }

    with open(output_path, 'w') as f:
        json.dump(payload, f, indent=2)


def save_dimensions(model, filepath: str):
    """Save all model dimensions + parameter dump into a text file."""
    with open(filepath, "w") as f:

        # Header
        f.write("------ MTLR MODEL DIMENSIONS ------\n")

        # Number of features
        f.write(f"n_features: {model.in_features}\n")

        # Time bins
        f.write(f"m (n_time_bins): {model.output_size}\n")
        f.write("time_bins:\n")
        time_bins_list = model.time_bins.cpu().numpy().tolist()
        f.write(",".join(str(x) for x in time_bins_list) + "\n")

        # Hidden layers
        if model.hidden_size:
            f.write(f"r (n_hidden_layers): {len(model.hidden_size)}\n")
            f.write(f"hidden_sizes: {model.hidden_size}\n")
        else:
            f.write("r (n_hidden_layers): 0\n")

        # Total parameters
        total_params = sum(p.numel() for p in model.parameters())
        f.write(f"DIM (total parameters): {total_params}\n\n")

        # Flatten parameters
        flat = torch.cat([
            p.detach().cpu().flatten() for p in model.parameters()
        ])

        # Parameter dump (one per line)
        f.write("PARAMETERS:\n")
        for i, val in enumerate(flat):
            f.write(f"{i+1}:{val.item()}\n")




if __name__ == '__main__':
    app.run(debug=True, host=API_HOST, port=API_PORT)
