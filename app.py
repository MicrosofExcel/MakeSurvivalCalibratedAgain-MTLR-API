import os
import json
import joblib
import dill
import uuid
from tqdm import trange
import time
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

# Import models and utilities
from model import MTLR
from icp import ConformalSurvDist, CSDiPOT
from icp.scorer import QuantileRegressionNC, SurvivalPredictionNC
from utils import set_seed, save_params
from utils.util_survival import survival_data_split, make_time_bins, xcal_from_hist
from SurvivalEVAL import QuantileRegEvaluator
from CondCalEvaluation import wsc_xcal



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_FOLDER'] = 'trained_models'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

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
n_features = 0

class Args:
    """Configuration object for model training"""
    def __init__(self, config):
        # Model parameters
        self.model = 'MTLR'
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
    if len(columns) < 2:
        raise ValueError(f"Dataset must have at least 2 columns. Found {len(columns)} columns.")

    data = data.rename(columns={columns[0]: 'time', columns[1]: 'censored'})
    data['event'] = ((data['censored'] == 0) | (data['censored'] == 'u')).astype(int)

    # ✅ Remove invalid zero-time rows (Time = 0 and Event = 1/occurred). Not plausible
    invalid_mask = (data["time"] <= 0)

    # # of invalid masks removed.
    # if invalid_mask.any():
    #     print(f"Removing {invalid_mask.sum()} samples with invalid time <= 0")

    data = data[~invalid_mask]

    data = data.drop(columns='censored')

    # Filter selected features
    feature_columns = [col for col in data.columns if col not in ['time', 'event']]
    if selected_features:
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

def make_strictly_increasing(x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Convert an array into a strictly increasing sequence by adding a tiny jitter
    to duplicate or non-increasing values.

    Parameters
    ----------
    x : np.ndarray
        Original array (may contain duplicates or non-increasing values)
    eps : float
        Small increment to enforce strict increase

    Returns
    -------
    x_new : np.ndarray
        Strictly increasing array
    """
    x_new = x.copy()
    for i in range(1, len(x_new)):
        if x_new[i] <= x_new[i - 1]:
            x_new[i] = x_new[i - 1] + eps
    return x_new




def train_mtlr_model(dataset_path, selected_features, args, i):
    """Train MTLR model with conformal prediction"""

    # ADD THIS AT THE START OF THE FUNCTION
    global ci, mae_hinge, mae_po, ibs, km_cal, xcal_stats, wsc_xcal_stats
    global dcal_chisquare, dcal_p_value_stat, train_times, infer_times, n_features

    # ✅ START TIMING
    # t0 = time.time()

    device = torch.device(args.device)

    # One unified seed per trial
    seed = args.seed + i

    # Set ALL RNG controls
    set_seed(seed, device)  # this should set torch + cuda + python.random
    np.random.seed(seed)    # explicitly set NumPy

    # t1 = time.time()
    # print(f"[Exp {i+1}] Setup & seeding: {t1-t0:.3f}s")
    
    # Prepare data AFTER seeding
    encoder, pct_train, pct_val, pct_test, data = prepare_data(
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


    # Separate features and survival labels
    X_train = data_train.drop(columns=['time', 'event'])
    y_train = data_train[['time', 'event']]

    X_val = data_val.drop(columns=['time', 'event']) if not data_val.empty else None
    y_val = data_val[['time', 'event']] if not data_val.empty else None

    X_test = data_test.drop(columns=['time', 'event'])
    y_test = data_test[['time', 'event']]

    # t4 = time.time()
    # print(f"[Exp {i+1}] Separate features/labels: {t4-t3:.3f}s")

    # ✅ Keep DataFrame form. Fit encoder on X_train only
    X_train = encoder.fit_transform(X_train)

    # t5 = time.time()
    # print(f"[Exp {i+1}] Encoder fit_transform: {t5-t4:.3f}s")

    if X_val is not None:
        X_val = encoder.transform(X_val)
    X_test = encoder.transform(X_test)

    # t6 = time.time()
    # print(f"[Exp {i+1}] Encoder transform val/test: {t6-t5:.3f}s")

    # ✅ REMOVE SLOW OPERATION - just use astype directly
    # The encoder already outputs numeric data, no need for pd.to_numeric
    X_train = X_train.astype("float32")

    # t7 = time.time()
    # print(f"[Exp {i+1}] X_train astype: {t7-t6:.3f}s")

    if X_val is not None:
        X_val = X_val.astype("float32")
    X_test = X_test.astype("float32")

    # t8 = time.time()
    # print(f"[Exp {i+1}] X_val/X_test astype: {t8-t7:.3f}s")

    # Extract survival labels
    t_train, e_train = y_train['time'].values, y_train['event'].values
    t_val, e_val = (y_val['time'].values, y_val['event'].values) if y_val is not None else (None, None)
    t_test, e_test = y_test['time'].values, y_test['event'].values
    
    
    n_features = X_train.shape[1]

    # Ensure numeric dtype for survival labels
    t_train = t_train.astype(float)
    e_train = e_train.astype(int)
    if t_val is not None:
        t_val = t_val.astype(float)
        e_val = e_val.astype(int)
    t_test = t_test.astype(float)
    e_test = e_test.astype(int)

    # t9 = time.time()
    # print(f"[Exp {i+1}] Extract/convert labels: {t9-t8:.3f}s")
    
    # Create time bins for MTLR
    discrete_bins = make_time_bins(t_train, event=e_train)

    # t10 = time.time()
    # print(f"[Exp {i+1}] make_time_bins: {t10-t9:.3f}s")

    
    
    # Build MTLR model
    model = MTLR(
        n_features=n_features,
        time_bins=discrete_bins,
        hidden_size=args.neurons,
        norm=args.norm,
        activation=args.activation,
        dropout=args.dropout
    )

    # t11 = time.time()
    # print(f"[Exp {i+1}] Build MTLR model: {t11-t10:.3f}s")
    
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
    
    # t12 = time.time()
    # print(f"[Exp {i+1}] Setup conformal prediction: {t12-t11:.3f}s")
    

    # Add 'time' and 'event' back to X_train
    data_train_for_fit = X_train.copy()
    data_train_for_fit['time'] = t_train
    data_train_for_fit['event'] = e_train

    data_val_for_fit = None
    data_val_for_cal = None
    if X_val is not None:
        data_val_for_fit = X_val.copy()
        data_val_for_cal = X_val.copy()
        data_val_for_fit['time'] = t_val
        data_val_for_fit['event'] = e_val
        data_val_for_cal['time'] = t_val
        data_val_for_cal['event'] = e_val
    
    # t13 = time.time()
    # print(f"[Exp {i+1}] Prepare data for fit: {t13-t12:.3f}s")

    # Train model using tuple-based data
    start_time = datetime.now()
    icp.fit(data_train_for_fit, data_val_for_fit)
    
    
    # t14 = time.time()
    # print(f"[Exp {i+1}] icp.fit(): {t14-t13:.3f}s")

    # Calibrate
    if args.use_train and X_val is not None:
        icp.calibrate(data_train_for_fit)  # use train data
    else:
        icp.calibrate(data_val_for_cal)


    mid_time = datetime.now()

    # t15 = time.time()
    # print(f"[Exp {i+1}] icp.calibrate(): {t15-t14:.3f}s")


    # Make predictions on test set
    # Convert DataFrame to Numpy array
    X_test_np = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
    quan_levels, quan_preds = icp.predict(X_test_np) 
    end_time = datetime.now()

    # t16 = time.time()
    # print(f"[Exp {i+1}] icp.predict(): {t16-t15:.3f}s")
    
    # Calculate training and inference times
    train_time = (mid_time - start_time).total_seconds()
    infer_time = (end_time - mid_time).total_seconds()
    
    # Evaluate performance
    t_train_val = np.concatenate((t_train, t_val)) if t_val is not None else t_train
    e_train_val = np.concatenate((e_train, e_val)) if e_val is not None else e_train
    
    # Make event times strictly increasing for PCHIP (x must be strictly increasing)
    unique_times_fixed = make_strictly_increasing(t_test)

    evaler = QuantileRegEvaluator(
        quan_preds, quan_levels, unique_times_fixed, e_test,
        t_train_val, e_train_val,
        predict_time_method="Median", interpolation=args.interpolate
    )

    # t17 = time.time()
    # print(f"[Exp {i+1}] Create evaluator: {t17-t16:.3f}s")


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
        wsc_xcal_score = float(wsc_xcal(X_test, e_test, pred_probs, random_state=seed))
    else:
        wsc_xcal_score = 0  # not enough data to compute the WSC

    # t18 = time.time()
    # print(f"[Exp {i+1}] Calculate metrics: {t18-t17:.3f}s")
    # print(f"[Exp {i+1}] === TOTAL TIME: {t18-t0:.3f}s ===\n")

    ci.append(c_index)
    ibs.append(ibs_score)
    mae_hinge.append(hinge_abs)
    mae_po.append(po_abs)
    km_cal.append(km_cal_score)
    xcal_stats.append(xcal_score)
    dcal_chisquare.append(float(dcal_chisquare_stat))
    dcal_p_value_stat.append(float(dcal_p_value))
    wsc_xcal_stats.append(wsc_xcal_score)
    train_times.append(train_time)
    infer_times.append(infer_time)


    # ✅ CLEANUP: Delete large objects
    del data_train, data_val, data_test
    del X_train, X_val, X_test
    del y_train, y_val, y_test
    del data_train_for_fit, data_val_for_fit, data_val_for_cal
    del t_train, e_train, t_val, e_val, t_test, e_test
    del quan_levels, quan_preds
    del evaler, pred_probs
    del model, nc_model
    del discrete_bins, unique_times_fixed
    
    # ✅ Force garbage collection
    import gc
    gc.collect()
    
    # ✅ Clear PyTorch cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"[Exp {i+1}] Cleanup complete")

    
    return icp, encoder


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


@app.route('/train', methods=['POST'])
def train_model():
    """
    Train MTLR model with ALL features in the dataset
    
    Expected CSV format:
    - First column: Time/Label (survival time or duration)
    - Second column: Censored (0=event occurred/uncensored, 1=censored)
    - Remaining columns: Features for prediction
    
    Expected form data:
    - dataset: CSV file with survival data in the format above
    - parameters: JSON object with model parameters (optional)
    
    This endpoint ALWAYS uses ALL features in the dataset (except first two columns).
    Use /retrain endpoint if you want to select specific features.
    
    Example JSON request:
    {
        "dataset_path": "/path/to/data.csv",
        "parameters": {"neurons": [64, 64], "dropout": 0.1}
    }
    """
    # --- Before training experiments in /train ---
    global ci, mae_hinge, mae_po, ibs, km_cal, xcal_stats, wsc_xcal_stats
    global dcal_chisquare, dcal_p_value_stat, train_times, infer_times, n_features

    # Reset metrics
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
    n_features = 0

    try:
        # Handle dataset - either as file upload or JSON path
        dataset_path = None
        
        # Option 1: File upload (multipart/form-data)
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
        
        # Option 2: JSON request with dataset path
        elif request.is_json:
            json_data = request.json
            dataset_path = json_data.get('dataset_path')
            if not dataset_path or not os.path.exists(dataset_path):
                return jsonify({'error': 'Invalid or missing dataset_path'}), 400
        
        else:
            return jsonify({'error': 'No dataset provided (either upload file or provide dataset_path)'}), 400
        
        # Parse user-selected configuration
        parameters = request.json.get('parameters', {}) if request.is_json else \
                     json.loads(request.form.get('parameters', '{}'))
        
        # Parse selected features
        selected_features = request.json.get('selected_features', None) if request.is_json else \
                            json.loads(request.form.get('selected_features', None))

        config = {'selected_features': selected_features } 
        
        # Merge parameters into config
        config.update(parameters)

        # Pass configurations + features into Args (has config and other default hyperparameters)
        args = Args(config)

        # safety: ensure n_exp is an int >= 1
        n_exp = max(1, int(getattr(args, 'n_exp', 1)))

        # Create model ID
        now = datetime.now()
        model_timestamp = now.strftime("%Y%m%d_%H%M%S")
        model_timestamp_date = now.date().isoformat()
        suffix = uuid.uuid4().hex[:6]  # 6-character random ID
        model_id = f"mtlr_{model_timestamp}_{suffix}"

        # Create model dir
        model_dir = os.path.join(app.config['MODEL_FOLDER'], model_id)
        os.makedirs(model_dir, exist_ok=True)

        
        # Training Loop 
        train_start = time.time()
        for i in trange(n_exp, disable=not args.verbose, desc='Experiment'):
            icp, encoder = train_mtlr_model(
                dataset_path, selected_features, args, i
            )

        train_end = time.time()

        train_duration = train_end - train_start

        # ----------------------------
        # 4️⃣ Save artifacts
        # ----------------------------

        # Save model weights (state_dict)
        model_path = os.path.join(model_dir, "model_weights.pth")
        torch.save({"model_state_dict": icp.nc_function.model.state_dict()}, model_path)
        
        # Save model config
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

        # Save trained at timestamp date
        trained_date_path = os.path.join(model_dir, "trained_at.txt")
        with open(trained_date_path, 'w') as f:
            f.write(model_timestamp_date)

        # Save encoder pipeline
        encoder_path = os.path.join(model_dir, "encoder.joblib")
        joblib.dump(encoder, encoder_path)

        # Save ICP state
        icp_state_path = os.path.join(model_dir, "icp_state.dill")
        with open(icp_state_path, "wb") as f:
            dill.dump(icp, f)

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

        # After saving artifacts, make base_url
        base_url = request.host_url.rstrip("/")  # e.g., http://localhost:5000
       

        return jsonify({
            "status": "success",
            "model_id": model_id,
            "metrics": metrics,
            "model_weights": f"{base_url}/models/{model_id}/model_weights.pth",
            "encoder": f"{base_url}/models/{model_id}/encoder.joblib",
            "icp_state": f"{base_url}/models/{model_id}/icp_state.dill",
            "model_config": f"{base_url}/models/{model_id}/model_config.json",
            "trained_at":f"{base_url}/models/{model_id}/trained_at.txt",
            "train_duration": train_duration,
            "timestamp": datetime.now().isoformat()
        }), 200


    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__
        }), 500


@app.route('/retrain', methods=['POST'])
def retrain_model():
    """
    Re-train MTLR model with SELECTED features
    
    Expected CSV format:
    - First column: Time/Label (survival time or duration)
    - Second column: Censored (0=event occurred/uncensored, 1=censored)
    - Remaining columns: Features for prediction
    
    Expected form data:
    - dataset: CSV file with survival data in the format above
    - features: JSON array of selected feature names (REQUIRED)
    - parameters: JSON object with model parameters
    
    This endpoint allows you to select specific features for training.
    Use /train endpoint if you want to use ALL features.
    
    Example JSON request:
    {
        "dataset_path": "/path/to/data.csv",
        "features": ["Height", "Weight", "Eye_Color"],
        "parameters": {"neurons": [64, 64], "dropout": 0.1}
    }
    """
    try:
        # Handle dataset - either as file upload or JSON path
        dataset_path = None
        
        # Option 1: File upload (multipart/form-data)
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
        
        # Option 2: JSON request with dataset path
        elif request.is_json:
            json_data = request.json
            dataset_path = json_data.get('dataset_path')
            if not dataset_path or not os.path.exists(dataset_path):
                return jsonify({'error': 'Invalid or missing dataset_path'}), 400
        
        else:
            return jsonify({'error': 'No dataset provided (either upload file or provide dataset_path)'}), 400
        
        
        # Parse configuration based on request type
        if request.is_json:
            # JSON request format
            json_data = request.json
            features = json_data.get('features', [])
            parameters = json_data.get('parameters', {})
            
            config = {
                'selected_features': features,
            }
        else:
            # Form data request format
            features = json.loads(request.form.get('features', '[]'))
            parameters = json.loads(request.form.get('parameters', '{}'))
            
            config = {
                'selected_features': features,
            }
        
        # Merge parameters into config
        config.update(parameters)
        
        # Extract selected features
        selected_features = config.get('selected_features', None)
        
        # Train model
        icp, encoder, metrics, args = train_mtlr_model(
            dataset_path, selected_features, config
        )
        
        # Generate model ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = f"mtlr_{timestamp}"
        
        # Save model
        model_path = os.path.join(app.config['MODEL_FOLDER'], f'{model_id}.pkl')
        model_data = {
            'icp': icp,
            'encoder': encoder,
            'config': config,
            'args': vars(args),
            'timestamp': datetime.now().isoformat(),
            'model_type': 'MTLR',
            'selected_features': selected_features
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Return response
        return jsonify({
            'status': 'success',
            'model_id': model_id,
            'model_path': os.path.abspath(model_path),
            'metrics': metrics,
            'config': config,
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__
        }), 500


@app.route('/predict', methods=['POST'])
def predict():
    """
    Make predictions using a trained model
    
    Expected JSON:
    {
        "model_id": "mtlr_20231101_120000",
        "features": {
            "feature1": value1,
            "feature2": value2,
            ...
        }
    }
    """
    try:
        data = request.json
        
        if 'model_id' not in data or 'features' not in data:
            return jsonify({'error': 'Missing model_id or features'}), 400
        
        model_id = data['model_id']
        model_path = os.path.join(app.config['MODEL_FOLDER'], f'{model_id}.pkl')
        
        if not os.path.exists(model_path):
            return jsonify({'error': 'Model not found'}), 404
        
        # Load model
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        icp = model_data['icp']
        encoder = model_data['encoder']
        
        # Prepare input data
        input_df = pd.DataFrame([data['features']])
        
        # Transform using encoder
        input_transformed = encoder.transform(input_df).values
        
        # Make prediction
        quan_levels, quan_preds = icp.predict(input_transformed)
        
        # Format response
        predictions = {
            'quantile_levels': quan_levels.tolist(),
            'quantile_predictions': quan_preds[0].tolist(),
            'median_survival_time': float(quan_preds[0][len(quan_preds[0])//2])
        }
        
        return jsonify({
            'status': 'success',
            'predictions': predictions,
            'model_id': model_id
        }), 200
    
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
            for artifact_name in ['model_weights.pth', 'encoder.joblib', 'icp_state.dill', 'model_config.json', 'trained_at.txt']:
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
                    timestamp = config_data.get('timestamp')

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


if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)