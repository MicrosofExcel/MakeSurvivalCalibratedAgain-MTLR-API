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
from flask_cors import CORS 


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
CORS(app, origins=["http://localhost:5174", "http://localhost:5173"], supports_credentials=True)


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

    # Remove invalid zero-time rows (Time = 0 and Event = 1/occurred). Not plausible
    invalid_mask = (data["time"] <= 0)
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
        
        for j in range(len(t_test)):
            # Median prediction (50th percentile)
            median_idx = np.argmin(np.abs(quan_levels - 0.5))
            median_pred = float(quan_preds[j, median_idx])
            median_preds.append(median_pred)
            
            # Mean prediction (integrate over survival curve)
            # Mean = integral of S(t) dt
            mean_pred = np.trapezoid(1 - quan_levels, quan_preds[j])
            mean_preds.append(float(mean_pred))
            
            # Probability of event at actual time
            actual_t = t_test[j]
            # Find CDF at actual time
            reached = quan_preds[j] <= actual_t
            if np.any(reached):
                cdf_at_t = np.max(quan_levels[reached])
                prob_event = float(cdf_at_t * 100)  # Convert to percentage
            else:
                prob_event = 0.0
            prob_at_actual_time.append(prob_event)
        
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


        # Train on all features by default
        selected_features = 'all'
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
        if all_fold_predictions:
            n_experiments = len(all_fold_predictions)
            n_samples = len(all_fold_predictions[0]['median_predictions'])
            n_quantiles = len(all_fold_predictions[0]['quantile_levels'])

            # Initialize arrays
            avg_median_preds = np.zeros(n_samples)
            avg_mean_preds = np.zeros(n_samples)
            avg_prob_at_t = np.zeros(n_samples)
            avg_quan_preds = np.zeros((n_samples, n_quantiles))

            for preds in all_fold_predictions:
                avg_median_preds += np.array(preds['median_predictions'])
                avg_mean_preds += np.array(preds['mean_predictions'])
                avg_prob_at_t += np.array(preds['prob_at_actual_time'])
                avg_quan_preds += np.array(preds['quantile_predictions'])

            # Compute averages
            avg_median_preds /= n_experiments
            avg_mean_preds /= n_experiments
            avg_prob_at_t /= n_experiments
            avg_quan_preds /= n_experiments

            aggregated_predictions = {
                'test_indices': all_fold_predictions[0]['test_indices'],
                'actual_times': all_fold_predictions[0]['actual_times'],
                'actual_events': all_fold_predictions[0]['actual_events'],
                'median_predictions': avg_median_preds.tolist(),
                'mean_predictions': avg_mean_preds.tolist(),
                'prob_at_actual_time': avg_prob_at_t.tolist(),
                'quantile_levels': all_fold_predictions[0]['quantile_levels'],
                'quantile_predictions': avg_quan_preds.tolist()
            }
        else:
            aggregated_predictions = None

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
    
        # Save CV_predictions
        cv_predictions_path = os.path.join(model_dir, "cv_predictions.json")
        with open(cv_predictions_path, 'w') as f:
            json.dump(aggregated_predictions, f, indent=2)
            
        # Also create a summary CSV for easy analysis
        cv_summary_path = os.path.join(model_dir, "cv_predictions_summary.csv")
        create_cv_summary_csv(aggregated_predictions, cv_summary_path)

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

        # Make base_url
        base_url = request.host_url.rstrip("/")  # e.g., http://localhost:5000
       

        response_data = {
            "status": "success",
            "model_id": model_id,
            "metrics": metrics,
            "selected_features": selected_features,
            "model_config": f"{base_url}/models/{model_id}/model_config.json",
            "model_file": {
                "encoder": f"{base_url}/models/{model_id}/encoder.joblib",
                "icp_state": f"{base_url}/models/{model_id}/icp_state.dill",
            },
            "cv_predictions": {
                "summary_csv": f"{base_url}/models/{model_id}/cv_predictions_summary.csv",
                "full_predictions": f"{base_url}/models/{model_id}/cv_predictions.json",
                "n_folds": len(all_fold_predictions),  # total experiments/folds
                "total_predictions": len(aggregated_predictions['actual_times'])  # total test samples
            },
            "trained_at": model_timestamp_date,
            "train_duration": train_duration,
            "timestamp": datetime.now().isoformat()
        }
        


        return jsonify(response_data), 200


    except Exception as e:
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
        selected_features_input = json_data.get('selected_features', 'inherit')
        
        if selected_features_input == 'inherit':
            # Not provided - inherit from parent
            selected_features = original_metadata.get('selected_features')
            selected_features_for_training = None if selected_features == 'all' else selected_features
            features_source = 'inherited'
        elif selected_features_input == 'all':
            # Explicitly use all features
            selected_features = 'all'
            selected_features_for_training = None
            features_source = 'all'
        elif isinstance(selected_features_input, list):
            # Specific feature list
            if len(selected_features_input) == 0:
                return jsonify({'error': 'selected_features list cannot be empty'}), 400
            selected_features = selected_features_input
            selected_features_for_training = selected_features_input
            features_source = 'selected'
        elif selected_features_input is None:
            # Explicitly set to None - inherit from parent
            selected_features = original_metadata.get('selected_features')
            selected_features_for_training = None if selected_features == 'all' else selected_features
            features_source = 'inherited'
        else:
            return jsonify({
                'error': 'selected_features must be "all", a list of features, null, or omitted to inherit'
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
            if key not in config:
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
        if all_fold_predictions:
            n_experiments = len(all_fold_predictions)
            n_samples = len(all_fold_predictions[0]['median_predictions'])
            n_quantiles = len(all_fold_predictions[0]['quantile_levels'])

            # Initialize arrays
            avg_median_preds = np.zeros(n_samples)
            avg_mean_preds = np.zeros(n_samples)
            avg_prob_at_t = np.zeros(n_samples)
            avg_quan_preds = np.zeros((n_samples, n_quantiles))

            for preds in all_fold_predictions:
                avg_median_preds += np.array(preds['median_predictions'])
                avg_mean_preds += np.array(preds['mean_predictions'])
                avg_prob_at_t += np.array(preds['prob_at_actual_time'])
                avg_quan_preds += np.array(preds['quantile_predictions'])

            # Compute averages
            avg_median_preds /= n_experiments
            avg_mean_preds /= n_experiments
            avg_prob_at_t /= n_experiments
            avg_quan_preds /= n_experiments

            aggregated_predictions = {
                'test_indices': all_fold_predictions[0]['test_indices'],
                'actual_times': all_fold_predictions[0]['actual_times'],
                'actual_events': all_fold_predictions[0]['actual_events'],
                'median_predictions': avg_median_preds.tolist(),
                'mean_predictions': avg_mean_preds.tolist(),
                'prob_at_actual_time': avg_prob_at_t.tolist(),
                'quantile_levels': all_fold_predictions[0]['quantile_levels'],
                'quantile_predictions': avg_quan_preds.tolist()
            }
        else:
            aggregated_predictions = None

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
        
        # Save CV Predictions
        cv_predictions_path = os.path.join(model_dir, "cv_predictions.json")
        with open(cv_predictions_path, 'w') as f:
            json.dump(aggregated_predictions, f, indent=2)
            
        # Also create a summary CSV for easy analysis
        cv_summary_path = os.path.join(model_dir, "cv_predictions_summary.csv")
        create_cv_summary_csv(aggregated_predictions, cv_summary_path)

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

        base_url = request.host_url.rstrip("/")

        response_data = {
            "status": "success",
            "model_id": new_model_id,
            "parent_model_id": model_id,
            "metrics": metrics,
            "selected_features": selected_features,
            "model_config": f"{base_url}/models/{model_id}/model_config.json",
            "model_file": {
                "encoder": f"{base_url}/models/{model_id}/encoder.joblib",
                "icp_state": f"{base_url}/models/{model_id}/icp_state.dill",
            },
            "trained_at": model_timestamp_date,
            "train_duration": train_duration,
            "retrained_from": model_id,
            "cv_predictions": {
                "summary_csv": f"{base_url}/models/{new_model_id}/cv_predictions_summary.csv",
                "full_predictions": f"{base_url}/models/{new_model_id}/cv_predictions.json",
                "n_folds": len(all_fold_predictions),
                "total_predictions": len(aggregated_predictions['actual_times'])
            },
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
    Make predictions using a trained model with conformal prediction intervals
    
    Expected JSON:
    {
        "model_id": "mtlr_20231101_120000_abc123",
        "features": {
            "feature1": value1,
            "feature2": value2,
            ...
        },
        "time_points": [0.5, 1, 1.5, 2, 2.5, 3],  # Optional: specific times for survival curve
        "alpha": 0.1  # Optional: significance level (default 0.1 for 90% prediction interval)
    }
    """
    try:
        data = request.json
        
        if 'model_id' not in data or 'features' not in data:
            return jsonify({'error': 'Missing model_id or features'}), 400
        
        model_id = data['model_id']
        model_dir = os.path.join(app.config['MODEL_FOLDER'], model_id)
        
        if not os.path.exists(model_dir):
            return jsonify({'error': f'Model {model_id} not found'}), 404
        
        # Load encoder
        encoder_path = os.path.join(model_dir, 'encoder.joblib')
        encoder = joblib.load(encoder_path)
        
        # Load ICP state (contains calibrated conformal scores)
        icp_state_path = os.path.join(model_dir, 'icp_state.dill')
        with open(icp_state_path, 'rb') as f:
            icp = dill.load(f)
        
        # Prepare input data as DataFrame
        input_df = pd.DataFrame([data['features']])
        
        # Transform using encoder
        input_transformed = encoder.transform(input_df)
        
        # Extract only feature columns
        feature_cols = [col for col in input_transformed.columns if col not in ['time', 'event']]
        x_input = input_transformed[feature_cols].values
        
        # Make prediction using ICP (conformal prediction)
        # This returns calibrated quantile predictions
        quan_levels, quan_preds = icp.predict(x_input)
        
        # quan_levels: array of quantile levels [0.01, 0.02, ..., 0.99]
        # quan_preds: shape (1, n_quantiles) - time at which each quantile is reached
        
        # Get time points for survival curve
        time_points = data.get('time_points', None)
        if time_points is None:
            # Create a grid of time points from 0 to max predicted time
            max_time = np.max(quan_preds[0])
            time_points = np.linspace(0, max_time, 100)
        else:
            time_points = np.array(time_points)
        
        # Convert quantile predictions to survival probabilities
        # The quantile predictions give us the full survival distribution
        survival_probs = convert_quantiles_to_survival(
            quan_levels, quan_preds[0], time_points
        )
        
        # Calculate median survival time (50th percentile)
        median_idx = np.argmin(np.abs(quan_levels - 0.5))
        median_survival = float(quan_preds[0, median_idx])
        
        # Calculate prediction intervals at specific confidence levels
        # For plotting uncertainty bands like in the image
        alpha = data.get('alpha', 0.1)  # 90% prediction interval by default
        
        # Get lower and upper bounds for survival curve
        # These come from the quantile predictions at different confidence levels
        lower_bound_survival = convert_quantiles_to_survival(
            quan_levels, quan_preds[0], time_points, bound='lower', alpha=alpha
        )
        upper_bound_survival = convert_quantiles_to_survival(
            quan_levels, quan_preds[0], time_points, bound='upper', alpha=alpha
        )
        
        # Format response
        predictions = {
            'survival_curve': {
                'time_points': time_points.tolist(),
                'survival_probabilities': survival_probs.tolist(),
                'lower_bound': lower_bound_survival.tolist(),
                'upper_bound': upper_bound_survival.tolist(),
                'confidence_level': 1 - alpha
            },
            'quantile_predictions': {
                'quantile_levels': quan_levels.tolist(),
                'time_points': quan_preds[0].tolist()
            },
            'key_statistics': {
                'median_survival_time': median_survival,
                '25th_percentile': float(quan_preds[0, np.argmin(np.abs(quan_levels - 0.25))]),
                '75th_percentile': float(quan_preds[0, np.argmin(np.abs(quan_levels - 0.75))])
            }
        }
        
        return jsonify({
            'status': 'success',
            'predictions': predictions,
            'model_id': model_id,
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__
        }), 500


def convert_quantiles_to_survival(quantile_levels, quantile_times, time_points, 
                                   bound='median', alpha=0.1):
    """
    Convert quantile predictions to survival probabilities with uncertainty bounds
    
    Parameters:
    -----------
    quantile_levels : array, shape (n_quantiles,)
        Quantile levels (e.g., [0.01, 0.02, ..., 0.99])
    quantile_times : array, shape (n_quantiles,)
        Time at which each quantile is reached
    time_points : array, shape (n_timepoints,)
        Time points at which to evaluate survival
    bound : str, one of ['lower', 'median', 'upper']
        Which bound of the prediction interval to return
    alpha : float
        Significance level for prediction intervals
        
    Returns:
    --------
    survival_probs : array, shape (n_timepoints,)
        Survival probability S(t) at each time point
    """
    survival_probs = np.ones(len(time_points))
    
    for i, t in enumerate(time_points):
        # Find the fraction of the distribution that has experienced the event by time t
        # This is the CDF: F(t) = P(T <= t)
        reached = quantile_times <= t
        
        if np.any(reached):
            # The CDF at time t is the largest quantile level reached
            cdf_at_t = np.max(quantile_levels[reached])
            
            # Survival function is complement of CDF: S(t) = 1 - F(t)
            survival_probs[i] = 1.0 - cdf_at_t
        else:
            # No events have occurred yet
            survival_probs[i] = 1.0
    
    return survival_probs


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



if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)