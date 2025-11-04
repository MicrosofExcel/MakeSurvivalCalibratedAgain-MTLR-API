# Make Survival Calibrated Again MTLR API with Inductive Conformal Prediction


## Introduction

This derived repository contains the code for The API for the MTLR model **with** inductive conformal prediction


## Overview

This repo **firstly** contains the MTLR API. This API currently only has a working /train endpoint and /models endpoint which trains and retrieves models. The rest will be implemented later.


This repository also contains the code for the experiments in the papers from the original repo. The code is organized as follows:
- `args/`: The arguments for the experiments.
- `CondCalEvaluation/`: The implementation of the conditional calibration score, $\text{Cal}_{\text{ws}}$.
- `data/`: The datasets used in the experiments.
- `figs/`: The figures presented in the papers.
- `icp/`: The implementation of inductive conformal prediction (ICP) for survival analysis. It contains two methods: **CSD** (class name: `ConformalSurvDist`) and **CiPOT** (class name: `CSDiPOT`).
- `models/`: Some self-implemented models (`DeepSurv`/`CoxPH`, `N-MTLR`/`MTLR`, `CQRNN`, `LogNormalNN`) used in the experiments.
- `posters/`: The poster presentations for both methods.
- `utils/`: The utility functions used in the experiments.
- `requirements.txt`: The requirements file to install the necessary packages.
- `README.md`: This file.
- `run.py`: The main script to run `CSD` or `CiPOT`.
- `run_baselines.py`: The script to run the non post-processed baselines.
- `run_sh`: The script to run the experiments in one go.

However, you can ignore a majority of everything above because this repo is primarily for using the MTLR-focused API


## Getting Started


### Prerequisites

Clone the repo, create a virtual environment with Python version >= 3.10.12 

```
python -m venv venv
```
#### Activate it:  

Windows (PowerShell):
```
.\venv\Scripts\Activate.ps1
```
macOS/Linux:
```
source venv/bin/activate
```
Then install the requirements.txt file
```
pip install -r requirements.txt
```


### Using the API

To run the server:
```bash
python app.py
```
Normally, the server is set to run on
http://localhost:5000

#### Configure
At the bottom of app.py, you can freely modify the port. Just be sure to reference that same port when accessing the URL of this API.

### API input

The API expects input in this form:
```
response = requests.post(
                f'{api_url}/train',
                files=files,
                data=data,
                timeout=600  # 10 minute timeout for training
            )
```
where files is
```
 with open(dataset_path, 'rb') as f:
        files = {
            'dataset': (os.path.basename(dataset_path), f, 'text/csv')
        }
```
and data is
```
data = {
        'parameters': 
                'neurons': [64,64],
                'dropout': 0.1,
                'seed': 0,
                'n_quantiles': 10,
                'lr': 1e-3,
                'batch_size': 256,
                'n_epochs': 500,
                'weight_decay': 1e-4 ,
                'n_exp': 10,
                 ....

}
```
The timeout will be based off celery for task processing which we will discuss later.

### Testing the API

I have included two default starting csv datasets that are formatted according to PSSP dataset requirements:
1. AML.csv
2. Breast_Cancer.csv

Utilizing any other dataset will not work unless you import your own correctly formatted dataset into /data and use that instead.

Please refer to the following link on dataset file format:
http://pssp.srv.ualberta.ca/home/file_format

To test the /train endpoint of the API:
```
python test_api.py
```
Which is set to run on AML.csv by default. To change that, scroll to the bottom of the test_api.py file and locate
```
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "data", "AML.csv")
```
Then switch out AML.csv for Breast_Cancer.csv or your own dataset file.

## Reference
[1] Shi-ang Qi, Yakun Yu, Russell Greiner. Conformalized Survival Distributions: A Generic Post-Process to Increase Calibration. ICML 2024. [[paper](https://proceedings.mlr.press/v235/qi24a.html)]

[2] Shi-ang Qi, Yakun Yu, Russell Greiner. Toward Conditional Distribution Calibration in Survival Prediction. NeurIPS 2024. [[paper]([https://openreview.net/forum?id=l8XnqbQYBK](https://proceedings.neurips.cc/paper_files/paper/2024/hash/9c8df8de46c1a1b39b30b9f74be69c02-Abstract-Conference.html))]

## Citation

We recommend you use the following to cite `CSD` method in your publications:

```
@InProceedings{pmlr-v235-qi24a,
  title =        {Conformalized Survival Distributions: A Generic Post-Process to Increase Calibration},
  author =       {Qi, Shi-Ang and Yu, Yakun and Greiner, Russell},
  booktitle = 	 {Proceedings of the 41st International Conference on Machine Learning},
  pages = 	     {41303--41339},
  year = 	       {2024},
  volume = 	     {235},
  series = 	     {Proceedings of Machine Learning Research},
  month = 	     {21--27 Jul},
  publisher =    {PMLR},
  url = 	       {https://proceedings.mlr.press/v235/qi24a.html},
}
```

We recommend you use the following to cite `CiPOT` method in your publications:
```
@inproceedings{qi2024toward,
 author =        {Qi, Shi-ang and Yu, Yakun and Greiner, Russell},
 booktitle =     {Advances in Neural Information Processing Systems},
 pages =         {86180--86225},
 publisher =     {Curran Associates, Inc.},
 title =         {Toward Conditional Distribution Calibration in Survival Prediction},
 url =           {https://proceedings.neurips.cc/paper_files/paper/2024/file/9c8df8de46c1a1b39b30b9f74be69c02-Paper-Conference.pdf},
 volume =        {37},
 year =          {2024}
}
```
