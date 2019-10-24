Kaggle: IEEE-CIS Vesta e-commerce fraud detection
==============================


This is a solution to the Kaggle competetion problem [IEEE-CIS Vesta Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/). The objective was to detect e-commerce fraud using transactional & web-analytics data. This involved data-cleaning, exploratory analysis, feature generation and predictive modeling. This repository contains the code, project details and some samples of exploratory analysis that it entailed. For details, see the [project notebook](./notebooks/vesta_ieee_cis_ecommerce_fraud_detection.ipynb).


Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to download or generate data
        │   └── make_dataset.py
        │
        ├── features       <- Scripts to turn raw data into features for modeling
        │   └── build_features.py
        │
        ├── models         <- Scripts to train models and then use trained models to make
        │   │                 predictions
        │   ├── predict_model.py
        │   └── train_model.py
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations
            └── visualize.py
  

Installation & Use
------------

1) Download or clone the project and `cd` to the project directory.

2) [Optional but Recommended] Create a virtual environment for the project. 

    ```python3.7 -m venv my_env```

3) [Required] Install dependencies

    ```pip install -r requirements.txt```

4) [Required] Install current package

    ```pip install .```

5) Sign up and download the datasets from [competetion website](https://www.kaggle.com/c/ieee-fraud-detection/) to `./data/raw/`.


6) Fire up a Jupter notebook: `(my_env)$ jupyter-notebook`. To see project details and run the code, open `./notebooks/vesta_ieee_cis_ecommerce_fraud_detection.ipynb`. 

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
