import os

# hide warnings
import warnings
warnings.filterwarnings('ignore')

# Data directory paths
raw_data_dir = os.path.relpath("../data/raw/")
interim_data_dir = os.path.relpath("../data/interim/")
processed_data_dir = os.path.relpath("../data/processed/")
submissions_dir = os.path.relpath("../data/submissions/")

# Models directory paths
models_dir = os.path.relpath("../models/")
