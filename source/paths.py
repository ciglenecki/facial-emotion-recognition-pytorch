# Paths
from pathlib import Path, PurePath
import random
from config import *
PATH_PROJECT = Path.cwd()

# DATASET ########################################################################
PATH_DATASET = Path(PATH_PROJECT, "dataset")
PATH_DATASET_CK = Path(PATH_DATASET, "ck+")

PATH_CK_EMOTIONS = Path(PATH_DATASET_CK, "emotions")
PATH_CK_FACS = Path(PATH_DATASET_CK, "facs")
PATH_CK_IMAGES = Path(PATH_DATASET_CK, "images")
PATH_CK_LANDMARKS = Path(PATH_DATASET_CK, "landmarks")

FILEPATHS_CK_EMOTIONS = PATH_CK_EMOTIONS.glob("*/*/*")
FILEPATHS_CK_FACS = PATH_CK_FACS.glob("*/*/*")
FILEPATHS_CK_IMAGES = PATH_CK_IMAGES.glob("*/*/*.png")
FILEPATHS_CK_LANDMARKS = PATH_CK_LANDMARKS.glob("*/*/*")

PATH_DATASET_GOOGLE = Path(PATH_DATASET, "google")


# NUMPY ########################################################################
PATH_NUMPY_CK = Path(PATH_DATASET, "numpy_ck")
PATH_NUMPY_CK_TEST = Path(PATH_NUMPY_CK, "test")

PATH_NUMPY_GOOGLE = Path(PATH_DATASET, "numpy_google")
PATH_NUMPY_GOOGLE_TEST = Path(PATH_NUMPY_GOOGLE, "test")

FILEPATHS_NUMPY_CK = list(PATH_NUMPY_CK.glob("*.npy"))
FILEPATHS_NUMPY_CK_TEST = list(PATH_NUMPY_CK_TEST.glob("*.npy"))

FILEPATHS_NUMPY_GOOGLE = list((PATH_NUMPY_GOOGLE.glob("*.npy")))
FILEPATHS_NUMPY_GOOGLE_TEST = list((PATH_NUMPY_GOOGLE_TEST.glob("*.npy")))

FILEPATHS_NUMPY = random.sample(FILEPATHS_NUMPY_GOOGLE, int(GOOGLE_TRAIN_SPLIT*len(FILEPATHS_NUMPY_GOOGLE))) + random.sample(FILEPATHS_NUMPY_CK, int(CK_TRAIN_SPLIT*len(FILEPATHS_NUMPY_CK)))

FILEPATHS_NUMPY_TEST = FILEPATHS_NUMPY_GOOGLE_TEST + FILEPATHS_NUMPY_CK_TEST

# MODELS ########################################################################
PATH_MODELS = Path(PATH_PROJECT, "models")
PATH_MODELS_TMP = Path(PATH_MODELS, "tmp")

MODEL_NAME = "resnet50"
MODEL_NAME_TRAINED = MODEL_NAME+"_trained"

PATH_MODEL = Path(PATH_MODELS, MODEL_NAME+".pth")
PATH_MODEL_STATE = Path(PATH_MODELS, MODEL_NAME_TRAINED)
PATH_MODEL_TMP = Path(PATH_MODELS_TMP, MODEL_NAME_TRAINED)
# VISUALS ########################################################################
PATH_VISUALS = Path(PATH_PROJECT, "visuals")

# LOGS ########################################################################
PATH_LOGS = Path(PATH_PROJECT, "logs")
TOTAL_EMO_CK = Path(PATH_LOGS, "total_emo_ck.txt")
TOTAL_EMO_GOOGLE = Path(PATH_LOGS, "total_emo_google.txt")
# SOURCE
PATH_SOURCE = Path(PATH_PROJECT, "source")
