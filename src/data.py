##
## PROJECT PRO, 2025
## TieBreaker
## File description:
## data
##

import sys
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import os
import joblib
import time

TRAINING_FILE = "training.pkl"
model = None

def train_model() -> None:
    global model


def load_training_data() -> None:
    global model
    if os.path.exists(TRAINING_FILE):
        model = joblib.load(TRAINING_FILE)
    return model