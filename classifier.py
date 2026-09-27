import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("./data/train_test_network.csv", sep=",")

features = [
    "src_port",
    "dst_port",
    "proto",
    "service",
    "duration"
]
