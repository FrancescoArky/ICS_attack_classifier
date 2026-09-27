import pandas as pd

df = pd.read_csv("./ICS_attack_classifier/data/train_test_network.csv", sep=",")

features = [
    "src_port",
    "dst_port",
    "proto",
    "service",
    "duration"
]
