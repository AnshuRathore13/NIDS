import argparse
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from src.model import MLPModel
from src.preprocessing import Preprocessor, balanced_sample_weight


def train(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    print(f"Loaded config: {cfg['project']['name']}")
    model_params = cfg["model"]["params"]
    model = MLPModel(model_params)
    print("Initialized Deep MLP Classifier:", model_params["hidden_layer_sizes"])
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    train(args.config)
