from components.make_dataframes import make_dataframes
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MaxNLocator

FIRST_FOLDER = "ProcessedData"
REFIT_FOLDER = "NewData"

sns.set(style="whitegrid", palette="muted", font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams["figure.figsize"] = 16, 10
pl.seed_everything(42)


def fitFirst():
    training_data, training_labels, test_data, test_labels = make_dataframes(FIRST_FOLDER)


def refit():
    training_data, training_labels, test_data, test_labels = make_dataframes(REFIT_FOLDER)


fitFirst()
