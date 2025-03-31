import torch
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as trans
import time
import os

torch.cuda.is_available
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

