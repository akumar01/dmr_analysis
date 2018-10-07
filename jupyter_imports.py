%load_ext autoreload
%autoreload 2

from pynwb import NWBHDF5IO
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.io import loadmat, savemat
from math import floor, ceil
# Script to add the entire project directory structure to the python path
import sys, os

# Find the root directory of the nse project
parent_path, current_dir = os.path.split(os.path.abspath('.'))
while current_dir != 'analysis':
	parent_path, current_dir = os.path.split(parent_path)
p = os.path.join(parent_path, current_dir)
# Add analysis
if p not in sys.path:
	sys.path.append(p)

# And standard list of subdirectories
for d in ["linear", "prelim", "utils"]:
	if 'p\\%s' % d not in sys.path:
		sys.path.append('p\\%s' % d)

# Now import all the other stuff:
from utils import preprocess
from utils import process
from utils import misc 