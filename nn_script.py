import filereader
from NearestNeighbor import *
import numpy as np

# Just some constants
LABELS = 0
DATA = 1
filename = "DSL-StrongPasswordData.csv"

# Read file
result = filereader.readFile(filename)

# Extract labels and data
labels = np.array(result[LABELS][:800])
data = np.array(result[DATA][:800])

# Shuffle the data
rng_state = np.random.get_state()
np.random.shuffle(labels)
np.random.set_state(rng_state)
np.random.shuffle(data)

# Train NN
nn = NearestNeighbor()
print nn.cross_validation(data, labels, 4)

# Test






