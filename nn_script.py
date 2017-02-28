import filereader

# Just some constants
LABELS = 0
DATA = 1

result = filereader.readFile("DSL-StrongPasswordData.csv")

labels = result[LABELS]
data = result[DATA]


