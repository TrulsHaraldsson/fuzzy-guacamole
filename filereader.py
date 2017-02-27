import csv

def readFile(filename):
    input_file_data = []
    with open(filename) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            input_file_data.append(row)
            
    input_file_data = input_file_data[1:] # Remove the first row consisting of what each element represent.
    input_labels = [] # One label for each row of data.
    input_data = [] # Rows of data consisting with index of label.
    for row in input_file_data:
        data = row[0]
        data = data.split(',')
        input_labels.append(data[0])
        input_data.append(data[3:])

    return input_labels, input_data # return the labels and the data in two lists.
