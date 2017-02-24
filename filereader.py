import csv
input_file_data = []

with open('DSL-StrongPasswordData.csv') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        #print ", ".join(row)
        input_file_data.append(row)
        
input_file_data = input_file_data[1:] # Remove the first row consisting of what each element represent.
input_labels = [] # One label for each row of data.
input_data = [] # Rows of data consisting with index of label.
for row in input_file_data:
    data = row[0]
    data = data.split(',')
    input_labels.append(data[0])
    input_data.append(data[3:])

print "label : ", input_labels[0], "\n data : ", input_data[0] # Showing test for first index of label and data.