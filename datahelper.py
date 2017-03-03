import csv


class FileReader(object):
    def __init__(self, filename_):
        self.filename = filename_


class CSVFileReader(FileReader):
    def __init__(self, filename_):
        super(CSVFileReader, self).__init__(filename_)

    def read_column_names(self):
        """
        Read the first row in the csv file which is assumed to contain the labels for each column.
        :return: A list of labels for each column.
        """
        with open(self.filename) as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_ALL) # Quote all since they are labels
            column_labels = next(reader)  # Read first row
        return column_labels

    def read_data(self):
        """
        Read the whole data set except the labels.
        :return: A list with rows.
        """
        with open(self.filename) as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
            data_rows = []
            for row in reader:
                data_rows.append(row)
            return data_rows[1:]  # Remove the first row consisting of labels


class DataManager(object):
    def __init__(self):
        self.subjects = []  # Contains a list of subjects with their associated labels and data
        self.number_of_subjects = 50
        self.number_of_repetitions = 400

    def get_subject(self, index):
        return self.subjects[index]

    def pre_process(self, filename):
        csv_reader = CSVFileReader(filename)
        data = csv_reader.read_data()
        for i in range(self.number_of_subjects):
            input_labels = []  # One label for each row of data.
            input_data = []  # Rows of data consisting with index of label.
            for j in range(self.number_of_repetitions):
                row = data[i * self.number_of_repetitions + j]
                input_labels.append(row[0])
                input_data.append([float(k) for k in row[3:]])
            self.subjects.append((input_labels, input_data))