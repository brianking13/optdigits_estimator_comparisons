from csv import reader
import numpy as np
from statistics import mode


# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    dataset = [[float(i) for i in j] for j in dataset]
    return dataset


def split_labels(data):
    remove_label = []
    labels = []
    for i in range(0, len(data)):
        remove_label.append(data[i][0:64])
        labels.append(data[i][64])
    data_no_label = np.array(remove_label)
    return data_no_label, labels


def mean(data):
    mean_value = np.zeros(shape = (np.shape(data)[0], 1))
    for i in range(0, np.shape(data)[0]):
        for j in range(0, np.shape(data)[1]):
            mean_value[i] = mean_value[i] + (data[i][j])
        mean_value[i] = mean_value[i] / np.shape(data)[1]
    return mean_value


def eigenvectors(data, m):
    covariance_matrix = np.cov(np.transpose(data))
    eig_values, eig_vectors = np.linalg.eig(covariance_matrix)
    W = eig_vectors[:, :2]
    x = np.transpose(data)
    z = np.matmul(np.transpose(W), (x - m))
    return z, W


# Import test data
test_data = load_csv('optdigits.tes')
training_data = load_csv('optdigits.tra')

# Create numpy arrays with all data points
training_data_no_label, train_labels = split_labels(training_data)
testing_data_no_label, test_labels = split_labels(test_data)

mu = mean(np.transpose(training_data_no_label))

# Find Eigenvectors of training data
z, W = eigenvectors(training_data_no_label, mu)

# Find Eigenvectors of test data
mu2 = mean(np.transpose(testing_data_no_label))
x = np.transpose(testing_data_no_label)
z2 = np.matmul(np.transpose(W),(x - mu2))

# Naive Estimator
h = 4
accuracy = 0
for i in range(len(test_labels)):
    if i % 50 == 0:
        print(str("{:.2f}".format(i/len(test_labels)*100))+"% complete")
    labels = []
    for j in range(len(train_labels)):
        if abs(z2[1, i] - z[1, j]) / h < 0.5 and abs(z2[0, i] - z[0, j]) / h < 0.5:
            labels.append(train_labels[j])
    try:
        if test_labels[i] == mode(labels):
            accuracy = accuracy + 1
    except:
        accuracy = accuracy

accuracy = accuracy/len(test_labels)
print("Accuracy: " + str("{:.2f}".format(accuracy*100)) + "%")



