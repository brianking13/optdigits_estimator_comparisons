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
    mean_value = np.zeros(shape=(np.shape(data)[0], 1))
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
    z = np.matmul(np.transpose(W),(x - m))
    return z, W


def eul_distance(data):
    add_val = data[0]**2+data[1]**2
    dist = np.sqrt(add_val)
    return dist


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


# kNN
K = 42
accuracy = 0
for i in range(np.shape(z2)[1]):
    if i%20 == 0:
        print("Calculating: " + str("{:.2f}".format(i/np.shape(z2)[1]*100))+"%")
    x = z2[:, i]
    distances = []
    numbers = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
    for j in range(np.shape(z)[1]):
        y = z[:,j]
        dist = y-x
        ptp_dist = eul_distance(dist)   # point-to-point distance
        distances.append(ptp_dist)
        numbers[train_labels[j]].append(ptp_dist)
    distances.sort()
    top = []

    # check which value in a certain key has the same value as one of the smallest distances
    for k in range(K):
        for key, value in numbers.items():
            for item in value:
                if item == distances[k]:
                    top.append(key)
    guess = mode(top)
    if guess == test_labels[i]:
        accuracy = accuracy + 1

print(str("{:.2f}".format(accuracy/np.shape(z2)[1]*100))+"%")


