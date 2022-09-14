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

# Create histogram
h = 4
dimension = {}
max = 40
max =int( round(max/h)*h*2)
step = int(max/h+1)
for i in range(len(z)):
    dimension[i] = np.linspace(0,max,step)-max/2

bins = np.array([[0,0]])
for i in range(np.shape(dimension[0])[0]):
    for j in range(np.shape(dimension[1])[0]):
        row_values = [dimension[0][i],dimension[1][j]]
        bins = np.vstack([bins, row_values])
bins = bins[1:,:]
bins = np.transpose(bins)


# Create Dictionaries to store training data and labels bins
in_bin2 ={}
for i in range(np.shape(bins)[1]):
    in_bin2[str("{:.2f}".format(bins[0][i]))+','+str("{:.2f}".format(bins[1][i]))] = []
in_bin = {k: [] for k in (range(np.shape(dimension[0])[0]**2))}
in_bin_value = {k: [] for k in (range(np.shape(dimension[0])[0]**2))}


# Check which bins training values fall into
for i in range(0,np.shape(z)[1]):
    number1 = round(z[0][i]/h)*h
    number2 = round(z[1][i]/h)*h
    in_bin2[str("{:.2f}".format(float(number1)))+','+str("{:.2f}".format(float(number2)))].append(train_labels[i])


# Find 2D of test data
mu2 = mean(np.transpose(testing_data_no_label))
x = np.transpose(testing_data_no_label)
z2 = np.matmul(np.transpose(W),(x - mu2))


# Predicting which bin test data falls into
accuracy = 0
for i in range(0,np.shape(z2)[1]):
    number1 = "{:.2f}".format(round(z2[0][i]/h)*h)
    number2 = "{:.2f}".format(round(z2[1][i]/h)*h)
    try:
        most_frequent = mode(in_bin2[str("{:.2f}".format(float(number1)))+','+str("{:.2f}".format(float(number2)))])
        if most_frequent == test_labels[i]:
            accuracy = accuracy + 1
    except:
        accuracy = accuracy

accuracy = accuracy/len(test_labels)
print("Accuracy: " + str("{:.2f}".format(accuracy*100)) + "%")
