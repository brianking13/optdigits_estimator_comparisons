from csv import reader
import numpy as np


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


def sort_data(data,z):
    sorted_data = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
    for i in range(len(data)):
        label = data[i][64]
        for j in range(10):
            if label == j:
                sorted_data[j].append(z[:,i])
    return sorted_data


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


def gaussian_kernel(x, x_i, d, h):
    u = (x-x_i)/h
    K = 1/(np.sqrt(2 * np.pi) ** d) * np.exp(-(np.sqrt(u[0]**2+u[1]**2)) ** 2 / 2)
    return K


def kernel_estimator(test, train, d, h):
    K_sum = 0
    for i in range(len(train)):
        K = gaussian_kernel(test, train[i], d, h)
        K_sum = K_sum + K
    prob = K_sum / (len(test)*h**d)
    return prob

def probability(test, train, d, h):
    probabilities = []
    for i in range(10):
        probabilities.append(kernel_estimator(test,train[i],d,h))
    max_value = max(probabilities)
    max_number = probabilities.index(max_value)
    return max_number


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

h = 1 #bin size
d = 10 #classes


# g = kernel_estimator(x,x_i,d,h)
sorted = sort_data(training_data,z) # dictionary with number as key and 2 dimensional arrays as values

accuracy = 0
for i in range(np.shape(z2)[1]):
    print("Calculating: " + str("{:.2f}".format(i/np.shape(z2)[1]*100))+"%")
    result = probability(z2[:,i],sorted,d,h)
    if result == test_data[i][64]:
        accuracy = accuracy + 1
accuracy = accuracy/np.shape(z2)[1]*100
print(accuracy)
