import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

####################### part 1 #######################
# Read the CSV file into a DataFrame
df = pd.read_csv('Wine.csv')

# Split the DataFrame into 3 types ,and sample 20 from each of them 
type1 = df.iloc[2:176]
type2 = df.iloc[177:381]
type3 = df.iloc[382:]
type1_sample = type1.sample(n=20)
type2_sample = type2.sample(n=20)
type3_sample = type3.sample(n=20)

# Write test.csv & train.csv files respectively
sampled_df = pd.concat([type1_sample, type2_sample, type3_sample])
sampled_df.to_csv('test.csv', index=False)
df = df.drop(sampled_df.index)
df.to_csv('train.csv', index=False)

####################### part 2 #######################
# Load the training and test data and convert to np arrays for training
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
x_train, y_train = train_df.iloc[:, 1:].values, train_df.iloc[:, 0].values # values of features, target
x_test, y_test = test_df.iloc[:, 1:].values, test_df.iloc[:, 0].values 
#print(y_train)

# Get labels and calculate the priors, means ,and stds
def fit(x, y): # "x" for x_train, "y" for y_train
    # Priors
    labels = np.unique(y)
    n_samples = len(y)
    priors = {}
    for label in labels:
        priors[label] = np.sum(y == label) / n_samples

    # Mean & stds
    parameters = {}
    for label in labels:
        label_data = x[y == label]
        means = np.mean(label_data, axis=0)
        stds = np.std(label_data, axis=0)
        parameters[label] = list(zip(means, stds))

    return labels, priors, parameters

# Make predictions on the test data
def predict(x, labels, priors, parameters):
    y_pred = []
    for data in x:
        posteriors = []
        for label in labels:
            likelihoods = []
            for i, (mean, sigma) in enumerate(parameters[label]):
                likelihoods.append(likelihood(data[i], mean, sigma))
            likelihood_product = np.product(likelihoods)
            posterior = priors[label] * likelihood_product
            posteriors.append(posterior)
        posterior_sum = np.sum(posteriors)
        posteriors /= posterior_sum
        label_idx = np.argmax(posteriors)
        y_pred.append(labels[label_idx])
    return np.array(y_pred)

def likelihood(x, mean, sigma):
    return np.exp(-((x - mean) ** 2 / (2 * sigma ** 2))) / (sigma * np.sqrt(2 * np.pi))

labels, priors, parameters = fit(x_train, y_train)
y_pred = predict(x_test, labels, priors, parameters)

#  Calculate the accuracy
correct_predictions = 0
for i in range(len(y_test)):
    if y_pred[i] == y_test[i]:
        correct_predictions += 1
accuracy = correct_predictions / len(y_test)
print(f'Accuracy rate: {accuracy:.2%}')

####################### part 3 #######################
# Load the data and preprocess it with standardization ,then use PCA
def plot_scatter(x, y, title):
    labels = np.unique(y) 
    colors = ['r', 'g', 'b']
    # Standardize the features
    x_std = StandardScaler().fit_transform(x)

    # Perform PCA to reduce the dimensionality
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x_std)

    # Plot the scatter plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    for label, color in zip(labels, colors):
        ax.scatter(x_pca[y==label, 0], x_pca[y==label, 1], color=color, label=label)
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')

    ax.set_title(title)
    ax.legend()
    plt.show()

plot_scatter(x_test, y_pred, 'Scatter plot of test data')
# plot_scatter(x_test, y_pred, 'Scatter plot of test data')

