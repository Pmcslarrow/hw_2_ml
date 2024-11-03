import matplotlib.pyplot as plt
import numpy as np

from starter import read_data, inplace_min_max_scaling, accuracy, kmeans

train_data = read_data('mnist_train.csv')
valid_data = read_data('mnist_valid.csv')
test_data = read_data('mnist_test.csv')

inplace_min_max_scaling(train_data)
inplace_min_max_scaling(test_data)
inplace_min_max_scaling(valid_data)

metrics = ['euclidean', 'cosim']

k_range = list(range(2, 11))
valid_accs = {metric: [] for metric in metrics}
train_accs = {metric: [] for metric in metrics}

for metric in metrics:
    for k in k_range:
        print()
        valid_accs[metric].append(accuracy(*kmeans(train_data, metric, k=k)))
        train_accs[metric].append(accuracy(*kmeans(train_data, metric, k=k)))
        print()

colors = ['red', 'blue']

plt.figure(figsize=(10, 6))

for idx, metric in enumerate(metrics):
    plt.plot(k_range, train_accs[metric], label=f"{metric} train", color=colors[idx], linestyle="-")

    plt.plot(k_range, valid_accs[metric], label=f"{metric} valid", color=colors[idx], linestyle="--")

    plt.xticks(k_range)

    plt.title('K-means Accuracy Over a Range of k Values')
    plt.xlabel('k')
    plt.ylabel('accuracy score')

    plt.legend()

plt.show()

best_k = {'euclidean': 6, 'cosim': 7}

for metric in metrics:
    print()
    print(f'TEST SET ACCURACY FOR {metric} WITH K={best_k[metric]}:', accuracy(*kmeans(train_data, metric, k=best_k[metric])))
    print()