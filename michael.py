import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_mutual_info_score, adjusted_rand_score

import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset

from starter import read_data, inplace_min_max_scaling, kmeans_helper

def format_data_for_clustering(data_list):
    cluster_data = [row[1].tolist() for row in data_list]
    true_labels = [row[0][0] for row in data_list]
    return cluster_data, true_labels

# Function to plot and save embeddings with unique colors for each label
def plot_and_save_embedding(embedding, labels, title, filepath, color_map=plt.colormaps.get_cmap("tab10")):
    # Convert labels to integers
    labels = np.array(labels, dtype=int)

    unique_labels = np.unique(labels)  # Get unique labels for the legend
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap=color_map, s=10, alpha=0.7)
    
    # Add legend with unique labels
    handles, _ = scatter.legend_elements()
    plt.legend(handles, unique_labels, title="Digits", loc="best", markerscale=2)

    plt.title(title)
    plt.colorbar(scatter, ticks=unique_labels, label="Digit")
    plt.savefig(filepath)
    plt.close()

def save_2d_embeddings(data, true_labels, subdir=''):
    # Directory to save embeddings
    embedding_dir = "embeddings"
    os.makedirs(os.path.join(embedding_dir, subdir), exist_ok=True)

    data = np.array(data)

    # PCA embedding
    pca = PCA(n_components=2)
    pca_embedding = pca.fit_transform(data)
    plot_and_save_embedding(pca_embedding, [int(l) for l in true_labels], "PCA Embedding", os.path.join(embedding_dir, subdir, "pca_embedding.png"))

    # UMAP embedding
    umap_model = umap.UMAP(n_components=2, random_state=42)
    umap_embedding = umap_model.fit_transform(data)
    plot_and_save_embedding(umap_embedding, [int(l) for l in true_labels], "UMAP Embedding", os.path.join(embedding_dir, subdir, "umap_embedding.png"))

    # t-SNE embedding
    tsne = TSNE(n_components=2, random_state=42)
    tsne_embedding = tsne.fit_transform(data)
    plot_and_save_embedding(tsne_embedding, [int(l) for l in true_labels], "t-SNE Embedding", os.path.join(embedding_dir, subdir, "tsne_embedding.png"))

    print(f"2D embeddings saved in {os.path.join(embedding_dir, subdir)}")

def compare_kmeans_implementations(data_list, metric, k=10, subdir=''):
    # Extract data for clustering and true labels
    cluster_data, true_labels = format_data_for_clustering(data_list)
    data = np.array(cluster_data)

    # Directory to save cluster center images
    output_dir = "optimal_cluster_centers"
    os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    print('CUSTOM')
    print('-'*5)

    # Custom KMeans clustering on original data
    cluster_memberships = kmeans_helper(cluster_data, metric, k=k)
    # X = [point for cluster in clusters for point in cluster]
    labels = cluster_memberships
    # for i, cluster in enumerate(clusters):
    #     labels.extend([i] * len(cluster))

    # for point in cluster_data:
    #     for i in range(k):
    #         if any(np.array_equal(point, cpoint) for cpoint in clusters):
    #             labels.append(i)
    #             break

    clusters = [[] for _ in range(k)]

    for data_point_idx, cluster in enumerate(cluster_memberships):
        clusters[cluster].append(cluster_data[data_point_idx])

    # Silhouette score for custom KMeans
    custom_score = silhouette_score(data, labels)
    cami = adjusted_mutual_info_score(true_labels, labels)
    cari = adjusted_rand_score(true_labels, labels)
    print(f"Silhouette Score with k={k} (custom KMeans):", custom_score)
    print("Adjusted Mutual Information (AMI):", cami)
    print("Adjusted Rand Index (ARI):", cari)

    print()

    print('SKLEARN')
    print('-'*5)
    
    # sklearn KMeans for comparison
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(data)
    sklearn_labels = kmeans.labels_

    # sklearn KMeans metrics
    silhouette_avg = silhouette_score(data, sklearn_labels)
    ami = adjusted_mutual_info_score(true_labels, sklearn_labels)
    ari = adjusted_rand_score(true_labels, sklearn_labels)

    print(f"Silhouette Score (sklearn KMeans) with k=10:", silhouette_avg)
    print("Adjusted Mutual Information (AMI):", ami)
    print("Adjusted Rand Index (ARI):", ari)

    # Compute average images for custom KMeans clusters
    avg_images_custom = []
    for cluster in clusters:
        avg_image = np.mean(cluster, axis=0)  # Calculate the mean image for each cluster
        avg_images_custom.append(avg_image.reshape(28, 28))  # Adjust shape if not 28x28

    # Plot all custom KMeans cluster average images in a single figure
    fig_custom, axes_custom = plt.subplots(1, k, figsize=(15, 5))
    fig_custom.suptitle("Custom KMeans Cluster Centers")

    for c_i, ax in enumerate(axes_custom):
        # Display each average image
        ax.imshow(avg_images_custom[c_i], cmap='gray')
        ax.axis('off')

    # Save the figure for custom KMeans
    fig_custom.savefig(os.path.join(output_dir, subdir, "custom_kmeans_cluster_centers.png"))
    plt.close(fig_custom)

    # Plot all sklearn KMeans cluster centers in a single figure
    fig_sklearn, axes_sklearn = plt.subplots(1, k, figsize=(15, 5))
    fig_sklearn.suptitle("Sklearn KMeans Cluster Centers")

    for c_i, ax in enumerate(axes_sklearn):
        # Reshape and plot each sklearn KMeans cluster center
        image_sklearn = kmeans.cluster_centers_[c_i].reshape(28, 28)  # Adjust shape if not 28x28
        ax.imshow(image_sklearn, cmap='gray')
        ax.axis('off')

    # Save the figure for sklearn KMeans
    fig_sklearn.savefig(os.path.join(output_dir, subdir, "sklearn_kmeans_cluster_centers.png"))
    plt.close(fig_sklearn)

    print(f"Cluster center figures saved in {os.path.join(output_dir, subdir)}")

if __name__ == '__main__':

    metric = 'cosim'

    # HW dataset
    print('HW dataset')
    print('-'*10)

    # show(filename, 'pixels')
    train_data = read_data('mnist_train.csv')
    valid_data = read_data('mnist_valid.csv')
    test_data = read_data('mnist_test.csv')

    inplace_min_max_scaling(train_data)
    inplace_min_max_scaling(test_data)
    inplace_min_max_scaling(valid_data)

    compare_kmeans_implementations(train_data, metric, subdir='hw')
    save_2d_embeddings(*format_data_for_clustering(train_data), 'hw')

    print()
    print()

    # FULL dataset (PyTorch)
    print('FULL dataset (PyTorch)')
    print('-'*10)

    # Load the MNIST dataset
    transform = transforms.ToTensor()
    train_data_mnist = datasets.MNIST(root="mnist_data", train=True, download=True, transform=transform)

    subset = 5000
    subset_indices = list(range(subset))
    train_subset = Subset(train_data_mnist, subset_indices)

    # Convert to desired format
    mnist_list = []
    for img, label in train_subset:
        # Flatten the image tensor and convert it to a list of pixel values
        pixels = img.view(-1).tolist()
        mnist_list.append([[label], pixels])

    inplace_min_max_scaling(mnist_list)

    compare_kmeans_implementations(mnist_list, metric, subdir=f'mnist_{str(subset)}')
    save_2d_embeddings(*format_data_for_clustering(mnist_list), f'mnist_{str(subset)}')
