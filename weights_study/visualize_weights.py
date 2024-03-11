import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

def imshow(img):
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def visualize_linear_layer_weights(layer, cmap='viridis'):
    # Check if the layer is indeed a linear layer
    if isinstance(layer, torch.nn.Linear):
        weights = layer.weight.data.numpy()
        n_output_features = weights.shape[0]
        n_rows = 2
        n_cols = int(np.ceil(n_output_features / n_rows))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
        for i in range(n_rows):
            for j in range(n_cols):
                ax = axes[i, j]
                feature_index = i * n_cols + j
                if feature_index < n_output_features:
                    ax.imshow(weights[feature_index].reshape(1, -1), cmap=cmap, aspect='auto')
                    ax.axis('off')
                else:
                    ax.axis('off')  # Hide axis if no more features to show
        plt.tight_layout()
        plt.show()
    else:
        print("The provided layer is not a linear layer.")

def visualize_conv_layer_weights(layer, cmap='viridis'):
    if isinstance(layer, torch.nn.Conv2d):
        weights = layer.weight.data.numpy()
        n_filters = weights.shape[0]
        n_rows = 2
        n_cols = int(np.ceil(n_filters / n_rows))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
        for i in range(n_rows):
            for j in range(n_cols):
                ax = axes[i, j]
                filter_index = i * n_cols + j
                if filter_index < n_filters:
                    ax.imshow(weights[filter_index][0], cmap=cmap)
                    ax.axis('off')
                else:
                    ax.axis('off')  # Hide axis if no more filters to show
        plt.show()
    else:
        print("The provided layer is not a convolutional layer.")