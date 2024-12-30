import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
from keras.datasets import mnist
import numpy as np
from openTSNE import TSNE, TSNEEmbedding, affinity, initialization

# Load MNIST data
(x_train, y_train), _ = mnist.load_data()

# Preprocess MNIST data
X = np.zeros((x_train.shape[0], 784))
for i in range(x_train.shape[0]):
    X[i] = x_train[i].flatten()
X = pd.DataFrame(X)
Y = pd.DataFrame(y_train)

# Shuffle dataset and take random 20% for visualization with t-SNE
X_sample = X.sample(frac=0.2, random_state=12).reset_index(drop=True)
Y_sample = Y.sample(frac=0.2, random_state=12).reset_index(drop=True)
X_sample['label'] = Y_sample

# Step 1: Define affinities
affinities = affinity.PerplexityBasedNN(
    X_sample.drop(columns='label').to_numpy(),
    n_jobs=-1,
    random_state=12,
    verbose=True
)

# Step 2: Define initial embedding
init = initialization.pca(X_sample.drop(columns='label').to_numpy(), random_state=12)

# Step 3: Construct TSNEEmbedding object
embedding = TSNEEmbedding(init, affinities, verbose=True)

# To store intermediate embeddings
frames = []

# Custom callback to store intermediate embeddings
def callback(iteration, error, embedding):
    frames.append(embedding.copy())

print("starting EE")
# Step 4: Early Exaggeration with a callback
EE_embedding = embedding.optimize(
    n_iter=250,
    exaggeration=12,
    callbacks=callback,
    callbacks_every_iters=2,
    verbose=True
)

# Step 5: Embedding with callback 
final_embedding = EE_embedding.optimize(
    n_iter=500,
    exaggeration=1,
    callbacks=callback,
    callbacks_every_iters=2,
    verbose=True
)
print("embedding done")

frames = np.array(frames)
print(frames.shape)

# Determine the bounds for the last frame
last_frame = frames[-1]
x_min, x_max = last_frame[:, 0].min(), last_frame[:, 0].max()
y_min, y_max = last_frame[:, 1].min(), last_frame[:, 1].max()
epsilon = max(abs(x_max - x_min), abs(y_max - y_min)) / 50 

# Create the animation figure and scatter plot
fig, ax = plt.subplots(figsize=(8, 8))
scatter = ax.scatter([], [], c=[], cmap='Paired', alpha=0.6)
ax.set_title("Standard t-SNE on MNIST")
ax.set_xlim(x_min - epsilon, x_max + epsilon)
ax.set_ylim(y_min - epsilon, y_max + epsilon)

# Update function for each frame
def update(frame_idx):
    tsne_results = frames[frame_idx]  # Get t-SNE results for the current frame
    scatter.set_offsets(tsne_results[:, :2])  # Update positions
    scatter.set_array(X_sample['label'].to_numpy())  # Update colors
    return [scatter]

# Create the animation
ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(frames),  # Total number of frames
    interval=5, 
    blit=True,
    repeat=True
)

# Specify the filename and writer
output_filename = "tsne_mnist_animation01_slower.mp4"
writer = FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)

# Save the animation
ani.save(output_filename, writer=writer)
print(f"Animation saved as {output_filename}")
