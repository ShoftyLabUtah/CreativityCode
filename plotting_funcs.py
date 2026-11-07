'''plotting funcs'''


import matplotlib.pyplot as plt
import math
import numpy as np
import cebra
from sklearn.linear_model import LinearRegression

import random

def remove_ticks_and_labels_3d(ax_object):
    # Remove ticks and labels
    ax_object.set_xticks([])
    ax_object.set_yticks([])
    ax_object.set_zticks([])
    ax_object.set_xlabel("")
    ax_object.set_ylabel("")
    ax_object.set_zlabel("")


def plot_embeddings_grid_fixed(embeddings_all, labels, label_to_color,  idx_order=[0,1,2]):

    # Determine grid size based on number of embeddings
    num_embeddings = len(embeddings_all)
    cols = math.ceil(math.sqrt(num_embeddings))  # Set columns based on square root
    rows = math.ceil(num_embeddings / cols)  # Compute necessary rows

    # unique_labels = np.unique(np.concatenate(labels))
    # if len(colors)==0:
    #     colors = list(map(plt.get_cmap(name='gist_rainbow'),np.linspace(0,1,len(unique_labels))))
    #     random.Random(4).shuffle(colors)
    # label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}    

    # Create figure and 3D subplots
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), subplot_kw={'projection': '3d'})

    # Flatten `ax` to handle cases where rows/cols > 1
    ax = np.ravel(ax)

    # Loop through embeddings and plot
    for i, embedding in enumerate(embeddings_all):
        embedding_colors = [label_to_color[label] for label in labels[i]]
        
        # Plot embedding in corresponding subplot
        scatter_pt = cebra.plot_embedding(
            embedding, 
            embedding_labels=None, 
            idx_order=idx_order, 
            markersize=5, 
            alpha=0.5, 
            color=embedding_colors, 
            ax=ax[i]
        )
        remove_ticks_and_labels_3d(scatter_pt)

    plt.subplots_adjust(wspace=0.01, hspace=0.01)  # Reduce spacing between subplots
    plt.tight_layout(pad=1.0)  # Reduce padding around figure



def plot_embeddings_grid(embeddings_all, labels, titles=None, colors=[], idx_order=[0,1,2]):

    # Determine grid size based on number of embeddings
    num_embeddings = len(embeddings_all)
    cols = math.ceil(math.sqrt(num_embeddings))  # Set columns based on square root
    rows = math.ceil(num_embeddings / cols)  # Compute necessary rows

    unique_labels = np.unique(np.concatenate(labels))
    if len(colors)==0:
        colors = list(map(plt.get_cmap(name='gist_rainbow'),np.linspace(0,1,len(unique_labels))))
        random.Random(4).shuffle(colors)
    label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}    

    # Create figure and 3D subplots
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), subplot_kw={'projection': '3d'})

    # Flatten `ax` to handle cases where rows/cols > 1
    ax = np.ravel(ax)

    # Loop through embeddings and plot
    for i, embedding in enumerate(embeddings_all):
        embedding_colors = [label_to_color[label] for label in labels[i]]
        
        # Plot embedding in corresponding subplot
        scatter_pt = cebra.plot_embedding(
            embedding, 
            embedding_labels=None, 
            idx_order=idx_order, 
            markersize=5, 
            alpha=0.5, 
            color=embedding_colors, 
            ax=ax[i]
        )
        remove_ticks_and_labels_3d(scatter_pt)
        try:
            ax[i].set_title(titles[i])   
        except: #no titles given
            ax[i].set_title('')
    plt.subplots_adjust(wspace=0.01, hspace=0.01)  # Reduce spacing between subplots
    plt.tight_layout(pad=1.0)  # Reduce padding around figure



def plot_polynomial_coding_axis(embed_df, creativity_values, dimensions=3, idx_order=[0,1,2], angle=[20, -30]):
    ''' 
        PLOT CURVED CODING AXIS IN EMBEDDING SPACE 
        fit polynomial regression for each dimension of embedding ("embed_df"), 
            then use this model with range of creativity scores (1D) as the input 
            to predict embeddings (e.g, 3D) that serve as coordinates for coding axis
    '''
    
    from scipy.interpolate import splprep, splev
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression


    ################# Fit separate polynomial regressions for each embedding dimension   ################# 

    ## 1. transform creativity values so next step will be polynomial and not linear regression
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(creativity_values.reshape(-1, 1)) #creativity scores are the input

    ## 2. Fit regression model for first 3 embedding dimension
    models = []
    for i in range(dimensions):  
        model = LinearRegression().fit(X_poly, embed_df.iloc[:, i])  
        models.append(model) #these are the fitted models

    ################# Predict 3D embedding coordinates from creativity scores ################# 

    ## 3. get creativity range: Generate list of linearly spaced points within observed range of creativity scores
    creativity_range = np.linspace(creativity_values.min(), creativity_values.max(), num=100)
    ## 4. change linear creativity range to polynomial curve 
    X_poly_range = poly.transform(creativity_range.reshape(-1, 1)) 

    ## 5. Predict corresponding 3D embedding coordinates
    x_smooth = models[idx_order[0]].predict(X_poly_range)
    y_smooth = models[idx_order[1]].predict(X_poly_range)
    z_smooth = models[idx_order[2]].predict(X_poly_range)

    ################# Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ## 6. Real embeddings scatter plot
    scatter = ax.scatter(embed_df.iloc[:, 0], embed_df.iloc[:, 1], embed_df.iloc[:, 2], 
                        s=5, c=creativity_values, cmap="cool", alpha=0.3)

    ## 7. Draw smooth nonlinear coding axis 
    ax.plot(x_smooth, y_smooth, z_smooth, color="k", linewidth=2, label="Nonlinear Coding Axis")

    # Labels and legend
    # ax.set_xlabel("Embedding Dim 1")
    # ax.set_ylabel("Embedding Dim 2")
    # ax.set_zlabel("Embedding Dim 3")
    # ax.legend()
    ax.view_init(elev=angle[0], azim=angle[1])  # Change angles for a better view
    plt.title("3D Embeddings with Polynomial Coding Axis")
    plt.colorbar(scatter, ax=ax, label="originality value")
    plt.show()


def plot_linear_coding_axis(embed_df, creativity_values, idx_order=[0,1,2], angle=[35,-55]):

    model = LinearRegression()
    model.fit(embed_df, creativity_values)
    # Get the coding axis as a unit vector
    coding_axis = model.coef_ / np.linalg.norm(model.coef_)
    creativity_projection = embed_df @ coding_axis

    min_proj, max_proj = creativity_projection.min(), creativity_projection.max()
    center = embed_df.mean(axis=0).values  # Mean embedding
    start_point = center + min_proj * coding_axis
    end_point = center + max_proj * coding_axis

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot with gradient colors
    scatter = ax.scatter(embed_df.iloc[:, 0], embed_df.iloc[:, 1], embed_df.iloc[:, 2], 
                        c=creativity_values, cmap="cool", alpha=0.25)


    # Draw the coding axis
    ax.plot(
        [start_point[0], end_point[0]],
        [start_point[1], end_point[1]],
        [start_point[2], end_point[2]],
        color="black", linewidth=2, label="Coding Axis"
    )

    # Labels and color bar
    # ax.set_xlabel("Embedding Dim 1")
    # ax.set_ylabel("Embedding Dim 2")
    # ax.set_zlabel("Embedding Dim 3")
    ax.view_init(elev=angle[0], azim=angle[1])  # Change angles for a better view
    plt.colorbar(scatter, label="Creativity Value")
    # ax.legend()
    plt.title("3D Embeddings with Linear Coding Axis")
    plt.show()