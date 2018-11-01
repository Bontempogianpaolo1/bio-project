
import warnings
import matplotlib.pyplot as plt
from itertools import islice, cycle
import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn import cluster, metrics
from sklearn.preprocessing import StandardScaler

seed = 241819


def plot_clusters(x, y, feature, counts):

    print("insert n_neighbors")
    n_neighbors = eval(input(''))
    print("insert n cluster")
    n_cluster = eval(input(''))
    print("insert n_components or percentile to keep")
    n_components = eval(input(''))
    default_base = {'n_neighbors': n_neighbors,
                    'n_clusters': n_cluster}
    x_stand = StandardScaler().fit_transform(x)
    # update parameters with dataset-specific values
    params = default_base.copy()
    if feature is not None:
        feature.setcomponents(n_components)
        feature.fit(x,y)
        xclustered = feature.transform(x_stand)
        xoriginal = feature.transform(x_stand)
    else:
        xclustered = x_stand
        xoriginal = x_stand
    # normalize dataset for easier parameter selection
    # estimate bandwidth for mean shift

    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(
        xclustered, n_neighbors=params['n_neighbors'], include_self=False)
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    # ============
    # Create cluster objects
    # ============
    two_means = cluster.KMeans(n_clusters=params['n_clusters'], precompute_distances=True,random_state=seed)
    ward = cluster.AgglomerativeClustering(
        n_clusters=params['n_clusters'], linkage='ward',
        connectivity=connectivity)
    complete = cluster.AgglomerativeClustering(
        linkage="complete",
        n_clusters=params['n_clusters'], connectivity=connectivity)
    average_linkage = cluster.AgglomerativeClustering(
        linkage="average",
        n_clusters=params['n_clusters'], connectivity=connectivity)
    clustering_algorithms = (
        ('KMeans', two_means),
        ('AgglomerativeClustering Using linkage=ward', ward),
        ('AgglomerativeClustering Using linkage=complete', complete),
        ('AgglomerativeClustering Using linkage=average', average_linkage),
    )

    for name, algorithm in clustering_algorithms:

        # catch warnings related to kneighbors_graph
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the " +
                "connectivity matrix is [0-9]{1,2}" +
                " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning)
            warnings.filterwarnings(
                "ignore",
                message="Graph is not fully connected, spectral embedding" +
                " may not work as expected.",
                category=UserWarning)

            # af = algorithm.fit(X_norm)
            # cluster_centers_indices = af.cluster_centers_indices_
            # labels = af.labels_
            # n_clusters_ = len(cluster_centers_indices)
            # print("Silhouette Coefficient: %0.3f"
            #      % metrics.silhouette_score(X_norm, labels, metric='sqeuclidean'))
            # std_clf = make_pipeline(StandardScaler(), PCA(n_components=100), algorithm)
            # std_clf.fit(X, y)
            # algorithm.fit(X)

        # if hasattr(algorithm, 'labels_'):
        # y_pred = af.labels_.astype(np.int)
        # if name == "DBSCAN" or name == "SpectralClustering":

        if hasattr(algorithm, 'fit_predict'):
            y_pred = algorithm.fit_predict(xclustered)
        else:
            algorithm.fit(xclustered)
            y_pred = algorithm.predict(xclustered)

        # else:
        # y_pred = af.predict(X_norm)
        labels = algorithm.labels_
        print("==============================================================")
        print(algorithm)
        print("==============================================================")

        n_cluster= len(set(labels))
        print("tot clusters {:d}".format(n_cluster))
        try:
            print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(xclustered, labels, metric='euclidean'))
        except:
            print("error to print the silhoutte")
        #  X_std = algorithm.transform(X2)
        if y_pred.size == 0:
            continue
        # plt.figure(figsize=(9 * 2 + 3, 12.5))
        # plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
        #                   hspace=.01)
        fig_size = (10, 7)
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=fig_size)

        plt.title(name, size=18)
        # ax = plt.subplot(len(datasets), 1, 1)
        colors = np.array(list(islice(cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k']),
                                      int(max(y_pred) + 1))))
        markers = np.array(list(islice(cycle(['o', 'v', '^', '<', '>', '8', 's']), int(max(y_pred) + 1))))
        for l, c, m in zip(range(0, max(y_pred+1)), colors, markers):
            ax1.scatter(xclustered[y_pred == l, 0], xclustered[y_pred == l, 1], color=c,
                        alpha=0.5, marker=m)
            if hasattr(algorithm, "cluster_centers_"):
                centers = algorithm.cluster_centers_
                ax1.scatter(centers[l, 0], centers[l, 1], color=c, alpha=0.5, marker=m, s=1000)

        colors = np.array(list(islice(cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k']),
                                      len(counts.index))))
        markers = np.array(list(islice(cycle(['o', 'v', '^', '<', '>', '8', 's']), len(counts.index))))
        for l, c, m in zip(counts.index, colors, markers):
            ax2.scatter(xoriginal[y == l, 0], xoriginal[y == l, 1], color=c, label='class %s' % l, alpha=0.5, marker=m)

        ax1.set_title(name)
        ax2.set_title('original')

        # plt.scatter(X_norm2[:, 0], X_norm2[:, 1], s=10, color=colors[y_pred],label='class %s' % y_pred, alpha=0.5)
        ax1.set_xlabel('1st principal component')
        ax1.set_ylabel('2nd principal component')
        ax2.set_xlabel('1st principal component')
        ax2.set_ylabel('2nd principal component')
        ax2.legend(loc='upper right')
        ax1.grid()
        ax2.grid()
        plt.pause(0.2)
