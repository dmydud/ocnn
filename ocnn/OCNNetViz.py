import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .OCNNetAnalyser import OCNNetAnalyser


class OCNNetViz:
    @staticmethod
    def plot_states(net_states, figsize=(5, 5), annot=False, fmt=".1f", square=True, cmap="flare_r",
                    vmax=1, vmin=-1, cbar=False, title="", ax=None, _show=True):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Neuron")
        ax.set_title(title)

        sns.heatmap(
            net_states,
            ax=ax,
            annot=annot,
            fmt=fmt,
            square=square,
            cmap=cmap,
            vmax=vmax,
            vmin=vmin,
            cbar=cbar
        )

        if _show:
            plt.show()

    @staticmethod
    def plot_hierarchy(net_states, n=100, _show=True):
        linkage_matrix = OCNNetAnalyser.get_linkage(net_states)

        fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))

        n_clusters = list()
        for theta_ in np.linspace(0, np.log(2), n):
            clusters = OCNNetAnalyser.cluster_by_linkage(linkage_matrix, theta=theta_)

            unique_, counts = np.unique(clusters, return_counts=True)
            n_clusters.append(len(unique_))
            ax1.scatter([theta_] * len(counts), counts, c="k", s=5, zorder=100)

        ax1.set_xlabel(r"$\theta$")
        ax1.set_ylabel("size of clusters", color='k')

        ax1_2 = ax1.twinx()
        sns.lineplot(x=np.linspace(0, np.log(2), n), y=n_clusters, ax=ax1_2, zorder=-1)
        ax1_2.set_ylabel('number of clusters, $K$', color=sns.color_palette("tab10")[0])
        ax1_2.grid(False)

        if _show:
            plt.show()

    @staticmethod
    def plot_clusters(input_data, net_states, theta, ax=None, figsize=(5, 5), xlabel="x",
                      ylabel="y", title="", _show=True):

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        clusters = OCNNetAnalyser.cluster(net_states, theta=theta)
        sns.scatterplot(x=input_data.T[0], y=input_data.T[1], c=clusters.astype(int),
                        cmap=sns.color_palette("tab10", as_cmap=True))

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        if _show:
            plt.show()
