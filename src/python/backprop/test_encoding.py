import sys, glob, re
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, StrMethodFormatter
from models.resnet import END_STORE_A
from backprop.test_embedding import encoding_function
from halutmatmul.maddness_legacy import learn_proto_and_hash_function
from halutmatmul.model import check_file_exists_and_return_path


def test_encoding(C=64):
    K = 16
    dim_per_C = 8
    # generated data
    a = 1.0
    # b = 1.0
    # X_train = np.random.rand(10000, dim_per_C * C) * a + b
    # X_test = np.random.rand(2000, dim_per_C * C) * a + b
    # # introduce some perturbations
    # for _ in range(5):
    #     for i in range(X_train.shape[0]):
    #         for c in range(C):
    #             X_train[i, np.random.randint(c * dim_per_C, dim_per_C * (c + 1))] += (
    #                 np.random.rand() * 5 - 2.5
    #             )

    # load data
    data_path = "/usr/scratch2/vilan1/janniss/halut/resnet18-cifar10-same-compression"
    layers_to_test = [
        "layer1.1.conv2",
        "layer2.1.conv2",
        "layer3.1.conv2",
        "layer4.1.conv2",
    ]
    l = "layer2.1.conv2"
    error_numbers = []
    zero_percentage = []
    bincount = []
    bincount_c = []
    error_per_c_k_all = []
    for l in layers_to_test:
        batch_size = 64
        files = glob.glob(data_path + f"/{l}_{batch_size}_{0}_*" + END_STORE_A)
        print(files)
        configs_reg = re.findall(r"(?<=_)(\d+)", files[0])
        iterations = int(configs_reg[2])
        a_parts = []
        for i in range(1, iterations):
            print(
                f"loading file {data_path}/{l}_{batch_size}_{i}_{iterations}{END_STORE_A}"
            )
            a_part = np.load(
                data_path + f"/{l}_{batch_size}_{i}_{iterations}" + END_STORE_A
            )
            a_parts.append(a_part)
        a_numpy = np.vstack(a_parts)
        print(a_numpy.shape)
        idx = np.arange(a_numpy.shape[0])
        np.random.shuffle(idx)
        X_train = a_numpy[idx]
        X_test = X_train[:10000]
        X_train = X_train[10000:]
        dim_per_C = X_train.shape[1] // C
        (
            _,
            all_prototypes,
            _,
            thresholds,
            dims,
        ) = learn_proto_and_hash_function(X_train, C, K)
        print(thresholds, dims)
        print(all_prototypes.shape, thresholds.shape)

        threshold_table = np.zeros((C * K)).astype(np.float16)
        for c in range(C):
            threshold_table[c * K : (c + 1) * K - 1] = thresholds[
                c * (K - 1) : (c + 1) * (K - 1)
            ]
        # encode with halut learned
        encoded, _, _ = encoding_function(
            threshold_table, X_test[:, dims].reshape((-1, C, 4)), tree_depth=4, K=K
        )
        print(np.bincount(encoded.flatten()))
        print(
            np.count_nonzero(X_test),
            X_test.shape[0] * X_test.shape[1],
            100
            - (np.count_nonzero(X_test) / (X_test.shape[0] * X_test.shape[1]) * 100),
        )
        print(encoded.shape)
        print(encoded)

        error_halut = X_test.copy()
        error_per_c_k = np.zeros((C, K))
        for c in range(C):
            error_halut -= all_prototypes[c, encoded[:, c], :]
            prototype_sel = all_prototypes[
                c, encoded[:, c], c * dim_per_C : (c + 1) * dim_per_C
            ]
            selection_error = np.mean(
                (prototype_sel - X_test[:, c * dim_per_C : (c + 1) * dim_per_C]) ** 2,
                axis=1,
            )
            for k in range(K):
                error_per_c_k[c, k] = np.sum(selection_error[encoded[:, c] == k])

        print("error per c k", error_per_c_k[0])
        error_per_c_k_all.append(error_per_c_k)
        error_halut = np.mean(error_halut**2, axis=0) / a
        print(error_halut.shape)
        print(error_halut)
        final_error_halut = np.mean(error_halut)
        print(final_error_halut)
        error_numbers.append(final_error_halut)
        zero_percentage.append(
            100 - (np.count_nonzero(X_test) / (X_test.shape[0] * X_test.shape[1]) * 100)
        )
        bincount.append(np.bincount(encoded.flatten()))
        bincount_per_class = []
        for c in range(C):
            bincount_per_class.append(np.bincount(encoded[:, c]))
        bincount_c.append(bincount_per_class)
        print("bincount", bincount_per_class)
    print("error numbers", error_numbers)
    print("zero percentage", zero_percentage)
    print("Layers", layers_to_test)
    # # PQ encoding
    #
    # X_train_reshaped = X_train.reshape(X_train.shape[0], C, dim_per_C)
    # kmeans_encoders = []
    # prototypes_pq = np.zeros((C, K, dim_per_C))
    # for c in range(C):
    #     print("learn clusters for c", c)
    #     kmeans = KMeans(n_clusters=K, random_state=0).fit(X_train_reshaped[:, c, :])
    #     prototypes_pq[c, :, :] = kmeans.cluster_centers_
    #     kmeans_encoders.append(kmeans)
    #
    # # encode with PQ
    # X_test_reshaped = X_test.reshape(X_test.shape[0], C, dim_per_C)
    # encoded_pq = np.zeros((X_test.shape[0], C))
    # for c in range(C):
    #     encoded_pq[:, c] = kmeans_encoders[c].predict(X_test_reshaped[:, c, :])
    #
    # print(encoded_pq.shape)
    # print(encoded_pq)
    #
    # encoded_pq = encoded_pq.astype(np.int32)
    # # encoding error PQ
    # error_pq = np.zeros((X_test.shape[0], X_test.shape[1]))
    # for c in range(C):
    #     error_pq[:, c * dim_per_C : (c + 1) * dim_per_C] += np.abs(
    #         X_test_reshaped[:, c, :] - prototypes_pq[c, :, :][encoded_pq[:, c]]
    #     )
    #
    # # error per dimension
    # print(error_pq.shape)
    # error_pq = np.mean(error_pq**2, axis=0) / a
    # # print(error_pq, error_pq.shape)
    # total_error_pq = np.mean(error_pq)
    # print(total_error_pq)
    #

    return error_numbers, zero_percentage, bincount, bincount_c, error_per_c_k_all


def run_tests():
    all_errors = []
    all_zero_percentage = []
    all_bincount = []
    all_bincount_c = []
    all_error_c_k = []
    for C in [32, 64, 128, 256, 512]:
        error_numbers, zero_percentage, bincount, bincount_c, error_c_k = test_encoding(
            C
        )
        all_errors.append(error_numbers)
        all_zero_percentage.append(zero_percentage)
        all_bincount.append(bincount)
        all_bincount_c.append(bincount_c)
        all_error_c_k.append(error_c_k)

    store_array = np.array(
        [all_errors, all_zero_percentage, all_bincount, all_bincount_c, all_error_c_k]
    )
    print(store_array)
    np.save("store_array_all.npy", store_array)


# pylint: disable=line-too-long
# source: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
def heatmap(
    data, row_labels, col_labels, ax=None, cbar_kw=None, cbarlabel="", **kwargs
):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(
    im,
    data=None,
    valfmt="{x:.2f}",
    textcolors=("black", "white"),
    threshold=None,
    **textkw,
):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0  # type: ignore

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):  # type: ignore
        for j in range(data.shape[1]):  # type: ignore
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])  # type: ignore
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)  # type: ignore
            texts.append(text)

    return texts


def plot():
    loaded = np.load("store_array_all.npy", allow_pickle=True)
    print(loaded.shape)
    all_errors = loaded[0]
    all_zero_percentage = loaded[1]
    all_bincount = loaded[2]
    all_bincount_c = loaded[3]
    all_error_c_k = loaded[4]

    zero_scaled_error = np.array(all_errors) / (
        (100 - np.array(all_zero_percentage)) / 100
    )
    print(zero_scaled_error)

    # plot error
    # line plot for zero scaled error and non scaled error
    for name, values in zip(
        ["zero_scaled", "non_zero_scaled"], [zero_scaled_error, all_errors]
    ):
        plt.style.use("seaborn-v0_8-poster")
        fig, ax = plt.figure(figsize=(10, 10)), plt.gca()
        Cs = [32, 64, 128, 256, 512]
        ax.plot(
            Cs,
            values[:, 0],
            label="layer1.1.conv2",
            marker="o",
            linestyle="--",
        )
        ax.plot(
            Cs,
            values[:, 1],
            label="layer2.1.conv2",
            linestyle="--",
            marker="o",
        )
        ax.plot(
            Cs,
            values[:, 2],
            label="layer3.1.conv2",
            marker="o",
            linestyle="--",
        )
        ax.plot(
            Cs,
            values[:, 3],
            label="layer4.1.conv2",
            marker="o",
            linestyle="--",
        )
        ax.plot(
            Cs[:4],
            values[[0, 1, 2, 3], [0, 1, 2, 3]],
            label="CW18",
            marker="x",
            linestyle="dotted",
        )
        ax.plot(
            Cs[1:],
            values[[1, 2, 3, 4], [0, 1, 2, 3]],
            label="CW9",
            marker="x",
            linestyle="dotted",
        )
        ax.set_xlabel("Number of Codebooks")
        ax.set_ylabel("Encoding Error (scaled for zero values)")
        ax.set_title("Encoding Error for different number of codebooks")
        ax.set_xscale("log")
        ax.set_xticks(Cs)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xticklabels(Cs)
        ax.legend()
        plt.savefig(f"results/figures/error_{name}.pdf", bbox_inches="tight", dpi=600)
        plt.savefig(f"results/figures/error_{name}.png", bbox_inches="tight", dpi=600)

    # heatmap for all bincounts

    for sel in [0]:  # range(len(Cs)):
        selection = sel
        C = Cs[selection]
        K = 16
        fig, axes = plt.subplots(1, 4, figsize=(23, 11 * 2**selection))
        fig.subplots_adjust(top=0.85)

        colormaps = ["Blues", "Oranges", "Greens", "Reds"]
        for i, ax in enumerate(axes):
            plot_array = np.zeros((C, K))
            looper = np.array(all_bincount_c[selection, i])
            for c in range(C):
                looper[c] = np.append(looper[c], [0 for _ in range(K - len(looper[c]))])
                plot_array[c, :] = looper[c]
            _, _ = heatmap(
                plot_array,
                [str(x) for x in np.arange(C) + 1],
                [str(x) for x in np.arange(K) + 1],
                ax=ax,
                cmap=colormaps[i],
                cbarlabel="# encoded inputs",
                cbar_kw={"fraction": 0.046, "pad": 0.04},
            )
            ax.set_title("layer" + str(i + 1) + ".1.conv2")
        fig.tight_layout()
        fig.suptitle(
            "# Encoded Inputs per codebook and prototype C=" + str(C),
            fontsize=20,
        )
        plt.savefig(
            f"results/figures/bincount_heatmap_{C}.pdf", bbox_inches="tight", dpi=600
        )
        plt.savefig(
            f"results/figures/bincount_heatmap_{C}.png", bbox_inches="tight", dpi=600
        )

    # heatmap for all bincounts
    print(all_bincount.shape)
    fig, axes = plt.subplots(2, 2, figsize=(17, 8))
    fig.subplots_adjust(top=0.7)
    colormaps = ["Blues", "Oranges", "Greens", "Reds"]
    for i, ax in enumerate([item for sublist in axes for item in sublist]):
        plot_array = np.zeros((len(Cs), K))
        looper = np.array(all_bincount[:, i])
        for c in range(len(Cs)):
            looper[c] = np.append(looper[c], [0 for _ in range(K - len(looper[c]))])
            plot_array[c, :] = looper[c]
        plot_array = plot_array / np.atleast_2d(Cs).T.repeat(K, axis=1)
        _, _ = heatmap(
            plot_array,
            [str(x) for x in np.array(Cs)],
            [str(x) for x in np.arange(K) + 1],
            ax=ax,
            cmap=colormaps[i],
            cbarlabel="# encoded inputs",
        )
        ax.set_title("layer" + str(i + 1) + ".1.conv2")
    fig.tight_layout()
    fig.suptitle("# of encoded inputs in each prototype for different Cs", fontsize=20)
    plt.savefig(
        "results/figures/bincount_heatmap_all.pdf", bbox_inches="tight", dpi=600
    )
    plt.savefig(
        "results/figures/bincount_heatmap_all.png", bbox_inches="tight", dpi=600
    )

    # plot error
    print(all_error_c_k.shape)
    print(all_error_c_k[0])

    colormaps = ["Blues", "Oranges", "Greens", "Reds"]
    selection = sel
    C = Cs[selection]
    K = 16
    colormaps = ["Blues", "Oranges", "Greens", "Reds"]
    for name in ["scaled", "unscaled"]:
        fig, axes = plt.subplots(1, 4, figsize=(23, 11 * 2**selection))
        fig.subplots_adjust(top=0.85)

        for i, ax in enumerate(axes):
            plot_array = np.zeros((C, K))
            looper = np.array(all_bincount_c[selection, i])
            looper2 = np.array(all_error_c_k[selection, i])
            for c in range(C):
                looper[c] = np.append(looper[c], [0 for _ in range(K - len(looper[c]))])
                looper2[c] = np.append(
                    looper2[c], [0 for _ in range(K - len(looper2[c]))]
                )
                if name == "scaled":
                    plot_array[c, :] = looper2[c] / (looper[c] + 1)
                else:
                    plot_array[c, :] = looper2[c]
            _, _ = heatmap(
                plot_array,
                [str(x) for x in np.arange(C) + 1],
                [str(x) for x in np.arange(K) + 1],
                ax=ax,
                cmap=colormaps[i],
                cbarlabel="encoding error scaled",
                cbar_kw={"fraction": 0.046, "pad": 0.04},
            )
            ax.set_title("layer" + str(i + 1) + ".1.conv2")
        fig.tight_layout()
        fig.suptitle(
            f"Encoding Error splitted by codebook and prototype {name}", fontsize=20
        )
        plt.savefig(
            f"results/figures/error_heatmap_{name}.pdf", bbox_inches="tight", dpi=600
        )
        plt.savefig(
            f"results/figures/error_heatmap_{name}.png", bbox_inches="tight", dpi=600
        )


if __name__ == "__main__":
    # run_tests()
    plot()
