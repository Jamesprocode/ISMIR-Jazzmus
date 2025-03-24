import os

import yaml
from matplotlib import pyplot as plt


def check_folders():
    if not os.path.exists("debug"):
        os.makedirs("debug")

    if not os.path.exists("logs"):
        os.makedirs("logs")


def load_config(file_path: str) -> dict:
    """Load a yaml configuration file.

    Args:
        file_path (str): Path to the yaml file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(file_path) as file:
        config = yaml.safe_load(file)
    return config


def print_smt_batch(dataloader, path_to_save="random_batch.pdf"):
    # gets the first batch of the dataloader
    batch = next(iter(dataloader))

    # extracts the images from the batch
    images = batch[0]

    fig = plt.figure()
    gs = fig.add_gridspec(len(images), hspace=0)
    axs = gs.subplots(sharex=True)
    fig.suptitle("Batch of images")

    for i in range(len(images)):
        # prints the shapes of the images
        print(images[i].shape)
        image_to_plot = images[i].numpy().squeeze(0)
        axs[i].imshow(image_to_plot, cmap="gray")
    for ax in axs:
        ax.label_outer()
    plt.savefig(path_to_save)


def list_files_recursively(directory):
    files = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            files.append(os.path.join(dirpath, filename))
    return sorted(files)


if __name__ == "__main__":
    # Example usage:
    config_dict = load_config("config.yaml")
    print(config_dict)
