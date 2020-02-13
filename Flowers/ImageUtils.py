import matplotlib.pyplot as plt

flower_classes = ['roses', 'daisy', 'dandelion', 'sunflowers', 'tulips']


def plot_images(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()
