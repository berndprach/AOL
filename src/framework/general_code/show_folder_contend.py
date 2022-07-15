
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def print_file(filepath):
    with open(filepath, "r") as f:
        print(f.read())


def display_image(filename, w, h):
    img = mpimg.imread(filename)
    plt.figure(figsize=(w, h))
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def display_folder_contend(foldername, w=20, h=6):
    if foldername is None:
        return None
    print(f"Showing all images in folder /{foldername.split('/')[-1]}:")
    for plot_name in sorted(os.listdir(foldername)):
        path = os.path.join(foldername, plot_name)
        if os.path.isdir(path):
            print(f"(Found sub-directory {plot_name}.)")
            continue
        if ".txt" in path:
            print(f"\nShowing text from {plot_name}:")
            print_file(path)
        elif ".png" in path:
            print(f"\nShowing {plot_name}:")
            display_image(path, w, h)
        else:
            print(f"Not sure what to do with this file: {path}")
