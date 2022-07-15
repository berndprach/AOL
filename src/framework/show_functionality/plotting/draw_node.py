
import matplotlib.pyplot as plt
import tensorflow as tf

import framework.show_functionality.functionality as fun
from framework.show_functionality.plotting.draw_image_with_channels import draw_image_with_channels
from framework.show_functionality.plotting.draw_grid import draw_grid


def draw_node_in_new_figure(node, title, max_width=10):
    """ Calls plt.figure() !! """
    pcd = (1, 3, 4)  # possiblt channel dimensions

    while len(node.shape) > 1 and node.shape[0] == 1:
        node = node[0]

    if len(node.shape) == 1:
        if node.shape[-1] > 20:
            # Find rectangle closest to square:
            max_h = int(node.shape[-1] ** (1 / 2))
            h = max_h
            while node.shape[-1] % h != 0:
                h -= 1
            node = tf.reshape(node, [h, -1])
            title += " reshaped"
        else:
            node = node[None, :]
        plt.figure(figsize=(10, 2))
        plt.title(title)
        plt.imshow(node)
        plt.colorbar()
    elif len(node.shape) == 2 and node.shape[-1] in pcd:
        node = node[None, :, :]
        plt.figure(figsize=(10, 2))
        plt.title(title)
        fun.imshow(node)
        plt.colorbar()
    elif len(node.shape) == 2:
        plt.figure()
        plt.title(title)
        fun.imshow(node)
        plt.colorbar()
    elif len(node.shape) == 3 and node.shape[-1] in pcd:
        if node.shape[-1] == 1:
            plt.figure()
            plt.title(title)
            fun.imshow(node)
            plt.colorbar()
        else:
            draw_image_with_channels(node, fig_subtitle=title)
    elif len(node.shape) == 3:
        node_t = tf.transpose(node, (2, 0, 1))[:, :, :, None]
        draw_grid(node_t, fig_subtitle=title, max_size=(max_width,))
    elif len(node.shape) == 4 and node.shape[-1] in pcd:
        node_t = tf.transpose(node, (2, 0, 1, 3))
        draw_grid(node_t, fig_subtitle=title, max_size=(max_width,))
    elif len(node.shape) == 4:
        node_t = tf.transpose(node, (2, 3, 0, 1))
        draw_grid(node_t, fig_subtitle=title, max_size=(2, max_width))
    elif len(node.shape) == 5 and node.shape[-1] in pcd:
        node_t = tf.transpose(node, (2, 3, 0, 1, 4))
        draw_grid(node_t, fig_subtitle=title, max_size=(2, max_width))
    elif len(node.shape) == 5 and node.shape[-2] in pcd:
        node_t = tf.transpose(node, (2, 4, 0, 1, 3))
        draw_grid(node_t, fig_subtitle=title, max_size=(2, max_width))
    else:
        print("!! I don't know how to handle the dimensions above yet !!")
        return False
    return True


def draw_node(node, title):
    pcd = (1, 3, 4)  # possiblt channel dimensions

    while len(node.shape) > 1 and node.shape[0] == 1:
        node = node[0]

    if len(node.shape) == 1:
        if node.shape[-1] > 20:
            # Find rectangle closest to square:
            max_h = int(node.shape[-1] ** (1 / 2))
            h = max_h
            while node.shape[-1] % h != 0:  # Loop ends definitely since n % 1 == 0.
                h -= 1
            node = tf.reshape(node, [h, -1])
            title += " reshaped"
        else:
            node = node[None, :]
        # plt.figure(figsize=(10, 2))
        plt.title(title)
        plt.imshow(node)
        # plt.colorbar()
    elif len(node.shape) == 2 and node.shape[-1] in pcd:
        node = node[None, :, :]
        # plt.figure(figsize=(10, 2))
        plt.title(title)
        fun.imshow(node)
        # plt.colorbar()
    elif len(node.shape) == 2:
        # plt.figure()
        plt.title(title)
        fun.imshow(node)
        # plt.colorbar()
    elif len(node.shape) == 3 and node.shape[-1] in 1:
        # plt.figure()
        plt.title(title)
        fun.imshow(node)
        # plt.colorbar()
    else:
        print("!! I need more than one subplot for this !!")
        return False
    return True


