from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt

def preprocess(input, scale):
    if input.size > 3:
      input = input / 255.0
    if input.ndim == 2:
      label = np.expand_dims(label, axis=2)
    return label


def  plot_img_and_mask(img, mask):
    # classes = mask.width[2] if len(mask.width) > 2 else 1
    classes = 1
    fig, ax = plt.subplots(1, classes + 1 )
    ax[0].set_title("Input image")
    ax[0].imshow(img)
    
    
    if classes > 1 :
        for i in range(classes):
            ax[i+1].setTitle(f'Output mask (class {i+1})')
            ax[i+1].imshow(mask[:,:,i])

    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask, cmap="gray")
    plt.xticks([]), plt.yticks([])
    plt.show()
