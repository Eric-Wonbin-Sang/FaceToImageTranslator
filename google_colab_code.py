import matplotlib
matplotlib.use('Agg')

import imageio
import numpy as np
from matplotlib import image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
from IPython.display import HTML

import cv2

driving_frame = None

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
frame = cv2.flip(frame, 0)

source_image = resize(image.imread("C:/Users/ericw/Desktop/sean.jpg"), (256, 256))[..., :3]
driving_video = [resize(frame, (256, 256))[..., :3] for frame in [frame]]


def display(source, driving, generated=None):
    fig = plt.figure(figsize=(8 + 4 * (generated is not None), 6))

    ims = []
    for i in range(len(driving)):
        cols = [source]
        cols.append(driving[i])
        if generated is not None:
            cols.append(generated[i])
        im = plt.imshow(np.concatenate(cols, axis=1), animated=True)
        plt.axis('off')
        ims.append([im])
        plt.show()

# display(source_image, driving_video)

# ----------------------------------------------
from demo import load_checkpoints
generator, kp_detector = load_checkpoints(config_path='config/vox-256.yaml',
                                          checkpoint_path='vox-cpk.pth.tar',
                                          cpu=True)
# ----------------------------------------------

from demo import make_animation
from skimage import img_as_ubyte

predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=True, cpu=True)

imageio.imsave("prediction.jpg", img_as_ubyte(predictions[0]))

predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=False, adapt_movement_scale=True, cpu=True)
imageio.imsave("prediction.jpg", img_as_ubyte(predictions[0]))
