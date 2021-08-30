import cv2

from Classes import PygameHelper, Image


class Camera:

    def __init__(self, index, upside_down=False):

        self.index = index
        self.cap = cv2.VideoCapture(index)
        self.upside_down = upside_down

    def get_webcam_image(self):
        ret, frame = self.cap.read()
        if frame is None:
            return frame

        init_height, init_width = frame.shape[0], frame.shape[1]
        new_height, new_width, y_start, y_end, x_start, x_end = get_resize_values(
            init_height, init_width, 255, 255)
        np_array = cv2.cvtColor(cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)[y_start:y_end, x_start:x_end], cv2.COLOR_BGR2RGB)
        return Image.Image(source_np=np_array)

    def to_pygame_display(self, display, x, y):
        PygameHelper.show_numpy_array(self.get_webcam_image().image_np, display, x=x, y=y, x_scale=1, y_scale=1)

    def __str__(self):
        return "Camera {}".format(
            self.index
        )


def get_resize_values(init_height, init_width, target_height, target_width):
    """ Values for transforming an image via scaling and cropping """
    new_height, new_width = int(target_height), int(init_width / init_height * target_width)
    y_start, y_end = 0, -1
    x_start, x_end = int((new_width - target_width) / 2), int(new_width - (new_width - target_width) / 2)
    return new_height, new_width, y_start, y_end, x_start, x_end
