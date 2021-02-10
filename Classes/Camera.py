import cv2

from Classes import PygameHelper, Image


class Camera:

    def __init__(self, index, upside_down=False):

        self.index = index
        self.cap = cv2.VideoCapture(index)
        self.upside_down = upside_down

    def get_webcam_image(self):
        ret, frame = self.cap.read()
        if frame is not None:
            # frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # frame = cv2.resize(frame, (int(frame.shape[1] * .6), int(frame.shape[0] * .6)))
            # x = 75
            # w = 256
            # y = 0 + 40
            # h = 256
            # frame = frame[y:y + h, x:x + w]
            pass
        return Image.Image(source_np=frame)

    def to_pygame_display(self, display, x, y):
        PygameHelper.show_numpy_array(self.get_webcam_image().image_np, display, x=x, y=y, x_scale=1, y_scale=1)

    def __str__(self):
        return "Camera {}".format(
            self.index
        )
