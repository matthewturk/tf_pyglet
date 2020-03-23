import array
import ctypes
import itertools
import pyglet
import numpy as np
from math import ceil, floor
import matplotlib.pyplot as plt


# https://bitbucket.org/pyglet/pyglet/issues/143/support-numpy-in-more-places
def get_numpy_data(arr):
     """
     :param arr: numpy array of float32
     :return: ctypes array of float32
     """
     # Accept any contiguous array of float32
     assert arr.flags["C_CONTIGUOUS"] or arr.flags["F_CONTIGUOUS"]
     assert arr.dtype == np.uint8
     return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8 * arr.size))[0]

class WindowWidget:
    def __init__(self, window, x0, y0, width, height):
        self.window = window
        self.x0 = x0
        self.y0 = y0
        self.width = width
        self.height = height
        self.x1 = x0 + width
        self.y1 = y0 + height
        self.setup()

    def local_pos(self, x, y):
        return x - self.x0, y - self.y0

    def local_scale(self, x, y):
        return (x - self.x0)/self.width, (y - self.y0)/self.height

class TransferFunctionWidget(WindowWidget):

    def setup(self):
        self.active = 'red'
        self.vals = {}
        self.colors = {}
        self.draw = {}
        self.N_bins = 128

        for i, color in enumerate(('red', 'green', 'blue')):
            self.vals[color] = np.vstack([np.mgrid[0.0:1.0:1j*self.N_bins],
                                     np.mgrid[0.0:1.0:self.N_bins*1j]]).copy(order="F")
            self.colors[color] = np.zeros((self.N_bins, 3), dtype="u1").copy(order="C")
            self.colors[color][:,i] = 255
            self.update_draw( color)

    def update_draw(self, color):
        self.draw[color] = self.vals[color] \
            * np.array([self.width, self.height])[:, None] \
            + np.array([self.x0, self.y0])[:, None]


    def compute_bin_range(self, x, dx):
        left_bound = floor(min(x + dx, x) * self.N_bins / self.width)
        right_bound = ceil(max(x + dx, x) * self.N_bins / self.width)
        return (max(min(_, self.N_bins -1), 0) for _ in (left_bound, right_bound))

    def compute_val_range(self, y, dy):
        yv1 = (y / self.height)
        yv2 = (y + dy) / self.height
        return (max(min(_, 1.0), 0.0) for _ in (yv1, yv2))

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if x < self.x0 or x > self.x1 or y < self.y0 or y > self.y1:
            return False
        xb1, xb2 = self.compute_bin_range(x - self.x0, dx)
        yv1, yv2 = self.compute_val_range(y - self.y0, dy)
        if dx < 0: yv2, yv1 = yv1, yv2
        self.vals[self.active][1,xb1:xb2] = np.mgrid[yv1:yv2:(xb2 - xb1)*1j]
        self.update_draw(self.active)
        return True

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.R:
            self.active = "red"
            return True
        elif symbol == pyglet.window.key.G:
            self.active = "green"
            return True
        elif symbol == pyglet.window.key.B:
            self.active = "blue"
            return True
        return False

    def on_draw(self):
        for color in ('red', 'green', 'blue'):
            pyglet.graphics.draw(self.N_bins, pyglet.gl.GL_LINE_STRIP,
                ('v2f', self.draw[color].ravel(order="F")),
                ('c3B', self.colors[color].ravel(order="C")))

class TransferFunctionImage(WindowWidget):
    def __init__(self, window, x0, y0, width, height, tf_widget):
        self.tf_widget = tf_widget
        super(TransferFunctionImage, self).__init__(window, x0, y0, width, height)

    def setup(self):
        ii = np.vstack(self.tf_widget.colors[c][:,2]
                         for c in ('red', 'green', 'blue')).copy(order="C")
        self.image_data = pyglet.image.ImageData(
            self.tf_widget.N_bins, 1,
            'RGB', get_numpy_data(ii),
            3,
        )
        self.sprite = pyglet.sprite.Sprite(self.image_data, self.x0, self.y0)
        self.sprite.update(scale_x = self.width / self.tf_widget.N_bins,
                           scale_y = self.height)

    def on_key_press(self, *args, **kwargs):
        pass

    def on_mouse_drag(self, *args, **kwargs):
        pass

    def update_sprite(self):
        bits = np.vstack(self.tf_widget.colors[c][:,2]
                         for c in ('red', 'green', 'blue')).copy(order="C")
        self.image_data.set_data("RGB", bits.shape[1]*3, get_numpy_data(bits))
        self.sprite.image.blit_into(self.image_data, 0, 0, 0)

    def on_draw(self):
        #self.update_sprite()
        self.sprite.draw()

class TransferFunctionWindow(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super(TransferFunctionWindow, self).__init__(*args, **kwargs)
        self.widgets = []

    def on_draw(self):
        self.clear()
        for w in self.widgets:
            w.on_draw()
        self.flip()

    def on_key_press(self, symbol, modifiers):
        for w in self.widgets:
            if w.on_key_press(symbol, modifiers):
                break
        else:
            super(TransferFunctionWindow, self).on_key_press(symbol, modifiers)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        for w in self.widgets:
            if w.on_mouse_drag(x, y, dx, dy, buttons, modifiers):
                break

window = TransferFunctionWindow()
tfw = TransferFunctionWidget(window, 0, 0, window.width, window.height - 100)
tfi = TransferFunctionImage(window, 0, window.height - 100, window.width, 100, tfw)
window.widgets.append(tfw)
window.widgets.append(tfi)
pyglet.app.run()
