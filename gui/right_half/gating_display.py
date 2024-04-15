import darkdetect
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from loguru import logger
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT


class GatingDisplay(FigureCanvasQTAgg):
    def __init__(self, main_window, parent=None, width=None, height=None, dpi=100):
        if darkdetect.isDark():
            plt.style.use('dark_background')

        width = main_window.config.display.image_size if width is None else width
        height = width // 2 if height is None else height
        width /= dpi  # convert pixels to inches
        height /= dpi
        fig = plt.figure(figsize=(width, height), dpi=dpi)
        super().__init__(fig)
        self.setParent(parent)
        self.toolbar = NavigationToolbar2QT(self, parent)