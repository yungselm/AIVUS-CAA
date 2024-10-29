from loguru import logger
from PyQt5.QtWidgets import QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsLineItem
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QPen
import matplotlib.pyplot as plt
import pandas as pd
pd.options.mode.chained_assignment = None # default='warn'

from scipy.ndimage import gaussian_filter1d

from report.report import report


class ResultsPlot(QMainWindow):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.pullback_speed = main_window.metadata.get('pullback_speed', 1)
        self.report_data = report(main_window, suppress_messages=True)
        
        if self.report_data is None:
            logger.error('No report data available to plot')
            self.close()  # Close the window if there's no data
            return
        
        self.setWindowTitle('Results Plot')
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.setCentralWidget(self.view)
        self.plot_results()

    def plot_results(self):
        logger.info('Plotting results')
        self.scene.clear()
        self.scene.setSceneRect(0, 0, 1000, 800)

        df = self.prep_data()

        # Create a matplotlib figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot lumen area by phase
        for phase, group in df.groupby('phase'):
            # Smooth the lumen_area
            smoothed_area = gaussian_filter1d(group['lumen_area'], sigma=2)  # Adjust sigma for smoothing
            ax1.plot(group['distance'], smoothed_area, label=f'Lumen Area - Phase {phase}')
        
        ax1.set_xlabel('Distance (mm)')
        ax1.set_ylabel('Lumen Area (mm²)')
        ax1.set_title('Lumen Area vs Distance by Phase')
        ax1.legend()

        # Plot elliptic ratio by phase
        for phase, group in df.groupby('phase'):
            # Smooth the elliptic_ratio
            smoothed_ratio = gaussian_filter1d(group['elliptic_ratio'], sigma=2)  # Adjust sigma for smoothing
            ax2.plot(group['distance'], smoothed_ratio, label=f'Elliptic Ratio - Phase {phase}')
        
        ax2.set_xlabel('Distance (mm)')
        ax2.set_ylabel('Elliptic Ratio')
        ax2.set_title('Elliptic Ratio vs Distance by Phase')
        ax2.legend()

        # Save the plot to a QImage
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        img = QImage(fig.canvas.buffer_rgba(), width, height, QImage.Format_ARGB32)

        self.scene.addPixmap(QPixmap.fromImage(img))


    def prep_data(self):
        # Filter to keep only 'phase' != '-'
        df = self.report_data[self.report_data['phase'] != '-'].copy()  # Use copy to avoid warnings

        df_dia = df[df['phase'] == 'D'].copy()  # Ensure a copy
        df_sys = df[df['phase'] == 'S'].copy()  # Ensure a copy

        # Calculate distance safely
        df_dia.loc[:, 'distance'] = (df_dia['frame'].max() - df_dia['frame']) / 30 * self.pullback_speed
        df_sys.loc[:, 'distance'] = (df_sys['frame'].max() - df_sys['frame']) / 30 * self.pullback_speed

        df = pd.concat([df_dia, df_sys])

        return df

    def closeEvent(self, event):
        self.main_window.results_plot = None
        event.accept()
