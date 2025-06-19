import sys
import hydra
import qdarktheme

from omegaconf import DictConfig
from PyQt5.QtWidgets import QApplication

from version import __version__
from gui.gui import Master

@hydra.main(version_base=None, config_path='.', config_name='config')
def main(config: DictConfig) -> None:
    qdarktheme.enable_hi_dpi()
    app = QApplication(sys.argv)
    app.setApplicationVersion(__version__)
    qdarktheme.setup_theme('auto')
    Master(config)
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
