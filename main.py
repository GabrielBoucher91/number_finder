"""
Main file for the project
"""
from PyQt5 import QtWidgets
import application

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ui = application.Application()
    ui.show()
    sys.exit(app.exec_())