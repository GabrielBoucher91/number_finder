"""
Main file for the project
"""
from PyQt5 import QtWidgets
import application
import applicationV2
import applicationV3
import applicationV4

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ui = applicationV4.ApplicationV4()
    ui.show()
    sys.exit(app.exec_())