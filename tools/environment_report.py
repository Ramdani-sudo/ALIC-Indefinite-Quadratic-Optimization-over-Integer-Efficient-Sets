import platform
import sys
import numpy
import scipy
import sympy
try:
    import xlsxwriter
    xw = xlsxwriter.__version__
except Exception:
    xw = "not installed"

print("Python:", sys.version.replace("\n", " "))
print("Platform:", platform.platform())
print("NumPy:", numpy.__version__)
print("SciPy:", scipy.__version__)
print("SymPy:", sympy.__version__)
print("XlsxWriter:", xw)
