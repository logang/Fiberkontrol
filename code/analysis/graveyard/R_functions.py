#!/usr/bin/env python                                                                               
import numpy as np

# R imports     
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri as np2ri; np2ri.activate()

# R functions for fitting principal curves and surfaces  
ro.r(
"""
library(ppcor)
""")
pcor = ro.r['pcor.test']
