import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir0 = os.path.dirname(parentdir)
sys.path.extend([parentdir, parentdir0])