"""
2D rendering framework using PyGame
"""
from __future__ import division
import os
import sys
import math
import numpy as np
from gym import error

try:
    import PyGame
except ImportError as e:
    raise ImportError('''
    Cannot import PyGame.
    HINT: You can install pygame directly via 'pip install pygame'.
    ''')

class Viewer(object):
    def __init__(self, width, height, display=None):