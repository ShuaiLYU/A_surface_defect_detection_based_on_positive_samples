
'''
This is  a  defect detection library.
'''
import os
def get_cur_path():
    return os.path.abspath(os.path.dirname(__file__))

from  .utils import *
from .utils import  _pair

from .transforms import *