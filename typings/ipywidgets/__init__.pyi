"""
This type stub file was generated by pyright.
"""

import os
from ._version import __jupyter_widgets_base_version__, __jupyter_widgets_controls_version__, __protocol_version__, __version__
from traitlets import dlink, link
from IPython import get_ipython
from comm import get_comm_manager
from .widgets import *

"""Interactive widgets for the Jupyter notebook.

Provide simple interactive controls in the notebook.
Each Widget corresponds to an object in Python and Javascript,
with controls on the page.

To put a Widget on the page, you can display it with Jupyter's display machinery::

    from ipywidgets import IntSlider
    slider = IntSlider(min=1, max=10)
    display(slider)

Moving the slider will change the value. Most Widgets have a current value,
accessible as a `value` attribute.
"""
def load_ipython_extension(ip): # -> None:
    """Set up Jupyter to work with widgets"""
    ...

def register_comm_target(kernel=...): # -> None:
    """Register the jupyter.widget comm target"""
    ...

