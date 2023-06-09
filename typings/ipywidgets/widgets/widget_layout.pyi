"""
This type stub file was generated by pyright.
"""

from traitlets import Instance
from .widget import Widget, register

"""Contains the Layout class"""
CSS_PROPERTIES = ...
@register
class Layout(Widget):
    """Layout specification

    Defines a layout that can be expressed using CSS.  Supports a subset of
    https://developer.mozilla.org/en-US/docs/Web/CSS/Reference

    When a property is also accessible via a shorthand property, we only
    expose the shorthand.

    For example:
    - ``flex-grow``, ``flex-shrink`` and ``flex-basis`` are bound to ``flex``.
    - ``flex-wrap`` and ``flex-direction`` are bound to ``flex-flow``.
    - ``margin-[top/bottom/left/right]`` values are bound to ``margin``, etc.
    """
    _view_name = ...
    _view_module = ...
    _view_module_version = ...
    _model_name = ...
    align_content = ...
    align_items = ...
    align_self = ...
    border_top = ...
    border_right = ...
    border_bottom = ...
    border_left = ...
    bottom = ...
    display = ...
    flex = ...
    flex_flow = ...
    height = ...
    justify_content = ...
    justify_items = ...
    left = ...
    margin = ...
    max_height = ...
    max_width = ...
    min_height = ...
    min_width = ...
    overflow = ...
    order = ...
    padding = ...
    right = ...
    top = ...
    visibility = ...
    width = ...
    object_fit = ...
    object_position = ...
    grid_auto_columns = ...
    grid_auto_flow = ...
    grid_auto_rows = ...
    grid_gap = ...
    grid_template_rows = ...
    grid_template_columns = ...
    grid_template_areas = ...
    grid_row = ...
    grid_column = ...
    grid_area = ...
    def __init__(self, **kwargs) -> None:
        ...
    
    border = ...


class LayoutTraitType(Instance):
    klass = Layout
    def validate(self, obj, value): # -> None:
        ...
    


