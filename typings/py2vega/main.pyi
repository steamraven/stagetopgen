"""
This type stub file was generated by pyright.
"""

import gast

"""Python to VegaExpression transpiler."""
class Variable:
    """Helper class for defining a variable in whitelisting."""
    def __init__(self, name, members) -> None:
        """Construct a Variable, given its name and available members."""
        ...
    


operator_mapping = ...
builtin_function_mapping = ...
class Py2VegaSyntaxError(SyntaxError):
    def __init__(self, message) -> None:
        ...
    


class Py2VegaNameError(NameError):
    def __init__(self, message) -> None:
        ...
    


def validate(nodes, origin_node): # -> None:
    """Check whether or not a list of nodes is valid.

    A list of nodes is considered valid when:
    - it is not empty
    - the last node is an `if` statement or a `return` statement
    - everything but the last element is not an `if` statement or a `return` statement
    """
    ...

def valid_attribute_impl(node, var): # -> bool:
    """Check the attribute access validity. Returns True if the member access is valid, False otherwise."""
    ...

def valid_attribute(node, whitelist): # -> bool:
    """Check the attribute access validity. Returns True if the member access is valid, False otherwise."""
    ...

class VegaExpressionVisitor(gast.NodeVisitor):
    """Visitor that turns a Node into a Vega expression."""
    def __init__(self, whitelist, scope=...) -> None:
        ...
    
    def generic_visit(self, node):
        """Throwing an error by default."""
        ...
    
    def visit_Return(self, node): # -> Any:
        """Turn a Python return statement into a Vega-expression."""
        ...
    
    def visit_If(self, node): # -> str:
        """Turn a Python if statement into a Vega-expression."""
        ...
    
    def visit_Constant(self, node): # -> str:
        """Turn a Python Constant node into a Vega-expression."""
        ...
    
    def visit_Tuple(self, node): # -> LiteralString:
        """Turn a Python tuple expression into a Vega-expression."""
        ...
    
    def visit_List(self, node): # -> LiteralString:
        """Turn a Python list expression into a Vega-expression."""
        ...
    
    def visit_Dict(self, node):
        """Turn a Python dict expression into a Vega-expression."""
        ...
    
    def visit_Assign(self, node): # -> Literal['null']:
        """Turn a Python assignment expression into a Vega-expression. And save the assigned variable in the current scope."""
        ...
    
    def visit_UnaryOp(self, node): # -> str:
        """Turn a Python unaryop expression into a Vega-expression."""
        ...
    
    def visit_BoolOp(self, node): # -> str:
        """Turn a Python boolop expression into a Vega-expression."""
        ...
    
    def visit_BinOp(self, node): # -> str:
        """Turn a Python binop expression into a Vega-expression."""
        ...
    
    def visit_IfExp(self, node): # -> str:
        """Turn a Python if expression into a Vega-expression."""
        ...
    
    def visit_Compare(self, node): # -> str:
        """Turn a Python compare expression into a Vega-expression."""
        ...
    
    def visit_Name(self, node): # -> _Identifier:
        """Turn a Python name expression into a Vega-expression."""
        ...
    
    def visit_Call(self, node): # -> str | LiteralString:
        """Turn a Python call expression into a Vega-expression."""
        ...
    
    def visit_Subscript(self, node): # -> str:
        """Turn a Python Subscript node into a Vega-expression."""
        ...
    
    def visit_Index(self, node): # -> Any:
        """Turn a Python subscript index into a Vega-expression."""
        ...
    
    def visit_Attribute(self, node): # -> str:
        """Turn a Python attribute expression into a Vega-expression."""
        ...
    


def py2vega(value, whitelist=...): # -> Any:
    """Convert Python code or Python function to a valid Vega expression."""
    ...

