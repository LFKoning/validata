"""Factory / base class for Operator classes.

Operator classes perform aggregations to reduce multiple columns to a
single one. Therefore the `__call__()` method of an Operater should
accept a DataFrame with 1 or more columns and return a DataFrame
with just a single column.

Operator classes should extend either the DataOperator or LogicalOperator
base class. DataOperator classes operate on the raw data values; for example,
they could compute the sum or the minimum value from a set of columns.

LogicalOperators operate on the boolean values returned by a Comparator. A
Comparator can return multiple columns with boolean values. A LogicalOperator
should reduce those to a single column, for example, by checking whether all
values are True.

All Operator classes should be initialized using the `Operator.get()` method.
"""


class Operator:
    """Abstract factory class for initializing Operator objects."""

    @classmethod
    def get(cls, symbol):
        """
        Method for getting a Operator object based on its symbol.

        Parameters
        ----------
        symbol : str
            String (lower case) identifying a comparator, for example "mean".

        Returns
        -------
        Operator
            Instance of the requested Operator subclass.
        """

        symbols = {op.symbol: op for op in cls.get_subclasses()}
        if symbol not in symbols:
            raise TypeError(f"Unknown Operator: {symbol}.")

        return symbols[symbol]()

    @classmethod
    def list(cls):
        """
        Returns a set of available Comparator symbols.

        Returns
        -------
        set
            Set of available Comparator symbols
        """

        return {op.symbol for op in cls.get_subclasses()}

    @classmethod
    def get_subclasses(cls):
        """Returns all subclasses recursively."""

        for subclass in cls.__subclasses__():
            if not hasattr(subclass, "symbol"):
                yield from subclass.get_subclasses()
            else:
                yield subclass


class DataOperator(Operator):
    """Base class for data operators."""

    type = "data"


class LogicalOperator(Operator):
    """Base class for logical operators."""

    type = "logical"
