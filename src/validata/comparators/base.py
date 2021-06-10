"""Factory / base class for Comparator classes.

Comparator classes perform logical comparisons to a provided target
value. Their `__call__()` method should accept a pandas DataFrame and
a comparison target value.

The method should compare all values in the DataFrame to the target and
return a DataFrame of identical shape containing only boolean values.

Comparator classes should always extend the `Comparators` base class and
be initialized via the `Comparators.get()` method.
"""


class Comparator:
    """Abstract factory class for initializing Comparator objects."""

    @classmethod
    def get(cls, symbol):
        """
        Method for getting a Comparator object based on its symbol.

        Parameters
        ----------
        symbol : str
            String (lower case) identifying a comparator, for example "==".

        Returns
        -------
        Comparator
            Instance of the requested Comparator subclass.
        """

        symbols = {comp.symbol: comp for comp in cls.get_subclasses()}
        if symbol not in symbols:
            raise TypeError(f"Unknown Comparator: {symbol}.")

        return symbols[symbol]()

    @classmethod
    def list(cls):
        """Returns a set of available Comparator symbols."""

        return {comp.symbol for comp in cls.get_subclasses()}

    @classmethod
    def get_subclasses(cls):
        """Returns all subclasses recursively."""

        for subclass in cls.__subclasses__():
            if not hasattr(subclass, "symbol"):
                yield from subclass.get_subclasses()
            else:
                yield subclass

    @staticmethod
    def _check_dtypes(df, dtype):
        """Check whether all columns match a certain data type."""

        invalid = set(df.columns) - set(df.select_dtypes(dtype))
        if invalid:
            raise TypeError(
                f"Columns did not match data type '{dtype}': {', '.join(invalid)}."
            )

    @staticmethod
    def _check_axis(provided, *valid):
        """Checks axis against valid value(s)."""

        if provided not in valid:
            raise ValueError("Axis should be either 0 (column-wise) or 1 (row-wise).")


class RowComparator(Comparator):
    """Base class for row-wise comparators."""


class ColumnComparator(Comparator):
    """Base class for columns-wise comparators."""
