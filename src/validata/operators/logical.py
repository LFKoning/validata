"""Module for logical operators."""

from validata.operators.base import LogicalOperator


class AnyOperator(LogicalOperator):
    """Returns True if any column contains True, False otherwise."""

    symbol = "any"

    def __call__(self, df):
        return df.any(axis=1).to_frame()


class AllOperator(LogicalOperator):
    """Returns True if all columns contain True, False otherwise."""

    symbol = "all"

    def __call__(self, df):
        return df.all(axis=1).to_frame()


class NoneOperator(LogicalOperator):
    """Returns True if all column contain False, True otherwise."""

    symbol = "none"

    def __call__(self, df):
        return ~df.any(axis=1).to_frame()
