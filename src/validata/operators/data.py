"""Module for data operators."""

from validata.operators.base import DataOperator


class MeanOperator(DataOperator):
    """Computes the mean value across columns."""

    symbol = "mean"

    def __call__(self, df):
        return df.mean(axis=1).to_frame()


class MedianOperator(DataOperator):
    """Computes the median value across columns."""

    symbol = "median"

    def __call__(self, df):
        return df.median(axis=1).to_frame()


class MinOperator(DataOperator):
    """Computes the minumum value across columns."""

    symbol = "min"

    def __call__(self, df):
        return df.min(axis=1).to_frame()


class MaxOperator(DataOperator):
    """Computes the maximum value across columns."""

    symbol = "max"

    def __call__(self, df):
        return df.max(axis=1).to_frame()


class SumOperator(DataOperator):
    """Computes the sum across columns."""

    symbol = "sum"

    def __call__(self, df):
        return df.sum(axis=1).to_frame()
