"""Module for mixin classes for shared functionality."""

import pandas as pd


class CompareMixin:
    """Mixin class for all comparison operators."""

    valid = "eq", "ne", "lt", "lte", "gt", "gte"

    @staticmethod
    def _compare_series(series, target, comp_func):

        # Create series of the same dtype
        if not isinstance(target, pd.Series):
            target = pd.Series(target, index=series.index, dtype=series.dtype)
        elif series.dtype != target.dtype:
            target = target.astype(series.dtype)

        return comp_func(series, target)

    def _compare(self, data, target, comp):

        # Get appropriate comparisson method
        if comp not in self.valid:
            raise ValueError(
                f"Invalid comparisson {comp!r}, choose from: " + ".".join(self.valid)
            )
        comp_func = getattr(pd.Series, comp)

        # Compare two series (ColumnComparator)
        if isinstance(data, pd.Series):
            return self._compare_series(data, target, comp_func)

        # Compare to all columns in a DataFrame (RowComparator)
        return data.apply(
            self._compare_series, target=target, comp_func=comp_func, axis=1
        )


class OutlierMixin:
    """Mixin class for common outlier detection functionality.
    Checks whether a record contains an outlier or extreme value given
    a provided detection method. Available methods are:

    - IQR: Uses Inter-Quartile Range (IQR) + whiskers (Tukey's method)
    - SD:  Uses mean + standard deviation (SD)
    - MAD: Uses Mean Absolute Deviation (MADe)

    See also: http://d-scholarship.pitt.edu/7948/1/Seo.pdf
    """

    @staticmethod
    def _outlier_iqr(series, whisker_low=1.5, whisker_high=1.5):
        """Marks outliers using the Inter-Quartile Range method."""

        # Calculate Q1, Q2 and IQR
        qlow = series.quantile(0.25)
        qhigh = series.quantile(0.75)
        iqr = qhigh - qlow

        # Compute filter using IQR with optional whiskers
        lim_low = -float("inf") if whisker_low is None else qlow - whisker_low * iqr
        lim_high = float("inf") if whisker_high is None else qhigh + whisker_high * iqr

        return (series <= lim_low) | (series >= lim_high)

    @staticmethod
    def _outlier_sd(series, whisker_low=2, whisker_high=2):
        """Marks outliers using mean and SD."""

        mean = series.mean()
        std = series.std()

        # Compute filter using mean and SD with optional whiskers
        lim_low = -float("inf") if whisker_low is None else mean - whisker_low * std
        lim_high = float("inf") if whisker_high is None else mean + whisker_high * std

        return (series <= lim_low) | (series >= lim_high)

    @staticmethod
    def _outlier_mad(series, whisker_low=2, whisker_high=2):
        """Marks outliers using Median Absolute Deviation."""

        median = series.median()
        made = 1.483 * (series - median).abs().median()

        # Compute filter using MAD and optional whiskers
        lim_low = -float("inf") if whisker_low is None else median - whisker_low * made
        lim_high = (
            float("inf") if whisker_high is None else median + whisker_high * made
        )

        return (series <= lim_low) | (series >= lim_high)
