"""
Module providing Comparator classes used in the data validation tool.

Comparator classes perform logical comparisons to a provided target
value. Their `__call__()` method should accept a pandas DataFrame and
a comparison target value.

The method should compare all values in the DataFrame to the target and
return a DataFrame of identical shape containing only boolean values.

Comparator classes should always extend the `Comparators` base class and
be initialized via the `Comparators.get()` method.
"""
import re
import pandas as pd

from validata.base_classes import Comparator


class EqComparator(Comparator):
    """Checks for identical values."""

    symbol = "=="

    def __call__(self, df, target):
        res = pd.DataFrame()

        if not isinstance(target, pd.Series):
            target = pd.Series(target, index=df.index)

        for col in df.columns:
            res[col] = df[col] == target.astype(df[col].dtype)

        return res


class UnEqComparator(Comparator):
    """Checks for non-identical values."""

    symbol = "!="

    def __call__(self, df, target):
        eq_comp = EqComparator()
        return ~eq_comp(df, target)


class GtComparator(Comparator):
    """Checks whether the data is greater than the target."""

    symbol = ">"

    def __call__(self, df, target):
        self._check_dtypes(df, "number")

        if not isinstance(target, pd.Series):
            target = pd.Series(target, index=df.index)

        return df.gt(target.astype(float), axis=0)


class GtEqComparator(Comparator):
    """Checks whether the data is greater than or equal to the target."""

    symbol = ">="

    def __call__(self, df, target):
        self._check_dtypes(df, "number")

        if not isinstance(target, pd.Series):
            target = pd.Series(target, index=df.index)

        return df.ge(target.astype(float), axis=0)


class LtComparator(Comparator):
    """Checks whether the data is less than the target."""

    symbol = "<"

    def __call__(self, df, target):
        self._check_dtypes(df, "number")

        if not isinstance(target, pd.Series):
            target = pd.Series(target, index=df.index)

        return df.lt(target.astype(float), axis=0)


class LtEqComparator(Comparator):
    """Checks whether the data is less than or equal to the target."""

    symbol = "<="

    def __call__(self, df, target):
        self._check_dtypes(df, "number")

        if not isinstance(target, pd.Series):
            target = pd.Series(target, index=df.index)

        return df.le(target.astype(float), axis=0)


class InComparator(Comparator):
    """Checks whether the data are present in the target list."""

    symbol = "in"

    def __call__(self, df, target):
        try:
            target = [t.strip() for t in target.split(",")]
        except AttributeError:
            raise RuntimeError(
                f"InComparator: Cannot construct list from target '{target}'."
            )

        # Comparing as string
        return df.astype(str).isin(target)


class BetweenComparator(Comparator):
    """Checks whether the data falls in the target range."""

    symbol = "between"

    def __call__(self, df, target):
        self._check_dtypes(df, "number")

        try:
            low, high = (float(t) for t in target.split(":"))
        except AttributeError:
            raise RuntimeError(
                f"BetweenComparator: Cannot split bound from target '{target}'."
            )
        except ValueError:
            raise RuntimeError(
                f"BetweenComparator: Non-numeric bounds in target '{target}'."
            )

        return df.applymap(lambda x: low <= float(x) <= high)


class NullComparator(Comparator):
    """Checks whether the data is missing (no target required)."""

    symbol = "missing"

    def __call__(self, df, target=None):
        return df.isna()


class NotNullComparator(Comparator):
    """Checks whether the data is not missing (no target required)."""

    symbol = "not missing"

    def __call__(self, df, target=None):
        return ~df.isna()


class RankComparator(Comparator):
    """Ranks records and indicates whether they fall above or below a threshold."""

    symbol = "ranks in"

    def __call__(self, df, target):
        self._check_dtypes(df, "number")

        match = re.match(
            r"(?P<from>top|bottom)\s+(?P<rank>[0-9]+)\s*(?P<pct>%)?", target
        )
        if not match:
            raise ValueError(
                f"RankComparator: Invalid target '{target}', "
                "use <top|bottom> <rank> (%)."
            )

        ascending = match.group("from") == "bottom"
        pct = match.group("pct") is not None
        rank = int(match.group("rank"))
        if pct:
            if not 0 < rank < 100:
                raise ValueError(
                    "RankComparator: Percentile rank must be between 0 - 100, "
                    f"got {rank} instead."
                )
            rank = rank / 100

        ranks = df.rank(ascending=ascending, pct=pct)
        return ranks <= rank


class ContainsComparator(Comparator):
    """
    Checks whether data matches a regular expression pattern.
    """

    symbol = "contains"

    def __call__(self, df, target):
        self._check_dtypes(df, "object")

        # Apply target as regex, mark missing values (ex. wrong data type) as False.
        return df.apply(lambda col: col.str.contains(target, regex=True, na=False))


class OutlierComparator(Comparator):
    """
    Checks whether a record contains an outlier or extreme value given
    a provided detection method. Available methods are:

    - IQR: Uses Inter-Quartile Range (IQR) + whiskers (Tukey's method)
    - SD:  Uses mean + standard deviation (SD)
    - MAD: Uses Mean Absolute Deviation (MADe)

    See also: http://d-scholarship.pitt.edu/7948/1/Seo.pdf
    """

    symbol = "is outlier by"

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

    def __call__(self, df, target="1.5 IQR"):
        match = re.match(
            r"(?P<side>\+|\-)?\s*(?P<whiskers>[0-9\.]+)\s+(?P<method>IQR|SD|MAD)",
            target,
            re.IGNORECASE,
        )
        if not match:
            raise ValueError(
                f"OutlierComparator: Invalid target '{target}', "
                "use (+|-)<whisker> <IQR|SD|MAD>."
            )

        # Set whiskers
        whisker_low = whisker_high = float(match.group("whiskers"))
        if match.group("side") == "+":
            whisker_low = None
        elif match.group("side") == "-":
            whisker_high = None

        # Select method

        if match.group("method").lower() == "sd":
            method = self._outlier_sd
        elif match.group("method").lower() == "mad":
            method = self._outlier_mad
        else:
            method = self._outlier_iqr

        return df.apply(
            method, whisker_low=whisker_low, whisker_high=whisker_high, axis=0
        )
