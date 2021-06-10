"""Module for row-wise comparator classes."""

import re

from validata.comparators.base import RowComparator
from validata.comparators.mixins import CompareMixin, OutlierMixin


class NullComparator(RowComparator):
    """Checks whether the data is missing (no target required)."""

    symbol = "missing"

    def __call__(self, df, target=None):
        return df.isna()


class NotNullComparator(RowComparator):
    """Checks whether the data is not missing (no target required)."""

    symbol = "not missing"

    def __call__(self, df, target=None):
        return ~df.isna()


class EqComparator(RowComparator, CompareMixin):
    """Checks for identical values."""

    symbol = "=="

    def __call__(self, df, target):
        return self._compare(df, target, "eq")


class UnEqComparator(RowComparator, CompareMixin):
    """Checks for non-identical values."""

    symbol = "!="

    def __call__(self, df, target):
        return self._compare(df, target, "ne")


class GtComparator(RowComparator, CompareMixin):
    """Checks whether the data is greater than the target."""

    symbol = ">"

    def __call__(self, df, target):
        return self._compare(df, target, "gt")


class GtEqComparator(RowComparator, CompareMixin):
    """Checks whether the data is greater than or equal to the target."""

    symbol = ">="

    def __call__(self, df, target):
        return self._compare(df, target, "gte")


class LtComparator(RowComparator, CompareMixin):
    """Checks whether the data is less than the target."""

    symbol = "<"

    def __call__(self, df, target):
        return self._compare(df, target, "lt")


class LtEqComparator(RowComparator, CompareMixin):
    """Checks whether the data is less than or equal to the target."""

    symbol = "<="

    def __call__(self, df, target):
        return self._compare(df, target, "lte")


class InComparator(RowComparator):
    """Checks whether the data are present in the target list."""

    symbol = "in"

    def __call__(self, df, target):
        try:
            target = [t.strip() for t in target.split(",")]
        except AttributeError as error:
            raise RuntimeError(
                f"InComparator: Cannot construct list from target '{target}'."
            ) from error

        # Comparing as string
        return df.astype(str).isin(target)


class BetweenComparator(RowComparator):
    """Checks whether the data falls in the target range."""

    symbol = "between"

    def __call__(self, df, target):
        self._check_dtypes(df, "number")

        try:
            low, high = (float(t) for t in target.split(":"))
        except AttributeError as error:
            raise RuntimeError(
                f"BetweenComparator: Cannot split bound from target '{target}'."
            ) from error
        except ValueError as error:
            raise RuntimeError(
                f"BetweenComparator: Non-numeric bounds in target '{target}'."
            ) from error

        return df.applymap(lambda x: low <= float(x) <= high)


class ContainsComparator(RowComparator):
    """Checks whether data matches a regular expression pattern."""

    symbol = "contains"

    def __call__(self, df, target):
        self._check_dtypes(df, "object")

        # Apply target as regex, mark missing values (ex. wrong data type) as False.
        return df.apply(lambda col: col.str.contains(target, regex=True, na=False))


class RankComparator(RowComparator):
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


class OutlierComparator(RowComparator, OutlierMixin):
    """Checks whether a record is an extreme value (outlier) given
    a provided detection method (IQR / SD / MAD).
    """

    symbol = "is outlier by"

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
