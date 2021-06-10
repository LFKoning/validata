"""Module containing the Parser class for complex boolean expressions."""

import logging
import numpy as np

from validata.tokenizer import Tokenizer
from validata.comparators import Comparator
from validata.operators import Operator


class Parser:
    """
    Parses for complex, nested boolean expressions.

    Parameters
    ----------
    expression : str
        Logical expression as string, some examples:
        - column_1 >= 30000
        - column_1 >= 2 and (column_2 missing or column_2 missing)
        - column_1 == 1 and any dummy_* == 1
    name : Optional[str]
        Name for the validation, defaults to "validation_result".
    """

    def __init__(self, expression, name="validation_result"):
        self._log = logging.getLogger(__name__)

        token_types = {op: "OPERATOR" for op in Operator.list()}
        token_types.update({comp: "COMPARATOR" for comp in Comparator.list()})
        self._tokenizer = Tokenizer(expression, token_types)

        self._name = name

    def evaluate(self, df):
        """
        Evaluate the logical expression against the provided pandas DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            Data set to evaluate the expression against.

        Returns
        -------
        pandas.DataFrame
            Single-column DataFrame with validation results.
        """

        result = None

        for token in self._tokenizer:

            self._log.debug("Processing token: %s [%s]", token.value, token.type)

            # Handle grouping / nesting
            if token.type == "GROUP_OPEN":
                self._log.debug("Entering nested expression.")
                result = self.evaluate(df)

            elif token.type == "GROUP_CLOSE":
                self._log.debug("Exiting nested expression.")
                break

            elif token.type in ["AND", "OR"]:
                self._log.debug("Processing %s expression.", token.value)

                if result is None:
                    raise ValueError("Use of AND / OR without left-hand expression.")

                # Get right hand side, compute results
                next_token = next(self._tokenizer)
                if next_token.type == "GROUP_OPEN":
                    self._log.debug("Entering nested right-hand expression.")
                    right_hand = self.evaluate(df)

                else:
                    self._log.debug("Evaluating right-hand partial expression.")

                    # Collect and evaluate the partial expression
                    self._tokenizer.rewind()
                    op, cols, comp, val = self._collect_partial()
                    self._log.debug(
                        "Operator: %s, Columns: %s, Comparator: %s, Value: %s",
                        op,
                        cols,
                        comp,
                        val,
                    )
                    right_hand = self._evaluate_partial(df, cols, comp, val, op)

                # Apply and / or operation
                if token.type == "AND":
                    # result = result and right_hand
                    result = np.all(
                        result.join(right_hand, rsuffix="_rh"), axis=1
                    ).to_frame()

                else:
                    # result = result or right_hand
                    result = np.any(
                        result.join(right_hand, rsuffix="_rh"), axis=1
                    ).to_frame()

            else:
                self._log.debug("Evaluating partial expression.")

                if result is not None:
                    raise ValueError("Expected and / or, got expression instead.")

                # Collect and evaluate the partial expression
                self._tokenizer.rewind()
                op, cols, comp, val = self._collect_partial()
                self._log.debug(
                    "Operator: %s, Columns: %s, Comparator: %s, Value: %s",
                    op,
                    cols,
                    comp,
                    val,
                )
                result = self._evaluate_partial(df, cols, comp, val, op)
        result.columns = [self._name]
        return result

    def _collect_partial(self):
        """
        Collect and classify the tokens of a partial expression.

        Returns
        -------
        Tuple[str, list, str, str]
            Tuple of operator, columns, comparator, and target value.
        """

        # Collect all tokens of the sub-expression
        end_sub = "AND", "OR", "GROUP_OPEN", "GROUP_CLOSE"
        tokens = []
        for token in self._tokenizer:
            tokens.append(token)
            next_token = self._tokenizer.peek()
            if next_token and next_token.type in end_sub:
                break

        # Initialize categories
        operator = None
        comparator = None
        columns = []
        value = []

        # Classify the sub-expression tokens
        for token in tokens:
            if token.type == "OPERATOR":
                operator = token.value
            elif token.type == "COMPARATOR":
                comparator = token.value
            elif not comparator:
                if token.value == "+":
                    continue
                columns.append(token)
            else:
                value.append(token.value)

        return operator, columns, comparator, " ".join(value)

    @staticmethod
    def _select_columns(df, column_tokens):
        """
        Select columns from a DataFrame based on al list of column tokens.
        """

        selected = []
        for token in column_tokens:

            # Match by regular expression
            if token.type == "REGEX":
                cols = df.columns[df.columns.str.contains(token.value, regex=True)]
                selected.extend(cols)

            # Match by name
            else:
                col_name = token.value
                if col_name.endswith("*"):
                    cols = {col for col in df.columns if col.startswith(col_name[:-1])}
                    if not cols:
                        raise RuntimeError(
                            f"No columns were selected using expression: '{col_name}'."
                        )
                    selected.extend(cols)

                else:
                    if col_name not in df.columns:
                        raise RuntimeError(
                            f"Column '{col_name}' not found in the data."
                        )
                    selected.append(col_name)

        return set(selected)

    # pylint: disable=too-many-arguments
    def _evaluate_partial(self, df, columns, comparator, target, operator=None):
        """
        Evaluate the partial expression against the provided data frame.

        Parameters
        ----------
        df : pandas.DataFrame
            Data set to evaluate the partial expression against.
        columns : list
            List of column names or regular expressions.
        comparator : str
            String identifying a Comparator class.
        target : str
            Target value to compare data against.
        operator : Optional[str]
            String identifying an Operator calss (if specified).

        Returns
        -------
        pandas.DataFrame
            Single-column DataFrame with validation results.
        """

        # Process and log partial expression components
        if operator:
            operator = Operator.get(operator)
            self._log.debug("Using operator: %s.", operator.__class__.__name__)

        columns = self._select_columns(df, columns)
        self._log.debug("Selected columns: %s.", ", ".join(columns))

        comparator = Comparator.get(comparator)
        self._log.debug("Using comparator: %s.", comparator.__class__.__name__)

        # Target is either constant or column reference
        if target.startswith("{") and target.endswith("}"):
            target = target[1:-1]
            self._log.debug("Using target column: %s.", target)

            if target not in df.columns:
                raise RuntimeError(f"Target column '{target}' does not exist.")
            target = df[target]

        else:
            self._log.debug("Using target: %s.", target)

        # Perform comparison and operation
        if operator:
            if operator.type == "logical":
                result = df[columns].pipe(comparator, target=target).pipe(operator)
            elif operator.type == "data":
                result = df[columns].pipe(operator).pipe(comparator, target=target)
            else:
                raise TypeError(
                    f"Unknown operator type: {operator.__class__.__name__}."
                )

        else:
            if len(columns) > 1:
                raise RuntimeError(
                    f"Too many columns ({len(columns)}) to compare, either select "
                    "a single column or reduce columns using an operator."
                )

            result = df[columns].pipe(comparator, target=target)

        return result
