import polars as pl
from polars._typing import PythonLiteral

IntoExpr = PythonLiteral | pl.Expr | pl.Series


def parse_into_expression(expr: IntoExpr) -> pl.Expr:
    """Parse a literal, an expression or a Series into a Polars expression."""
    if isinstance(expr, pl.Expr):
        return expr
    if isinstance(expr, str):
        return pl.col(expr)
    return pl.lit(expr)


def mpe(y_true: IntoExpr, y_pred: IntoExpr) -> pl.Expr:
    """Return an expression to compute the Mean Percentage Error."""
    y_true = parse_into_expression(y_true)
    y_pred = parse_into_expression(y_pred)

    # Add small epsilon to avoid zero division
    return (
        (((y_true - y_pred) / (y_true + 1e-5)) * 100).mean().round(1).alias("MPE (%)")
    )


def mae(y_true: IntoExpr, y_pred: IntoExpr) -> pl.Expr:
    """Return an expression to compute the Mean Absolute Error."""
    y_true = parse_into_expression(y_true)
    y_pred = parse_into_expression(y_pred)

    return (y_true - y_pred).abs().mean().round(1).alias("MAE (days)")


def mape(y_true: IntoExpr, y_pred: IntoExpr) -> pl.Expr:
    """Return an expression to compute the Mean Absolute Percentage Error."""
    y_true = parse_into_expression(y_true)
    y_pred = parse_into_expression(y_pred)

    # Add small epsilon to avoid zero division
    return (
        (((y_true - y_pred).abs() / (y_true + 1e-5)) * 100)
        .mean()
        .round(1)
        .alias("MAPE (%)")
    )


def r2_score(y_true: IntoExpr, y_pred: IntoExpr) -> pl.Expr:
    """Return an expression to compute the R-squared score."""
    y_true = parse_into_expression(y_true)
    y_pred = parse_into_expression(y_pred)

    # Sum of Squared Residuals (SS_res)
    ss_res = (y_true - y_pred).pow(2).sum()

    # Total Sum of Squares (SS_tot)
    ss_tot = (y_true - y_true.mean()).pow(2).sum()

    # Calculate R2
    return (1 - (ss_res / (ss_tot + 1e-9))).mul(100).round(1).alias("R2 (%)")


def pinball_loss(y_true: IntoExpr, y_pred: IntoExpr, alpha: float) -> pl.Expr:
    """Return an expression to compute the Pinball loss."""
    y_true = parse_into_expression(y_true)
    y_pred = parse_into_expression(y_pred)

    residual = y_true - y_pred
    loss = pl.max_horizontal(alpha * residual, (alpha - 1) * residual)
    return loss.mean().round(2).alias(f"Pinball q_{alpha}")


def coverage(
    lower_bound: IntoExpr,
    upper_bound: IntoExpr,
    value: IntoExpr = "True Duration",
) -> pl.Expr:
    """Return an expression to compute the coverage of an interval over values."""
    value = parse_into_expression(value)
    lower_bound = parse_into_expression(lower_bound)
    upper_bound = parse_into_expression(upper_bound)

    return (
        ((value >= lower_bound) & (value <= upper_bound))
        .mean()
        .mul(100)
        .round(1)
        .cast(pl.Utf8)
        + pl.lit("%")
    ).alias("Coverage")


def quantile_crossing(
    lower_bound_col: str = "Lower Bound",
    predicted_col: str = "Predicted Duration",
    upper_bound_col: str = "Upper Bound",
) -> pl.Expr:
    """Return an expression to compute the number of quantile crossings."""
    return (
        pl.any_horizontal(
            pl.col(lower_bound_col) >= pl.col(predicted_col),
            pl.col(lower_bound_col) >= pl.col(upper_bound_col),
            pl.col(predicted_col) >= pl.col(upper_bound_col),
        )
        .sum()
        .alias("Quantile Crossing")
    )
