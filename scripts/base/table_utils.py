#!/usr/bin/env python3
"""
Common utilities for table formatting and output in CSV and LaTeX formats.
"""

import pandas as pd
from typing import Optional


def calculate_significance_stars(estimator: float, std_error: float) -> str:
    """Calculate significance stars based on z-score.

    Args:
        estimator: The coefficient estimate
        std_error: The standard error

    Returns:
        String with significance stars: *** (p<0.01), ** (p<0.05), * (p<0.10), or empty
    """
    if estimator is None or std_error is None or std_error == 0:
        return ""

    z_score = abs(estimator / std_error)

    if z_score > 2.576:  # p < 0.01
        return "***"
    elif z_score > 1.96:  # p < 0.05
        return "**"
    elif z_score > 1.645:  # p < 0.10
        return "*"
    else:
        return ""


def format_coefficient(
    estimator: Optional[float], std_error: Optional[float], decimal_places: int = 3
) -> str:
    """Format a coefficient with standard error and significance stars.

    Args:
        estimator: The coefficient estimate
        std_error: The standard error
        decimal_places: Number of decimal places to display

    Returns:
        Formatted string like "0.123 (0.045)**"
    """
    if estimator is None or std_error is None:
        return ""

    stars = calculate_significance_stars(estimator, std_error)
    return f"{estimator:.{decimal_places}f} ({std_error:.{decimal_places}f}){stars}"


def dataframe_to_latex(
    df: pd.DataFrame,
    caption: str = "",
    label: str = "",
    adjustbox: bool = True,
    add_midrule_before_row: Optional[str] = None,
) -> str:
    """Convert DataFrame to LaTeX table format.

    Args:
        df: The DataFrame to convert
        caption: Table caption
        label: Table label for references
        adjustbox: Whether to use adjustbox for wide tables
        add_midrule_before_row: Add \\midrule before row with this name/index

    Returns:
        Complete LaTeX table environment as string
    """
    latex = df.to_latex(
        escape=False, column_format="l" + "c" * len(df.columns), bold_rows=True
    )

    if add_midrule_before_row:
        lines = latex.split("\n")
        for i, line in enumerate(lines):
            if add_midrule_before_row in line or (
                add_midrule_before_row == ""
                and line.strip() == r" &  &  &  &  &  &  \\"[: len(line.strip())]
            ):
                lines.insert(i, r"\midrule")
                break
        latex = "\n".join(lines)

    if adjustbox:
        latex_full = f"""\\begin{{table}}[htbp]
\\centering
\\adjustbox{{width=\\textwidth}}{{
{latex}
}}
\\caption{{{caption}}}
\\label{{{label}}}
\\end{{table}}"""
    else:
        latex_full = f"""\\begin{{table}}[htbp]
\\centering
{latex}
\\caption{{{caption}}}
\\label{{{label}}}
\\end{{table}}"""

    return latex_full


def save_table(
    df: pd.DataFrame,
    output_path: str,
    format: str = "csv",
    caption: str = "",
    label: str = "",
    **kwargs,
) -> None:
    """Save a DataFrame as CSV or LaTeX table.

    Args:
        df: The DataFrame to save
        output_path: Path to save the file
        format: Output format ('csv' or 'latex')
        caption: Table caption (for LaTeX)
        label: Table label (for LaTeX)
        **kwargs: Additional arguments passed to dataframe_to_latex
    """
    if format.lower() == "latex":
        latex_output = dataframe_to_latex(df, caption, label, **kwargs)
        with open(output_path, "w") as f:
            f.write(latex_output)
    else:
        df.to_csv(output_path, index=False)
