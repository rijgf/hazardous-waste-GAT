"""Rebuild the revised-parameter manuscript only after all v4 results exist.

Importing this entry does not generate data, train, search, or write reports.
The revision runner supplies the identical source lock and registry routing used
by the formal runs; the shared report still enforces complete numerical replay.
"""
from pathlib import Path


def configure():
    import run_parameter_revision as revision
    import run_pareto_experiments as engine
    import report_pareto_experiments as report
    import plot_pareto_sensitivity as plot

    engine.OUT = report.OUT = plot.OUT = revision.OUT
    report.ENTRY_SCRIPT = Path(__file__).resolve()
    return report


def main():
    configure().build()


if __name__ == '__main__':
    main()
