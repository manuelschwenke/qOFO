"""HTML reports for the tuning module.

Two public entry points:
    write_tuning_report     -- summarise an Optuna study.
    write_validation_report -- summarise a validation-set run.
"""

from tuning.reports.tuning_report import write_tuning_report
from tuning.reports.validation_report import write_validation_report

__all__ = ["write_tuning_report", "write_validation_report"]
