"""Educational input screening for the demo.

**This is not a process-safety system.** It performs range checks on synthetic
inputs and raises teaching flags so a learner notices *that* certain quantities
matter. It has no hazard model, no site data, no instrumentation, and no
standing whatsoever in a real hazard assessment. Real screening is done through
HAZOP/LOPA, applicable codes, and qualified engineering review.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

__all__ = [
    "Flag",
    "SafetyAssessment",
    "Severity",
    "assess_demo_constraints",
    "evaluate_demo_safety",
]

VALID_MESSAGE = (
    "Valid for educational simulation only. "
    "Real plant decisions require formal process-safety review."
)


class Severity(StrEnum):
    """Teaching-flag severity. Ordered least to most attention-worthy."""

    INFO = "info"
    WARNING = "warning"
    SERIOUS = "serious"


@dataclass(frozen=True)
class Flag:
    """One educational observation about an input value."""

    severity: Severity
    title: str
    detail: str


@dataclass(frozen=True)
class SafetyAssessment:
    """Outcome of screening one synthetic operating point.

    Attributes:
        is_valid: False only when an input is physically impossible, which is a
            *modelling* error, not a safety verdict.
        message: Human-readable summary.
        flags: Educational observations; never a clearance to operate.
    """

    is_valid: bool
    message: str
    flags: list[Flag] = field(default_factory=list)

    @property
    def highest_severity(self) -> Severity | None:
        """The most attention-worthy severity present, or None."""
        order = [Severity.INFO, Severity.WARNING, Severity.SERIOUS]
        present = [f.severity for f in self.flags]
        return max(present, key=order.index) if present else None


# Thresholds are round teaching numbers chosen to make the dashboard respond
# visibly. They are NOT limits from any standard, code, or datasheet.
H2S_NOTICEABLE_PPM = 100.0
H2S_ELEVATED_PPM = 500.0
HIGH_PRESSURE_BAR = 16.0
HIGH_CO2_FRACTION = 0.15
LOW_METHANE_FRACTION = 0.40


def evaluate_demo_safety(
    pressure_bar: float,
    h2s_ppm: float,
    methane_fraction: float,
    co2_fraction: float | None = None,
) -> SafetyAssessment:
    """Range-check inputs and attach educational flags.

    Args:
        pressure_bar: Operating pressure, bar. Must be > 0.
        h2s_ppm: Hydrogen sulphide concentration, ppm. Must be >= 0.
        methane_fraction: Methane fraction, must be within [0, 1].
        co2_fraction: Optional CO₂ fraction, checked when provided.

    Returns:
        A :class:`SafetyAssessment`. ``is_valid=False`` means the numbers cannot
        describe a real stream at all; it does not mean anything else is "safe".
    """
    if pressure_bar <= 0:
        return SafetyAssessment(False, "Invalid pressure input.")
    if not 0 <= methane_fraction <= 1:
        return SafetyAssessment(False, "Methane fraction must be between 0 and 1.")
    if h2s_ppm < 0:
        return SafetyAssessment(False, "H2S concentration cannot be negative.")
    if co2_fraction is not None and not 0 <= co2_fraction <= 1:
        return SafetyAssessment(False, "CO2 fraction must be between 0 and 1.")

    flags: list[Flag] = []

    if h2s_ppm >= H2S_ELEVATED_PPM:
        flags.append(
            Flag(
                Severity.SERIOUS,
                "Elevated H₂S in the synthetic stream",
                "Hydrogen sulphide is acutely toxic and corrosive. Any real stream "
                "in this range drives material selection, detection, ventilation, "
                "and personnel protection decisions that only a qualified team "
                "can make. Nothing here substitutes for that.",
            )
        )
    elif h2s_ppm >= H2S_NOTICEABLE_PPM:
        flags.append(
            Flag(
                Severity.WARNING,
                "H₂S present in the synthetic stream",
                "Sour-gas service changes metallurgy and monitoring requirements. "
                "In this demo it only reduces the modelled recovery fraction.",
            )
        )

    if pressure_bar >= HIGH_PRESSURE_BAR:
        flags.append(
            Flag(
                Severity.WARNING,
                "High modelled pressure",
                "Higher pressure improves the toy recovery correlation, but real "
                "pressure ratings, relief sizing, and compression duty are outside "
                "this model entirely.",
            )
        )

    if co2_fraction is not None and co2_fraction >= HIGH_CO2_FRACTION:
        flags.append(
            Flag(
                Severity.WARNING,
                "High CO₂ fraction",
                "CO₂ dilutes heating value and, with water present, drives acid-gas "
                "corrosion. This demo captures only the heating-value dilution.",
            )
        )

    if methane_fraction <= LOW_METHANE_FRACTION:
        flags.append(
            Flag(
                Severity.INFO,
                "Lean gas",
                "At low methane fraction the recoverable energy shrinks fast, which "
                "is usually what makes a recovery project stop paying back.",
            )
        )

    return SafetyAssessment(True, VALID_MESSAGE, flags)


def assess_demo_constraints(
    pressure_bar: float, h2s_ppm: float, methane_fraction: float
) -> tuple[bool, str]:
    """Backward-compatible wrapper returning ``(is_valid, message)``.

    Prefer :func:`evaluate_demo_safety`, which also returns the teaching flags.
    """
    result = evaluate_demo_safety(pressure_bar, h2s_ppm, methane_fraction)
    return result.is_valid, result.message
