"""HPC Astronomy Authority - Certified Runtime Enforcement.

A0.3 - fail-closed verification of the running scientific runtime.

This module owns POLICY: the specific component values this deployment is
certified against. The MACHINERY it relies on - version collection, path
resolution and file hashing - lives in scientific_environment.py and is
imported, never reimplemented.

TWO RESPONSIBILITIES, KEPT SEPARATE

    collect_runtime_scientific_components()
        Observation. Retrieves the live evidence.

    verify_runtime_scientific_components()
        Assertion gate. Compares evidence against the certified
        expectations and returns None on success. It never returns the
        observed mapping; callers that need the evidence call the collector.

THREE DISTINCT FAILURE CATEGORIES

    RUNTIME COMPONENT COLLECTION FAILED
        Evidence could not be obtained at all.

    RUNTIME EVIDENCE MALFORMED
        Evidence was supplied but its field set is not exactly the
        expected one.

    ENVIRONMENT DRIFT
        Evidence was successfully collected and is well formed, but
        differs from the certified expectations.

An inability to collect evidence is never reported as drift.

SCOPE OF THE CLAIM

Verification here establishes only that the running software components and
the bundled IERS/Delta-T data match the certified A0 reference expectations
before astronomy is loaded.

It does NOT establish:
    - full live scientificEnvironmentId certification;
    - BSP disk integrity;
    - ephemeris dataset verification;
    - clean-room reconstruction;
    - Linux/EC2 certification;
    - cross-platform bit-identical astronomy.

The pinned HPC reference BSP artifacts are NOT read here. Full BSP startup
integrity is a later increment.

Cheap by construction: the only file read is the bundled IERS/Delta-T table
(62 KB). No ephemeris kernel is opened and no Skyfield timescale is built.
There is no fallback, no warning-and-continue, no repair, no package
install and no network access.
"""

from scientific_environment import (
    ScientificEnvironmentError,
    distribution_version,
    python_version_string,
    resolve_iers_data_path,
    sha256_file,
)

CERTIFIED_PYTHON_VERSION = "CPython 3.13.7"
CERTIFIED_SKYFIELD_VERSION = "1.54"
CERTIFIED_NUMPY_VERSION = "2.4.3"
CERTIFIED_JPLEPHEM_VERSION = "2.24"
CERTIFIED_IERS_DATA_SHA256 = (
    "C7D7536D898DFA9F8CD43E8044FF51E108CC8289675A13FEE9822010A1C4935C"
)

# Sorted, so every diagnostic and every comparison is deterministic.
EXPECTED_COMPONENT_FIELDS = (
    "iersDataSha256",
    "jplephemVersion",
    "numpyVersion",
    "pythonVersion",
    "skyfieldVersion",
)

CERTIFIED_RUNTIME_SCIENTIFIC_COMPONENTS = {
    "iersDataSha256": CERTIFIED_IERS_DATA_SHA256,
    "jplephemVersion": CERTIFIED_JPLEPHEM_VERSION,
    "numpyVersion": CERTIFIED_NUMPY_VERSION,
    "pythonVersion": CERTIFIED_PYTHON_VERSION,
    "skyfieldVersion": CERTIFIED_SKYFIELD_VERSION,
}


class RuntimeEnvironmentError(ScientificEnvironmentError):
    """Scientific runtime certification failed closed."""


def _collect_iers_data_sha256():
    """Hash the IERS/Delta-T table inside the RUNNING Skyfield installation."""
    return sha256_file(resolve_iers_data_path())


_COMPONENT_COLLECTORS = (
    ("iersDataSha256", _collect_iers_data_sha256),
    ("jplephemVersion", lambda: distribution_version("jplephem")),
    ("numpyVersion", lambda: distribution_version("numpy")),
    ("pythonVersion", python_version_string),
    ("skyfieldVersion", lambda: distribution_version("skyfield")),
)


def collect_runtime_scientific_components():
    """Read the cheap scientific components from the running process.

    Returns a mapping carrying exactly EXPECTED_COMPONENT_FIELDS.

    Any collection failure is fail-closed: a missing Skyfield, an absent or
    unreadable IERS file, or an uninstalled distribution all raise
    RuntimeEnvironmentError with the original exception preserved as the
    cause.

    Exception is caught broadly on purpose - collection is entirely
    environment-dependent, and an inability to determine a component is
    exactly as disqualifying as a wrong value. BaseException is NOT caught,
    so KeyboardInterrupt and SystemExit still propagate.
    """
    components = {}

    for field, collector in _COMPONENT_COLLECTORS:
        try:
            components[field] = collector()
        except Exception as error:
            raise RuntimeEnvironmentError(
                "RUNTIME COMPONENT COLLECTION FAILED - cannot collect %s "
                "from the running runtime; refusing to start: %s: %s"
                % (field, type(error).__name__, error)
            ) from error

    return components


def verify_runtime_scientific_components(components=None):
    """Assert that the running scientific components are the certified ones.

    This is a gate, not an accessor. On success it returns None; it never
    returns the observed mapping. Callers needing the evidence itself call
    collect_runtime_scientific_components().

    ``components`` exists for testing: passing an explicit mapping verifies
    the comparison logic without mutating the running environment. When it
    is None the values are collected from the live process.

    Raises RuntimeEnvironmentError on any collection failure, malformed
    field set, or differing value. There is no warning-and-continue, no
    fallback and no auto-repair path.
    """
    observed = (
        collect_runtime_scientific_components() if components is None else components
    )

    if not isinstance(observed, dict):
        raise RuntimeEnvironmentError(
            "RUNTIME EVIDENCE MALFORMED - components must be a dict, "
            "received %s; refusing to start" % type(observed).__name__
        )

    expected_fields = set(EXPECTED_COMPONENT_FIELDS)
    missing = sorted(expected_fields - set(observed))
    unexpected = sorted(set(observed) - expected_fields)

    if missing or unexpected:
        detail = []
        if missing:
            detail.append("missing %s" % missing)
        if unexpected:
            detail.append("unexpected %s" % unexpected)
        raise RuntimeEnvironmentError(
            "RUNTIME EVIDENCE MALFORMED - component field set is not the "
            "expected set; refusing to start: " + "; ".join(detail)
        )

    mismatches = []
    for field in EXPECTED_COMPONENT_FIELDS:
        certified = CERTIFIED_RUNTIME_SCIENTIFIC_COMPONENTS[field]
        running = observed[field]
        if running != certified:
            mismatches.append(
                "%s: certified %r, running %r" % (field, certified, running)
            )

    if mismatches:
        raise RuntimeEnvironmentError(
            "ENVIRONMENT DRIFT - the running scientific runtime is not the "
            "certified runtime; refusing to start. " + "; ".join(mismatches)
        )

    return None
