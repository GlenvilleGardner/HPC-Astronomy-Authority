"""A0.3 verification for runtime_enforcement.py.

Standard library only: unittest and unittest.mock.

INVOCATION
----------
Run from the repository root with the repository interpreter:

    .\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v

EVIDENCE DISCIPLINE
-------------------
PROVED      - follows from the module's own control flow.
MEASURED    - a value observed on this machine under a known environment.
ASSUMED / REQUIRES FUTURE VERIFICATION
            - not established here.

INDEPENDENT ORACLE DISCIPLINE
-----------------------------
The expected component values below are declared as test-local literals.
They are deliberately NOT imported from runtime_enforcement. If a production
CERTIFIED_* constant were wrong, importing it would make the tests agree
with the defect; declaring it independently makes them detect it.

SCOPE
-----
These tests exercise the A0.3 runtime certification gate only. They do not
hash the pinned HPC reference BSP artifacts, do not verify the ephemeris
dataset against disk, and do not certify a full live
scientificEnvironmentId. Those remain later increments.

No network. No service startup. No package, ephemeris, IERS or
configuration file is mutated; the failure-path tests patch module
attributes only, and unittest.mock restores them automatically.
"""

import unittest
from unittest import mock

import runtime_enforcement
from runtime_enforcement import (
    RuntimeEnvironmentError,
    collect_runtime_scientific_components,
    verify_runtime_scientific_components,
)
from scientific_environment import ScientificEnvironmentError


# ---------------------------------------------------------------------------
# Independent test-local expectations (NOT imported from production)
# ---------------------------------------------------------------------------

EXPECTED_PYTHON_VERSION = "CPython 3.13.7"
EXPECTED_SKYFIELD_VERSION = "1.54"
EXPECTED_NUMPY_VERSION = "2.4.3"
EXPECTED_JPLEPHEM_VERSION = "2.24"
EXPECTED_IERS_DATA_SHA256 = (
    "C7D7536D898DFA9F8CD43E8044FF51E108CC8289675A13FEE9822010A1C4935C"
)

EXPECTED_FIELDS = (
    "iersDataSha256",
    "jplephemVersion",
    "numpyVersion",
    "pythonVersion",
    "skyfieldVersion",
)

EXPECTED_CERTIFIED_MAPPING = {
    "iersDataSha256": EXPECTED_IERS_DATA_SHA256,
    "jplephemVersion": EXPECTED_JPLEPHEM_VERSION,
    "numpyVersion": EXPECTED_NUMPY_VERSION,
    "pythonVersion": EXPECTED_PYTHON_VERSION,
    "skyfieldVersion": EXPECTED_SKYFIELD_VERSION,
}

# Plausible wrong values, one per field. The IERS value is the MEASURED hash
# of the Skyfield 1.53 table, so the drift cases mirror a real mis-launch.
DRIFTED_VALUES = {
    "iersDataSha256": (
        "B263DAD4A3E5C6DA30F9461271ABFCA56218FECC0A458960A0BB2AD0D672D652"
    ),
    "jplephemVersion": "2.23",
    "numpyVersion": "2.4.4",
    "pythonVersion": "CPython 3.13.6",
    "skyfieldVersion": "1.53",
}

DRIFT_PREFIX = "ENVIRONMENT DRIFT"
MALFORMED_PREFIX = "RUNTIME EVIDENCE MALFORMED"
COLLECTION_PREFIX = "RUNTIME COMPONENT COLLECTION FAILED"


def certified_evidence(**overrides):
    """A fresh copy of the independently declared certified mapping."""
    evidence = dict(EXPECTED_CERTIFIED_MAPPING)
    evidence.update(overrides)
    return evidence


# ---------------------------------------------------------------------------


class TestErrorContract(unittest.TestCase):
    """PROVED - callers catching the A0.1 error type still catch A0.3."""

    def test_runtime_error_is_a_scientific_environment_error(self):
        self.assertTrue(
            issubclass(RuntimeEnvironmentError, ScientificEnvironmentError)
        )


class TestCertifiedConstants(unittest.TestCase):
    """The independent oracle for the production certified constants."""

    def test_certified_constants_match_independent_literals(self):
        self.assertEqual(
            runtime_enforcement.CERTIFIED_PYTHON_VERSION, EXPECTED_PYTHON_VERSION
        )
        self.assertEqual(
            runtime_enforcement.CERTIFIED_SKYFIELD_VERSION, EXPECTED_SKYFIELD_VERSION
        )
        self.assertEqual(
            runtime_enforcement.CERTIFIED_NUMPY_VERSION, EXPECTED_NUMPY_VERSION
        )
        self.assertEqual(
            runtime_enforcement.CERTIFIED_JPLEPHEM_VERSION, EXPECTED_JPLEPHEM_VERSION
        )
        self.assertEqual(
            runtime_enforcement.CERTIFIED_IERS_DATA_SHA256, EXPECTED_IERS_DATA_SHA256
        )

    def test_certified_mapping_matches_independent_mapping(self):
        self.assertEqual(
            runtime_enforcement.CERTIFIED_RUNTIME_SCIENTIFIC_COMPONENTS,
            EXPECTED_CERTIFIED_MAPPING,
        )

    def test_expected_component_fields_are_exactly_the_five(self):
        self.assertEqual(runtime_enforcement.EXPECTED_COMPONENT_FIELDS, EXPECTED_FIELDS)
        self.assertEqual(len(runtime_enforcement.EXPECTED_COMPONENT_FIELDS), 5)


class TestGateAcceptsCertifiedEvidence(unittest.TestCase):

    def test_exact_certified_mapping_passes_and_returns_none(self):
        result = verify_runtime_scientific_components(certified_evidence())
        self.assertIsNone(result)


class TestEnvironmentDrift(unittest.TestCase):

    def test_each_single_field_drift_is_rejected_and_named(self):
        for field in EXPECTED_FIELDS:
            with self.subTest(field=field):
                evidence = certified_evidence(**{field: DRIFTED_VALUES[field]})
                with self.assertRaises(RuntimeEnvironmentError) as caught:
                    verify_runtime_scientific_components(evidence)
                message = str(caught.exception)
                self.assertTrue(message.startswith(DRIFT_PREFIX), message)
                self.assertIn(field, message)

    def test_multiple_drift_reports_every_mismatched_field(self):
        drifted = ("iersDataSha256", "numpyVersion", "skyfieldVersion")
        evidence = certified_evidence(
            **{field: DRIFTED_VALUES[field] for field in drifted}
        )
        with self.assertRaises(RuntimeEnvironmentError) as caught:
            verify_runtime_scientific_components(evidence)
        message = str(caught.exception)
        self.assertTrue(message.startswith(DRIFT_PREFIX), message)
        for field in drifted:
            self.assertIn(field, message)
        # The two undrifted fields must not be reported as mismatches.
        self.assertNotIn("jplephemVersion", message)
        self.assertNotIn("pythonVersion", message)


class TestMalformedEvidence(unittest.TestCase):

    def test_missing_field_rejected(self):
        evidence = certified_evidence()
        evidence.pop("numpyVersion")
        with self.assertRaises(RuntimeEnvironmentError) as caught:
            verify_runtime_scientific_components(evidence)
        message = str(caught.exception)
        self.assertTrue(message.startswith(MALFORMED_PREFIX), message)
        self.assertIn("numpyVersion", message)

    def test_unexpected_field_rejected(self):
        evidence = certified_evidence(fastapiVersion="0.135.1")
        with self.assertRaises(RuntimeEnvironmentError) as caught:
            verify_runtime_scientific_components(evidence)
        message = str(caught.exception)
        self.assertTrue(message.startswith(MALFORMED_PREFIX), message)
        self.assertIn("fastapiVersion", message)

    def test_missing_and_unexpected_reported_together(self):
        evidence = certified_evidence(pipVersion="25.2")
        evidence.pop("skyfieldVersion")
        with self.assertRaises(RuntimeEnvironmentError) as caught:
            verify_runtime_scientific_components(evidence)
        message = str(caught.exception)
        self.assertTrue(message.startswith(MALFORMED_PREFIX), message)
        self.assertIn("skyfieldVersion", message)
        self.assertIn("pipVersion", message)

    def test_non_dict_evidence_rejected(self):
        for evidence in ([], (), "", 0, object()):
            with self.subTest(evidence=type(evidence).__name__):
                with self.assertRaises(RuntimeEnvironmentError) as caught:
                    verify_runtime_scientific_components(evidence)
                self.assertTrue(
                    str(caught.exception).startswith(MALFORMED_PREFIX),
                    str(caught.exception),
                )


class TestCollectionFailure(unittest.TestCase):
    """Collection failure is a distinct category from drift."""

    def test_iers_collection_failure_is_wrapped_and_named(self):
        original = FileNotFoundError(2, "No such file or directory", "iers.npz")
        with mock.patch.object(
            runtime_enforcement, "resolve_iers_data_path", return_value="<patched>"
        ), mock.patch.object(
            runtime_enforcement, "sha256_file", side_effect=original
        ):
            with self.assertRaises(RuntimeEnvironmentError) as caught:
                collect_runtime_scientific_components()

        message = str(caught.exception)
        self.assertTrue(message.startswith(COLLECTION_PREFIX), message)
        self.assertIn("iersDataSha256", message)
        self.assertIs(caught.exception.__cause__, original)

    def test_distribution_collection_failure_is_wrapped_and_named(self):
        original = ModuleNotFoundError("No package metadata was found")
        with mock.patch.object(
            runtime_enforcement, "resolve_iers_data_path", return_value="<patched>"
        ), mock.patch.object(
            runtime_enforcement, "sha256_file", return_value=EXPECTED_IERS_DATA_SHA256
        ), mock.patch.object(
            runtime_enforcement, "distribution_version", side_effect=original
        ):
            with self.assertRaises(RuntimeEnvironmentError) as caught:
                collect_runtime_scientific_components()

        message = str(caught.exception)
        self.assertTrue(message.startswith(COLLECTION_PREFIX), message)
        self.assertIn("jplephemVersion", message)
        self.assertIs(caught.exception.__cause__, original)


class TestBaseExceptionBoundary(unittest.TestCase):
    """The broad catch is Exception only, never BaseException.

    A control-flow signal such as SystemExit or KeyboardInterrupt must pass
    through untouched, so an operator interrupt is never disguised as a
    certification result.
    """

    def test_base_exception_from_a_collector_is_not_converted(self):
        for signal in (SystemExit, KeyboardInterrupt):
            with self.subTest(signal=signal.__name__):
                with mock.patch.object(
                    runtime_enforcement,
                    "resolve_iers_data_path",
                    return_value="<patched>",
                ), mock.patch.object(
                    runtime_enforcement, "sha256_file", side_effect=signal
                ):
                    with self.assertRaises(signal):
                        collect_runtime_scientific_components()


class TestLiveRepositoryVenv(unittest.TestCase):
    """ENVIRONMENT-BOUND. A failure here requires classification.

    The live environment may have drifted, collection may have failed, or an
    implementation/test defect may exist. Do not change certification values
    merely to make this test pass.

    MEASURED - the expected values were observed on this machine under the
    pinned repository .venv. Cheap by construction: only the 62 KB IERS file
    is hashed; no BSP artifact is read.
    """

    def test_live_collection_matches_independent_certified_mapping(self):
        collected = collect_runtime_scientific_components()
        self.assertIsInstance(collected, dict)
        self.assertEqual(collected, EXPECTED_CERTIFIED_MAPPING)

    def test_live_gate_succeeds_and_returns_none(self):
        self.assertIsNone(verify_runtime_scientific_components())


if __name__ == "__main__":
    unittest.main(verbosity=2)
