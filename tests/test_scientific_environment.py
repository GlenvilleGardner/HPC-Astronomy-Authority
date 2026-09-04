"""A0.1 verification for scientific_environment.py.

Standard library only. No third-party test dependency is introduced.

INVOCATION
----------
Run from the repository root with the repository interpreter:

    .\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v

``python -m`` places the current working directory on sys.path, which is how
``import scientific_environment`` resolves. There is no tests/__init__.py and
none is required for this layout.

EVIDENCE DISCIPLINE
-------------------
PROVED      - follows from the canonical serialization rules themselves.
MEASURED    - a value observed on this machine under a known environment.
EMPIRICALLY CHARACTERIZED
            - observed over a sample; not a universal claim.
ASSUMED / REQUIRES FUTURE VERIFICATION
            - not established here.

INDEPENDENT ORACLE DISCIPLINE
-----------------------------
Expected canonical bytes are LITERALS transcribed from the A0.1b
verification, not values produced by the module under test. Expected
identifiers are recomputed in these tests with bare hashlib over those
literals, never by calling the module's own canonical_digest. A production
helper is therefore never validated solely against another production helper
that shares its implementation path.

SCOPE
-----
The 3.4 GB pinned HPC reference BSP set is NOT hashed here. Those artifact
hashes enter as fixed constants; verifying them against the files on disk
belongs to the later startup-integrity increment.

This file does NOT certify a full live scientificEnvironmentId. Until the
BSP artifacts are verified at startup, only the cheap running scientific
components are checked. Full live environment certification is
ASSUMED / REQUIRES FUTURE VERIFICATION.
"""

import hashlib
import unittest

import scientific_environment as se


# ---------------------------------------------------------------------------
# Fixed canonical constants, established and accepted in A0.1b.
# ---------------------------------------------------------------------------

# MEASURED - SHA-256 of the pinned HPC reference BSP artifacts on this
# machine. NOT claimed to be independently verified NASA/JPL publication
# hashes; establishing that correspondence is a separate provenance gate.
PINNED_ARTIFACTS = {
    "primary": {
        "routingName": "de440.bsp",
        "sha256": "A4CE9BF9B3282BECC9F4B2AC3CEBE03A2AE7599981AABD7265FD8482FFF7C4B5",
    },
    "ancient": {
        "routingName": "de441_part-1.bsp",
        "sha256": "13757827F5DB41B835A24BBD637488636CE79A8CA754062FED17844F7D5B618E",
    },
    "future": {
        "routingName": "de441_part-2.bsp",
        "sha256": "3ABB17DAE2D78DD34880377544AACB54892104A0D4462B322CB9F4454D4887F6",
    },
}

# MEASURED - SHA-256 of skyfield/data/iers.npz in the pinned Skyfield 1.54.
PINNED_IERS_SHA256 = (
    "C7D7536D898DFA9F8CD43E8044FF51E108CC8289675A13FEE9822010A1C4935C"
)

PINNED_PYTHON = "CPython 3.13.7"
PINNED_SKYFIELD = "1.54"
PINNED_NUMPY = "2.4.3"
PINNED_JPLEPHEM = "2.24"

# Canonical bytes transcribed verbatim from the A0.1b verification. These are
# the external oracle; the module must reproduce them exactly.
EXPECTED_MANIFEST_BYTES = (
    b'{"ancient":{"routingName":"de441_part-1.bsp",'
    b'"sha256":"13757827F5DB41B835A24BBD637488636CE79A8CA754062FED17844F7D5B618E"},'
    b'"future":{"routingName":"de441_part-2.bsp",'
    b'"sha256":"3ABB17DAE2D78DD34880377544AACB54892104A0D4462B322CB9F4454D4887F6"},'
    b'"primary":{"routingName":"de440.bsp",'
    b'"sha256":"A4CE9BF9B3282BECC9F4B2AC3CEBE03A2AE7599981AABD7265FD8482FFF7C4B5"},'
    b'"schemaVersion":"1"}'
)
EXPECTED_MANIFEST_LENGTH = 376

EXPECTED_ENVIRONMENT_BYTES = (
    b'{"authoritySolverGeneration":"hpc-authority-solver-v1",'
    b'"ephemerisDataSetId":'
    b'"sha256:4a363dccb92d868a33543c5a2272658dded49d02db4711f9203563603594f84b",'
    b'"horizonModelGeneration":"hpc-apparent-sunset-v1",'
    b'"iersDataSha256":'
    b'"C7D7536D898DFA9F8CD43E8044FF51E108CC8289675A13FEE9822010A1C4935C",'
    b'"jplephemVersion":"2.24","numpyVersion":"2.4.3",'
    b'"pythonVersion":"CPython 3.13.7","schemaVersion":"1","skyfieldVersion":"1.54"}'
)
EXPECTED_ENVIRONMENT_LENGTH = 410

# Governance-fixed identifiers, independently verified in A0.1b.
EXPECTED_DATASET_ID = (
    "sha256:4a363dccb92d868a33543c5a2272658dded49d02db4711f9203563603594f84b"
)
EXPECTED_ENVIRONMENT_ID = (
    "sha256:c223962508d19a9c969a9d62508497f6d0a418244c06509116668267e250fde0"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def independent_digest(raw):
    """Recompute an identifier with bare hashlib, bypassing the module."""
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def pinned_artifacts_copy():
    """A copy deep enough that mutation tests cannot leak into the constant."""
    return {role: dict(entry) for role, entry in PINNED_ARTIFACTS.items()}


def build_manifest():
    return se.build_ephemeris_manifest(PINNED_ARTIFACTS)


def build_environment(**overrides):
    """Assemble the canonical environment for tests.

    The dataset identifier defaults to the independently established
    EXPECTED_DATASET_ID literal. It is deliberately NOT obtained through
    derive_ephemeris_data_set_id(): the environment tests must not depend on
    the manifest helper they are supposed to be independent of.
    """
    values = {
        "python_version": PINNED_PYTHON,
        "skyfield_version": PINNED_SKYFIELD,
        "numpy_version": PINNED_NUMPY,
        "jplephem_version": PINNED_JPLEPHEM,
        "iers_data_sha256": PINNED_IERS_SHA256,
        "ephemeris_data_set_id": EXPECTED_DATASET_ID,
    }
    values.update(overrides)
    return se.build_scientific_environment(**values)


class FailClosedMixin:
    def assert_fails_closed(self, callable_object, *args):
        """Reject, raising ScientificEnvironmentError, returning nothing."""
        with self.assertRaises(se.ScientificEnvironmentError):
            callable_object(*args)


# ---------------------------------------------------------------------------
# Canonical serialization
# ---------------------------------------------------------------------------


class TestCanonicalSerialization(unittest.TestCase):
    """PROVED - properties that follow from the canonical serialization rules."""

    def test_keys_are_sorted(self):
        self.assertEqual(
            se.canonical_json_bytes({"b": "2", "a": "1"}), b'{"a":"1","b":"2"}'
        )

    def test_insertion_order_is_irrelevant(self):
        self.assertEqual(
            se.canonical_json_bytes({"b": "2", "a": "1"}),
            se.canonical_json_bytes({"a": "1", "b": "2"}),
        )

    def test_no_insignificant_whitespace(self):
        raw = se.canonical_json_bytes({"a": "1", "b": "2"})
        self.assertNotIn(b" ", raw)
        self.assertNotIn(b"\n", raw)

    def test_no_trailing_newline(self):
        self.assertFalse(se.canonical_json_bytes({"a": "1"}).endswith(b"\n"))

    def test_non_finite_floats_are_rejected(self):
        # allow_nan=False. Emitting NaN or Infinity would produce tokens that
        # standards-conformant parsers reject, silently breaking independent
        # recomputation in another language or on another platform. The
        # underlying ValueError is wrapped so the module presents a single
        # failure type across its whole surface.
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaises(se.ScientificEnvironmentError):
                    se.canonical_json_bytes({"a": value})

    def test_digest_is_prefixed_lowercase_hex(self):
        digest = se.canonical_digest({"a": "1"})
        self.assertTrue(digest.startswith("sha256:"))
        self.assertEqual(len(digest), 71)
        self.assertEqual(digest, digest.lower())


# ---------------------------------------------------------------------------
# Manifest canonical bytes
# ---------------------------------------------------------------------------


class TestEphemerisManifestCanonicalBytes(unittest.TestCase):
    """The module must reproduce the A0.1b literals exactly."""

    def test_canonical_bytes_are_exact(self):
        self.assertEqual(
            se.canonical_json_bytes(build_manifest()), EXPECTED_MANIFEST_BYTES
        )

    def test_canonical_length_is_376(self):
        self.assertEqual(
            len(se.canonical_json_bytes(build_manifest())), EXPECTED_MANIFEST_LENGTH
        )

    def test_dataset_id_recomputed_independently(self):
        # Three-way: bare hashlib over the literal oracle, the governance-fixed
        # value, and the module derivation must all agree. Any two agreeing
        # while the third differs fails here.
        recomputed = independent_digest(EXPECTED_MANIFEST_BYTES)
        self.assertEqual(recomputed, EXPECTED_DATASET_ID)
        self.assertEqual(se.derive_ephemeris_data_set_id(build_manifest()), recomputed)


# ---------------------------------------------------------------------------
# Environment canonical bytes
# ---------------------------------------------------------------------------


class TestScientificEnvironmentCanonicalBytes(unittest.TestCase):

    def test_canonical_bytes_are_exact(self):
        self.assertEqual(
            se.canonical_json_bytes(build_environment()), EXPECTED_ENVIRONMENT_BYTES
        )

    def test_canonical_length_is_410(self):
        self.assertEqual(
            len(se.canonical_json_bytes(build_environment())),
            EXPECTED_ENVIRONMENT_LENGTH,
        )

    def test_environment_id_recomputed_independently(self):
        recomputed = independent_digest(EXPECTED_ENVIRONMENT_BYTES)
        self.assertEqual(recomputed, EXPECTED_ENVIRONMENT_ID)
        self.assertEqual(
            se.derive_scientific_environment_id(build_environment()), recomputed
        )


# ---------------------------------------------------------------------------
# Identity sensitivity
# ---------------------------------------------------------------------------


class TestIdentitySensitivity(unittest.TestCase):
    """The identifiers must actually move when their inputs move."""

    def test_role_swap_changes_dataset_id(self):
        # Same three hashes, wrong roles. A bare set of hashes would not
        # detect this, which is why roles are bound to bytes.
        artifacts = pinned_artifacts_copy()
        artifacts["ancient"]["sha256"], artifacts["future"]["sha256"] = (
            artifacts["future"]["sha256"],
            artifacts["ancient"]["sha256"],
        )
        manifest = se.build_ephemeris_manifest(artifacts)
        self.assertNotEqual(
            se.derive_ephemeris_data_set_id(manifest), EXPECTED_DATASET_ID
        )

    def test_each_bsp_hash_is_load_bearing(self):
        for role in se.EPHEMERIS_ROLES:
            with self.subTest(role=role):
                artifacts = pinned_artifacts_copy()
                artifacts[role]["sha256"] = "0" * 64
                manifest = se.build_ephemeris_manifest(artifacts)
                self.assertNotEqual(
                    se.derive_ephemeris_data_set_id(manifest), EXPECTED_DATASET_ID
                )

    def test_dataset_id_change_propagates_to_environment_id(self):
        environment = build_environment(ephemeris_data_set_id="sha256:" + "0" * 64)
        self.assertNotEqual(
            se.derive_scientific_environment_id(environment), EXPECTED_ENVIRONMENT_ID
        )


# ---------------------------------------------------------------------------
# Fail-closed: manifest
# ---------------------------------------------------------------------------


class TestFailClosedManifest(FailClosedMixin, unittest.TestCase):

    def test_missing_role_rejected(self):
        artifacts = pinned_artifacts_copy()
        artifacts.pop("future")
        self.assert_fails_closed(se.build_ephemeris_manifest, artifacts)

    def test_unexpected_role_rejected(self):
        artifacts = pinned_artifacts_copy()
        artifacts["legacy"] = {"routingName": "de421.bsp", "sha256": "A" * 64}
        self.assert_fails_closed(se.build_ephemeris_manifest, artifacts)

    def test_unexpected_role_field_rejected(self):
        # sizeBytes and coverage are audit metadata, never canonical identity.
        artifacts = pinned_artifacts_copy()
        artifacts["primary"]["sizeBytes"] = "119799808"
        self.assert_fails_closed(se.build_ephemeris_manifest, artifacts)

    def test_lowercase_artifact_sha256_rejected(self):
        artifacts = pinned_artifacts_copy()
        artifacts["primary"]["sha256"] = artifacts["primary"]["sha256"].lower()
        self.assert_fails_closed(se.build_ephemeris_manifest, artifacts)

    def test_malformed_artifact_sha256_rejected(self):
        for bad in ("ABCD", "A" * 63, "A" * 65, "G" * 64):
            with self.subTest(value=bad):
                artifacts = pinned_artifacts_copy()
                artifacts["primary"]["sha256"] = bad
                self.assert_fails_closed(se.build_ephemeris_manifest, artifacts)

    def test_wrong_manifest_schema_version_rejected(self):
        manifest = dict(build_manifest())
        manifest["schemaVersion"] = "2"
        self.assert_fails_closed(se.validate_ephemeris_manifest, manifest)

    def test_whitespace_padded_routing_name_rejected_not_trimmed(self):
        for padded in (" de440.bsp", "de440.bsp ", "\tde440.bsp", "de440.bsp\n"):
            with self.subTest(value=padded):
                artifacts = pinned_artifacts_copy()
                artifacts["primary"]["routingName"] = padded
                # Rejected outright. No trimmed variant is ever produced.
                self.assert_fails_closed(se.build_ephemeris_manifest, artifacts)

    def test_empty_and_non_string_values_rejected(self):
        for bad in ("", None, 440, b"de440.bsp", ["de440.bsp"]):
            with self.subTest(value=bad):
                artifacts = pinned_artifacts_copy()
                artifacts["primary"]["routingName"] = bad
                self.assert_fails_closed(se.build_ephemeris_manifest, artifacts)


# ---------------------------------------------------------------------------
# Fail-closed: environment
# ---------------------------------------------------------------------------


class TestFailClosedEnvironment(FailClosedMixin, unittest.TestCase):

    def test_missing_field_rejected(self):
        environment = {
            key: value
            for key, value in build_environment().items()
            if key != "numpyVersion"
        }
        self.assert_fails_closed(se.validate_scientific_environment, environment)

    def test_unexpected_field_rejected(self):
        environment = dict(build_environment())
        environment["fastapiVersion"] = "0.135.1"
        self.assert_fails_closed(se.validate_scientific_environment, environment)

    def test_malformed_derived_digest_rejected(self):
        for bad in ("md5:" + "0" * 64, "0" * 64, "sha256:" + "0" * 63, "sha256:"):
            with self.subTest(value=bad):
                environment = dict(build_environment())
                environment["ephemerisDataSetId"] = bad
                self.assert_fails_closed(
                    se.validate_scientific_environment, environment
                )

    def test_uppercase_derived_digest_rejected(self):
        environment = dict(build_environment())
        environment["ephemerisDataSetId"] = "sha256:" + "A" * 64
        self.assert_fails_closed(se.validate_scientific_environment, environment)

    def test_wrong_schema_version_rejected(self):
        environment = dict(build_environment())
        environment["schemaVersion"] = "2"
        self.assert_fails_closed(se.validate_scientific_environment, environment)

    def test_wrong_solver_generation_rejected(self):
        environment = dict(build_environment())
        environment["authoritySolverGeneration"] = "hpc-authority-solver-v2"
        self.assert_fails_closed(se.validate_scientific_environment, environment)

    def test_wrong_horizon_generation_rejected(self):
        environment = dict(build_environment())
        environment["horizonModelGeneration"] = "hpc-apparent-sunset-v2"
        self.assert_fails_closed(se.validate_scientific_environment, environment)

    def test_whitespace_padded_version_rejected_not_trimmed(self):
        for padded in (" 1.54", "1.54 ", "1.54\n"):
            with self.subTest(value=padded):
                self.assert_fails_closed(
                    lambda value=padded: build_environment(skyfield_version=value)
                )

    def test_empty_and_non_string_values_rejected(self):
        for bad in ("", None, 154, ["1.54"]):
            with self.subTest(value=bad):
                self.assert_fails_closed(
                    lambda value=bad: build_environment(skyfield_version=value)
                )

    def test_non_dict_environment_rejected(self):
        self.assert_fails_closed(
            se.validate_scientific_environment, ["not", "a", "dict"]
        )


# ---------------------------------------------------------------------------
# Running scientific components
# ---------------------------------------------------------------------------


class TestRunningScientificComponents(unittest.TestCase):
    """ENVIRONMENT-BOUND. Expected to fail on a different runtime, and such a
    failure means ENVIRONMENT DRIFT, not a defect in the module.

    Scope is deliberately narrow. This class checks only the cheap running
    scientific components: the Python identity, the three scientifically
    causal distribution versions, and the active IERS/Delta-T data file
    (62 KB).

    It does NOT certify a full live scientificEnvironmentId. The pinned BSP
    artifacts are not verified against the files on disk here, so live
    environment certification remains
    ASSUMED / REQUIRES FUTURE VERIFICATION until the startup-integrity
    increment hashes the BSP set.

    MEASURED - every expected value below was observed on this machine under
    the pinned runtime.
    """

    def test_python_version_matches(self):
        self.assertEqual(se.python_version_string(), PINNED_PYTHON)

    def test_skyfield_version_matches(self):
        self.assertEqual(se.distribution_version("skyfield"), PINNED_SKYFIELD)

    def test_numpy_version_matches(self):
        self.assertEqual(se.distribution_version("numpy"), PINNED_NUMPY)

    def test_jplephem_version_matches(self):
        self.assertEqual(se.distribution_version("jplephem"), PINNED_JPLEPHEM)

    def test_active_iers_sha256_matches(self):
        # EMPIRICALLY CHARACTERIZED - the bundled IERS/Delta-T table is the
        # proven cause of the measured 42 ms sunset shift between Skyfield
        # 1.53 and 1.54, which is why its byte identity is checked directly
        # rather than inferred from the Skyfield version alone.
        self.assertEqual(
            se.sha256_file(se.resolve_iers_data_path()), PINNED_IERS_SHA256
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
