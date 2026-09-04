"""HPC Astronomy Authority - Scientific Environment Identity.

A0.1 - scientific reproducibility foundation.

This module is PURE IDENTITY MACHINERY. It performs no astronomy, reads no
ephemeris data at import time, and has no side effects on import.

Two canonical artifacts are defined here:

    ephemeris dataset manifest  ->  ephemerisDataSetId
        WHICH pinned HPC reference BSP artifacts this Authority declares.

    scientific environment      ->  scientificEnvironmentId
        WHICH scientific specification this Authority declares.

Both identifiers describe DECLARED MATERIAL ONLY. Neither is evidence that
any computation ran, that the declared artifacts were verified against the
files on disk, or that a particular numerical result was produced. Verifying
declared material against reality is a separate, later step.

Neither identifier states WHICH kernel a particular computation used. That
remains per-computation provenance under W-3: it is reported on each
astronomical response and is never inferred from configuration or from the
identities defined here.

The BSP artifact hashes handled here are pinned HPC reference artifact
identities. They are NOT claimed to be independently verified NASA/JPL
publication hashes; establishing that correspondence is a separate
provenance gate.

Canonical serialization rules. Changing any of them is a schema change:

    - JSON, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
      allow_nan=False
    - UTF-8, no trailing newline in the hashed bytes
    - every identity scalar is a non-empty string
    - leading and trailing whitespace is REJECTED, never trimmed
    - SHA-256 of artifact file content is UPPERCASE hex
    - derived identifiers are "sha256:" + LOWERCASE hex

SHA-256 here provides deterministic identity, drift detection and
reproducibility comparison. It is NOT authentication, NOT signing, and NOT
proof of issuance: any party can compute these digests.

Standard library only. No third-party package is imported at module import.
"""

import hashlib
import json
import platform
import sys
from importlib.metadata import version as _distribution_version
from pathlib import Path

SCIENTIFIC_ENVIRONMENT_SCHEMA_VERSION = "1"
EPHEMERIS_MANIFEST_SCHEMA_VERSION = "1"

AUTHORITY_SOLVER_GENERATION = "hpc-authority-solver-v1"
HORIZON_MODEL_GENERATION = "hpc-apparent-sunset-v1"

# Logical roles, not filenames. Routing names are bound to bytes inside the
# manifest so that exchanging two artifacts is detectable.
EPHEMERIS_ROLES = ("ancient", "future", "primary")
EPHEMERIS_ROLE_FIELDS = ("routingName", "sha256")

SCIENTIFIC_ENVIRONMENT_FIELDS = (
    "authoritySolverGeneration",
    "ephemerisDataSetId",
    "horizonModelGeneration",
    "iersDataSha256",
    "jplephemVersion",
    "numpyVersion",
    "pythonVersion",
    "schemaVersion",
    "skyfieldVersion",
)

DIGEST_PREFIX = "sha256:"

_UPPER_HEX = frozenset("0123456789ABCDEF")
_LOWER_HEX = frozenset("0123456789abcdef")
_FILE_READ_BLOCK = 1024 * 1024


class ScientificEnvironmentError(Exception):
    """Canonical material is missing, malformed, or unexpected."""


def canonical_json_bytes(obj):
    """Return the exact bytes hashed for any canonical artifact.

    Sorted keys make the result independent of insertion order; the compact
    separators remove insignificant whitespace; no trailing newline is added.

    allow_nan=False rejects non-finite floats. Emitting NaN or Infinity would
    produce tokens that standards-conformant parsers reject, which would
    silently break independent recomputation in another language or on
    another platform.
    """
    try:
        text = json.dumps(
            obj,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ScientificEnvironmentError(
            "value is not canonically serializable: %s" % error
        ) from error

    return text.encode("utf-8")


def canonical_digest(obj):
    """Return DIGEST_PREFIX + lowercase SHA-256 hex of the canonical bytes."""
    return DIGEST_PREFIX + hashlib.sha256(canonical_json_bytes(obj)).hexdigest()


def _require_identity_string(value, field):
    """Every canonical scalar must be a non-empty, unpadded string.

    Leading and trailing whitespace is rejected rather than trimmed: silently
    normalizing an input would let two different declared environments
    collapse onto the same identity.
    """
    if not isinstance(value, str):
        raise ScientificEnvironmentError(
            "%s must be a string, received %s" % (field, type(value).__name__)
        )
    if not value:
        raise ScientificEnvironmentError("%s must not be empty" % field)
    if value != value.strip():
        raise ScientificEnvironmentError(
            "%s must not carry leading or trailing whitespace, received %r"
            % (field, value)
        )
    return value


def _require_upper_sha256(value, field):
    """Artifact content hashes are exactly 64 uppercase hex characters."""
    _require_identity_string(value, field)
    if len(value) != 64 or not set(value) <= _UPPER_HEX:
        raise ScientificEnvironmentError(
            "%s must be 64 uppercase hex characters, received %r" % (field, value)
        )
    return value


def _require_digest(value, field):
    """Derived identifiers are DIGEST_PREFIX + 64 lowercase hex characters."""
    _require_identity_string(value, field)
    if not value.startswith(DIGEST_PREFIX):
        raise ScientificEnvironmentError(
            "%s must start with %r, received %r" % (field, DIGEST_PREFIX, value)
        )
    body = value[len(DIGEST_PREFIX):]
    if len(body) != 64 or not set(body) <= _LOWER_HEX:
        raise ScientificEnvironmentError(
            "%s must carry 64 lowercase hex characters, received %r" % (field, value)
        )
    return value


def _require_exact_keys(mapping, expected, what):
    """Reject any mapping whose key set is not exactly ``expected``."""
    missing = sorted(set(expected) - set(mapping))
    unexpected = sorted(set(mapping) - set(expected))
    if missing:
        raise ScientificEnvironmentError("%s missing keys: %s" % (what, missing))
    if unexpected:
        raise ScientificEnvironmentError(
            "%s has unexpected keys: %s" % (what, unexpected)
        )


def validate_ephemeris_manifest(manifest):
    """Fail closed unless the manifest is exactly the canonical shape."""
    if not isinstance(manifest, dict):
        raise ScientificEnvironmentError(
            "ephemeris manifest must be a dict, received %s" % type(manifest).__name__
        )

    _require_exact_keys(
        manifest, set(EPHEMERIS_ROLES) | {"schemaVersion"}, "ephemeris manifest"
    )

    schema_version = _require_identity_string(
        manifest["schemaVersion"], "ephemeris manifest schemaVersion"
    )
    if schema_version != EPHEMERIS_MANIFEST_SCHEMA_VERSION:
        raise ScientificEnvironmentError(
            "ephemeris manifest schemaVersion must be %r, received %r"
            % (EPHEMERIS_MANIFEST_SCHEMA_VERSION, schema_version)
        )

    for role in EPHEMERIS_ROLES:
        entry = manifest[role]
        if not isinstance(entry, dict):
            raise ScientificEnvironmentError(
                "ephemeris manifest role %r must be a dict, received %s"
                % (role, type(entry).__name__)
            )
        _require_exact_keys(
            entry, EPHEMERIS_ROLE_FIELDS, "ephemeris manifest role %r" % role
        )
        _require_identity_string(entry["routingName"], "%s.routingName" % role)
        _require_upper_sha256(entry["sha256"], "%s.sha256" % role)

    return manifest


def build_ephemeris_manifest(artifacts):
    """Build the canonical manifest of pinned HPC reference BSP artifacts.

    ``artifacts`` maps each required role to a mapping carrying exactly
    ``routingName`` and ``sha256``.

    ``sizeBytes`` and coverage are deliberately NOT canonical identity
    material: SHA-256 already establishes byte identity, and coverage is
    derivable from the bytes. Both remain useful for operational audit and
    are reported outside this manifest.
    """
    if not isinstance(artifacts, dict):
        raise ScientificEnvironmentError(
            "artifacts must be a dict, received %s" % type(artifacts).__name__
        )

    _require_exact_keys(artifacts, EPHEMERIS_ROLES, "artifacts")

    manifest = {"schemaVersion": EPHEMERIS_MANIFEST_SCHEMA_VERSION}
    for role in EPHEMERIS_ROLES:
        entry = artifacts[role]
        if not isinstance(entry, dict):
            raise ScientificEnvironmentError(
                "artifacts[%r] must be a dict, received %s"
                % (role, type(entry).__name__)
            )
        _require_exact_keys(entry, EPHEMERIS_ROLE_FIELDS, "artifacts[%r]" % role)
        manifest[role] = {
            "routingName": _require_identity_string(
                entry["routingName"], "%s.routingName" % role
            ),
            "sha256": _require_upper_sha256(entry["sha256"], "%s.sha256" % role),
        }

    return validate_ephemeris_manifest(manifest)


def derive_ephemeris_data_set_id(manifest):
    """Derive ephemerisDataSetId from a validated canonical manifest."""
    return canonical_digest(validate_ephemeris_manifest(manifest))


def validate_scientific_environment(environment):
    """Fail closed unless the environment is exactly the canonical shape."""
    if not isinstance(environment, dict):
        raise ScientificEnvironmentError(
            "scientific environment must be a dict, received %s"
            % type(environment).__name__
        )

    _require_exact_keys(
        environment, SCIENTIFIC_ENVIRONMENT_FIELDS, "scientific environment"
    )

    for field in SCIENTIFIC_ENVIRONMENT_FIELDS:
        _require_identity_string(environment[field], field)

    for field, approved in (
        ("schemaVersion", SCIENTIFIC_ENVIRONMENT_SCHEMA_VERSION),
        ("authoritySolverGeneration", AUTHORITY_SOLVER_GENERATION),
        ("horizonModelGeneration", HORIZON_MODEL_GENERATION),
    ):
        if environment[field] != approved:
            raise ScientificEnvironmentError(
                "%s must be %r, received %r" % (field, approved, environment[field])
            )

    _require_upper_sha256(environment["iersDataSha256"], "iersDataSha256")
    _require_digest(environment["ephemerisDataSetId"], "ephemerisDataSetId")

    return environment


def build_scientific_environment(
    *,
    python_version,
    skyfield_version,
    numpy_version,
    jplephem_version,
    iers_data_sha256,
    ephemeris_data_set_id,
):
    """Assemble the canonical scientific environment material.

    Every value is supplied by the caller. This function never inspects the
    running process, which keeps it deterministic and independently testable.
    Collecting live values is the caller's decision, not a side effect here.
    """
    environment = {
        "authoritySolverGeneration": AUTHORITY_SOLVER_GENERATION,
        "ephemerisDataSetId": ephemeris_data_set_id,
        "horizonModelGeneration": HORIZON_MODEL_GENERATION,
        "iersDataSha256": iers_data_sha256,
        "jplephemVersion": jplephem_version,
        "numpyVersion": numpy_version,
        "pythonVersion": python_version,
        "schemaVersion": SCIENTIFIC_ENVIRONMENT_SCHEMA_VERSION,
        "skyfieldVersion": skyfield_version,
    }
    return validate_scientific_environment(environment)


def derive_scientific_environment_id(environment):
    """Derive scientificEnvironmentId from a validated canonical environment."""
    return canonical_digest(validate_scientific_environment(environment))


def sha256_file(path):
    """Stream a file and return its UPPERCASE hex SHA-256.

    Never called at import. Callers decide when to pay the cost; hashing the
    pinned HPC reference BSP set is a startup-time operation introduced in a
    later A0 step, not here.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_FILE_READ_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def python_version_string():
    """Return the canonical Python identity, for example "CPython 3.13.7".

    Deliberately excludes sys.version, which embeds the build tag, build
    timestamp and compiler string. Those are platform noise and would make
    the identity un-recomputable on another operating system.
    """
    return "%s %d.%d.%d" % (
        platform.python_implementation(),
        sys.version_info[0],
        sys.version_info[1],
        sys.version_info[2],
    )


def distribution_version(distribution_name):
    """Return an installed distribution version without importing the package."""
    return _distribution_version(distribution_name)


def resolve_iers_data_path():
    """Locate the IERS/Delta-T file inside the RUNNING Skyfield installation.

    Skyfield is imported lazily, inside this function, so that importing this
    module stays standard-library only. No absolute path is hard-coded: the
    location is derived from the package actually in use.
    """
    import skyfield

    return Path(skyfield.__file__).resolve().parent / "data" / "iers.npz"
