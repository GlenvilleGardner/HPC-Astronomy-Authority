"""HPC Astronomy Authority - Scientific Provenance Surface.

A5 - the scientific identity this process actually established, collected once
and then read.

WHAT THIS MODULE IS

The seam between A0.1's pure identity machinery and a running Authority. A0.1
defines WHAT the canonical scientific environment is and how it is hashed; it
deliberately never inspects the process, which is what keeps it deterministic
and independently testable. Something has to look at the running installation
and the artifacts on disk. That is this module, and nothing else in it.

It performs NO astronomy. No ephemeris is loaded, no timescale is built, no
kernel is selected, no crossing is solved and no observer is validated. It
reads version strings and hashes files.

WHY EVERYTHING IS COLLECTED AT STARTUP

The scientific identity of this process cannot change while it runs, so it is
established once and then only read. That is not a cache: a cache is an
optimization over a value that could be recomputed, and recomputing this one
per request would hash 3.3 GB of pinned NASA/JPL artifacts to re-derive a
constant.

The cost is real and is accepted deliberately. Hashing the pinned artifact set
takes roughly nineteen seconds, and the Authority pays it before it serves
anything. An Authority that cannot state its own scientific identity must not
answer scientific questions, so establishing that identity is a precondition
of becoming operational rather than a feature of the first request.

FAILS CLOSED AT STARTUP, NOT AT REQUEST TIME

If collection or validation fails, establishment raises and the process does
not start. There is deliberately no degraded mode, no partially initialised
provenance, and no endpoint that reports its own unavailability: a running
Authority either knows what it is or does not exist.

Because establishment succeeded before any route was reachable, the request
path has no failure mode of its own. It reads an already-validated value.

THE BUILDER IS PURE, THE ESTABLISHMENT IS NOT

build_scientific_provenance takes the artifact directory and the routing names
from its caller and returns a record; it holds no state and can be called
repeatedly with supplied values, which is what makes every property below
testable without a process. establish_scientific_provenance is the one place
module state is written, exactly once, at startup.

Standard library and A0.1 only. This module deliberately does not import
astronomy_solver: the artifact directory and the routing names are supplied by
the caller, so nothing here can build a timescale, and the scientific-runtime
gate's ordering guarantee is not something this module can weaken.
"""

import copy
import os

from scientific_environment import (
    EPHEMERIS_ROLES,
    build_ephemeris_manifest,
    build_scientific_environment,
    derive_ephemeris_data_set_id,
    derive_scientific_environment_id,
    distribution_version,
    python_version_string,
    resolve_iers_data_path,
    sha256_file,
)

_PROVENANCE = None


class ScientificProvenanceUnavailable(RuntimeError):
    """The scientific identity was never established.

    Raised only by a read that precedes establishment. It cannot occur on a
    served request: establishment happens at startup and a failure there stops
    the process, so no route ever becomes reachable without it.
    """


def build_scientific_provenance(*, ephemeris_dir, routing_names):
    """Collect and validate the complete scientific provenance record.

    ``ephemeris_dir`` is the directory holding the pinned artifacts and
    ``routing_names`` maps each canonical role to the artifact filename that
    role currently routes to. Both are supplied rather than discovered, for
    the same reason A0.1 supplies every field to its builder: a function that
    inspects its own environment cannot be tested against an environment it
    does not have.

    Returns a record carrying the canonical environment, its derived
    identifier, and the canonical ephemeris manifest the identifier was
    derived from. The manifest is published alongside the identity rather than
    folded into it: ``ephemerisDataSetId`` inside the environment IS the
    manifest's digest, so publishing the manifest lets an independent consumer
    recompute that identifier instead of taking it on trust. It is the
    identity's preimage, not a second copy of it.

    Every value is validated by A0.1 before it is returned. Nothing is
    repaired, defaulted or trimmed.
    """
    if set(routing_names) != set(EPHEMERIS_ROLES):
        raise ValueError(
            "routing_names must cover exactly %r, received %r"
            % (sorted(EPHEMERIS_ROLES), sorted(routing_names))
        )

    manifest = build_ephemeris_manifest(
        {
            role: {
                "routingName": routing_names[role],
                # Content identity of the artifact this role actually routes
                # to. Binding the name to the bytes is what makes exchanging
                # two artifacts detectable rather than invisible.
                "sha256": sha256_file(
                    os.path.join(ephemeris_dir, routing_names[role])
                ),
            }
            for role in EPHEMERIS_ROLES
        }
    )

    environment = build_scientific_environment(
        python_version=python_version_string(),
        skyfield_version=distribution_version("skyfield"),
        numpy_version=distribution_version("numpy"),
        jplephem_version=distribution_version("jplephem"),
        # The IERS/Delta-T table inside the RUNNING installation, located
        # through A0.1 rather than by any path stated here.
        iers_data_sha256=sha256_file(resolve_iers_data_path()),
        ephemeris_data_set_id=derive_ephemeris_data_set_id(manifest),
    )

    return {
        "environment": environment,
        "scientificEnvironmentId": derive_scientific_environment_id(environment),
        "ephemerisManifest": manifest,
    }


def establish_scientific_provenance(*, ephemeris_dir, routing_names):
    """Establish this process's scientific identity. Called once, at startup.

    The only writer of module state here. A second establishment is refused
    rather than allowed to replace an identity that responses may already have
    been issued under.

    Raises out of startup on any collection or validation failure, which stops
    the process. That is the intended behaviour and must not be softened.
    """
    global _PROVENANCE

    if _PROVENANCE is not None:
        raise RuntimeError(
            "scientific provenance is already established and is immutable"
        )

    _PROVENANCE = build_scientific_provenance(
        ephemeris_dir=ephemeris_dir, routing_names=routing_names
    )


def scientific_provenance():
    """Return the established scientific provenance.

    A DEEP COPY, so no consumer can mutate the identity this process is
    serving under. The established record itself is written once and never
    again.

    Performs no astronomy, opens no file, hashes nothing and discovers
    nothing. It reads an already-validated value and copies it.
    """
    if _PROVENANCE is None:
        raise ScientificProvenanceUnavailable(
            "scientific provenance has not been established"
        )

    return copy.deepcopy(_PROVENANCE)
