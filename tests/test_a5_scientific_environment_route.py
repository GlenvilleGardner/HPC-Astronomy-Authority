"""A5 verification for the scientific environment provenance surface.

Covers scientific_provenance.py and GET /scientific-environment in server.py:
startup establishment, immutability, the published response shape,
independently reproducible derivations, mutation sensitivity of every
scientific component, serialization independence, fail-closed behaviour, the
absence of astronomy and filesystem access on the request path, the security
surface, and preservation of every previously published route.

Test framework: standard-library unittest.
Scientific execution uses the repository's certified Skyfield runtime and the
real pinned Authority ephemeris artifacts.

INVOCATION
----------
Run from the repository root with the repository interpreter:

    .\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v

ROUTE SEAM

The route function is invoked directly, following A1c, A1d, A3c-1..4 and A4.
httpx is not installed, so starlette's TestClient is unavailable, and
installing a dependency in order to test is not permitted.

WHY MOST TESTS DO NOT HASH THE ARTIFACT SET

Hashing the pinned artifacts takes roughly nineteen seconds. The process
already paid that once, at import, and that established record is asserted
against directly. Everything that must vary an input - a changed version, a
changed artifact hash, a malformed field - drives the PURE builder with
sha256_file patched, which is what makes those properties testable at all: the
real artifacts cannot be mutated to order.

INDEPENDENT ORACLE DISCIPLINE

The canonical field set, the route path, the digest prefix and the expected
role names are declared here as test-local literals rather than imported from
the modules under test. Digests are recomputed with hashlib and json directly,
never by calling the module's own canonical_digest, so a defective canonical
serializer cannot agree with itself.
"""

import hashlib
import inspect
import json
import re
import unittest
from unittest import mock

import astronomy_solver
import scientific_provenance
import server
from scientific_environment import ScientificEnvironmentError


# --- Test-local literals. Never imported from the modules under test. ------

ENVIRONMENT_PATH = "/scientific-environment"

EXPECTED_TOP_LEVEL_KEYS = (
    "environment",
    "scientificEnvironmentId",
    "ephemerisManifest",
)

EXPECTED_ENVIRONMENT_KEYS = (
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

EXPECTED_ROLES = ("ancient", "future", "primary")
EXPECTED_ROLE_KEYS = ("routingName", "sha256")

DIGEST_PREFIX = "sha256:"
UPPER_HEX = set("0123456789ABCDEF")
LOWER_HEX = set("0123456789abcdef")

# A synthetic artifact set. Deliberately not the real filenames, so a test
# cannot pass by accidentally reading the real installation.
STUB_ROUTING = {
    "ancient": "stub-ancient.bsp",
    "future": "stub-future.bsp",
    "primary": "stub-primary.bsp",
}
STUB_HASHES = {
    "stub-ancient.bsp": "A" * 64,
    "stub-future.bsp": "B" * 64,
    "stub-primary.bsp": "C" * 64,
}
STUB_IERS = "D" * 64


def independent_digest(obj):
    """Recompute a canonical digest here, not through the module under test."""
    text = json.dumps(
        obj, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    )
    return DIGEST_PREFIX + hashlib.sha256(text.encode("utf-8")).hexdigest()


def stub_sha256_file(path):
    """Stand in for artifact hashing, keyed on the filename component."""
    name = str(path).replace("\\", "/").rsplit("/", 1)[-1]

    if name in STUB_HASHES:
        return STUB_HASHES[name]

    return STUB_IERS


def build_stub(**overrides):
    """Build a provenance record from synthetic inputs, hashing nothing.

    ``hashes`` and ``iers`` are consumed here; every remaining keyword patches
    the same-named attribute of the module under test. They are removed BEFORE
    the closure is built, so a lazily consumed keyword can never also be
    mistaken for a patch target.
    """
    hashes = dict(STUB_HASHES)
    hashes.update(overrides.pop("hashes", {}))
    iers = overrides.pop("iers", STUB_IERS)

    def sha256(path):
        name = str(path).replace("\\", "/").rsplit("/", 1)[-1]
        return hashes.get(name, iers)

    with mock.patch.object(scientific_provenance, "sha256_file", sha256):
        with mock.patch.object(
            scientific_provenance, "resolve_iers_data_path",
            lambda: "/nowhere/iers.npz",
        ):
            patches = []
            for name, value in overrides.items():
                patches.append(
                    mock.patch.object(scientific_provenance, name, value)
                )
            for patch in patches:
                patch.start()
            try:
                return scientific_provenance.build_scientific_provenance(
                    ephemeris_dir="/nowhere",
                    routing_names=dict(STUB_ROUTING),
                )
            finally:
                for patch in reversed(patches):
                    patch.stop()


def registered_routes():
    found = {}
    for route in server.app.routes:
        endpoint = getattr(route, "endpoint", None)
        if endpoint is None:
            continue
        found[route.path] = (
            set(route.methods),
            endpoint.__name__,
            list(inspect.signature(endpoint).parameters),
        )
    return found


# --- 1. The published response -------------------------------------------


class TestPublishedResponse(unittest.TestCase):
    def setUp(self):
        self.body = server.scientific_environment()

    def test_the_route_is_registered_and_takes_no_parameters(self):
        routes = registered_routes()

        self.assertIn(ENVIRONMENT_PATH, routes)

        methods, name, parameters = routes[ENVIRONMENT_PATH]

        self.assertIn("GET", methods)
        self.assertEqual(name, "scientific_environment")
        self.assertEqual(parameters, [])

    def test_the_shape_is_exactly_the_published_contract(self):
        self.assertEqual(tuple(self.body), EXPECTED_TOP_LEVEL_KEYS)
        self.assertEqual(
            tuple(self.body["environment"]), EXPECTED_ENVIRONMENT_KEYS
        )

        manifest = self.body["ephemerisManifest"]

        self.assertEqual(manifest["schemaVersion"], "1")
        self.assertEqual(
            tuple(sorted(k for k in manifest if k != "schemaVersion")),
            EXPECTED_ROLES,
        )
        for role in EXPECTED_ROLES:
            self.assertEqual(tuple(manifest[role]), EXPECTED_ROLE_KEYS)

    def test_every_identity_scalar_has_the_governed_spelling(self):
        environment = self.body["environment"]

        self.assertTrue(
            self.body["scientificEnvironmentId"].startswith(DIGEST_PREFIX)
        )
        self.assertTrue(
            set(self.body["scientificEnvironmentId"][len(DIGEST_PREFIX):])
            <= LOWER_HEX
        )
        self.assertTrue(
            environment["ephemerisDataSetId"].startswith(DIGEST_PREFIX)
        )
        self.assertTrue(set(environment["iersDataSha256"]) <= UPPER_HEX)
        self.assertEqual(len(environment["iersDataSha256"]), 64)

        for role in EXPECTED_ROLES:
            entry = self.body["ephemerisManifest"][role]
            self.assertEqual(len(entry["sha256"]), 64)
            self.assertTrue(set(entry["sha256"]) <= UPPER_HEX)

        for field in EXPECTED_ENVIRONMENT_KEYS:
            value = environment[field]
            self.assertIsInstance(value, str)
            self.assertTrue(value)
            self.assertEqual(value, value.strip())

    def test_the_identity_is_independently_reproducible(self):
        """Recomputed here with hashlib and json, not by the module."""
        self.assertEqual(
            independent_digest(self.body["environment"]),
            self.body["scientificEnvironmentId"],
        )

    def test_the_manifest_independently_reproduces_the_data_set_id(self):
        self.assertEqual(
            independent_digest(self.body["ephemerisManifest"]),
            self.body["environment"]["ephemerisDataSetId"],
        )

    def test_the_manifest_describes_the_artifacts_actually_routed_to(self):
        routed = {
            "ancient": astronomy_solver.ANCIENT_KERNEL,
            "future": astronomy_solver.FUTURE_KERNEL,
            "primary": astronomy_solver.PRIMARY_KERNEL,
        }

        for role, name in routed.items():
            self.assertEqual(
                self.body["ephemerisManifest"][role]["routingName"], name
            )

    def test_repeated_reads_are_identical(self):
        self.assertEqual(self.body, server.scientific_environment())

    def test_the_published_record_cannot_be_mutated_by_a_consumer(self):
        self.body["environment"]["skyfieldVersion"] = "0.0.0"
        self.body["ephemerisManifest"]["primary"]["sha256"] = "0" * 64

        fresh = server.scientific_environment()

        self.assertNotEqual(fresh["environment"]["skyfieldVersion"], "0.0.0")
        self.assertNotEqual(
            fresh["ephemerisManifest"]["primary"]["sha256"], "0" * 64
        )


# --- 2. Mutation sensitivity ---------------------------------------------


class TestIdentitySensitivity(unittest.TestCase):
    """Anything scientifically material must move the identity."""

    def setUp(self):
        self.baseline = build_stub()

    def test_the_stub_builder_is_non_vacuous(self):
        self.assertEqual(
            tuple(self.baseline["environment"]), EXPECTED_ENVIRONMENT_KEYS
        )
        self.assertEqual(
            independent_digest(self.baseline["environment"]),
            self.baseline["scientificEnvironmentId"],
        )
        self.assertEqual(
            self.baseline["ephemerisManifest"]["primary"]["sha256"],
            STUB_HASHES["stub-primary.bsp"],
        )

    def test_each_scientific_component_version_changes_the_identity(self):
        for name, replacement in (
            ("python_version_string", lambda: "CPython 9.9.9"),
            ("distribution_version", lambda d: "0.0.0"),
        ):
            with self.subTest(component=name):
                mutated = build_stub(**{name: replacement})

                self.assertNotEqual(
                    mutated["scientificEnvironmentId"],
                    self.baseline["scientificEnvironmentId"],
                )

    def test_each_distribution_independently_changes_the_identity(self):
        real = scientific_provenance.distribution_version

        for distribution in ("skyfield", "numpy", "jplephem"):
            with self.subTest(distribution=distribution):
                def one_changed(name, _target=distribution, _real=real):
                    return "0.0.0" if name == _target else _real(name)

                mutated = build_stub(distribution_version=one_changed)

                self.assertNotEqual(
                    mutated["scientificEnvironmentId"],
                    self.baseline["scientificEnvironmentId"],
                )

    def test_a_changed_iers_table_changes_the_identity(self):
        mutated = build_stub(iers="E" * 64)

        self.assertNotEqual(
            mutated["environment"]["iersDataSha256"],
            self.baseline["environment"]["iersDataSha256"],
        )
        self.assertNotEqual(
            mutated["scientificEnvironmentId"],
            self.baseline["scientificEnvironmentId"],
        )

    def test_changed_ephemeris_CONTENT_changes_both_derived_identities(self):
        """The decisive artifact-drift case: same filenames, different bytes."""
        for role, filename in STUB_ROUTING.items():
            with self.subTest(role=role):
                mutated = build_stub(hashes={filename: "F" * 64})

                self.assertEqual(
                    mutated["ephemerisManifest"][role]["routingName"],
                    self.baseline["ephemerisManifest"][role]["routingName"],
                )
                self.assertNotEqual(
                    mutated["environment"]["ephemerisDataSetId"],
                    self.baseline["environment"]["ephemerisDataSetId"],
                )
                self.assertNotEqual(
                    mutated["scientificEnvironmentId"],
                    self.baseline["scientificEnvironmentId"],
                )

    def test_exchanging_two_artifacts_is_detectable(self):
        """Routing names bound to bytes: swapping them must be visible."""
        swapped = build_stub(hashes={
            "stub-ancient.bsp": STUB_HASHES["stub-future.bsp"],
            "stub-future.bsp": STUB_HASHES["stub-ancient.bsp"],
        })

        self.assertNotEqual(
            swapped["scientificEnvironmentId"],
            self.baseline["scientificEnvironmentId"],
        )

    def test_serialization_order_cannot_change_the_identity(self):
        environment = self.baseline["environment"]
        reordered = dict(reversed(list(environment.items())))

        self.assertNotEqual(list(reordered), list(environment))
        self.assertEqual(
            independent_digest(reordered), independent_digest(environment)
        )

    def test_the_identity_is_deterministic_across_rebuilds(self):
        self.assertEqual(build_stub(), self.baseline)


# --- 3. Fail-closed behaviour --------------------------------------------


class TestFailsClosed(unittest.TestCase):
    def test_a_malformed_component_is_refused(self):
        for name, replacement in (
            ("python_version_string", lambda: ""),
            ("python_version_string", lambda: "  CPython 3.13.7  "),
            ("distribution_version", lambda d: ""),
        ):
            with self.subTest(replacement=replacement):
                with self.assertRaises(ScientificEnvironmentError):
                    build_stub(**{name: replacement})

    def test_a_malformed_artifact_hash_is_refused(self):
        for bad in ("", "abc", "a" * 64, "G" * 64, "A" * 63):
            with self.subTest(hash=bad):
                with self.assertRaises(ScientificEnvironmentError):
                    build_stub(hashes={"stub-primary.bsp": bad})

    def test_a_malformed_iers_hash_is_refused(self):
        with self.assertRaises(ScientificEnvironmentError):
            build_stub(iers="not-a-hash")

    def test_an_incomplete_role_set_is_refused(self):
        with mock.patch.object(
            scientific_provenance, "sha256_file", stub_sha256_file
        ):
            for routing in (
                {"ancient": "a.bsp", "future": "f.bsp"},
                dict(STUB_ROUTING, extra="x.bsp"),
                {},
            ):
                with self.subTest(routing=sorted(routing)):
                    with self.assertRaises(ValueError):
                        scientific_provenance.build_scientific_provenance(
                            ephemeris_dir="/nowhere", routing_names=routing
                        )

    def test_establishment_is_refused_a_second_time(self):
        """The identity responses were issued under is never replaced."""
        with self.assertRaises(RuntimeError):
            scientific_provenance.establish_scientific_provenance(
                ephemeris_dir="/nowhere", routing_names=dict(STUB_ROUTING)
            )

    def test_reading_before_establishment_fails_closed(self):
        with mock.patch.object(scientific_provenance, "_PROVENANCE", None):
            with self.assertRaises(
                scientific_provenance.ScientificProvenanceUnavailable
            ):
                scientific_provenance.scientific_provenance()


# --- 4. The request path does nothing -------------------------------------


class TestRequestPathIsInert(unittest.TestCase):
    def test_no_astronomy_is_performed(self):
        with mock.patch.object(
            astronomy_solver.almanac, "find_discrete"
        ) as discrete, mock.patch.object(
            astronomy_solver, "load_kernel"
        ) as kernel:
            body = server.scientific_environment()

        discrete.assert_not_called()
        kernel.assert_not_called()
        self.assertEqual(tuple(body), EXPECTED_TOP_LEVEL_KEYS)

    def test_no_file_is_opened(self):
        real_open = open

        with mock.patch("builtins.open", side_effect=real_open) as opener:
            body = server.scientific_environment()

        opener.assert_not_called()
        self.assertEqual(tuple(body), EXPECTED_TOP_LEVEL_KEYS)

    def test_nothing_is_hashed_or_discovered(self):
        with mock.patch.object(
            scientific_provenance, "sha256_file"
        ) as hasher, mock.patch.object(
            scientific_provenance, "resolve_iers_data_path"
        ) as resolver, mock.patch.object(
            scientific_provenance, "distribution_version"
        ) as versions, mock.patch.object(
            scientific_provenance, "build_scientific_provenance"
        ) as builder:
            server.scientific_environment()

        hasher.assert_not_called()
        resolver.assert_not_called()
        versions.assert_not_called()
        builder.assert_not_called()


# --- 5. Security surface ---------------------------------------------------


class TestSecuritySurface(unittest.TestCase):
    def setUp(self):
        self.body = server.scientific_environment()
        self.text = json.dumps(self.body)

    def test_no_filesystem_path_is_published(self):
        for marker in ("/", "\\", ":\\", "C:", "Users", "site-packages",
                       "ephemeris/", ".venv", "home", "tmp"):
            with self.subTest(marker=marker):
                if marker == "/":
                    # "sha256:" prefixes are not paths; check separators only.
                    continue
                self.assertNotIn(marker, self.text)

    def test_artifact_names_are_bare_filenames(self):
        for role in EXPECTED_ROLES:
            name = self.body["ephemerisManifest"][role]["routingName"]

            self.assertNotIn("/", name)
            self.assertNotIn("\\", name)
            self.assertTrue(name.endswith(".bsp"))

    def test_no_host_port_process_or_environment_information(self):
        import os
        import socket

        lowered = self.text.lower()

        for forbidden in ("host", "port", "pid", "cwd", "user", "token",
                          "secret", "password", "credential", "key=",
                          "authorization", "url", "127.0.0.1", "localhost"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, lowered)

        self.assertNotIn(socket.gethostname().lower(), lowered)
        self.assertNotIn(str(os.getpid()), self.text)

    def test_no_environment_variable_value_is_published(self):
        import os

        for name, value in os.environ.items():
            if not value or len(value) < 6:
                continue
            with self.subTest(variable=name):
                self.assertNotIn(value, self.text)

    def test_only_the_three_published_keys_exist(self):
        self.assertEqual(tuple(self.body), EXPECTED_TOP_LEVEL_KEYS)


# --- 6. Structural controls ------------------------------------------------


class TestSourceGuard(unittest.TestCase):
    BLOCK_MARKER = "# A5 - scientific environment provenance route."
    GOVERNED_BLOCK_MARKER = re.compile(
        r"^# A\d+[a-z]?(?:-[0-9a-z]+)? - ", re.MULTILINE
    )

    @property
    def server_source(self):
        return inspect.getsource(server)

    @staticmethod
    def executable(function):
        source = inspect.getsource(function)
        source = re.sub(r'"""[\s\S]*?"""', "", source, count=1)
        return "\n".join(line.split("#", 1)[0] for line in source.split("\n"))

    def test_the_marker_obeys_the_governed_grammar(self):
        source = self.server_source
        markers = self.GOVERNED_BLOCK_MARKER.findall(source)

        self.assertGreater(len(markers), 4)

        own = source.find(self.BLOCK_MARKER)
        self.assertNotEqual(own, -1, "A5 block marker not found")
        self.assertIsNotNone(self.GOVERNED_BLOCK_MARKER.match(source, own))
        self.assertIsNone(self.GOVERNED_BLOCK_MARKER.search(source, own + 1))
        self.assertGreater(source.find("def scientific_environment("), own)

    def test_the_preceding_governed_block_is_still_delimited(self):
        source = self.server_source
        previous = source.find("# A4 - ABC-1 bulk sunset count route.")

        self.assertNotEqual(previous, -1)
        self.assertLess(previous, source.find(self.BLOCK_MARKER))

        following = self.GOVERNED_BLOCK_MARKER.search(source, previous + 1)

        self.assertIsNotNone(following)
        self.assertEqual(following.start(), source.find(self.BLOCK_MARKER))

    def test_the_route_body_only_reads(self):
        body = self.executable(server.scientific_environment)

        self.assertIn("scientific_provenance()", body)
        self.assertGreater(len(body), 20)

        for forbidden in (
            "sha256", "open(", "Path", "os.", "almanac", "load_kernel",
            "build_scientific", "derive_", "establish_", "if ", "for ",
            "try", "HTTPException",
        ):
            with self.subTest(token=forbidden):
                self.assertNotIn(forbidden, body)

    def test_the_source_scanner_is_non_vacuous(self):
        planted = "def f():\n    return sha256_file(open('/etc/passwd'))\n"
        stripped = re.sub(r'"""[\s\S]*?"""', "", planted, count=1)

        self.assertIn("sha256", stripped)
        self.assertIn("open(", stripped)

    def test_establishment_happens_before_the_application_object(self):
        """A route can never be reachable without an established identity."""
        source = self.server_source

        established = source.find("establish_scientific_provenance(")
        app_created = source.find('app = FastAPI(')

        self.assertNotEqual(established, -1)
        self.assertNotEqual(app_created, -1)
        self.assertLess(established, app_created)

    def test_establishment_follows_the_runtime_gate(self):
        source = self.server_source

        gate = source.find("verify_runtime_scientific_components()")
        established = source.find("establish_scientific_provenance(")

        self.assertNotEqual(gate, -1)
        self.assertLess(gate, established)

    def test_the_provenance_module_performs_no_astronomy(self):
        """Executable source only.

        The module documents at length which astronomy it does NOT perform,
        and that prose must never fail this scan. Docstrings are removed
        before anything is inspected, then line comments.
        """
        source = inspect.getsource(scientific_provenance)
        executable = re.sub(r'"""[\s\S]*?"""', "", source)
        executable = "\n".join(
            line.split("#", 1)[0] for line in executable.split("\n")
        )

        # Non-vacuity: stripping must not have emptied the module.
        self.assertGreater(len(executable), 600)

        for forbidden in (
            "import astronomy_solver", "from astronomy_solver",
            "almanac", "find_discrete", "wgs84", "timescale", "tt_jd",
            "load_kernel", "import skyfield", "fastapi", "HTTPException",
        ):
            with self.subTest(token=forbidden):
                self.assertNotIn(forbidden, executable)

        # Non-vacuity: it really is the module that builds the record.
        self.assertIn("build_scientific_environment", executable)
        self.assertIn("build_ephemeris_manifest", executable)

    def test_sampling_policy_is_absent_from_the_environment(self):
        """Ruled out of the A0.1 schema; covered by skyfieldVersion."""
        body = server.scientific_environment()

        for absent in ("samplingStepDays", "stepDays", "step_days",
                       "countContract", "sunsetCount"):
            self.assertNotIn(absent, json.dumps(body))


# --- 7. Previously published routes preserved -----------------------------


class TestPublishedRoutesPreserved(unittest.TestCase):
    PREVIOUS = {
        "/health": "health",
        "/delta-t/{year}": "delta_t",
        "/delta-t-bce/{year}": "delta_t_bce",
        "/equinox/{year}": "equinox",
        "/equinox-bce/{year}": "equinox_bce",
        "/season-events": "season_events",
        "/season-events-bce/{year}": "season_events_bce",
        "/solar_longitude": "solar",
        "/subsolar-point": "subsolar",
        "/sunset": "sunset",
        "/sunset-after": "sunset_after",
        "/sunset-successor": "sunset_successor",
        "/solar-longitude-event-before": "solar_longitude_event_before",
        "/solar-longitude-event-after": "solar_longitude_event_after",
        "/sunset-bracket": "sunset_bracket",
        "/solar-longitude-event-in-year": "solar_longitude_event_in_year",
        "/sunset-event-after": "sunset_event_after",
        "/sunset-count": "sunset_count",
    }

    def test_every_previously_published_route_is_still_registered(self):
        routes = registered_routes()

        for path, endpoint in self.PREVIOUS.items():
            with self.subTest(path=path):
                self.assertIn(path, routes)
                self.assertEqual(routes[path][1], endpoint)

    # Application routes added by governed increments AFTER A5.
    #
    # A5's closed-world claim is about the surface AS OF A5: that this
    # increment added exactly one route. A later additive increment does not
    # weaken that claim, but it does make an unqualified count of the whole
    # application factually wrong. Subtracting the later additions keeps the
    # claim exactly as strong as it was while letting it stay true.
    #
    # Each entry is owned by the increment that added it, and its own suite
    # owns the closed world as of that increment.
    POST_A5_ROUTES = frozenset({
        "/earth-rotation", "/solar-regime", "/night-start-after",
        "/solar-crossing", "/civil-instant",
    })

    def test_exactly_one_route_was_added(self):
        routes = registered_routes()
        published = {
            path for path in routes
            if not path.startswith(("/openapi", "/docs", "/redoc"))
        } - self.POST_A5_ROUTES

        self.assertEqual(published - set(self.PREVIOUS), {ENVIRONMENT_PATH})

    def test_an_existing_astronomical_route_still_answers(self):
        body = server.health()

        self.assertEqual(body["status"], "ok")
        self.assertTrue(body["defaultKernel"].endswith(".bsp"))


if __name__ == "__main__":
    unittest.main()
