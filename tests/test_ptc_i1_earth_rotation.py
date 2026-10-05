"""PTC-I1 verification for the Earth rotation evidence surface.

Covers earth_rotation.py and GET /earth-rotation in server.py: the published
response shape, the scientific relationships among TT, Delta T, UT1, UT1-UTC
and the Earth Rotation Angle, provenance regime derivation, determinism,
validation and fail-closed behaviour, the security surface, the absence of any
observer or calendar concept, and preservation of every previously published
route.

Test framework: standard-library unittest.
Scientific execution uses the repository's certified Skyfield runtime.

INVOCATION
----------
Run from the repository root with the repository interpreter:

    .\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v

ROUTE SEAM

The route function is invoked directly, following A1c, A1d, A3c-1..4, A4 and
A5. httpx is not installed, so starlette's TestClient is unavailable, and
installing a dependency in order to test is not permitted.

INDEPENDENT ORACLE DISCIPLINE

The Earth Rotation Angle is recomputed here from the conventional relation
published in the IERS Conventions, with the constants written as test-local
literals, never by calling the same routine the module under test calls. A
defective ERA implementation therefore cannot agree with itself.

The field set, the route path, the convention string and the regime labels are
likewise declared as test-local literals rather than imported from the module
under test, so a renamed field is a failure rather than a silent rename.

NO SCRATCHPAD NUMBERS ARE TRUSTED

Reference instants are named by their astronomical meaning and encoded through
the certified transport at test time. No decimal value carried over from an
investigation report is asserted as authority.
"""

import inspect
import math
import unittest

from fastapi import HTTPException

import earth_rotation
import server
from exact_time_transport import encode_tt_bits

# --- Independent oracle: IERS Conventions relation, IAU 2000 B1.8 ----------
#
# ERA(Tu) = 2*pi*(0.7790572732640 + 1.00273781191135448 * Tu)
# Tu = (Julian UT1 date - 2451545.0)
#
# Written here in rotations rather than radians, because that is the form the
# published field carries. These constants are the test's own oracle and are
# deliberately not imported from anywhere in the repository.
ERA_AT_J2000_ROTATIONS = 0.7790572732640
ERA_ROTATIONS_PER_UT1_DAY = 1.00273781191135448
J2000_TT = 2451545.0

ROTATION_PATH = "/earth-rotation"

EXPECTED_KEYS = (
    "ttBits",
    "tt",
    "utc",
    "ut1JulianDate",
    "deltaTSeconds",
    "deltaTProvenance",
    "dut1Seconds",
    "dut1Provenance",
    "eraRotations",
    "rotationConvention",
)

REGIME_TABULATED = "TABULATED"
REGIME_MODELLED = "MODELLED"
REGIME_UNSUPPORTED = "UNSUPPORTED"

# TT - TAI, the fixed offset of Terrestrial Time from atomic time.
TT_MINUS_TAI_SECONDS = 32.184

SECONDS_PER_DAY = 86400.0


def bits_for(*utc_parts):
    """The published transport encoding of a named UTC instant's TT state."""
    return encode_tt_bits(float(earth_rotation.ts.utc(*utc_parts).tt))


def era_rotations_oracle(ut1_julian_date):
    """ERA on [0, 1) from the conventional relation, computed independently."""
    tu = ut1_julian_date - J2000_TT
    return (ERA_AT_J2000_ROTATIONS + ERA_ROTATIONS_PER_UT1_DAY * tu) % 1.0


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


# --- Reference instants ----------------------------------------------------
#
# Named by astronomical meaning, encoded at test time through the certified
# runtime. The 2019 case sits in the HPC epoch era; the 2026 cases are the
# modern references used throughout the PTC investigation; the fourth is an
# ordinary instant with no special standing, present so the contract is not
# only exercised at interesting states.

EPOCH_ERA_2019 = (2019, 3, 20, 21, 58, 26)
MODERN_2026_EQUINOX = (2026, 3, 20, 14, 45, 57)
MODERN_2026_POLAR_SUNSET = (2026, 9, 25, 3, 19, 33)
ORDINARY_2024 = (2024, 7, 4, 6, 30, 0)

REFERENCE_INSTANTS = (
    EPOCH_ERA_2019,
    MODERN_2026_EQUINOX,
    MODERN_2026_POLAR_SUNSET,
    ORDINARY_2024,
)


# --- 1. The published response --------------------------------------------


class TestPublishedResponse(unittest.TestCase):
    def setUp(self):
        self.body = server.earth_rotation(bits_for(*MODERN_2026_EQUINOX))

    def test_the_route_is_registered_and_takes_only_an_exact_state(self):
        routes = registered_routes()

        self.assertIn(ROTATION_PATH, routes)

        methods, name, parameters = routes[ROTATION_PATH]

        self.assertIn("GET", methods)
        self.assertEqual(name, "earth_rotation")

        # DECISIVE: no observer. Earth Rotation Angle has no observer term, and
        # at the exact poles no unique meridian exists to supply one.
        self.assertEqual(parameters, ["ttBits"])

    def test_the_shape_is_exactly_the_published_contract(self):
        self.assertEqual(tuple(self.body), EXPECTED_KEYS)

    def test_the_instant_projection_matches_the_exact_route_family(self):
        bits = bits_for(*MODERN_2026_EQUINOX)

        self.assertEqual(self.body["ttBits"], bits)
        self.assertIsInstance(self.body["tt"], float)
        self.assertIsInstance(self.body["utc"], str)

    def test_scalars_carry_their_scientific_types(self):
        self.assertIsInstance(self.body["ut1JulianDate"], float)
        self.assertIsInstance(self.body["deltaTSeconds"], float)
        self.assertIsInstance(self.body["eraRotations"], float)
        self.assertIsInstance(self.body["rotationConvention"], str)

    def test_the_convention_is_named_on_the_wire(self):
        convention = self.body["rotationConvention"]

        # A consumer must be able to tell WHICH definition produced the angle
        # without reading this repository.
        self.assertIn("B1.8", convention)
        self.assertIn("IERS", convention)

    def test_the_response_is_deterministic(self):
        again = server.earth_rotation(bits_for(*MODERN_2026_EQUINOX))

        self.assertEqual(self.body, again)


# --- 2. Scientific relationships ------------------------------------------


class TestScientificRelationships(unittest.TestCase):
    def test_ut1_is_tt_less_delta_t(self):
        for parts in REFERENCE_INSTANTS:
            with self.subTest(instant=parts):
                body = server.earth_rotation(bits_for(*parts))

                expected = body["tt"] - body["deltaTSeconds"] / SECONDS_PER_DAY

                # Delta T IS defined as TT - UT1, so this is an identity and
                # should hold to representation precision, not to a tolerance
                # chosen for comfort.
                self.assertAlmostEqual(
                    body["ut1JulianDate"], expected, places=9
                )

    def test_era_agrees_with_the_conventional_relation(self):
        for parts in REFERENCE_INSTANTS:
            with self.subTest(instant=parts):
                body = server.earth_rotation(bits_for(*parts))

                oracle = era_rotations_oracle(body["ut1JulianDate"])

                # The certified runtime and the published IERS relation must
                # agree far below any level a consumer could act on.
                self.assertLess(abs(body["eraRotations"] - oracle), 1e-9)

    def test_era_is_a_rotation_fraction(self):
        for parts in REFERENCE_INSTANTS:
            with self.subTest(instant=parts):
                body = server.earth_rotation(bits_for(*parts))

                self.assertGreaterEqual(body["eraRotations"], 0.0)
                self.assertLess(body["eraRotations"], 1.0)

    def test_era_advances_at_the_conventional_sidereal_rate(self):
        # One mean solar day of UT1 advances ERA by slightly more than one
        # rotation. This is the distinction PTC research turned on, so it is
        # asserted rather than assumed.
        first = server.earth_rotation(bits_for(2026, 3, 20, 0, 0, 0))
        second = server.earth_rotation(bits_for(2026, 3, 21, 0, 0, 0))

        elapsed_ut1 = second["ut1JulianDate"] - first["ut1JulianDate"]
        advance = (
            second["eraRotations"] - first["eraRotations"]
        ) % 1.0 + math.floor(ERA_ROTATIONS_PER_UT1_DAY * elapsed_ut1)

        self.assertAlmostEqual(
            advance, ERA_ROTATIONS_PER_UT1_DAY * elapsed_ut1, places=9
        )

    def test_dut1_closes_the_atomic_time_identity(self):
        # delta_t = (TT - TAI) + (TAI - UTC) - (UT1 - UTC)
        # so (TAI - UTC), the accumulated leap offset, must come out an integer
        # number of seconds wherever UT1-UTC is published at all.
        body = server.earth_rotation(bits_for(*MODERN_2026_EQUINOX))

        self.assertIsNotNone(body["dut1Seconds"])

        leap_offset = (
            body["deltaTSeconds"] + body["dut1Seconds"] - TT_MINUS_TAI_SECONDS
        )

        self.assertAlmostEqual(leap_offset, round(leap_offset), places=6)
        self.assertGreater(leap_offset, 0)

    def test_era_wraps_without_discontinuity_in_the_underlying_angle(self):
        # Cross a wrap deliberately: the published value must fall back into
        # [0, 1) while the conventional relation continues to agree.
        before = server.earth_rotation(bits_for(2026, 3, 20, 12, 0, 0))
        after = server.earth_rotation(bits_for(2026, 3, 20, 13, 0, 0))

        self.assertGreater(before["eraRotations"], 0.9)
        self.assertLess(after["eraRotations"], 0.1)

        for body in (before, after):
            oracle = era_rotations_oracle(body["ut1JulianDate"])
            self.assertLess(abs(body["eraRotations"] - oracle), 1e-9)


# --- 3. Provenance --------------------------------------------------------


class TestProvenance(unittest.TestCase):
    def test_modern_states_are_reported_as_tabulated(self):
        body = server.earth_rotation(bits_for(*MODERN_2026_EQUINOX))

        self.assertEqual(body["deltaTProvenance"], REGIME_TABULATED)
        self.assertEqual(body["dut1Provenance"], REGIME_TABULATED)

    def test_deep_past_is_reported_as_modelled_and_withholds_dut1(self):
        body = server.earth_rotation(bits_for(-3000, 1, 1, 0, 0, 0))

        self.assertEqual(body["deltaTProvenance"], REGIME_MODELLED)

        # DECISIVE: the runtime WOULD return a difference here - tens of
        # thousands of seconds - and publishing it as UT1-UTC would be false.
        self.assertIsNone(body["dut1Seconds"])
        self.assertEqual(body["dut1Provenance"], REGIME_UNSUPPORTED)

    def test_far_future_is_reported_as_modelled_and_withholds_dut1(self):
        body = server.earth_rotation(bits_for(3000, 1, 1, 0, 0, 0))

        self.assertEqual(body["deltaTProvenance"], REGIME_MODELLED)
        self.assertIsNone(body["dut1Seconds"])
        self.assertEqual(body["dut1Provenance"], REGIME_UNSUPPORTED)

    def test_delta_t_is_still_published_outside_the_table(self):
        # Modelled is a statement about provenance, not a refusal. The value
        # remains available; only its confidence changes.
        body = server.earth_rotation(bits_for(1600, 1, 1, 0, 0, 0))

        self.assertIsInstance(body["deltaTSeconds"], float)
        self.assertEqual(body["deltaTProvenance"], REGIME_MODELLED)

    def test_the_regime_boundary_follows_the_pinned_table(self):
        # The boundary is derived from the table this process actually built,
        # not from a year written beside it, so it moves when the pinned
        # artifact moves.
        span = earth_rotation._TABULATED_SPAN

        self.assertIsNotNone(span)

        start, end = span

        self.assertEqual(earth_rotation.delta_t_regime(start), REGIME_TABULATED)
        self.assertEqual(earth_rotation.delta_t_regime(end), REGIME_TABULATED)
        self.assertEqual(
            earth_rotation.delta_t_regime(start - 1.0), REGIME_MODELLED
        )
        self.assertEqual(
            earth_rotation.delta_t_regime(end + 1.0), REGIME_MODELLED
        )


# --- 4. Validation and fail-closed behaviour ------------------------------


class TestValidation(unittest.TestCase):
    def assert_governed_400(self, ttBits):
        with self.assertRaises(HTTPException) as caught:
            server.earth_rotation(ttBits)

        self.assertEqual(caught.exception.status_code, 400)

        detail = caught.exception.detail

        self.assertIsInstance(detail, dict)
        self.assertIn("reason", detail)
        self.assertIn("message", detail)

        return detail

    def test_malformed_exact_state_is_refused(self):
        for value in ("", "not-hex", "4142A9B0", "4142a9b000000000", "Z" * 16,
                      "4142A9B0000000000"):
            with self.subTest(ttBits=value):
                self.assert_governed_400(value)

    def test_non_finite_states_are_refused(self):
        # Infinity and NaN are not times. The transport codec refuses both
        # spellings before any time-scale evaluation is attempted.
        for value in ("7FF0000000000000", "FFF0000000000000",
                      "7FF8000000000000"):
            with self.subTest(ttBits=value):
                self.assert_governed_400(value)

    def test_the_refusal_names_a_stable_machine_readable_reason(self):
        detail = self.assert_governed_400("not-hex")

        self.assertEqual(detail["reason"], "TT_BITS_INVALID")


# --- 5. Security surface --------------------------------------------------


class TestSecuritySurface(unittest.TestCase):
    def test_no_response_field_discloses_internal_topology(self):
        for parts in REFERENCE_INSTANTS:
            with self.subTest(instant=parts):
                serialized = repr(server.earth_rotation(bits_for(*parts)))

                for leak in ("C:\\", "/home", "/var", ".bsp", ".npz",
                             "127.0.0.1", "localhost", "http://", "https://",
                             "Traceback", "site-packages", "ephemeris"):
                    self.assertNotIn(leak, serialized)

    def test_a_refusal_discloses_nothing_beyond_the_governed_envelope(self):
        with self.assertRaises(HTTPException) as caught:
            server.earth_rotation("not-hex")

        serialized = repr(caught.exception.detail)

        for leak in ("C:\\", "/home", "Traceback", "site-packages",
                     ".bsp", ".npz", "File \""):
            self.assertNotIn(leak, serialized)


# --- 6. The Authority knows nothing about calendars -----------------------


class TestNoCalendarConcepts(unittest.TestCase):
    FORBIDDEN = (
        "hpc", "telma", "abib", "sabbath", "creation", "weekday",
        "continuity", "polar", "sce", "month",
    )

    def test_the_module_source_names_no_calendar_concept_as_an_identifier(self):
        source = inspect.getsource(earth_rotation)

        # Executable lines only: the module docstring legitimately explains
        # what it refuses to do, and those refusals must remain documented.
        executable = "\n".join(
            line for line in source.splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
        executable = executable.split('"""')
        executable = "".join(executable[::2]).lower()

        for token in self.FORBIDDEN:
            self.assertNotIn(token, executable)

    def test_no_published_field_is_a_calendar_quantity(self):
        body = server.earth_rotation(bits_for(*MODERN_2026_EQUINOX))

        for key in body:
            for token in self.FORBIDDEN:
                self.assertNotIn(token, key.lower())


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
        "/scientific-environment": "scientific_environment",
    }

    def test_every_previously_published_route_is_still_registered(self):
        routes = registered_routes()

        for path, name in self.PREVIOUS.items():
            with self.subTest(path=path):
                self.assertIn(path, routes)
                self.assertEqual(routes[path][1], name)

    # Application routes added by governed increments AFTER PTC-I1.
    #
    # PTC-I1's closed-world claim is about the surface AS OF PTC-I1: that this
    # increment added exactly one route. A later additive increment does not
    # weaken that claim, but it does make an unqualified count of the whole
    # application factually wrong. Subtracting the later additions keeps the
    # claim exactly as strong as it was while letting it stay true.
    #
    # Each entry is owned by the increment that added it, and its own suite
    # owns the closed world as of that increment.
    POST_PTC_I1_ROUTES = frozenset({"/solar-regime", "/night-start-after"})

    def test_this_increment_added_exactly_one_route(self):
        routes = registered_routes()

        # /openapi.json, /docs, /docs/oauth2-redirect and /redoc are generated
        # by FastAPI itself and belong to no increment. Every comparable
        # route-surface test in this repository filters them by the same
        # prefixes, and only those prefixes: an application route is never
        # excluded here.
        published = {
            path for path in routes
            if not path.startswith(("/openapi", "/docs", "/redoc"))
        } - self.POST_PTC_I1_ROUTES

        added = published - set(self.PREVIOUS)

        self.assertEqual(added, {ROTATION_PATH})


if __name__ == "__main__":
    unittest.main()
