"""PTC-I2 verification for the certified solar regime surface.

Covers determine_solar_regime in astronomy_solver.py, project_solar_regime in
astronomical_event_transport.py and GET /solar-regime in server.py: fidelity
to the certified event predicate, the published response shape, the ordinary
control, certified continuous daylight and darkness, marginal and
indeterminate policy, the decisive distinction between a bounded search that
found nothing and a certified absence, exact-pole behaviour, the equinox
distinction, interval and observer validation, the security surface, and
preservation of every previously published route.

Test framework: standard-library unittest.
Scientific execution uses the repository's certified Skyfield runtime.

INVOCATION
----------
Run from the repository root with the repository interpreter:

    .\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v

ROUTE SEAM

Route functions are invoked directly, following A1c, A1d, A3c-1..4, A4, A5 and
PTC-I1. httpx is not installed, so starlette's TestClient is unavailable, and
installing a dependency in order to test is not permitted.

INDEPENDENT ORACLE DISCIPLINE

Three claims are verified against something other than the code that makes
them.

The event threshold and the altitude margin are checked against
almanac.sunrise_sunset itself: the library predicate is called directly and
its boolean is compared with the sign of the published margin, sample by
sample. A margin that did not reproduce the certified predicate would fail
here even though both come from the same ephemeris.

The resolution guard is recomputed from test-local literals rather than
imported, so a changed constant is a failure rather than a silent
redefinition.

The altitude rate bound the guard rests on is measured, not assumed: the
margin's own finite differences are taken on a grid far finer than the
published one and asserted to stay below the published bound. That assertion
is the standing verification of the one numerical claim this increment adds.

The field set and the regime vocabulary are declared as test-local literals,
so a renamed field or regime is a failure rather than a silent rename.

NO SCRATCHPAD NUMBERS ARE TRUSTED

Reference observers and intervals are named by their geography and season and
encoded through the certified transport at test time. Every astronomical
quantity asserted is recomputed through the certified runtime; no decimal
carried over from an investigation report is asserted as authority.
"""

import inspect
import math
import unittest

import numpy
from fastapi import HTTPException
from skyfield import almanac
from skyfield.api import wgs84

import astronomy_solver
import server
from exact_time_transport import encode_tt_bits

REGIME_PATH = "/solar-regime"

EXPECTED_KEYS = (
    "regime",
    "crossingPresent",
    "sunsets",
    "sunrises",
    "crossingEnumerationAgrees",
    "requested",
    "covered",
    "complete",
    "truncationReason",
    "eventThresholdDegrees",
    "eventConvention",
    "minimumAltitudeMarginDegrees",
    "maximumAltitudeMarginDegrees",
    "resolutionGuardDegrees",
    "sampleStepDays",
    "ephemerisRole",
)

ORDINARY = "ORDINARY"
CONTINUOUS_DAYLIGHT = "CONTINUOUS_DAYLIGHT"
CONTINUOUS_DARKNESS = "CONTINUOUS_DARKNESS"
MARGINAL = "MARGINAL"
INDETERMINATE = "INDETERMINATE"

EVERY_REGIME = (ORDINARY, CONTINUOUS_DAYLIGHT, CONTINUOUS_DARKNESS,
                MARGINAL, INDETERMINATE)

# --- The test's own copies of the published constants ---------------------
#
# Declared here, never imported. The point of the assertions below is that the
# module's values equal these, so importing them would make the check vacuous.
THRESHOLD_DEGREES = -0.8333
SAMPLE_STEP_DAYS = 0.01
MAX_SPAN_DAYS = 400.0
RATE_ROTATION_DEG_PER_DAY = 370.0
RATE_DRIFT_DEG_PER_DAY = 1.0

EPHEMERIS_ROLES = ("ancient", "future", "primary")

# --- Reference observers -------------------------------------------------
#
# Named by geography. The mid-latitude control has ordinary crossings every
# day; the transition observers sit in the band where the Sun grazes the
# threshold; the two poles are the geometric limiting cases.
NEW_YORK = (40.7406, -73.9865)
TRANSITION_CROSSING = (65.5, 0.0)
TRANSITION_GRAZING = (66.0, 0.0)
TRANSITION_CLEAR = (67.0, 0.0)
LONGYEARBYEN = (78.2232, 15.6267)
ROSS_REGION = (-77.8463, 166.6683)
NORTH_POLE = (90.0, 0.0)
SOUTH_POLE = (-90.0, 0.0)


def bits_for(*utc_parts):
    """The published transport encoding of a named UTC instant's TT state."""
    return encode_tt_bits(float(astronomy_solver.ts.utc(*utc_parts).tt))


def tt_of(*utc_parts):
    return float(astronomy_solver.ts.utc(*utc_parts).tt)


def regime_of(start, end, observer):
    latitude, longitude = observer
    return server.solar_regime(
        bits_for(*start), bits_for(*end), latitude, longitude
    )


def guard_oracle(latitude, step_days):
    """The resolution guard, recomputed from the test's own literals."""
    rate = (
        RATE_ROTATION_DEG_PER_DAY * abs(math.cos(math.radians(latitude)))
        + RATE_DRIFT_DEG_PER_DAY
    )
    return rate * step_days / 2.0


def certified_predicate(observer):
    """almanac.sunrise_sunset itself, built the way the Authority builds it."""
    latitude, longitude = observer
    return almanac.sunrise_sunset(
        astronomy_solver.load_kernel(astronomy_solver.PRIMARY_KERNEL),
        wgs84.latlon(latitude, longitude),
    )


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


# --- 1. Fidelity to the certified event predicate -------------------------


class TestPredicateFidelity(unittest.TestCase):
    """The margin must BE the certified predicate, not resemble it."""

    OBSERVERS = (NEW_YORK, TRANSITION_CROSSING, TRANSITION_GRAZING,
                 LONGYEARBYEN, ROSS_REGION, NORTH_POLE, SOUTH_POLE)

    def test_the_published_threshold_is_the_predicates_own_threshold(self):
        # Not -0.83333...: the library compares against -0.8333 exactly, and
        # a module carrying the rounder-looking constant would disagree with
        # the certified predicate near the horizon.
        self.assertEqual(
            astronomy_solver.EVENT_THRESHOLD_DEGREES, THRESHOLD_DEGREES
        )

    def test_the_margin_sign_reproduces_the_predicate_sample_by_sample(self):
        lo = tt_of(2026, 1, 1)
        hi = tt_of(2027, 1, 1)
        grid = numpy.linspace(lo, hi, 20001)

        for observer in self.OBSERVERS:
            with self.subTest(observer=observer):
                latitude, longitude = observer

                margin = astronomy_solver._altitude_margin_degrees(
                    astronomy_solver.PRIMARY_KERNEL, grid, latitude, longitude
                )
                library = numpy.asarray(
                    certified_predicate(observer)(
                        astronomy_solver.ts.tt_jd(grid)
                    )
                )

                self.assertEqual(
                    int(numpy.count_nonzero(library != (margin >= 0.0))), 0
                )

    def test_the_declared_resolution_and_admission_bound_are_unchanged(self):
        self.assertEqual(
            astronomy_solver.REGIME_SAMPLE_STEP_DAYS, SAMPLE_STEP_DAYS
        )
        self.assertEqual(
            astronomy_solver.REGIME_MAX_SPAN_DAYS, MAX_SPAN_DAYS
        )

    def test_the_regime_vocabulary_is_exactly_the_published_five(self):
        published = {
            astronomy_solver.REGIME_ORDINARY,
            astronomy_solver.REGIME_CONTINUOUS_DAYLIGHT,
            astronomy_solver.REGIME_CONTINUOUS_DARKNESS,
            astronomy_solver.REGIME_MARGINAL,
            astronomy_solver.REGIME_INDETERMINATE,
        }

        self.assertEqual(published, set(EVERY_REGIME))


# --- 2. The altitude rate bound the guard rests on ------------------------


class TestAltitudeRateBound(unittest.TestCase):
    """The one numerical claim this increment adds, measured rather than
    assumed."""

    def test_the_bound_is_never_exceeded_by_the_measured_rate(self):
        # A grid twenty times finer than the published one, so the measured
        # finite difference approaches the true supremum from below.
        fine_step = SAMPLE_STEP_DAYS / 20.0

        for observer in (NEW_YORK, TRANSITION_CROSSING, LONGYEARBYEN,
                         ROSS_REGION, NORTH_POLE, SOUTH_POLE):
            with self.subTest(observer=observer):
                latitude, longitude = observer

                grid = numpy.arange(tt_of(2026, 1, 1), tt_of(2027, 1, 1),
                                    fine_step)
                margin = astronomy_solver._altitude_margin_degrees(
                    astronomy_solver.PRIMARY_KERNEL, grid, latitude, longitude
                )
                measured = float(
                    numpy.abs(numpy.diff(margin)).max() / fine_step
                )
                published = (
                    astronomy_solver.altitude_rate_bound_degrees_per_day(
                        latitude
                    )
                )

                self.assertLess(measured, published)

    def test_the_bound_collapses_to_the_drift_term_at_the_poles(self):
        # At the pole the Sun's altitude is its declination, so only the
        # non-rotational term survives. This is what makes a polar regime
        # certifiable to five thousandths of a degree.
        for latitude in (90.0, -90.0):
            with self.subTest(latitude=latitude):
                bound = (
                    astronomy_solver.altitude_rate_bound_degrees_per_day(
                        latitude
                    )
                )

                self.assertAlmostEqual(bound, RATE_DRIFT_DEG_PER_DAY, places=9)

    def test_the_bound_matches_the_independently_recomputed_formula(self):
        for latitude in (0.0, 40.7406, 65.5, 78.2232, 90.0, -90.0):
            with self.subTest(latitude=latitude):
                expected = (
                    RATE_ROTATION_DEG_PER_DAY
                    * abs(math.cos(math.radians(latitude)))
                    + RATE_DRIFT_DEG_PER_DAY
                )

                self.assertAlmostEqual(
                    astronomy_solver.altitude_rate_bound_degrees_per_day(
                        latitude
                    ),
                    expected,
                    places=12,
                )


# --- 3. The published response --------------------------------------------


class TestPublishedResponse(unittest.TestCase):
    def setUp(self):
        self.body = regime_of((2026, 6, 1), (2026, 6, 8), NEW_YORK)

    def test_the_route_is_registered_with_an_interval_and_an_observer(self):
        routes = registered_routes()

        self.assertIn(REGIME_PATH, routes)

        methods, name, parameters = routes[REGIME_PATH]

        self.assertIn("GET", methods)
        self.assertEqual(name, "solar_regime")
        self.assertEqual(
            parameters, ["ttLoBits", "ttHiBits", "latitude", "longitude"]
        )

    def test_the_shape_is_exactly_the_published_contract(self):
        self.assertEqual(tuple(self.body), EXPECTED_KEYS)
        self.assertEqual(tuple(self.body["requested"]), ("lo", "hi"))
        self.assertEqual(tuple(self.body["covered"]), ("lo", "hi"))
        for side in ("requested", "covered"):
            for bound in ("lo", "hi"):
                self.assertEqual(
                    tuple(self.body[side][bound]), ("ttBits", "tt", "utc")
                )

    def test_the_regime_is_always_one_of_the_published_five(self):
        self.assertIn(self.body["regime"], EVERY_REGIME)

    def test_the_declared_interval_is_carried_back_unaltered(self):
        self.assertEqual(
            self.body["requested"]["lo"]["ttBits"], bits_for(2026, 6, 1)
        )
        self.assertEqual(
            self.body["requested"]["hi"]["ttBits"], bits_for(2026, 6, 8)
        )

    def test_the_observer_is_not_transported(self):
        # The repository's rule, asserted rather than assumed: no projection in
        # the event transport carries an observer. The route still accepts one;
        # it simply does not travel back, so there is no second binding that
        # could drift from the one the geometry ran on.
        self.assertNotIn("observer", self.body)

        for absent in ("latitude", "longitude"):
            self.assertNotIn(absent, repr(self.body))

    def test_the_event_definition_travels_with_the_answer(self):
        self.assertEqual(self.body["eventThresholdDegrees"], THRESHOLD_DEGREES)

        convention = self.body["eventConvention"]

        # A consumer must be able to tell WHICH event without reading this
        # repository: whose centre, at what altitude, under what corrections.
        self.assertIn("-0.8333", convention)
        self.assertIn("topocentric", convention)
        self.assertIn("refraction", convention)
        self.assertIn("semidiameter", convention)

    def test_the_resolution_is_published_with_every_classification(self):
        self.assertLessEqual(self.body["sampleStepDays"], SAMPLE_STEP_DAYS)
        self.assertGreater(self.body["sampleStepDays"], 0.0)
        self.assertAlmostEqual(
            self.body["resolutionGuardDegrees"],
            guard_oracle(NEW_YORK[0], self.body["sampleStepDays"]),
            places=12,
        )

    def test_the_margin_extremes_bracket_each_other(self):
        self.assertLessEqual(
            self.body["minimumAltitudeMarginDegrees"],
            self.body["maximumAltitudeMarginDegrees"],
        )

    def test_provenance_is_a_role_and_never_a_filename(self):
        self.assertIn(self.body["ephemerisRole"], EPHEMERIS_ROLES)

    def test_the_response_is_deterministic(self):
        again = regime_of((2026, 6, 1), (2026, 6, 8), NEW_YORK)

        self.assertEqual(self.body, again)


# --- 4. Ordinary control --------------------------------------------------


class TestOrdinaryControl(unittest.TestCase):
    def test_a_mid_latitude_week_has_genuine_crossings(self):
        body = regime_of((2026, 6, 1), (2026, 6, 8), NEW_YORK)

        self.assertEqual(body["regime"], ORDINARY)
        self.assertIs(body["crossingPresent"], True)
        self.assertEqual(body["sunsets"], 7)
        self.assertEqual(body["sunrises"], 7)
        self.assertTrue(body["crossingEnumerationAgrees"])

    def test_a_mid_latitude_week_is_never_called_continuous(self):
        body = regime_of((2026, 6, 1), (2026, 6, 8), NEW_YORK)

        self.assertNotIn(
            body["regime"], (CONTINUOUS_DAYLIGHT, CONTINUOUS_DARKNESS)
        )

    def test_the_margin_straddles_the_threshold_where_crossings_exist(self):
        body = regime_of((2026, 6, 1), (2026, 6, 8), NEW_YORK)

        self.assertLess(body["minimumAltitudeMarginDegrees"], 0.0)
        self.assertGreater(body["maximumAltitudeMarginDegrees"], 0.0)


# --- 5. Certified continuous daylight -------------------------------------


class TestContinuousDaylight(unittest.TestCase):
    CASES = (
        ("Longyearbyen", LONGYEARBYEN, (2026, 5, 1), (2026, 8, 1)),
        ("North Pole", NORTH_POLE, (2026, 4, 15), (2026, 9, 1)),
        ("South Pole", SOUTH_POLE, (2026, 11, 1), (2027, 1, 15)),
        ("67 N midsummer", TRANSITION_CLEAR, (2026, 6, 18), (2026, 6, 25)),
    )

    def test_continuous_daylight_is_certified_not_inferred(self):
        for label, observer, start, end in self.CASES:
            with self.subTest(case=label):
                body = regime_of(start, end, observer)

                self.assertEqual(body["regime"], CONTINUOUS_DAYLIGHT)
                self.assertIs(body["crossingPresent"], False)
                self.assertTrue(body["complete"])
                self.assertIsNone(body["truncationReason"])

    def test_no_sunset_is_fabricated_in_continuous_daylight(self):
        for label, observer, start, end in self.CASES:
            with self.subTest(case=label):
                body = regime_of(start, end, observer)

                self.assertEqual(body["sunsets"], 0)
                self.assertEqual(body["sunrises"], 0)

    def test_the_margin_clears_the_guard_by_the_published_amounts(self):
        for label, observer, start, end in self.CASES:
            with self.subTest(case=label):
                body = regime_of(start, end, observer)

                # This inequality IS the certification. It is asserted rather
                # than assumed so a weakened guard cannot pass unnoticed.
                self.assertGreater(
                    body["minimumAltitudeMarginDegrees"],
                    body["resolutionGuardDegrees"],
                )


# --- 6. Certified continuous darkness -------------------------------------


class TestContinuousDarkness(unittest.TestCase):
    CASES = (
        ("Longyearbyen", LONGYEARBYEN, (2026, 11, 20), (2027, 1, 20)),
        ("Ross region", ROSS_REGION, (2026, 5, 15), (2026, 7, 15)),
        ("North Pole", NORTH_POLE, (2026, 10, 15), (2027, 2, 15)),
        ("South Pole", SOUTH_POLE, (2026, 4, 15), (2026, 9, 1)),
    )

    def test_continuous_darkness_is_certified_not_inferred(self):
        for label, observer, start, end in self.CASES:
            with self.subTest(case=label):
                body = regime_of(start, end, observer)

                self.assertEqual(body["regime"], CONTINUOUS_DARKNESS)
                self.assertIs(body["crossingPresent"], False)
                self.assertTrue(body["complete"])
                self.assertIsNone(body["truncationReason"])

    def test_no_sunrise_is_fabricated_in_continuous_darkness(self):
        for label, observer, start, end in self.CASES:
            with self.subTest(case=label):
                body = regime_of(start, end, observer)

                self.assertEqual(body["sunrises"], 0)
                self.assertEqual(body["sunsets"], 0)

    def test_the_margin_clears_the_guard_on_the_dark_side(self):
        for label, observer, start, end in self.CASES:
            with self.subTest(case=label):
                body = regime_of(start, end, observer)

                self.assertLess(
                    body["maximumAltitudeMarginDegrees"],
                    -body["resolutionGuardDegrees"],
                )


# --- 7. Marginal policy ---------------------------------------------------


class TestMarginalPolicy(unittest.TestCase):
    """Where the science does not support a stronger claim, none is made."""

    def test_a_grazing_geometry_is_reported_marginal(self):
        body = regime_of((2026, 6, 18), (2026, 6, 25), TRANSITION_GRAZING)

        self.assertEqual(body["regime"], MARGINAL)

        # DECISIVE: not False. The Authority does not know whether a crossing
        # occurs, and "unknown" must not be readable as "none".
        self.assertIsNone(body["crossingPresent"])

    def test_a_grazing_geometry_is_never_collapsed_into_continuity(self):
        body = regime_of((2026, 6, 18), (2026, 6, 25), TRANSITION_GRAZING)

        self.assertNotIn(
            body["regime"], (CONTINUOUS_DAYLIGHT, CONTINUOUS_DARKNESS)
        )

    def test_marginal_is_exactly_the_case_the_guard_does_not_clear(self):
        body = regime_of((2026, 6, 18), (2026, 6, 25), TRANSITION_GRAZING)

        # The Sun's lowest point really is above the threshold here, and the
        # margin really is inside the guard. Both halves are asserted so the
        # classification cannot be right by accident.
        self.assertGreater(body["minimumAltitudeMarginDegrees"], 0.0)
        self.assertLess(
            body["minimumAltitudeMarginDegrees"],
            body["resolutionGuardDegrees"],
        )

    def test_the_neighbouring_latitude_still_resolves_ordinarily(self):
        # Half a degree south the Sun genuinely sets, and the contract says so
        # plainly. The band is narrow, which is why a latitude rule would be
        # wrong rather than merely crude.
        body = regime_of((2026, 6, 18), (2026, 6, 25), TRANSITION_CROSSING)

        self.assertEqual(body["regime"], ORDINARY)
        self.assertIs(body["crossingPresent"], True)


# --- 8. No latitude rule --------------------------------------------------


class TestClassificationDerivesFromAstronomy(unittest.TestCase):
    def test_one_observer_yields_three_different_regimes(self):
        # DECISIVE against any `if latitude >= X: polar` rule. The same exact
        # observer is continuous daylight, continuous darkness and ordinary,
        # depending only on the interval declared.
        daylight = regime_of((2026, 4, 15), (2026, 9, 1), NORTH_POLE)
        darkness = regime_of((2026, 10, 15), (2027, 2, 15), NORTH_POLE)
        ordinary = regime_of((2026, 2, 15), (2026, 4, 15), NORTH_POLE)

        self.assertEqual(daylight["regime"], CONTINUOUS_DAYLIGHT)
        self.assertEqual(darkness["regime"], CONTINUOUS_DARKNESS)
        self.assertEqual(ordinary["regime"], ORDINARY)

    def test_a_mid_latitude_observer_is_never_continuous_in_any_season(self):
        for start, end in (((2026, 1, 1), (2026, 1, 8)),
                           ((2026, 6, 18), (2026, 6, 25)),
                           ((2026, 12, 18), (2026, 12, 25))):
            with self.subTest(interval=(start, end)):
                body = regime_of(start, end, NEW_YORK)

                self.assertEqual(body["regime"], ORDINARY)


# --- 9. Bounded search is not absence ------------------------------------


class TestBoundedSearchIsNotAbsence(unittest.TestCase):
    """The decisive regression: a horizon that found nothing is not an empty
    sky."""

    ANCHOR = (2026, 12, 15)

    def test_the_bounded_successor_route_finds_no_event_at_the_anchor(self):
        # Establish the premise rather than assuming it. Both polar observers
        # are in polar night, and the published directional route reaches only
        # its own certified horizon.
        for observer in (LONGYEARBYEN, NORTH_POLE):
            with self.subTest(observer=observer):
                latitude, longitude = observer

                with self.assertRaises(HTTPException) as caught:
                    server.sunset_event_after(
                        ttBits=bits_for(*self.ANCHOR),
                        latitude=latitude,
                        longitude=longitude,
                    )

                self.assertEqual(caught.exception.status_code, 404)

    def test_a_wide_certified_interval_proves_the_crossing_exists(self):
        # DECISIVE. The same anchor, the same observer, a declared interval
        # wide enough to contain the genuine event: the Authority reports a
        # crossing present rather than repeating the bounded route's silence.
        anchor = tt_of(*self.ANCHOR)

        for observer in (LONGYEARBYEN, NORTH_POLE):
            with self.subTest(observer=observer):
                latitude, longitude = observer

                body = server.solar_regime(
                    encode_tt_bits(anchor),
                    encode_tt_bits(anchor + 200.0),
                    latitude,
                    longitude,
                )

                self.assertEqual(body["regime"], ORDINARY)
                self.assertIs(body["crossingPresent"], True)
                self.assertTrue(body["complete"])

    def test_a_narrow_interval_claims_only_its_own_territory(self):
        # A three-day interval in polar night is genuinely continuous
        # darkness, and that is a true statement about three days. What makes
        # it safe is that the interval it applies to is published with it, so
        # it can never be read as a claim about the season.
        anchor = tt_of(*self.ANCHOR)

        body = server.solar_regime(
            encode_tt_bits(anchor),
            encode_tt_bits(anchor + astronomy_solver.SUCCESSOR_SEARCH_SPAN_DAYS),
            LONGYEARBYEN[0],
            LONGYEARBYEN[1],
        )

        self.assertEqual(body["regime"], CONTINUOUS_DARKNESS)
        self.assertEqual(
            body["requested"]["hi"]["ttBits"],
            encode_tt_bits(
                anchor + astronomy_solver.SUCCESSOR_SEARCH_SPAN_DAYS
            ),
        )
        self.assertEqual(
            body["covered"]["hi"]["ttBits"], body["requested"]["hi"]["ttBits"]
        )

    def test_absence_over_an_incomplete_frontier_is_indeterminate(self):
        # The distinction stated directly at the solver: when the examined
        # territory is not the whole declared interval and nothing was found,
        # the answer is that nothing is established - never that nothing
        # happens.
        source = inspect.getsource(astronomy_solver.determine_solar_regime)

        self.assertIn("REGIME_INDETERMINATE", source)
        self.assertIn("frontier.complete", source)


# --- 10. Exact poles -----------------------------------------------------


class TestExactPoles(unittest.TestCase):
    def test_both_exact_poles_are_accepted_and_classified(self):
        for latitude in (90.0, -90.0):
            with self.subTest(latitude=latitude):
                body = server.solar_regime(
                    bits_for(2026, 6, 18), bits_for(2026, 6, 25),
                    latitude, 0.0
                )

                self.assertIn(body["regime"], EVERY_REGIME)

    def test_longitude_changes_no_classification_at_the_exact_poles(self):
        # Investigated rather than assumed, and asserted rather than special
        # cased: the existing geodetic contract accepts a longitude at the
        # pole, and the published answer is unmoved by it. No meridian is
        # chosen for the pole anywhere in this increment.
        for latitude in (90.0, -90.0):
            baseline = server.solar_regime(
                bits_for(2026, 3, 1), bits_for(2026, 4, 1), latitude, 0.0
            )
            for longitude in (-180.0, -73.9865, 35.2137, 179.9):
                with self.subTest(latitude=latitude, longitude=longitude):
                    other = server.solar_regime(
                        bits_for(2026, 3, 1), bits_for(2026, 4, 1),
                        latitude, longitude
                    )

                    self.assertEqual(other["regime"], baseline["regime"])
                    self.assertIs(
                        other["crossingPresent"], baseline["crossingPresent"]
                    )
                    self.assertEqual(other["sunsets"], baseline["sunsets"])
                    self.assertEqual(other["sunrises"], baseline["sunrises"])
                    self.assertAlmostEqual(
                        other["minimumAltitudeMarginDegrees"],
                        baseline["minimumAltitudeMarginDegrees"],
                        places=9,
                    )

    def test_the_polar_guard_is_the_drift_term_alone(self):
        body = server.solar_regime(
            bits_for(2026, 4, 15), bits_for(2026, 9, 1), 90.0, 0.0
        )

        self.assertAlmostEqual(
            body["resolutionGuardDegrees"],
            RATE_DRIFT_DEG_PER_DAY * body["sampleStepDays"] / 2.0,
            places=9,
        )


# --- 11. The equinox is not a sunrise ------------------------------------


class TestEquinoxDistinction(unittest.TestCase):
    def test_the_apparent_polar_event_is_not_the_equinox(self):
        # Recomputed through the certified runtime, both quantities. The
        # apparent event lags or leads the equinox by more than two days at
        # the poles, because solar semidiameter and refraction move it and the
        # equinox is a geocentric longitude crossing that knows no observer.
        for kind, observer in (
            (astronomy_solver.SOLAR_LONGITUDE_000, NORTH_POLE),
            (astronomy_solver.SOLAR_LONGITUDE_180, NORTH_POLE),
            (astronomy_solver.SOLAR_LONGITUDE_000, SOUTH_POLE),
            (astronomy_solver.SOLAR_LONGITUDE_180, SOUTH_POLE),
        ):
            with self.subTest(kind=kind, observer=observer):
                equinox = astronomy_solver.find_solar_longitude_event_in_year(
                    2026, kind
                ).tt
                latitude, longitude = observer

                predicate = certified_predicate(observer)
                times, _events = almanac.find_discrete(
                    astronomy_solver.ts.tt_jd(equinox - 20.0),
                    astronomy_solver.ts.tt_jd(equinox + 20.0),
                    predicate,
                )

                self.assertEqual(len(times), 1)
                self.assertGreater(abs(float(times[0].tt) - equinox), 1.0)

    def test_the_equinox_instant_is_inside_an_ordinary_polar_regime(self):
        # The regime around the equinox contains a genuine crossing. The
        # equinox is a checkpoint in that interval, not the crossing itself,
        # and this contract publishes no equinox at all.
        equinox = astronomy_solver.find_solar_longitude_event_in_year(
            2026, astronomy_solver.SOLAR_LONGITUDE_000
        ).tt

        body = server.solar_regime(
            encode_tt_bits(equinox - 20.0),
            encode_tt_bits(equinox + 20.0),
            NORTH_POLE[0],
            NORTH_POLE[1],
        )

        self.assertEqual(body["regime"], ORDINARY)
        self.assertEqual(body["sunrises"], 1)
        self.assertEqual(body["sunsets"], 0)

        # No equinox, no season and no solar longitude crosses this wire.
        serialized = repr(body).lower()
        for absent in ("equinox", "season", "solstice", "longitude_000"):
            self.assertNotIn(absent, serialized)


# --- 12. Interval and observer validation --------------------------------


class TestValidation(unittest.TestCase):
    def assert_governed_400(self, *arguments, **keywords):
        with self.assertRaises(HTTPException) as caught:
            server.solar_regime(*arguments, **keywords)

        self.assertEqual(caught.exception.status_code, 400)

        detail = caught.exception.detail

        self.assertIsInstance(detail, dict)
        self.assertIn("reason", detail)
        self.assertIn("message", detail)

        return detail

    def test_a_malformed_bound_is_refused(self):
        good = bits_for(2026, 6, 1)

        for lo, hi in (("not-hex", good), (good, "not-hex"),
                       ("", good), ("4142A9B0", good),
                       ("4142a9b000000000", good),
                       ("7FF0000000000000", good),
                       (good, "7FF8000000000000")):
            with self.subTest(lo=lo, hi=hi):
                detail = self.assert_governed_400(lo, hi, 40.0, 0.0)

                self.assertEqual(detail["reason"], "TT_BITS_INVALID")

    def test_a_reversed_interval_is_refused(self):
        detail = self.assert_governed_400(
            bits_for(2026, 6, 8), bits_for(2026, 6, 1), 40.0, 0.0
        )

        self.assertEqual(detail["reason"], "INSTANT_STATE_INVALID")

    def test_a_zero_width_interval_is_refused(self):
        same = bits_for(2026, 6, 1)
        detail = self.assert_governed_400(same, same, 40.0, 0.0)

        self.assertEqual(detail["reason"], "INSTANT_STATE_INVALID")

    def test_an_over_long_interval_is_refused_before_any_astronomy(self):
        anchor = tt_of(2026, 1, 1)
        detail = self.assert_governed_400(
            encode_tt_bits(anchor),
            encode_tt_bits(anchor + MAX_SPAN_DAYS + 1.0),
            40.0,
            0.0,
        )

        self.assertEqual(detail["reason"], "REGIME_INTERVAL_TOO_LONG")

    def test_the_admission_bound_itself_is_admitted(self):
        anchor = tt_of(2026, 1, 1)

        body = server.solar_regime(
            encode_tt_bits(anchor),
            encode_tt_bits(anchor + MAX_SPAN_DAYS),
            NORTH_POLE[0],
            NORTH_POLE[1],
        )

        self.assertIn(body["regime"], EVERY_REGIME)

    def test_an_observer_outside_the_geodetic_domain_is_refused(self):
        lo = bits_for(2026, 6, 1)
        hi = bits_for(2026, 6, 8)

        for latitude, longitude in ((90.5, 0.0), (-90.5, 0.0),
                                    (40.0, 180.5), (40.0, -180.5),
                                    (float("nan"), 0.0),
                                    (40.0, float("inf"))):
            with self.subTest(latitude=latitude, longitude=longitude):
                detail = self.assert_governed_400(lo, hi, latitude, longitude)

                self.assertEqual(detail["reason"], "OBSERVER_OUT_OF_DOMAIN")

    def test_the_domain_boundaries_themselves_are_admitted(self):
        lo = bits_for(2026, 6, 1)
        hi = bits_for(2026, 6, 8)

        for latitude, longitude in ((90.0, 180.0), (-90.0, -180.0)):
            with self.subTest(latitude=latitude, longitude=longitude):
                body = server.solar_regime(lo, hi, latitude, longitude)

                self.assertIn(body["regime"], EVERY_REGIME)


# --- 13. Security surface -------------------------------------------------


class TestSecuritySurface(unittest.TestCase):
    LEAKS = ("C:\\", "/home", "/var", ".bsp", ".npz", "de440", "de441",
             "ephemeris/", "127.0.0.1", "localhost", "http://", "https://",
             "Traceback", "site-packages", "File \"")

    def test_no_response_field_discloses_an_artifact_or_a_path(self):
        for observer, start, end in (
            (NEW_YORK, (2026, 6, 1), (2026, 6, 8)),
            (NORTH_POLE, (2026, 4, 15), (2026, 9, 1)),
            (SOUTH_POLE, (2026, 4, 15), (2026, 9, 1)),
            (TRANSITION_GRAZING, (2026, 6, 18), (2026, 6, 25)),
        ):
            with self.subTest(observer=observer):
                serialized = repr(regime_of(start, end, observer))

                for leak in self.LEAKS:
                    self.assertNotIn(leak, serialized)

    def test_the_published_role_is_not_a_filename(self):
        body = regime_of((2026, 6, 1), (2026, 6, 8), NEW_YORK)
        role = body["ephemerisRole"]

        self.assertIn(role, EPHEMERIS_ROLES)
        self.assertNotIn(".", role)
        self.assertNotEqual(role, astronomy_solver.PRIMARY_KERNEL)

    def test_a_refusal_discloses_nothing_beyond_the_governed_envelope(self):
        with self.assertRaises(HTTPException) as caught:
            server.solar_regime("not-hex", "not-hex", 40.0, 0.0)

        serialized = repr(caught.exception.detail)

        for leak in self.LEAKS:
            self.assertNotIn(leak, serialized)


# --- 14. No calendar concepts --------------------------------------------


class TestNoCalendarConcepts(unittest.TestCase):
    FORBIDDEN = ("hpc", "telma", "abib", "sabbath", "creation", "weekday",
                 "continuity", "ptc", "polar", "month", "ordinal")

    def test_no_published_field_is_a_calendar_quantity(self):
        for observer, start, end in ((NEW_YORK, (2026, 6, 1), (2026, 6, 8)),
                                     (NORTH_POLE, (2026, 4, 15), (2026, 9, 1))):
            body = regime_of(start, end, observer)

            def walk(node, path=""):
                if isinstance(node, dict):
                    for key, value in node.items():
                        for token in self.FORBIDDEN:
                            self.assertNotIn(token, key.lower())
                        walk(value, path + "/" + key)

            walk(body)

    def test_no_published_value_names_a_calendar_concept(self):
        body = regime_of((2026, 4, 15), (2026, 9, 1), NORTH_POLE)
        serialized = repr(body).lower()

        for token in ("telma", "sabbath", "creation", "weekday", "hpc"):
            self.assertNotIn(token, serialized)


# --- 15. Previously published routes preserved ---------------------------


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
        "/earth-rotation": "earth_rotation",
    }

    # Application routes added by governed increments AFTER PTC-I2.
    #
    # PTC-I2's closed-world claim is about the surface AS OF PTC-I2: that this
    # increment added exactly one route. A later additive increment does not
    # weaken that claim, but it does make an unqualified count of the whole
    # application factually wrong. Subtracting the later additions keeps the
    # claim exactly as strong as it was while letting it stay true.
    #
    # Each entry is owned by the increment that added it, and its own suite
    # owns the closed world as of that increment.
    POST_PTC_I2_ROUTES = frozenset({"/night-start-after", "/solar-crossing"})

    def test_every_previously_published_route_is_still_registered(self):
        routes = registered_routes()

        for path, name in self.PREVIOUS.items():
            with self.subTest(path=path):
                self.assertIn(path, routes)
                self.assertEqual(routes[path][1], name)

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
        }

        self.assertEqual(
            published - set(self.PREVIOUS) - self.POST_PTC_I2_ROUTES,
            {REGIME_PATH},
        )

    def test_an_existing_solar_event_route_still_answers_unchanged(self):
        body = server.sunset_event_after(
            ttBits=bits_for(2026, 6, 1),
            latitude=NEW_YORK[0],
            longitude=NEW_YORK[1],
        )

        self.assertEqual(tuple(body), ("ttBits", "tt", "utc", "kernel"))

    def test_the_existing_count_route_still_answers_unchanged(self):
        body = server.sunset_count(
            ttLoBits=bits_for(2026, 6, 1),
            ttHiBits=bits_for(2026, 6, 8),
            latitude=NEW_YORK[0],
            longitude=NEW_YORK[1],
        )

        self.assertEqual(body["sunsetCount"], 7)


if __name__ == "__main__":
    unittest.main()
