"""DT-R4-A verification for the certified night-start primitive.

Covers find_night_start_after in astronomy_solver.py and the certified
skeleton beneath it: the governed constants, the R4 certified latitude domain
and its edges, the ordinary sunset night-start and its bit identity with the
DT-A1 directional sunset, the vanished-night minimum at the DT-O4 polar
fixtures, the absence of a distant-season re-anchor, the R4 seam, tangency,
polar night, Delta-T knots and the failure taxonomy.

Test framework: standard-library unittest.
Scientific execution uses the repository's certified Skyfield runtime and the
real pinned Authority ephemeris artifacts.

INVOCATION
----------
Run from the repository root with the repository interpreter:

    .\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v

INDEPENDENT ORACLE DISCIPLINE
-----------------------------

Reason codes, night-start kinds, the certified latitude limit and the
fixtures are declared here as test-local literals. They are deliberately NOT
imported from astronomy_solver: importing a constant to check that same
constant would make the test agree with a defective value instead of
detecting it.

PROVENANCE OF THE FIXTURES
--------------------------

The polar and control fixtures are DT-O4 R4 anchor evaluations (hours after
the 2019 March equinox to the night-start), recorded under Skyfield 1.53 and
reconfirmed one observer at a time with DT-O4's own reference implementation
on the governed Skyfield 1.54 runtime (DT-R4-A1): 8/8 identical in kind,
rotation offset and hours to four decimals.
"""

import math
import unittest

import numpy

import astronomy_solver
from astronomy_solver import (
    SunsetChronologyError,
    SunsetEvent,
    find_night_start_after,
    find_solar_longitude_event_in_year,
    find_sunset_from_instant,
)

# --- Governed literals -------------------------------------------------------

LATITUDE_LIMIT = 89.739
SEARCH_SPAN_DAYS = 3.0

GENUINE_SUNSET = "GENUINE_SUNSET"
VANISHED_NIGHT_MINIMUM = "VANISHED_NIGHT_MINIMUM"
TANGENCY_UNRESOLVED = "TANGENCY_UNRESOLVED"

LATITUDE_UNCERTIFIED = "NIGHT_START_LATITUDE_UNCERTIFIED"
ORDER_AMBIGUOUS = "NIGHT_START_ORDER_AMBIGUOUS"
POLAR_NIGHT = "NIGHT_START_POLAR_NIGHT"
UNRESOLVED = "NIGHT_START_UNRESOLVED"
DETECTION_AMBIGUOUS = "SUNSET_DETECTION_AMBIGUOUS"
DETECTION_AMBIGUOUS_DT_KNOT = "SUNSET_DETECTION_AMBIGUOUS_DT_KNOT"
DETECTION_INCONSISTENT = "SUNSET_DETECTION_INCONSISTENT"
OBSERVER_OUT_OF_DOMAIN = "OBSERVER_OUT_OF_DOMAIN"
INSTANT_STATE_INVALID = "INSTANT_STATE_INVALID"

# The certified root finder's convergence width (Skyfield EPSILON).
FIND_DISCRETE_EPSILON_DAYS = 0.001 / 86400.0

# DT-O4 fixtures, reconfirmed on Skyfield 1.54: (latitude, longitude, kind,
# hours from the 2019 March equinox to the night-start, rounded to 1e-4 h).
ORDINARY_FIXTURES = (
    ("New York", 40.7406, -73.9586, 1.1548),
    ("Jerusalem", 31.7683, 35.2137, 17.8762),
)
VANISHED_FIXTURES = (
    ("89.5N 0E", 89.5, 0.0, 1.6684),
    ("89.25S 0E", -89.25, 0.0, 2.4702),
    ("89.5S 0E", -89.5, 0.0, 2.6311),
    ("89.5S 56.9W", -89.5, -56.9, 6.4236),
    ("89.5S 90W", -89.5, -90.0, 8.6298),
    ("89.5S 180", -89.5, 180.0, 14.6284),
)

# Rounding of the DT-O4 hours (5e-5 h) plus slack, in days.
FIXTURE_TOLERANCE_DAYS = 0.2 / 86400.0

# DT-A1 tangency evidence: at this latitude, longitude 0, the deepest 2019
# northern-midsummer night inside the June solstice +- 2 day window just
# vanishes.
TANGENT_LATITUDE = 65.74293204295881

SEAM_LATITUDE = 89.5


def equinox_2019():
    return find_solar_longitude_event_in_year(2019, "SOLAR_LONGITUDE_000").tt


def reason_of(callable_, *args):
    try:
        callable_(*args)
    except SunsetChronologyError as error:
        return error.reason
    return None


# --- 1. Governed constants --------------------------------------------------


class TestGovernedConstants(unittest.TestCase):
    def test_constants_are_the_governed_values(self):
        self.assertEqual(
            astronomy_solver.NIGHT_START_LATITUDE_LIMIT_DEGREES, LATITUDE_LIMIT
        )
        self.assertEqual(
            astronomy_solver.NIGHT_START_SEARCH_SPAN_DAYS, SEARCH_SPAN_DAYS
        )
        self.assertEqual(
            astronomy_solver.NIGHT_START_GENUINE_SUNSET, GENUINE_SUNSET
        )
        self.assertEqual(
            astronomy_solver.NIGHT_START_VANISHED_NIGHT_MINIMUM,
            VANISHED_NIGHT_MINIMUM,
        )
        self.assertEqual(
            astronomy_solver.NIGHT_START_TANGENCY_UNRESOLVED, TANGENCY_UNRESOLVED
        )

    def test_reasons_are_the_governed_codes(self):
        for name, value in (
            ("REASON_NIGHT_START_LATITUDE_UNCERTIFIED", LATITUDE_UNCERTIFIED),
            ("REASON_NIGHT_START_ORDER_AMBIGUOUS", ORDER_AMBIGUOUS),
            ("REASON_NIGHT_START_POLAR_NIGHT", POLAR_NIGHT),
            ("REASON_NIGHT_START_UNRESOLVED", UNRESOLVED),
        ):
            with self.subTest(name=name):
                self.assertEqual(getattr(astronomy_solver, name), value)

    def test_the_horizon_is_the_directional_sunset_horizon(self):
        """One frontier, and so one artifact, for both searches."""
        self.assertEqual(
            astronomy_solver.NIGHT_START_SEARCH_SPAN_DAYS,
            astronomy_solver.SUCCESSOR_SEARCH_SPAN_DAYS,
        )


# --- 2. Certified latitude domain -------------------------------------------


class TestCertifiedDomain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.equinox = equinox_2019()

    def test_both_domain_edges_are_certified(self):
        for latitude in (LATITUDE_LIMIT, -LATITUDE_LIMIT):
            with self.subTest(latitude=latitude):
                record = find_night_start_after(self.equinox, latitude, 0.0)
                self.assertEqual(record.kind, VANISHED_NIGHT_MINIMUM)
                self.assertGreater(record.lo, self.equinox)
                self.assertEqual(record.certified_domain_limit_degrees, LATITUDE_LIMIT)

    def test_the_next_latitude_beyond_either_edge_is_refused(self):
        for latitude in (
            math.nextafter(LATITUDE_LIMIT, math.inf),
            math.nextafter(-LATITUDE_LIMIT, -math.inf),
        ):
            with self.subTest(latitude=latitude):
                self.assertEqual(
                    reason_of(find_night_start_after, self.equinox, latitude, 0.0),
                    LATITUDE_UNCERTIFIED,
                )

    def test_the_exact_poles_are_refused(self):
        for latitude in (90.0, -90.0):
            with self.subTest(latitude=latitude):
                self.assertEqual(
                    reason_of(find_night_start_after, self.equinox, latitude, 0.0),
                    LATITUDE_UNCERTIFIED,
                )

    def test_the_geodetic_domain_is_still_checked_first(self):
        for latitude in (90.5, math.nan, True):
            with self.subTest(latitude=latitude):
                self.assertEqual(
                    reason_of(find_night_start_after, self.equinox, latitude, 0.0),
                    OBSERVER_OUT_OF_DOMAIN,
                )

    def test_a_malformed_instant_is_refused(self):
        self.assertEqual(
            reason_of(find_night_start_after, math.nan, 40.0, 0.0),
            INSTANT_STATE_INVALID,
        )

    def test_the_general_sunset_domain_is_not_narrowed(self):
        """The exact pole remains a valid observer for the sunset routes."""
        self.assertIsNone(
            reason_of(find_sunset_from_instant, self.equinox, 90.0, 0.0)
        )


# --- 3. Ordinary night-start: the DT-A1 sunset, bit for bit -----------------


class TestOrdinaryNightStart(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.equinox = equinox_2019()

    def test_the_night_start_is_the_directional_sunset(self):
        for name, latitude, longitude, hours in ORDINARY_FIXTURES:
            with self.subTest(observer=name):
                record = find_night_start_after(self.equinox, latitude, longitude)
                sunset = find_sunset_from_instant(self.equinox, latitude, longitude)

                self.assertEqual(record.kind, GENUINE_SUNSET)
                self.assertEqual(record.event_tt, sunset.tt)
                self.assertEqual(record.lo, sunset.tt)
                self.assertEqual(record.hi, sunset.tt)
                self.assertEqual(record.kernel, sunset.kernel)
                self.assertIsNone(record.minimum_margin_degrees)
                self.assertAlmostEqual(
                    record.event_tt, self.equinox + hours / 24.0,
                    delta=FIXTURE_TOLERANCE_DAYS,
                )

    def test_provenance_travels_with_the_answer(self):
        record = find_night_start_after(self.equinox, 40.7406, -73.9586)
        self.assertEqual(record.kernel, "de440.bsp")
        self.assertEqual(record.ephemeris_role, "primary")
        self.assertEqual(record.event_threshold_degrees, -0.8333)
        self.assertEqual(
            record.event_convention, astronomy_solver.EVENT_CONVENTION
        )


# --- 4. Vanished night: the DT-O4 polar fixtures ----------------------------


class TestVanishedNightMinimum(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.equinox = equinox_2019()
        cls.records = {
            name: find_night_start_after(cls.equinox, latitude, longitude)
            for name, latitude, longitude, _ in VANISHED_FIXTURES
        }

    def test_every_fixture_is_a_vanished_night_minimum(self):
        for name, _, _, _ in VANISHED_FIXTURES:
            with self.subTest(observer=name):
                record = self.records[name]
                self.assertEqual(record.kind, VANISHED_NIGHT_MINIMUM)
                self.assertIsNone(record.event_tt)
                self.assertGreater(record.minimum_margin_degrees, 0.0)

    def test_the_bracket_holds_the_dt_o4_minimum(self):
        for name, _, _, hours in VANISHED_FIXTURES:
            with self.subTest(observer=name):
                record = self.records[name]
                instant = self.equinox + hours / 24.0
                self.assertLessEqual(record.lo - FIXTURE_TOLERANCE_DAYS, instant)
                self.assertLessEqual(instant, record.hi + FIXTURE_TOLERANCE_DAYS)

    def test_the_bracket_is_strictly_after_the_instant_and_narrow(self):
        for name, _, _, _ in VANISHED_FIXTURES:
            with self.subTest(observer=name):
                record = self.records[name]
                self.assertGreater(record.lo, self.equinox)
                self.assertLessEqual(record.lo, record.hi)
                self.assertLess((record.hi - record.lo) * 86400.0, 60.0)

    def test_no_sunset_is_fabricated(self):
        """89.5N has no sunset in the directional horizon at all."""
        self.assertIsNone(find_sunset_from_instant(self.equinox, 89.5, 0.0))
        self.assertEqual(self.records["89.5N 0E"].kind, VANISHED_NIGHT_MINIMUM)


class TestNoDistantSeasonReAnchor(unittest.TestCase):
    def test_the_first_night_is_used_not_the_returning_sunset(self):
        """The literal rule's first sunset at 89.5N is months later (DT-O3
        Stage 4); the night-start is the first night after the instant."""
        equinox = equinox_2019()
        record = find_night_start_after(equinox, 89.5, 0.0)
        self.assertLess(record.hi - equinox, 1.0)


# --- 5. The R4 seam ----------------------------------------------------------


class TestSeam(unittest.TestCase):
    """Where the altitude minimum falls exactly at the instant, which side of
    the instant the night begins cannot be certified."""

    @classmethod
    def setUpClass(cls):
        cls.equinox = equinox_2019()
        kernel = astronomy_solver.select_kernel_for_interval(
            cls.equinox - 1.0, cls.equinox + 1.0
        )

        def slope(longitude):
            margin = astronomy_solver._detection_margin(
                kernel, SEAM_LATITUDE, longitude
            )
            below, above = margin([cls.equinox - 1e-3, cls.equinox + 1e-3])
            return above - below

        lo, hi = 0.0, 60.0
        assert slope(lo) < 0.0 < slope(hi)
        for _ in range(45):
            middle = 0.5 * (lo + hi)
            if slope(middle) < 0.0:
                lo = middle
            else:
                hi = middle
        cls.seam = 0.5 * (lo + hi)

    def test_the_seam_is_where_dt_o4_places_it(self):
        """Near 1.6684 h x 15 deg/h east of the 89.5N fixture's meridian."""
        self.assertAlmostEqual(self.seam, 25.03, delta=0.05)

    def test_the_seam_fails_closed(self):
        self.assertEqual(
            reason_of(
                find_night_start_after, self.equinox, SEAM_LATITUDE, self.seam
            ),
            ORDER_AMBIGUOUS,
        )

    def test_either_side_certifies_one_rotation_apart(self):
        west = find_night_start_after(self.equinox, SEAM_LATITUDE, self.seam - 0.05)
        east = find_night_start_after(self.equinox, SEAM_LATITUDE, self.seam + 0.05)
        for record in (west, east):
            self.assertEqual(record.kind, VANISHED_NIGHT_MINIMUM)
            self.assertGreater(record.lo, self.equinox)
        self.assertLess(west.hi - self.equinox, 1.0 / 24.0)
        self.assertGreater(east.lo - self.equinox, 23.0 / 24.0)


# --- 6. Tangency --------------------------------------------------------------


class TestTangency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        solstice = find_solar_longitude_event_in_year(
            2019, "SOLAR_LONGITUDE_090"
        ).tt
        cls.anchor = solstice - 2.0
        cls.records = {
            delta: find_night_start_after(cls.anchor, TANGENT_LATITUDE + delta, 0.0)
            for delta in (3e-6, 0.0, -3e-6)
        }

    def test_each_side_of_the_tangency_is_certified(self):
        self.assertEqual(self.records[3e-6].kind, VANISHED_NIGHT_MINIMUM)
        self.assertEqual(self.records[-3e-6].kind, GENUINE_SUNSET)

    def test_the_tangency_itself_stays_explicitly_unresolved(self):
        record = self.records[0.0]
        self.assertEqual(record.kind, TANGENCY_UNRESOLVED)
        self.assertIsNone(record.event_tt)
        self.assertGreater(record.lo, self.anchor)
        self.assertLessEqual(record.lo, record.hi)

    def test_the_night_start_is_continuous_through_the_tangency(self):
        starts = [record.lo for record in self.records.values()]
        self.assertLess((max(starts) - min(starts)) * 86400.0, 60.0)


# --- 7. Polar night and Delta-T knots ---------------------------------------


class TestRefusals(unittest.TestCase):
    def test_polar_night_is_refused(self):
        solstice = find_solar_longitude_event_in_year(
            2019, "SOLAR_LONGITUDE_270"
        ).tt
        self.assertEqual(
            reason_of(find_night_start_after, solstice, 80.0, 0.0), POLAR_NIGHT
        )

    def test_a_frontier_touching_a_delta_t_knot_is_refused(self):
        knot = 1721045.0 + 365.25 * 1895.0
        self.assertEqual(
            reason_of(find_night_start_after, knot - 1.0, 40.0, 0.0),
            DETECTION_AMBIGUOUS_DT_KNOT,
        )


# --- 8. The certified skeleton on synthetic margins --------------------------


class TestSyntheticSkeleton(unittest.TestCase):
    """The skeleton applied to known margins, isolating each outcome.

    Each margin respects the governed bounds at latitude 40 (|G'| <= L1,
    |G''| <= L2), and the instants are small so binary64 spacing stays far
    below epsilon_G.
    """

    LATITUDE = 40.0
    ANCHOR = 10.0

    @staticmethod
    def wave(offset, amplitude, phase=0.0):
        def margin(values):
            t = numpy.asarray(values, dtype=float)
            return offset + amplitude * numpy.cos(
                2.0 * numpy.pi * (t - 10.0 - phase)
            )
        return margin

    def locate(self, margin, first_sunset):
        return astronomy_solver._locate_night_start(
            margin, self.LATITUDE, self.ANCHOR, self.ANCHOR + 3.0, first_sunset
        )

    def no_sunset_search(self):
        self.fail("a sunset search was made where the skeleton needs none")

    def test_a_vanished_night_brackets_its_minimum(self):
        kind, lo, hi, event_tt, least = self.locate(
            self.wave(0.25, 0.2), self.no_sunset_search
        )
        self.assertEqual(kind, VANISHED_NIGHT_MINIMUM)
        self.assertLessEqual(lo, self.ANCHOR + 0.5)
        self.assertLessEqual(self.ANCHOR + 0.5, hi)
        self.assertIsNone(event_tt)
        self.assertGreater(least, 0.0)

    def test_an_ordinary_night_starts_at_the_dt_a1_sunset(self):
        root = self.ANCHOR + 0.25
        kind, lo, hi, event_tt, _ = self.locate(
            self.wave(0.0, 0.5), lambda: SunsetEvent(tt=root, kernel="test")
        )
        self.assertEqual((kind, lo, hi, event_tt),
                         (GENUINE_SUNSET, root, root, root))

    def test_a_sunset_outside_the_certified_crossing_is_inconsistent(self):
        with self.assertRaises(SunsetChronologyError) as raised:
            self.locate(
                self.wave(0.0, 0.5),
                lambda: SunsetEvent(tt=self.ANCHOR + 0.6, kernel="test"),
            )
        self.assertEqual(raised.exception.reason, DETECTION_INCONSISTENT)

    def test_polar_night_is_refused_without_a_sunset_search(self):
        with self.assertRaises(SunsetChronologyError) as raised:
            self.locate(self.wave(-0.3, 0.2), self.no_sunset_search)
        self.assertEqual(raised.exception.reason, POLAR_NIGHT)

    def test_a_minimum_at_the_instant_is_order_ambiguous(self):
        with self.assertRaises(SunsetChronologyError) as raised:
            self.locate(self.wave(0.25, 0.2, phase=0.5), self.no_sunset_search)
        self.assertEqual(raised.exception.reason, ORDER_AMBIGUOUS)

    def test_a_minimum_just_after_the_instant_is_certified(self):
        kind, lo, _, _, _ = self.locate(
            self.wave(0.25, 0.2, phase=0.5 + 0.01), self.no_sunset_search
        )
        self.assertEqual(kind, VANISHED_NIGHT_MINIMUM)
        self.assertGreater(lo, self.ANCHOR)


# --- 9. Non-interference ------------------------------------------------------


class TestNonInterference(unittest.TestCase):
    def test_the_sunset_instant_comes_only_from_the_dt_a1_route_solver(self):
        import inspect

        source = inspect.getsource(astronomy_solver.find_night_start_after)
        self.assertIn("find_sunset_from_instant(anchor, latitude, longitude)",
                      source)
        self.assertNotIn("find_discrete", source)

    def test_no_published_sunset_solver_refers_to_the_night_start(self):
        import inspect

        for function in (
            astronomy_solver.find_sunset_from_instant,
            astronomy_solver.count_sunsets_in_interval,
            astronomy_solver.determine_solar_regime,
            astronomy_solver._certified_sunset_structure,
            astronomy_solver._certified_sunsets_in_frontier,
            astronomy_solver._certified_first_sunset_after,
        ):
            with self.subTest(function=function.__name__):
                self.assertNotIn("night_start", inspect.getsource(function))


if __name__ == "__main__":
    unittest.main()
