"""PTC-A1 verification for the certified directional solar-crossing locator.

Covers find_solar_crossing in astronomy_solver.py: the governed constants
and request validation, ordinary sunrises and sunsets in both directions,
the overlap with /sunset-event-after, strict ordering at an exact-root
anchor, a northern polar day, a northern polar night, the southern
hemisphere, the DT-A1 tangency, deep-time and Delta-T-knot representatives,
truncated frontiers and provenance.

Test framework: standard-library unittest.
Scientific execution uses the repository's certified Skyfield runtime and the
real pinned Authority ephemeris artifacts.

INVOCATION
----------
Run from the repository root with the repository interpreter:

    .\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v

INDEPENDENT ORACLE DISCIPLINE
-----------------------------

Reason codes, spellings, the horizon bound, the event convention and the
fixtures are declared here as test-local literals. They are deliberately NOT
imported from astronomy_solver: importing a constant to check that same
constant would make the test agree with a defective value instead of
detecting it. Orientation is checked against the certified predicate
itself, read on both sides of each reported crossing.

EVENT IDENTITY ACROSS BRACKETS
------------------------------

Roots found under different search brackets may differ in their final
bits, so no assertion here compares two independently bracketed roots for
identity. Bit identity is asserted only where the frontier - and therefore
the bracket - is the same. Elsewhere the assertions are ordinal: which side
of a state a crossing lies on, and how far apart two DIFFERENT crossings
are.

PROVENANCE OF THE FIXTURES
--------------------------

The 70N states are DT-CB-E1 recorded Authority evidence (Engine
tests/helpers/r4-authority-evidence.ts, Authority b6f1147): 4142C1FDBCD4FDAC
is the last sunset before the 2019 polar day (the walk from it found no
sunset for 36 consecutive three-day probes), 4142C221BCD4FDAC is the probe
that found the returning sunset 4142C2223EE20421, and 4142C21014ECB620 /
4142C26BD729D16B are the 2019 June and December solstices. The tangency
latitude, the 1945 knot observer and the 65N / -4018 territory are the
DT-A1 certification fixtures.
"""

import math
import struct
import unittest

from skyfield import almanac
from skyfield.api import wgs84

import astronomy_solver
from astronomy_solver import (
    SunsetChronologyError,
    find_solar_crossing,
    find_solar_longitude_event_in_year,
    find_sunset_from_instant,
    find_sunset_predecessor,
    supported_search_frontier,
    ts,
)

# --- Governed literals -------------------------------------------------------

RISING = "RISING"
SETTING = "SETTING"
AFTER = "AFTER"
BEFORE = "BEFORE"

MAX_HORIZON_DAYS = 400.0
SUNSET_SEARCH_SPAN_DAYS = 3.0
THRESHOLD_DEGREES = -0.8333
EVENT_CONVENTION = (
    "USNO apparent sunrise/sunset: the Sun's centre at -0.8333 degrees "
    "apparent topocentric altitude, combining 16 arcminutes of solar "
    "semidiameter with 34 arcminutes of mean refraction; sea-level observer, "
    "no terrain"
)

ORIENTATION_INVALID = "SOLAR_CROSSING_ORIENTATION_INVALID"
DIRECTION_INVALID = "SOLAR_CROSSING_DIRECTION_INVALID"
HORIZON_INVALID = "SOLAR_CROSSING_HORIZON_INVALID"
HORIZON_TOO_LONG = "SOLAR_CROSSING_HORIZON_TOO_LONG"
DETECTION_AMBIGUOUS = "SUNSET_DETECTION_AMBIGUOUS"
DETECTION_AMBIGUOUS_DT_KNOT = "SUNSET_DETECTION_AMBIGUOUS_DT_KNOT"
INSTANT_STATE_INVALID = "INSTANT_STATE_INVALID"
OBSERVER_OUT_OF_DOMAIN = "OBSERVER_OUT_OF_DOMAIN"
COVERAGE_EXHAUSTED = "EPHEMERIS_COVERAGE_EXHAUSTED"
REACH_EXHAUSTED = "EPHEMERIS_REACH_EXHAUSTED"

PRIMARY = "de440.bsp"
ANCIENT = "de441_part-1.bsp"
FUTURE = "de441_part-2.bsp"

# The certified root finder's convergence width (Skyfield EPSILON).
FIND_DISCRETE_EPSILON_DAYS = 0.001 / 86400.0

NEW_YORK = (40.7406, -73.9586)
JERUSALEM = (31.7683, 35.2137)

# PTC-A1 BEFORE identity investigation: the 2019 New York sunrise reported by
# RISING / AFTER from the March equinox over a one-day horizon.
NEW_YORK_SUNRISE_ROOT = "4142C1E1FA973B45"

# DT-CB-E1 recorded 70N evidence (see module docstring).
FINAL_SUNSET_70N = "4142C1FDBCD4FDAC"
RETURN_PROBE_70N = "4142C221BCD4FDAC"
RETURNING_SUNSET_70N = "4142C2223EE20421"
JUNE_SOLSTICE_2019 = "4142C21014ECB620"
DECEMBER_SOLSTICE_2019 = "4142C26BD729D16B"

# DT-A1 fixtures.
TANGENT_LATITUDE = 65.74293204295881
KNOT_1945_LATITUDE = 40.7406
KNOT_1945_LONGITUDE = -158.8306544325173


def tt_of_bits(spelling):
    """The binary64 a ttBits spelling names, decoded independently."""
    return struct.unpack(">d", bytes.fromhex(spelling))[0]


def bits_of(value):
    return struct.pack(">d", value).hex().upper()


def spring(year):
    return find_solar_longitude_event_in_year(year, "SOLAR_LONGITUDE_000").tt


def june(year):
    return find_solar_longitude_event_in_year(year, "SOLAR_LONGITUDE_090").tt


def crossing(anchor, observer, orientation, direction, horizon):
    return find_solar_crossing(
        anchor, observer[0], observer[1], orientation, direction, horizon
    )


def event_of(anchor, observer, orientation, direction, horizon):
    record = crossing(anchor, observer, orientation, direction, horizon)
    return record.event_tt


def reason_of(callable_, *args):
    try:
        callable_(*args)
    except SunsetChronologyError as error:
        return error.reason
    return None


def predicate(tt, observer):
    frontier = supported_search_frontier(tt - 0.5, tt + 0.5, *observer)
    return almanac.sunrise_sunset(
        astronomy_solver.load_kernel(frontier.kernel), wgs84.latlon(*observer)
    )


def assert_orientation(test, tt, observer, orientation):
    """The certified predicate changes sign at the event, in that direction.

    The root finder reports the upper end of its converged bracket, so the
    state one convergence width earlier lies on the near side.
    """
    up = predicate(tt, observer)
    before = bool(up(ts.tt_jd(tt - FIND_DISCRETE_EPSILON_DAYS)))
    after = bool(up(ts.tt_jd(tt)))
    if orientation == RISING:
        test.assertEqual((before, after), (False, True))
    else:
        test.assertEqual((before, after), (True, False))


# --- 1. Governed constants --------------------------------------------------


class TestGovernedConstants(unittest.TestCase):
    def test_spellings_and_bound_are_the_governed_values(self):
        self.assertEqual(astronomy_solver.SOLAR_CROSSING_RISING, RISING)
        self.assertEqual(astronomy_solver.SOLAR_CROSSING_SETTING, SETTING)
        self.assertEqual(astronomy_solver.SOLAR_CROSSING_AFTER, AFTER)
        self.assertEqual(astronomy_solver.SOLAR_CROSSING_BEFORE, BEFORE)
        self.assertEqual(
            astronomy_solver.SOLAR_CROSSING_MAX_HORIZON_DAYS, MAX_HORIZON_DAYS
        )

    def test_reasons_are_the_governed_codes(self):
        for name, value in (
            ("REASON_SOLAR_CROSSING_ORIENTATION_INVALID", ORIENTATION_INVALID),
            ("REASON_SOLAR_CROSSING_DIRECTION_INVALID", DIRECTION_INVALID),
            ("REASON_SOLAR_CROSSING_HORIZON_INVALID", HORIZON_INVALID),
            ("REASON_SOLAR_CROSSING_HORIZON_TOO_LONG", HORIZON_TOO_LONG),
        ):
            with self.subTest(name=name):
                self.assertEqual(getattr(astronomy_solver, name), value)

    def test_the_predicate_and_its_convention_are_unchanged(self):
        self.assertEqual(astronomy_solver.EVENT_THRESHOLD_DEGREES, THRESHOLD_DEGREES)
        self.assertEqual(astronomy_solver.EVENT_CONVENTION, EVENT_CONVENTION)

    def test_the_bound_is_a_separate_constant_with_the_admission_value(self):
        self.assertEqual(astronomy_solver.COUNT_MAX_SPAN_DAYS, MAX_HORIZON_DAYS)
        self.assertEqual(astronomy_solver.REGIME_MAX_SPAN_DAYS, MAX_HORIZON_DAYS)


# --- 2. Request validation --------------------------------------------------


class TestRequestValidation(unittest.TestCase):
    ANCHOR = 2458563.5

    def refusal(self, *, anchor=None, observer=NEW_YORK, orientation=SETTING,
                direction=AFTER, horizon=3.0):
        return reason_of(
            find_solar_crossing,
            self.ANCHOR if anchor is None else anchor,
            observer[0], observer[1], orientation, direction, horizon,
        )

    def test_unpublished_orientations_are_refused(self):
        for value in ("rising", "Sunrise", "", "SUNRISE", None, 1):
            with self.subTest(orientation=value):
                self.assertEqual(self.refusal(orientation=value), ORIENTATION_INVALID)

    def test_unpublished_directions_are_refused(self):
        for value in ("after", "FORWARD", "", None, -1):
            with self.subTest(direction=value):
                self.assertEqual(self.refusal(direction=value), DIRECTION_INVALID)

    def test_invalid_horizons_are_refused_not_clamped(self):
        for value in (0.0, -0.0, -1.0, math.nan, math.inf, -math.inf, True,
                      "3", None):
            with self.subTest(horizon=value):
                self.assertEqual(self.refusal(horizon=value), HORIZON_INVALID)

    def test_a_horizon_beyond_the_bound_is_refused(self):
        for value in (math.nextafter(MAX_HORIZON_DAYS, math.inf), 401.0, 1e9):
            with self.subTest(horizon=value):
                self.assertEqual(self.refusal(horizon=value), HORIZON_TOO_LONG)

    def test_the_bound_itself_is_admitted(self):
        record = crossing(self.ANCHOR, NEW_YORK, SETTING, BEFORE, MAX_HORIZON_DAYS)
        self.assertEqual(record.horizon_days, MAX_HORIZON_DAYS)
        self.assertTrue(record.complete)
        self.assertIsNotNone(record.event_tt)

    def test_an_integer_horizon_is_admitted_as_its_value(self):
        record = crossing(self.ANCHOR, NEW_YORK, SETTING, AFTER, 3)
        self.assertEqual(record.horizon_days, 3.0)

    def test_a_malformed_anchor_is_the_substrate_refusal(self):
        for value in (math.nan, math.inf):
            with self.subTest(anchor=value):
                self.assertEqual(self.refusal(anchor=value), INSTANT_STATE_INVALID)

    def test_an_observer_outside_the_domain_is_the_substrate_refusal(self):
        self.assertEqual(self.refusal(observer=(91.0, 0.0)), OBSERVER_OUT_OF_DOMAIN)


# --- 3. Ordinary sunrises and sunsets ---------------------------------------


class TestOrdinaryCrossings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.anchor = spring(2019)

    def test_all_four_requests_answer_with_the_requested_crossing(self):
        for name, observer in (("New York", NEW_YORK), ("Jerusalem", JERUSALEM)):
            for orientation in (RISING, SETTING):
                for direction in (AFTER, BEFORE):
                    with self.subTest(observer=name, orientation=orientation,
                                      direction=direction):
                        record = crossing(
                            self.anchor, observer, orientation, direction,
                            SUNSET_SEARCH_SPAN_DAYS,
                        )
                        self.assertTrue(record.complete)
                        self.assertIsNone(record.truncation_reason)
                        self.assertEqual(record.orientation, orientation)
                        self.assertEqual(record.direction, direction)
                        if direction == AFTER:
                            self.assertGreater(record.event_tt, self.anchor)
                        else:
                            self.assertLess(record.event_tt, self.anchor)
                        assert_orientation(self, record.event_tt, observer,
                                           orientation)

    def test_the_two_directions_bracket_the_anchor_one_rotation_apart(self):
        for name, observer in (("New York", NEW_YORK), ("Jerusalem", JERUSALEM)):
            for orientation in (RISING, SETTING):
                with self.subTest(observer=name, orientation=orientation):
                    later = event_of(self.anchor, observer, orientation, AFTER, 3.0)
                    earlier = event_of(self.anchor, observer, orientation, BEFORE, 3.0)
                    self.assertLess(earlier, self.anchor)
                    self.assertGreater(later, self.anchor)
                    self.assertGreater(later - earlier, 0.9)
                    self.assertLess(later - earlier, 1.1)

    def test_sunrise_and_sunset_alternate(self):
        sunset = event_of(self.anchor, NEW_YORK, SETTING, AFTER, 3.0)
        sunrise = event_of(sunset, NEW_YORK, RISING, AFTER, 3.0)
        next_sunset = event_of(sunrise, NEW_YORK, SETTING, AFTER, 3.0)
        self.assertLess(sunset, sunrise)
        self.assertLess(sunrise, next_sunset)
        self.assertLess(next_sunset - sunset, 1.1)


# --- 4. Overlap with the existing sunset route -------------------------------


class TestOverlapWithSunsetEventAfter(unittest.TestCase):
    """Over the same three-day frontier the bracket is the same, so bits are."""

    def test_setting_after_is_bit_identical_to_the_dt_a1_successor(self):
        start = spring(2019)
        for name, observer in (("New York", NEW_YORK), ("Jerusalem", JERUSALEM)):
            for offset in (0.0, 0.3, 0.71, 1.9):
                with self.subTest(observer=name, offset=offset):
                    anchor = start + offset
                    expected = find_sunset_from_instant(anchor, *observer).tt
                    found = event_of(anchor, observer, SETTING, AFTER,
                                     SUNSET_SEARCH_SPAN_DAYS)
                    self.assertEqual(bits_of(found), bits_of(expected))

    def test_the_recorded_returning_sunset_is_reproduced_bit_for_bit(self):
        found = event_of(tt_of_bits(RETURN_PROBE_70N), (70.0, 0.0), SETTING,
                         AFTER, SUNSET_SEARCH_SPAN_DAYS)
        self.assertEqual(bits_of(found), RETURNING_SUNSET_70N)

    def test_certified_before_agrees_where_the_predecessor_is_already_right(self):
        """Ordinary geometry: the uncertified A2-3b answer is correct there."""
        anchor = spring(2019)
        for name, observer in (("New York", NEW_YORK), ("Jerusalem", JERUSALEM)):
            with self.subTest(observer=name):
                expected = find_sunset_predecessor(anchor, *observer).tt
                found = event_of(anchor, observer, SETTING, BEFORE,
                                 SUNSET_SEARCH_SPAN_DAYS)
                self.assertEqual(bits_of(found), bits_of(expected))


# --- 5. Strict ordering at an exact-root anchor ------------------------------


class TestExactRootAnchor(unittest.TestCase):
    def test_a_root_anchor_is_excluded_from_both_directions(self):
        start = spring(2019)
        for orientation in (RISING, SETTING):
            root = event_of(start, NEW_YORK, orientation, AFTER, 3.0)
            with self.subTest(orientation=orientation):
                later = event_of(root, NEW_YORK, orientation, AFTER, 3.0)
                earlier = event_of(root, NEW_YORK, orientation, BEFORE, 3.0)
                # The neighbouring crossings, not the anchor's own.
                self.assertGreater(later - root, 0.5)
                self.assertGreater(root - earlier, 0.5)
                assert_orientation(self, later, NEW_YORK, orientation)
                assert_orientation(self, earlier, NEW_YORK, orientation)

    def test_the_backward_exclusion_is_one_convergence_width(self):
        strictly_before = astronomy_solver._strictly_before
        anchor = 2458563.5
        self.assertFalse(strictly_before(anchor, anchor))
        self.assertFalse(strictly_before(anchor + 1e-9, anchor))
        self.assertFalse(
            strictly_before(anchor - FIND_DISCRETE_EPSILON_DAYS / 2.0, anchor)
        )
        self.assertTrue(
            strictly_before(anchor - 4.0 * FIND_DISCRETE_EPSILON_DAYS, anchor)
        )

    def test_a_cross_bracket_representation_of_the_anchor_is_not_before_it(self):
        """PTC-A1 BEFORE identity regression (recorded investigation case).

        NEW_YORK_SUNRISE_ROOT is the 2019 New York sunrise as RISING / AFTER
        reports it from the March equinox over a one-day horizon. Refined
        again over the three-day BEFORE frontier ending at that root, the same
        physical crossing is reported a few units in the last place EARLIER -
        a state a literal ``<`` would accept as a sunrise before the anchor.
        The governed identity rule must exclude it and return the previous
        day's sunrise instead.
        """
        root = tt_of_bits(NEW_YORK_SUNRISE_ROOT)
        self.assertEqual(
            bits_of(event_of(spring(2019), NEW_YORK, RISING, AFTER, 1.0)),
            NEW_YORK_SUNRISE_ROOT,
        )

        # The alternate representation really exists in the BEFORE frontier.
        frontier = supported_search_frontier(root, root - 3.0, *NEW_YORK)
        up = almanac.sunrise_sunset(
            astronomy_solver.load_kernel(frontier.kernel), wgs84.latlon(*NEW_YORK)
        )
        times, events = almanac.find_discrete(
            ts.tt_jd(frontier.tt_lo), ts.tt_jd(frontier.tt_hi), up
        )
        alternates = [
            float(t.tt) for t, u in zip(times, events)
            if bool(u) and 0.0 < root - float(t.tt) <= FIND_DISCRETE_EPSILON_DAYS
        ]
        self.assertEqual(len(alternates), 1)

        found = event_of(root, NEW_YORK, RISING, BEFORE, 3.0)
        self.assertLess(found, alternates[0])
        self.assertGreater(root - found, 0.9)
        self.assertLess(root - found, 1.1)
        assert_orientation(self, found, NEW_YORK, RISING)

    def test_the_identity_window_is_narrower_than_any_certified_leaf(self):
        """The proof the BEFORE identity rule rests on, in seconds.

        A certified DT-A2 leaf is split only when wider than W_MIN, and at a
        fraction no smaller than 0.5 - 1/16, so every certified leaf is wider
        than (0.5 - 1/16) * W_MIN. Two same-orientation crossings need an
        opposite crossing in a leaf of its own between them, so they are at
        least that far apart. The identity window is the root finder's 1 ms
        reporting resolution; it must be far narrower, or the rule could
        merge two distinct crossings.
        """
        smallest_split_fraction = 0.5 - 1.0 / 16.0
        leaf_floor_seconds = (
            smallest_split_fraction * astronomy_solver.DETECTION_W_MIN_DAYS * 86400.0
        )
        identity_window_seconds = (
            astronomy_solver._FIND_DISCRETE_EPSILON_DAYS * 86400.0
        )

        self.assertEqual(smallest_split_fraction, 0.4375)
        self.assertAlmostEqual(identity_window_seconds, 0.001, places=12)
        self.assertGreater(leaf_floor_seconds, 0.57)
        self.assertGreater(leaf_floor_seconds, identity_window_seconds)
        self.assertGreater(leaf_floor_seconds / identity_window_seconds, 500.0)


# --- 6. Northern polar day (70N, 2019) --------------------------------------


class TestPolarDay70N(unittest.TestCase):
    OBSERVER = (70.0, 0.0)

    @classmethod
    def setUpClass(cls):
        cls.final_sunset = tt_of_bits(FINAL_SUNSET_70N)
        cls.solstice = tt_of_bits(JUNE_SOLSTICE_2019)
        cls.onset_sunrise = event_of(cls.final_sunset, cls.OBSERVER, RISING,
                                     AFTER, 3.0)

    def test_the_onset_sunrise_ends_the_last_night_within_hours(self):
        self.assertGreater(self.onset_sunrise, self.final_sunset)
        self.assertLess(self.onset_sunrise - self.final_sunset, 0.25)
        assert_orientation(self, self.onset_sunrise, self.OBSERVER, RISING)

    def test_no_crossing_of_either_orientation_follows_for_sixty_days(self):
        for orientation in (RISING, SETTING):
            with self.subTest(orientation=orientation):
                record = crossing(self.onset_sunrise, self.OBSERVER, orientation,
                                  AFTER, 60.0)
                self.assertIsNone(record.event_tt)
                self.assertTrue(record.complete)
                self.assertIsNone(record.truncation_reason)

    def test_one_long_search_finds_the_returning_sunset(self):
        returning = event_of(self.onset_sunrise, self.OBSERVER, SETTING, AFTER,
                             100.0)
        self.assertGreater(returning - self.onset_sunrise, 60.0)
        assert_orientation(self, returning, self.OBSERVER, SETTING)
        # The same physical sunset the 36-probe walk found: the recorded state
        # lies on the far side of this crossing within the same night.
        recorded = tt_of_bits(RETURNING_SUNSET_70N)
        self.assertLess(abs(returning - recorded), 0.25)

    def test_backward_from_midsummer_finds_the_final_sunset_and_onset_sunrise(self):
        final = event_of(self.solstice, self.OBSERVER, SETTING, BEFORE, 60.0)
        onset = event_of(self.solstice, self.OBSERVER, RISING, BEFORE, 60.0)
        self.assertLess(final, onset)
        self.assertLess(onset, self.solstice)
        self.assertLess(onset - final, 0.25)
        self.assertLess(abs(final - self.final_sunset), 0.25)
        assert_orientation(self, final, self.OBSERVER, SETTING)
        assert_orientation(self, onset, self.OBSERVER, RISING)

    def test_backward_from_the_returning_sunset_skips_the_polar_day(self):
        returning = tt_of_bits(RETURNING_SUNSET_70N)
        previous = event_of(returning, self.OBSERVER, SETTING, BEFORE, 100.0)
        self.assertLess(previous, self.onset_sunrise)
        self.assertLess(abs(previous - self.final_sunset), 0.25)


# --- 7. Northern polar night (80N, 2019-2020) -------------------------------


class TestPolarNight80N(unittest.TestCase):
    OBSERVER = (80.0, 0.0)

    @classmethod
    def setUpClass(cls):
        cls.solstice = tt_of_bits(DECEMBER_SOLSTICE_2019)

    def test_absence_is_certified_either_side_of_midwinter(self):
        for orientation in (RISING, SETTING):
            for direction in (AFTER, BEFORE):
                with self.subTest(orientation=orientation, direction=direction):
                    record = crossing(self.solstice, self.OBSERVER, orientation,
                                      direction, 30.0)
                    self.assertIsNone(record.event_tt)
                    self.assertTrue(record.complete)

    def test_the_final_sunrise_precedes_the_onset_sunset_on_the_last_day(self):
        final = event_of(self.solstice, self.OBSERVER, RISING, BEFORE, 100.0)
        onset = event_of(self.solstice, self.OBSERVER, SETTING, BEFORE, 100.0)
        self.assertLess(final, onset)
        self.assertLess(onset - final, 0.5)
        self.assertLess(onset, self.solstice - 30.0)
        assert_orientation(self, final, self.OBSERVER, RISING)
        assert_orientation(self, onset, self.OBSERVER, SETTING)

    def test_the_returning_sunrise_precedes_the_first_sunset_after_it(self):
        returning = event_of(self.solstice, self.OBSERVER, RISING, AFTER, 100.0)
        first_sunset = event_of(self.solstice, self.OBSERVER, SETTING, AFTER, 100.0)
        self.assertGreater(returning, self.solstice + 30.0)
        self.assertLess(returning, first_sunset)
        self.assertLess(first_sunset - returning, 0.5)
        assert_orientation(self, returning, self.OBSERVER, RISING)
        assert_orientation(self, first_sunset, self.OBSERVER, SETTING)


# --- 8. Southern hemisphere -------------------------------------------------


class TestSouthernHemisphere(unittest.TestCase):
    """The same primitive, with the seasons reversed and no code path changed."""

    OBSERVER = (-70.0, 0.0)

    @classmethod
    def setUpClass(cls):
        cls.june = june(2019)
        cls.december = tt_of_bits(DECEMBER_SOLSTICE_2019)

    def test_polar_night_around_the_june_solstice(self):
        for orientation in (RISING, SETTING):
            with self.subTest(orientation=orientation):
                record = crossing(self.june, self.OBSERVER, orientation, AFTER, 20.0)
                self.assertIsNone(record.event_tt)
                self.assertTrue(record.complete)

        final = event_of(self.june, self.OBSERVER, RISING, BEFORE, 60.0)
        onset = event_of(self.june, self.OBSERVER, SETTING, BEFORE, 60.0)
        returning = event_of(self.june, self.OBSERVER, RISING, AFTER, 60.0)
        self.assertLess(final, onset)
        self.assertLess(onset - final, 0.5)
        self.assertLess(onset, self.june)
        self.assertGreater(returning, self.june + 20.0)
        assert_orientation(self, final, self.OBSERVER, RISING)
        assert_orientation(self, onset, self.OBSERVER, SETTING)
        assert_orientation(self, returning, self.OBSERVER, RISING)

    def test_polar_day_around_the_december_solstice(self):
        for orientation in (RISING, SETTING):
            with self.subTest(orientation=orientation):
                record = crossing(self.december, self.OBSERVER, orientation,
                                  AFTER, 20.0)
                self.assertIsNone(record.event_tt)
                self.assertTrue(record.complete)


# --- 9. Tangency ------------------------------------------------------------


class TestTangency(unittest.TestCase):
    """DT-A1: certified when the proof supports it, refused when it does not."""

    @classmethod
    def setUpClass(cls):
        cls.lo = june(2019) - 2.0

    def requests(self, latitude):
        observer = (latitude, 0.0)
        return (
            (SETTING, AFTER, self.lo, observer),
            (RISING, AFTER, self.lo, observer),
            (SETTING, BEFORE, self.lo + 3.0, observer),
            (RISING, BEFORE, self.lo + 3.0, observer),
        )

    def test_the_tangency_itself_is_refused_never_forced(self):
        for delta in (0.0, 1e-7):
            for orientation, direction, anchor, observer in self.requests(
                    TANGENT_LATITUDE + delta):
                with self.subTest(delta=delta, orientation=orientation,
                                  direction=direction):
                    self.assertEqual(
                        reason_of(find_solar_crossing, anchor, observer[0],
                                  observer[1], orientation, direction, 3.0),
                        DETECTION_AMBIGUOUS,
                    )

    def test_a_clearly_vanished_night_is_certified_absent(self):
        for orientation, direction, anchor, observer in self.requests(
                TANGENT_LATITUDE + 3e-6):
            with self.subTest(orientation=orientation, direction=direction):
                record = crossing(anchor, observer, orientation, direction, 3.0)
                self.assertIsNone(record.event_tt)
                self.assertTrue(record.complete)

    def test_a_clearly_present_night_is_certified_found(self):
        observer = (TANGENT_LATITUDE - 3e-6, 0.0)
        sunset = event_of(self.lo, observer, SETTING, AFTER, 3.0)
        sunrise = event_of(self.lo, observer, RISING, AFTER, 3.0)
        self.assertLess(sunset, sunrise)
        self.assertLess(sunrise - sunset, 0.01)
        assert_orientation(self, sunset, observer, SETTING)
        assert_orientation(self, sunrise, observer, RISING)


# --- 10. Deep time and Delta-T knots ----------------------------------------


class TestDeepTimeRepresentatives(unittest.TestCase):
    def test_the_minus_4018_short_nights_recover_their_missed_sunrises(self):
        """Around the -4018 June solstice at 65N the nights are shorter than
        the root finder's 0.04 day sampling step, so it skips both crossings
        of some nights. Each recovered night's sunrise must be certified."""
        observer = (65.0, 0.0)
        solstice = june(-4018)
        frontier = supported_search_frontier(solstice - 8.0, solstice + 8.0,
                                             *observer)
        self.assertEqual(frontier.kernel, ANCIENT)
        up = almanac.sunrise_sunset(
            astronomy_solver.load_kernel(frontier.kernel), wgs84.latlon(*observer)
        )
        times, events = almanac.find_discrete(
            ts.tt_jd(frontier.tt_lo), ts.tt_jd(frontier.tt_hi), up
        )
        reported_sets = [float(t.tt) for t, u in zip(times, events) if not bool(u)]
        reported_rises = [float(t.tt) for t, u in zip(times, events) if bool(u)]
        certified, reason = astronomy_solver._certified_sunsets_in_frontier(
            frontier, observer[0], observer[1], up, reported_sets
        )
        self.assertIsNone(reason)
        recovered = sorted(set(certified) - set(reported_sets))
        self.assertGreaterEqual(len(recovered), 1)

        for sunset in recovered[:3]:
            with self.subTest(sunset=sunset):
                sunrise = event_of(sunset, observer, RISING, AFTER, 1.0)
                self.assertGreater(sunrise, sunset)
                self.assertLess(sunrise - sunset, 0.04)
                assert_orientation(self, sunrise, observer, RISING)
                # The root finder alone never reported this sunrise.
                self.assertFalse(any(
                    sunset < value <= sunrise + FIND_DISCRETE_EPSILON_DAYS
                    for value in reported_rises
                ))
                # And the certified backward search returns to that night.
                back = event_of(sunrise, observer, SETTING, BEFORE, 1.0)
                self.assertLess(back, sunrise)
                self.assertLess(sunrise - back, 0.04)

    def test_an_ancient_kernel_representative(self):
        anchor = spring(-4018)
        for orientation in (RISING, SETTING):
            with self.subTest(orientation=orientation):
                record = crossing(anchor, JERUSALEM, orientation, AFTER, 3.0)
                self.assertEqual(record.kernel, ANCIENT)
                self.assertEqual(record.ephemeris_role, "ancient")
                self.assertTrue(record.complete)
                assert_orientation(self, record.event_tt, JERUSALEM, orientation)

    def test_a_future_kernel_representative(self):
        anchor = spring(10000)
        for orientation in (RISING, SETTING):
            with self.subTest(orientation=orientation):
                record = crossing(anchor, NEW_YORK, orientation, AFTER, 3.0)
                self.assertEqual(record.kernel, FUTURE)
                self.assertEqual(record.ephemeris_role, "future")
                self.assertTrue(record.complete)
                assert_orientation(self, record.event_tt, NEW_YORK, orientation)

    def test_a_delta_t_knot_fails_closed_in_both_orientations_and_directions(self):
        knot = 1721045.0 + 365.25 * 1945.0
        for orientation, direction, anchor in (
            (SETTING, AFTER, knot - 0.25),
            (RISING, AFTER, knot - 0.25),
            (SETTING, BEFORE, knot + 0.25),
        ):
            with self.subTest(orientation=orientation, direction=direction):
                self.assertEqual(
                    reason_of(find_solar_crossing, anchor, KNOT_1945_LATITUDE,
                              KNOT_1945_LONGITUDE, orientation, direction, 0.5),
                    DETECTION_AMBIGUOUS_DT_KNOT,
                )


# --- 11. Truncated frontiers ------------------------------------------------


class TestTruncatedFrontier(unittest.TestCase):
    def test_a_crossing_inside_a_truncated_forward_frontier_is_returned(self):
        end = astronomy_solver.kernel_coverage_tt(astronomy_solver.FUTURE_KERNEL).tt_end
        record = crossing(end - 1.0, NEW_YORK, SETTING, AFTER, 10.0)
        self.assertIsNotNone(record.event_tt)
        self.assertFalse(record.complete)
        self.assertEqual(record.truncation_reason, COVERAGE_EXHAUSTED)
        self.assertEqual(record.covered_hi, end)

        # From that crossing nothing remains before coverage ends, and that is
        # never reported as absence.
        self.assertEqual(
            reason_of(find_solar_crossing, record.event_tt, *NEW_YORK, SETTING,
                      AFTER, 10.0),
            COVERAGE_EXHAUSTED,
        )

    def test_a_truncated_backward_frontier_behaves_the_same_way(self):
        start = astronomy_solver.kernel_coverage_tt(
            astronomy_solver.ANCIENT_KERNEL
        ).tt_start
        record = crossing(start + 1.0, NEW_YORK, SETTING, BEFORE, 10.0)
        self.assertIsNotNone(record.event_tt)
        self.assertFalse(record.complete)
        self.assertIn(record.truncation_reason, (COVERAGE_EXHAUSTED, REACH_EXHAUSTED))
        self.assertIn(
            reason_of(find_solar_crossing, record.event_tt, *NEW_YORK, SETTING,
                      BEFORE, 10.0),
            (COVERAGE_EXHAUSTED, REACH_EXHAUSTED),
        )


# --- 12. Provenance and the record ------------------------------------------


class TestProvenance(unittest.TestCase):
    def test_the_record_states_its_request_frontier_and_predicate(self):
        anchor = spring(2019)
        forward = crossing(anchor, NEW_YORK, RISING, AFTER, 3.0)
        backward = crossing(anchor, NEW_YORK, RISING, BEFORE, 3.0)

        self.assertEqual(forward.anchor, anchor)
        self.assertEqual(forward.requested_bound, anchor + 3.0)
        self.assertEqual(forward.covered_lo, anchor)
        self.assertEqual(forward.covered_hi, anchor + 3.0)
        self.assertEqual(backward.requested_bound, anchor - 3.0)
        self.assertEqual(backward.covered_lo, anchor - 3.0)
        self.assertEqual(backward.covered_hi, anchor)

        for record in (forward, backward):
            with self.subTest(direction=record.direction):
                self.assertEqual(record.kernel, PRIMARY)
                self.assertEqual(record.ephemeris_role, "primary")
                self.assertEqual(record.event_threshold_degrees, THRESHOLD_DEGREES)
                self.assertEqual(record.event_convention, EVENT_CONVENTION)

    def test_the_record_carries_no_observer_and_no_calendar(self):
        fields = set(astronomy_solver.SolarCrossing.__dataclass_fields__)
        for absent in ("latitude", "longitude", "observer", "weekday", "hpc",
                       "rotation", "regime", "polar", "sabbath"):
            with self.subTest(field=absent):
                self.assertFalse(any(absent in name for name in fields))


if __name__ == "__main__":
    unittest.main()
