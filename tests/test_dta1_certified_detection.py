"""DT-A1 verification for certified sunset-event detection.

Covers the DT-A1 block of astronomy_solver and its two live consumers,
count_sunsets_in_interval (/sunset-count) and find_sunset_from_instant
(/sunset-event-after): the governed DT-A2 / DT-A2C constants, the 19 Delta-T
value knots of the pinned runtime, the interval certification, the ambiguity
outcomes, the mandatory 65N / astronomical -4018 regression (356 -> 365), the
preserved 56-row known-undercount corpus, preservation of every event the root
finder already reported, exactly-once ownership across a kernel seam, and
truthful completeness.

Test framework: standard-library unittest.
Scientific execution uses the repository's certified Skyfield runtime and the
real pinned Authority ephemeris artifacts.

INVOCATION
----------
Run from the repository root with the repository interpreter:

    .\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v

INDEPENDENT ORACLE DISCIPLINE
-----------------------------

The governed constants, the knot years and the corpus are declared here as
test-local literals. They are deliberately NOT imported from astronomy_solver:
importing a constant to check that same constant would make the test agree
with a defective value instead of detecting it.

PROVENANCE OF THE CORPUS
------------------------

KNOWN_UNDERCOUNTS is the DT-O3 Authority defect ledger (56 rows, longitude 0):
every row where the pre-DT-A1 Authority count disagreed with the DT-O3
certified per-rotation count. It was re-confirmed under the governed Skyfield
1.54 runtime by DT-A1 / DT-A2 (168/168 counts, 56/56 recovered). Each row is
(astronomical year, latitude, pre-DT-A1 Authority count, certified count),
counted over the governed contract (spring crossing of the year, spring
crossing of the next year].
"""

import math
import unittest

import numpy
from skyfield import almanac
from skyfield.api import wgs84
from skyfield.functions import load_bundled_npy

import astronomy_solver
from astronomy_solver import (
    SunsetChronologyError,
    count_sunsets_in_interval,
    find_solar_longitude_event_in_year,
    find_sunset_from_instant,
    supported_search_frontier,
    ts,
)

# --- Governed constants (DT-A2 / DT-A2C), restated as literals ---------------

EPSILON_G = 1e-10
BOUNDARY_GUARD = 2e-10
BAND_DEGREES = 1e-6
W_SEED_DAYS = 2.0 ** -5
W_MIN_DAYS = 2.0 ** -16
L1_COEFFICIENTS = (6.5, 0.02)
L2_COEFFICIENTS = (42.0, 0.02)
J0_COEFFICIENT = 7.292122438822127e-08
J1_COEFFICIENT = 3.9912377712236665e-08
SLOPE_BREAKS = 2
# Ratified by DT-A2K-R: every routed spline boundary with an analytic value
# jump above 1e-9 s, selected without any slope pre-filter (19, not 17).
DT_KNOT_YEARS = (
    -100.0, 1150.0, 1650.0, 1720.0, 1830.0, 1850.0, 1855.0, 1865.0, 1875.0,
    1890.0, 1895.0, 1905.0, 1915.0, 1920.0, 1935.0, 1940.0, 1945.0, 1950.0,
    1965.0,
)
THRESHOLD_DEGREES = -0.8333

# The certified root finder's convergence width (Skyfield EPSILON).
FIND_DISCRETE_EPSILON_DAYS = 0.001 / 86400.0

# --- Evidence --------------------------------------------------------------

MANDATORY_CASE = dict(year=-4018, latitude=65.0, longitude=0.0,
                      before=356, certified=365, recovered=9)

KNOWN_UNDERCOUNTS = (
    (-4018, -84.0, 59, 60), (-4018, -82.0, 81, 82), (-4018, -79.0, 113, 114),
    (-4018, -77.0, 135, 136), (-4018, -76.0, 147, 148),
    (-4018, -75.0, 159, 160), (-4018, -72.0, 199, 200),
    (-4018, -70.0, 231, 232), (-4018, -69.0, 249, 250),
    (-4018, -68.0, 271, 272), (-4018, -66.0, 333, 334),
    (-4018, 65.0, 356, 365), (-4018, 66.0, 332, 333), (-4018, 67.0, 300, 302),
    (-4018, 68.0, 270, 272), (-4018, 69.0, 249, 250), (-4018, 70.0, 231, 232),
    (-4018, 73.0, 185, 186), (-4018, 82.0, 81, 82),
    (2019, -84.0, 61, 62), (2019, -80.0, 105, 106), (2019, -79.0, 116, 117),
    (2019, -77.0, 139, 140), (2019, -74.0, 177, 178), (2019, -73.0, 192, 193),
    (2019, -71.0, 223, 224), (2019, -70.0, 241, 242), (2019, -68.0, 288, 289),
    (2019, -67.0, 328, 329), (2019, -66.0, 347, 348), (2019, 66.0, 344, 347),
    (2019, 67.0, 325, 327), (2019, 68.0, 286, 288), (2019, 70.0, 239, 241),
    (2019, 71.0, 222, 223), (2019, 72.0, 205, 207), (2019, 74.0, 176, 177),
    (2019, 84.0, 61, 62),
    (10000, -85.0, 52, 53), (10000, -82.0, 85, 86), (10000, -76.0, 157, 158),
    (10000, -74.0, 184, 185), (10000, -71.0, 233, 234),
    (10000, -70.0, 252, 254), (10000, -69.0, 277, 280),
    (10000, -68.0, 321, 322), (10000, -67.0, 339, 340),
    (10000, 67.0, 340, 342), (10000, 69.0, 279, 280), (10000, 71.0, 233, 235),
    (10000, 72.0, 215, 217), (10000, 74.0, 184, 185), (10000, 76.0, 157, 158),
    (10000, 77.0, 144, 145), (10000, 78.0, 132, 133), (10000, 80.0, 109, 110),
)

# DT-A1 tangency evidence: at this latitude, longitude 0, the deepest 2019
# northern-midsummer night inside the June solstice +- 2 day window just
# vanishes (minimum altitude == threshold).
TANGENT_LATITUDE = 65.74293204295881

NEW_YORK = (40.7406, -73.9586)


def spring(year):
    return find_solar_longitude_event_in_year(year, "SOLAR_LONGITUDE_000").tt


def predicate(frontier, latitude, longitude):
    return almanac.sunrise_sunset(
        astronomy_solver.load_kernel(frontier.kernel),
        wgs84.latlon(latitude, longitude),
    )


def root_finder_sunsets(tt_lo, tt_hi, latitude, longitude):
    """The pre-DT-A1 enumeration: the certified root finder alone."""
    frontier = supported_search_frontier(tt_lo, tt_hi, latitude, longitude)
    times, events = almanac.find_discrete(
        ts.tt_jd(frontier.tt_lo), ts.tt_jd(frontier.tt_hi),
        predicate(frontier, latitude, longitude),
    )
    return frontier, [
        float(t.tt) for t, up in zip(times, events)
        if not bool(up) and frontier.tt_lo < float(t.tt) <= frontier.tt_hi
    ]


def certified_sunsets(tt_lo, tt_hi, latitude, longitude):
    """The production DT-A1 event list over the same admitted frontier."""
    frontier, reported = root_finder_sunsets(tt_lo, tt_hi, latitude, longitude)
    events, reason = astronomy_solver._certified_sunsets_in_frontier(
        frontier, latitude, longitude,
        predicate(frontier, latitude, longitude), reported,
    )
    return frontier, reported, list(events), reason


# --- 1. Governed constants --------------------------------------------------


class TestGovernedConstants(unittest.TestCase):
    def test_constants_are_the_governed_values(self):
        self.assertEqual(astronomy_solver.DETECTION_EPSILON_G, EPSILON_G)
        self.assertEqual(astronomy_solver.DETECTION_BOUNDARY_GUARD, BOUNDARY_GUARD)
        self.assertEqual(astronomy_solver.DETECTION_BAND_DEGREES, BAND_DEGREES)
        self.assertEqual(astronomy_solver.DETECTION_W_SEED_DAYS, W_SEED_DAYS)
        self.assertEqual(astronomy_solver.DETECTION_W_MIN_DAYS, W_MIN_DAYS)
        self.assertEqual(astronomy_solver.DETECTION_L1_COEFFICIENTS, L1_COEFFICIENTS)
        self.assertEqual(astronomy_solver.DETECTION_L2_COEFFICIENTS, L2_COEFFICIENTS)
        self.assertEqual(astronomy_solver.DETECTION_J0_COEFFICIENT, J0_COEFFICIENT)
        self.assertEqual(astronomy_solver.DETECTION_J1_COEFFICIENT, J1_COEFFICIENT)
        self.assertEqual(astronomy_solver.DETECTION_SLOPE_BREAKS, SLOPE_BREAKS)
        self.assertEqual(astronomy_solver.DETECTION_DT_KNOT_YEARS, DT_KNOT_YEARS)

    def test_bounds_are_evaluated_from_the_governed_form(self):
        for latitude in (0.0, 40.0, 65.0, -89.5, 90.0):
            c = abs(math.cos(math.radians(latitude)))
            self.assertEqual(astronomy_solver.detection_l1(latitude), 6.5 * c + 0.02)
            self.assertEqual(astronomy_solver.detection_l2(latitude), 42.0 * c + 0.02)

    def test_band_is_one_microdegree_in_g_units(self):
        s0 = math.sin(math.radians(THRESHOLD_DEGREES))
        band = min(
            abs(math.sin(math.radians(THRESHOLD_DEGREES + s * BAND_DEGREES)) - s0)
            for s in (1.0, -1.0)
        )
        self.assertEqual(astronomy_solver._DETECTION_BAND_G, band)
        self.assertAlmostEqual(band, 1.745145e-8, delta=1e-13)

    def test_seed_satisfies_lemma_s(self):
        self.assertLess(W_SEED_DAYS, 0.2903 / (6.5 + 0.02))


# --- 2. Delta-T knots of the pinned runtime ---------------------------------


class TestDeltaTKnots(unittest.TestCase):
    def test_every_knot_is_a_one_millisecond_value_jump(self):
        for year in DT_KNOT_YEARS:
            tt = 1721045.0 + 365.25 * year
            jump = float(ts.tt_jd(tt + 1e-6).delta_t - ts.tt_jd(tt - 1e-6).delta_t)
            with self.subTest(year=year):
                self.assertGreater(abs(jump), 0.9e-3)

    def test_no_other_s15_boundary_is_a_value_jump(self):
        """The knot list is complete against the runtime's own table."""
        table = load_bundled_npy("delta_t.npz")["Table-S15.2020.txt"]
        iers_start_year = (float(ts.delta_t_table[0][0]) - 1721045.0) / 365.25
        jumps = []
        for boundary in sorted(set(float(x) for x in table[1])):
            if not boundary < iers_start_year:
                continue
            tt = 1721045.0 + 365.25 * boundary
            jump = float(ts.tt_jd(tt + 1e-6).delta_t - ts.tt_jd(tt - 1e-6).delta_t)
            if abs(jump) > 1e-7:
                jumps.append(boundary)
        self.assertEqual(tuple(jumps), DT_KNOT_YEARS)


# --- 3. Interval certification (synthetic margins) --------------------------


class TestIntervalCertification(unittest.TestCase):
    """The structure applies the DT-A2 theorems to whatever margin it is given.

    Synthetic margins isolate each outcome; the astronomy is tested below.
    """

    LO, HI = 2460000.0, 2460001.0

    def structure(self, margin, latitude=40.0):
        return astronomy_solver._certified_sunset_structure(
            margin, latitude, self.LO, self.HI, BOUNDARY_GUARD
        )

    def test_one_monotone_crossing_is_one_setting_bracket(self):
        middle = self.LO + 0.4
        result = self.structure(lambda x: -0.5 * (numpy.asarray(x, float) - middle))
        self.assertEqual(len(result.setting), 1)
        self.assertEqual(result.rising, ())
        self.assertEqual(result.ambiguities, ())
        lo, hi = result.setting[0]
        self.assertLess(lo, middle)
        self.assertLess(middle, hi)

    def test_a_clear_interval_has_no_bracket(self):
        result = self.structure(lambda x: 0.3 + 0.0 * numpy.asarray(x, float))
        self.assertEqual((result.setting, result.rising, result.ambiguities), ((), (), ()))

    def test_a_tangency_inside_the_band_is_ambiguous_not_classified(self):
        middle = self.LO + 0.4
        result = self.structure(
            lambda x: 1e-9 + 2.0 * (numpy.asarray(x, float) - middle) ** 2
        )
        self.assertEqual(result.setting, ())
        self.assertEqual(result.rising, ())
        self.assertTrue(result.ambiguities)
        self.assertTrue(all(kind.startswith("AMBIGUOUS") for kind, _, _ in result.ambiguities))

    def test_an_indeterminate_boundary_is_ambiguous(self):
        result = self.structure(lambda x: numpy.asarray(x, float) - self.LO)
        self.assertEqual(result.ambiguities[0][0], "AMBIGUOUS_BOUNDARY")
        self.assertEqual(result.setting, ())

    def test_no_certification_crosses_a_delta_t_knot(self):
        knot = 1721045.0 + 365.25 * 1850.0

        def margin(x):
            x = numpy.asarray(x, dtype=float)
            return numpy.where(numpy.abs(x - knot) < 2e-6, 1e-12, 0.3)

        result = astronomy_solver._certified_sunset_structure(
            margin, 40.0, knot - 1.0, knot + 1.0, BOUNDARY_GUARD
        )
        kinds = [kind for kind, _, _ in result.ambiguities]
        self.assertIn("AMBIGUOUS_DT_KNOT", kinds)
        self.assertEqual(result.setting, ())


# --- 3b. Delta-T knot adversarial regression (DT-A2K-R) ---------------------


def _setting_observer_longitude(margin_at, t_eval, target, centre, half_width=0.05):
    """Bisect the longitude at which margin_at(longitude, t_eval) == target.

    Deterministic: a fixed bracket around the DT-A2K-R provenance longitude
    and a fixed number of halvings.
    """
    lo, hi = centre - half_width, centre + half_width
    f_lo = margin_at(lo, t_eval) - target
    f_hi = margin_at(hi, t_eval) - target
    assert (f_lo >= 0) != (f_hi >= 0), "provenance bracket does not straddle"
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        f_mid = margin_at(mid, t_eval) - target
        if (f_mid >= 0) == (f_lo >= 0):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class TestDeltaTKnotAdversarial(unittest.TestCase):
    """The 1895 knot can manufacture a second sunset within a millisecond.

    At the 1895 knot the pinned Delta-T steps by +1 ms, so the Earth's
    rotation effectively steps back by 1 ms. For a setting observer whose
    altitude margin sits half the resulting G jump below zero just before the
    switch, the model sets, rises AT the switch, and sets again about half a
    millisecond later: two genuine sets of the certified predicate. Under the
    old 17-knot inventory the smooth-interval theorem was applied across this
    knot and certified ONE set with no ambiguity (DT-A2K-R). With the knot
    partitioned, the detector must fail closed.

    Construction (DT-A2K-R adversarial3, flip case): latitude 40.7406, the
    longitude solved so that G at the last representable instant before the
    switch equals -J/2, J being the measured jump in G across the switch.
    """

    LATITUDE = 40.7406
    PROVENANCE_LONGITUDE_1895 = 160.48564904892623
    PROVENANCE_LONGITUDE_1945 = -158.8306544325173

    @classmethod
    def setUpClass(cls):
        cls.knot = 1721045.0 + 365.25 * 1895.0
        cls.kernel = astronomy_solver.select_kernel_for_interval(
            cls.knot - 1.0, cls.knot + 1.0
        )
        cls.left = math.nextafter(cls.knot, -math.inf)

        def margin_at(longitude, tt):
            margin = astronomy_solver._detection_margin(
                cls.kernel, cls.LATITUDE, longitude
            )
            return float(margin([tt])[0])

        cls.margin_at = staticmethod(margin_at)
        root = _setting_observer_longitude(
            margin_at, cls.left, 0.0, cls.PROVENANCE_LONGITUDE_1895
        )
        cls.jump = margin_at(root, cls.knot) - margin_at(root, cls.left)
        cls.flip = _setting_observer_longitude(
            margin_at, cls.left, -cls.jump / 2.0, cls.PROVENANCE_LONGITUDE_1895
        )
        cls.outside = _setting_observer_longitude(
            margin_at, cls.knot - 2.3e-5, 0.0, cls.PROVENANCE_LONGITUDE_1895
        )

    def test_the_knot_is_a_one_millisecond_switch_and_is_partitioned(self):
        step = float(ts.tt_jd(self.knot).delta_t - ts.tt_jd(self.left).delta_t)
        self.assertAlmostEqual(step, 1e-3, delta=1e-9)
        self.assertGreater(self.jump, 0.0)
        self.assertIn(self.knot, astronomy_solver.DETECTION_DT_KNOT_TT)
        self.assertIn(
            1721045.0 + 365.25 * 1945.0, astronomy_solver.DETECTION_DT_KNOT_TT
        )

    def test_the_construction_reproduces_the_dt_a2k_r_observer(self):
        self.assertAlmostEqual(self.flip, self.PROVENANCE_LONGITUDE_1895, delta=1e-6)

    def test_the_model_really_sets_twice_within_a_millisecond(self):
        """Every representable instant within 400 ulps either side."""
        grid = [self.left]
        for _ in range(400):
            grid.insert(0, math.nextafter(grid[0], -math.inf))
        grid.append(self.knot)
        for _ in range(400):
            grid.append(math.nextafter(grid[-1], math.inf))
        margin = astronomy_solver._detection_margin(self.kernel, self.LATITUDE, self.flip)
        up = margin(grid) >= 0
        kinds = ["set" if up[i] else "rise" for i in range(len(grid) - 1) if up[i] != up[i + 1]]
        self.assertEqual(kinds, ["set", "rise", "set"])

    def test_the_count_fails_closed_at_the_knot(self):
        record = count_sunsets_in_interval(
            self.knot - 0.25, self.knot + 0.25, self.LATITUDE, self.flip
        )
        self.assertFalse(record.complete)
        self.assertEqual(record.truncation_reason, "SUNSET_DETECTION_AMBIGUOUS_DT_KNOT")

    def test_the_directional_search_fails_closed_at_the_knot(self):
        with self.assertRaises(SunsetChronologyError) as raised:
            find_sunset_from_instant(self.knot - 0.25, self.LATITUDE, self.flip)
        self.assertEqual(raised.exception.reason, "SUNSET_DETECTION_AMBIGUOUS_DT_KNOT")

    def test_the_knot_is_isolated_not_searched_through(self):
        margin = astronomy_solver._detection_margin(self.kernel, self.LATITUDE, self.flip)
        structure = astronomy_solver._certified_sunset_structure(
            margin, self.LATITUDE, self.knot - 0.25, self.knot + 0.25, BOUNDARY_GUARD
        )
        self.assertIn(
            ("AMBIGUOUS_DT_KNOT", self.knot - 1e-6, self.knot + 1e-6),
            structure.ambiguities,
        )
        for lo, hi in structure.setting + structure.rising:
            self.assertFalse(lo <= self.knot <= hi)

    def test_a_crossing_outside_the_knot_window_still_certifies(self):
        record = count_sunsets_in_interval(
            self.knot - 0.25, self.knot + 0.25, self.LATITUDE, self.outside
        )
        self.assertTrue(record.complete)
        self.assertIsNone(record.truncation_reason)
        self.assertEqual(record.count, 1)

    def test_the_1945_negative_step_is_likewise_isolated(self):
        knot = 1721045.0 + 365.25 * 1945.0
        record = count_sunsets_in_interval(
            knot - 0.25, knot + 0.25, self.LATITUDE, self.PROVENANCE_LONGITUDE_1945
        )
        self.assertFalse(record.complete)
        self.assertEqual(record.truncation_reason, "SUNSET_DETECTION_AMBIGUOUS_DT_KNOT")


# --- 4. Mandatory regression: 65N, astronomical -4018 -----------------------


class TestMandatoryRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        case = MANDATORY_CASE
        cls.lo, cls.hi = spring(case["year"]), spring(case["year"] + 1)
        cls.frontier, cls.reported, cls.events, cls.reason = certified_sunsets(
            cls.lo, cls.hi, case["latitude"], case["longitude"]
        )
        cls.record = count_sunsets_in_interval(
            cls.lo, cls.hi, case["latitude"], case["longitude"]
        )

    def test_the_root_finder_alone_reports_356(self):
        self.assertEqual(len(self.reported), MANDATORY_CASE["before"])

    def test_the_production_count_is_365_and_complete(self):
        self.assertEqual(self.record.count, MANDATORY_CASE["certified"])
        self.assertTrue(self.record.complete)
        self.assertIsNone(self.record.truncation_reason)
        self.assertIsNone(self.reason)

    def test_exactly_nine_sunsets_are_recovered(self):
        recovered = sorted(set(self.events) - set(self.reported))
        self.assertEqual(len(recovered), MANDATORY_CASE["recovered"])

    def test_every_reported_event_keeps_its_bits(self):
        self.assertTrue(set(self.reported) <= set(self.events))

    def test_events_are_strictly_ordered_and_unique(self):
        self.assertEqual(len(set(self.events)), len(self.events))
        self.assertTrue(all(b > a for a, b in zip(self.events, self.events[1:])))
        self.assertTrue(all(self.lo < t <= self.hi for t in self.events))

    def test_every_recovered_event_is_a_genuine_setting_transition(self):
        up = predicate(self.frontier, MANDATORY_CASE["latitude"], MANDATORY_CASE["longitude"])
        for tt in sorted(set(self.events) - set(self.reported)):
            with self.subTest(tt=tt):
                self.assertTrue(bool(up(ts.tt_jd(tt - FIND_DISCRETE_EPSILON_DAYS))))
                self.assertFalse(bool(up(ts.tt_jd(tt))))

    def test_the_directional_walk_no_longer_skips_a_night(self):
        """Sunset to sunset through the recovered nights, as the Engine walks."""
        recovered = sorted(set(self.events) - set(self.reported))
        latitude, longitude = MANDATORY_CASE["latitude"], MANDATORY_CASE["longitude"]
        walk = [find_sunset_from_instant(recovered[0] - 3.0, latitude, longitude).tt]
        while walk[-1] < recovered[-1] + 2.0:
            walk.append(find_sunset_from_instant(walk[-1], latitude, longitude).tt)
        gaps = [(b - a) * 24.0 for a, b in zip(walk, walk[1:])]
        self.assertTrue(all(20.0 <= g <= 28.0 for g in gaps), gaps)
        self.assertEqual(
            sum(any(abs(r - w) < 1e-6 for w in walk) for r in recovered),
            len(recovered),
        )


# --- 5. The preserved 56-row corpus -----------------------------------------


class TestKnownUndercountCorpus(unittest.TestCase):
    def test_every_known_undercount_is_recovered(self):
        self.assertEqual(len(KNOWN_UNDERCOUNTS), 56)
        for year, latitude, before, certified in KNOWN_UNDERCOUNTS:
            with self.subTest(year=year, latitude=latitude):
                record = count_sunsets_in_interval(
                    spring(year), spring(year + 1), latitude, 0.0
                )
                self.assertEqual(record.count, certified)
                self.assertTrue(record.complete)
                self.assertGreater(certified, before)


# --- 6. Ambiguity at a constructed tangency ---------------------------------


class TestTangency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        solstice = find_solar_longitude_event_in_year(2019, "SOLAR_LONGITUDE_090").tt
        cls.lo, cls.hi = solstice - 2.0, solstice + 2.0

    def outcome(self, delta):
        record = count_sunsets_in_interval(
            self.lo, self.hi, TANGENT_LATITUDE + delta, 0.0
        )
        try:
            event = find_sunset_from_instant(self.lo, TANGENT_LATITUDE + delta, 0.0)
            walk = "FOUND" if event is not None else "NONE_CERTIFIED"
        except SunsetChronologyError as error:
            walk = error.reason
        return record, walk

    def test_a_clearly_vanished_night_is_certified_absent(self):
        record, walk = self.outcome(+3e-6)
        self.assertEqual(record.count, 0)
        self.assertTrue(record.complete)
        self.assertEqual(walk, "NONE_CERTIFIED")

    def test_a_clearly_present_night_is_certified_found(self):
        record, walk = self.outcome(-3e-6)
        self.assertEqual(record.count, 1)
        self.assertTrue(record.complete)
        self.assertEqual(walk, "FOUND")

    def test_the_tangency_itself_is_ambiguous_never_forced(self):
        for delta in (1e-7, 0.0):
            with self.subTest(delta=delta):
                record, walk = self.outcome(delta)
                self.assertFalse(record.complete)
                self.assertEqual(record.truncation_reason, "SUNSET_DETECTION_AMBIGUOUS")
                self.assertEqual(walk, "SUNSET_DETECTION_AMBIGUOUS")


# --- 7. Ordinary non-regression and exactly-once ownership ------------------


class TestOrdinaryNonRegression(unittest.TestCase):
    def test_ordinary_event_identity_is_bit_identical(self):
        for name, (latitude, longitude), year in (
            ("New York", NEW_YORK, 2019),
            ("Quito", (-0.1807, -78.4678), 2019),
            ("Jerusalem", (31.7683, 35.2137), -1),
            ("40S", (-40.0, 60.0), 5000),
        ):
            with self.subTest(observer=name, year=year):
                _, reported, events, reason = certified_sunsets(
                    spring(year), spring(year + 1), latitude, longitude
                )
                self.assertEqual(events, reported)
                self.assertIsNone(reason)

    def test_the_directional_answer_is_unchanged_where_it_was_correct(self):
        anchor = spring(2019)
        frontier = supported_search_frontier(
            anchor, anchor + astronomy_solver.SUCCESSOR_SEARCH_SPAN_DAYS, *NEW_YORK
        )
        times, events = almanac.find_discrete(
            ts.tt_jd(frontier.tt_lo), ts.tt_jd(frontier.tt_hi),
            predicate(frontier, *NEW_YORK),
        )
        first = next(float(t.tt) for t, up in zip(times, events)
                     if not bool(up) and float(t.tt) > anchor)
        self.assertEqual(find_sunset_from_instant(anchor, *NEW_YORK).tt, first)

    def test_counts_tile_exactly_once_across_a_kernel_seam(self):
        middle = spring(1550)
        lo, hi = middle - 165.0, middle + 100.0
        seam = astronomy_solver.kernel_coverage_tt("de440.bsp").tt_start
        self.assertLess(lo, seam)
        self.assertLess(seam, hi)
        first = count_sunsets_in_interval(lo, middle, *NEW_YORK)
        second = count_sunsets_in_interval(middle, hi, *NEW_YORK)
        whole = count_sunsets_in_interval(lo, hi, *NEW_YORK)
        self.assertNotEqual(first.kernel, second.kernel)
        self.assertEqual(first.count + second.count, whole.count)


if __name__ == "__main__":
    unittest.main()
