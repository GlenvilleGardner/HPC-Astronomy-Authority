"""ABC-1 Operation 1 verification for the bulk observer-local sunset count.

Covers count_sunsets_in_interval and SunsetCount in astronomy_solver: the
half-open interval contract, exact-state validation, the governed span
admission bound, reuse of the published frontier substrate, kernel truncation,
boundary coincidence at binary64 equality, the complete zero count,
determinism, semantic equivalence with the certified successor enumeration,
sampling characterization against an independent finer grid, and the
structural controls that keep arithmetic shortcuts and event identity out.

Test framework: standard-library unittest.
Scientific execution uses the repository's certified Skyfield runtime and the
real pinned Authority ephemeris artifacts.

INVOCATION
----------
Run from the repository root with the repository interpreter:

    .\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v

TWO CLASSES OF EVIDENCE, DELIBERATELY SEPARATE
----------------------------------------------

CLASS A - SEMANTIC EQUIVALENCE. The bulk integer count is compared against the
published certified successor enumeration, walked one sunset at a time, over
territory where that walk is valid. The assertion is on INTEGERS ONLY. The two
paths solve their roots under different brackets and the Authority does not
guarantee those roots carry the same bits, so no root is ever compared to
another root here - not by equality, not by tolerance, not at all.

CLASS B - SAMPLING CHARACTERIZATION. The production grid is compared against a
test-local finer grid built from the same certified predicate. This is a
CHARACTERIZATION oracle, not a completeness proof: it can show that the
production grid missed a state longer than its own spacing, and it can show
nothing about states shorter than that. It shares the production predicate's
blind spot at a smaller scale, and the report says so rather than implying a
guarantee. The successor walk is NOT usable for this purpose because it reads
the same step_days the production count does.

The two are never mixed. A Class-A agreement says the count matches the
certified path; it says nothing about whether the certified path is complete.

INDEPENDENT ORACLE DISCIPLINE
-----------------------------

Stable reason codes, pinned artifact filenames, the governed span bound, the
expected field set and the observer inventory are declared here as test-local
literals. They are deliberately NOT imported from astronomy_solver: importing
a constant to check that same constant would make the test agree with a
defective value instead of detecting it.
"""

import inspect
import math
import re
import unittest
from unittest import mock

from skyfield import almanac
from skyfield.api import wgs84
from skyfield.searchlib import find_discrete

import astronomy_solver
from astronomy_solver import (
    SunsetCount,
    count_sunsets_in_interval,
    find_sunset_from_instant,
    ts,
)
from scientific_environment import ScientificEnvironmentError


# --- Test-local literals. Never imported from the module under test. -------

REASON_INTERVAL_TOO_LONG = "COUNT_INTERVAL_TOO_LONG"
REASON_STATE_INVALID = "INSTANT_STATE_INVALID"
REASON_OBSERVER_OUT_OF_DOMAIN = "OBSERVER_OUT_OF_DOMAIN"
REASON_COVERAGE_EXHAUSTED = "EPHEMERIS_COVERAGE_EXHAUSTED"

MAX_SPAN_DAYS = 400.0
PRIMARY_ARTIFACT = "de440.bsp"

EXPECTED_FIELDS = (
    "count",
    "requested_lo",
    "requested_hi",
    "covered_lo",
    "covered_hi",
    "complete",
    "truncation_reason",
    "boundary_coincident",
    "kernel",
)

# The production sampling grid, restated here rather than read from Skyfield,
# so a change to it is detected instead of tracked.
PRODUCTION_STEP_DAYS = 0.04

# The characterization grid. Finer than production by a factor of forty.
FINE_STEP_DAYS = 0.001

# Observers inside the currently constructible calendar domain.
NEW_YORK = (40.7128, -74.0060)
SYDNEY = (-33.8688, 151.2093)
EQUATOR = (0.0, 0.0)
LAPLAND_65N = (65.0, 20.0)
USHUAIA = (-54.8019, -68.3030)
TOKYO = (35.6762, 139.6503)

CONSTRUCTIBLE_OBSERVERS = (
    ("New York", NEW_YORK),
    ("Sydney", SYDNEY),
    ("Equator", EQUATOR),
    ("Lapland 65N", LAPLAND_65N),
    ("Ushuaia", USHUAIA),
    ("Tokyo", TOKYO),
)

# Observers ABOVE the constructible calendar domain, where the production grid
# is known to resolve fewer transitions than a finer one. Pinned so the
# limitation is recorded rather than discovered later.
POLAR_TRANSITION_OBSERVERS = (
    ("68N", (68.0, 20.0)),
    ("68S", (-68.0, 20.0)),
)


def tt_of(year, month, day):
    return float(ts.utc(year, month, day).tt)


def sunsets_on_grid(latitude, longitude, tt_lo, tt_hi, step_days):
    """Sunset states in [tt_lo, tt_hi], enumerated on a caller-chosen grid.

    Test-local. It wraps the SAME certified predicate the solver uses and
    changes only the sampling metadata find_discrete reads, which is how the
    repository already bounds a search grid elsewhere. No alternate
    astronomical model is introduced, and nothing here reaches production.
    """
    base = almanac.sunrise_sunset(
        astronomy_solver.load_kernel(PRIMARY_ARTIFACT),
        wgs84.latlon(latitude, longitude),
    )

    def sampled(t, _base=base):
        return _base(t)

    sampled.step_days = step_days

    times, events = find_discrete(ts.tt_jd(tt_lo), ts.tt_jd(tt_hi), sampled)

    return [
        float(t.tt)
        for t, sun_is_up in zip(times, events)
        if not bool(sun_is_up)
    ]


def walked_sunsets(latitude, longitude, tt_lo, tt_hi):
    """Sunsets in (tt_lo, tt_hi], walked with the certified successor solver.

    This is the Class-A oracle. It is the published directional operation,
    stepped one sunset at a time exactly as a consumer would, and it is
    deliberately a completely different traversal from the single bulk scan.

    Returns None when the walk cannot complete - at high latitude the governed
    three-day directional horizon legitimately reports an absence - so a
    caller can tell "the oracle disagrees" from "the oracle does not apply".
    """
    walked = []
    cursor = tt_lo

    for _ in range(600):
        event = find_sunset_from_instant(cursor, latitude, longitude)

        if event is None:
            return None

        if event.tt > tt_hi:
            return walked

        walked.append(event.tt)
        cursor = event.tt

    raise AssertionError("successor walk did not terminate")


class SunsetCountAssertions(unittest.TestCase):
    """Shared structural assertions for every returned count."""

    def assert_well_formed(self, record):
        self.assertIsInstance(record, SunsetCount)
        self.assertIsInstance(record.count, int)
        self.assertGreaterEqual(record.count, 0)
        self.assertIsInstance(record.complete, bool)
        self.assertIsInstance(record.boundary_coincident, bool)
        self.assertIsInstance(record.kernel, str)

        # Completeness and its reason are one statement, never two.
        if record.complete:
            self.assertIsNone(record.truncation_reason)
        else:
            self.assertIsInstance(record.truncation_reason, str)

        # The examined territory never exceeds, and never leaves, what was
        # asked for; the lower bound is never moved.
        self.assertEqual(record.covered_lo, record.requested_lo)
        self.assertLessEqual(record.covered_hi, record.requested_hi)

        if record.complete:
            self.assertEqual(record.covered_hi, record.requested_hi)


# --- 1. The interval contract ----------------------------------------------


class TestIntervalContract(SunsetCountAssertions):
    """tt_lo < sunset.tt <= tt_hi, by exact binary64 comparison."""

    def test_counts_a_known_year_for_a_reference_observer(self):
        lo, hi = tt_of(2027, 3, 21), tt_of(2028, 3, 21)
        record = count_sunsets_in_interval(lo, hi, *NEW_YORK)

        self.assert_well_formed(record)
        self.assertTrue(record.complete)
        self.assertEqual(record.kernel, PRIMARY_ARTIFACT)

        # Non-vacuity: a whole year really was counted, not an empty interval.
        self.assertGreater(record.count, 300)

    def test_lower_bound_is_exclusive_and_upper_bound_inclusive(self):
        """Place the bounds exactly on two determined sunsets.

        The lower one must be excluded and the upper one included. Both roots
        are obtained from the published solver, so this exercises the
        comparison the solver actually performs rather than a contrived value.
        """
        anchor = tt_of(2027, 6, 1)
        first = find_sunset_from_instant(anchor, *NEW_YORK)
        second = find_sunset_from_instant(first.tt, *NEW_YORK)
        third = find_sunset_from_instant(second.tt, *NEW_YORK)

        # (first, third] should hold exactly the second and the third, PROVIDED
        # the bulk scan re-reports those roots at the same bits. That is not
        # guaranteed across brackets, so the assertion is bounded rather than
        # exact: the count is 2 or differs by at most one boundary root.
        record = count_sunsets_in_interval(first.tt, third.tt, *NEW_YORK)

        self.assert_well_formed(record)
        self.assertIn(record.count, (2, 3))

        # The decisive, bracket-independent statement: widening the upper
        # bound past a further sunset adds exactly one.
        fourth = find_sunset_from_instant(third.tt, *NEW_YORK)
        wider = count_sunsets_in_interval(first.tt, fourth.tt, *NEW_YORK)

        self.assertEqual(wider.count, record.count + 1)

    def test_adjacent_intervals_tile_without_double_counting(self):
        """The half-open form is what makes composition exact.

        A sunset lying on a shared boundary belongs to the earlier interval
        and to that one only, so two abutting counts sum to the whole.
        """
        lo, mid, hi = tt_of(2027, 4, 1), tt_of(2027, 7, 1), tt_of(2027, 10, 1)

        whole = count_sunsets_in_interval(lo, hi, *NEW_YORK)
        first = count_sunsets_in_interval(lo, mid, *NEW_YORK)
        second = count_sunsets_in_interval(mid, hi, *NEW_YORK)

        for record in (whole, first, second):
            self.assert_well_formed(record)
            self.assertTrue(record.complete)

        self.assertEqual(first.count + second.count, whole.count)
        self.assertGreater(whole.count, 150)

    def test_repeated_calls_are_deterministic(self):
        lo, hi = tt_of(2027, 3, 21), tt_of(2027, 6, 21)
        first = count_sunsets_in_interval(lo, hi, *SYDNEY)
        second = count_sunsets_in_interval(lo, hi, *SYDNEY)

        self.assertEqual(first, second)


# --- 2. Interval and observer validation -----------------------------------


class TestIntervalValidation(unittest.TestCase):
    def test_zero_width_interval_fails_closed(self):
        anchor = tt_of(2027, 3, 21)

        with self.assertRaises(ScientificEnvironmentError) as caught:
            count_sunsets_in_interval(anchor, anchor, *NEW_YORK)

        self.assertEqual(caught.exception.reason, REASON_STATE_INVALID)

    def test_reversed_interval_fails_closed(self):
        lo, hi = tt_of(2027, 3, 21), tt_of(2027, 6, 21)

        with self.assertRaises(ScientificEnvironmentError) as caught:
            count_sunsets_in_interval(hi, lo, *NEW_YORK)

        self.assertEqual(caught.exception.reason, REASON_STATE_INVALID)

    def test_no_epsilon_equality_is_introduced(self):
        """One unit in the last place is a genuine ascending interval."""
        anchor = tt_of(2027, 3, 21)
        just_above = math.nextafter(anchor, math.inf)

        self.assertNotEqual(anchor, just_above)

        record = count_sunsets_in_interval(just_above, anchor + 2.0, *NEW_YORK)

        self.assertIsInstance(record, SunsetCount)

        # ...and the reversed ULP is still reversed.
        with self.assertRaises(ScientificEnvironmentError):
            count_sunsets_in_interval(just_above, anchor, *NEW_YORK)

    def test_non_finite_bounds_fail_closed(self):
        anchor = tt_of(2027, 3, 21)

        for bad in (math.nan, math.inf, -math.inf):
            for pair in ((bad, anchor), (anchor, bad)):
                with self.assertRaises(ScientificEnvironmentError) as caught:
                    count_sunsets_in_interval(*pair, *NEW_YORK)

                self.assertEqual(caught.exception.reason, REASON_STATE_INVALID)

    def test_observer_validation_is_the_substrate_s(self):
        lo, hi = tt_of(2027, 3, 21), tt_of(2027, 4, 21)

        for latitude, longitude in (
            (90.5, 0.0),
            (-90.5, 0.0),
            (0.0, 180.5),
            (0.0, -180.5),
            (math.nan, 0.0),
            (0.0, math.inf),
            (True, 0.0),
        ):
            with self.assertRaises(ScientificEnvironmentError) as caught:
                count_sunsets_in_interval(lo, hi, latitude, longitude)

            self.assertEqual(
                caught.exception.reason, REASON_OBSERVER_OUT_OF_DOMAIN
            )


# --- 3. The governed span admission bound ----------------------------------


class TestResourceBound(SunsetCountAssertions):
    def test_exactly_the_bound_is_admitted(self):
        lo = tt_of(2027, 3, 21)
        record = count_sunsets_in_interval(
            lo, lo + MAX_SPAN_DAYS, *NEW_YORK
        )

        self.assert_well_formed(record)
        self.assertTrue(record.complete)

    def test_just_below_the_bound_is_admitted(self):
        lo = tt_of(2027, 3, 21)
        record = count_sunsets_in_interval(
            lo, lo + MAX_SPAN_DAYS - 0.1, *NEW_YORK
        )

        self.assert_well_formed(record)

    def test_just_above_the_bound_is_refused(self):
        lo = tt_of(2027, 3, 21)

        with self.assertRaises(ScientificEnvironmentError) as caught:
            count_sunsets_in_interval(
                lo, lo + MAX_SPAN_DAYS + 0.1, *NEW_YORK
            )

        self.assertEqual(caught.exception.reason, REASON_INTERVAL_TOO_LONG)

    def test_refusal_costs_no_astronomy(self):
        """Admission is decided before any search is performed."""
        lo = tt_of(2027, 3, 21)
        recorder = mock.Mock()

        with mock.patch.object(
            astronomy_solver.almanac, "find_discrete", recorder
        ):
            with self.assertRaises(ScientificEnvironmentError):
                count_sunsets_in_interval(lo, lo + 500.0, *NEW_YORK)

        recorder.assert_not_called()

    def test_every_governed_year_interval_fits_inside_the_bound(self):
        """The bound admits the interval the intended consumer needs.

        Successive governed spring crossings, measured here rather than
        assumed, must lie comfortably inside the admission bound.
        """
        seasons = almanac.seasons(
            astronomy_solver.load_kernel(PRIMARY_ARTIFACT)
        )

        def spring(year):
            times, events = find_discrete(
                ts.utc(year, 1, 1), ts.utc(year, 12, 31), seasons
            )
            return next(
                float(t.tt)
                for t, event in zip(times, events)
                if int(event) == 0
            )

        for year in (1600, 2000, 2027, 2300, 2600):
            span = spring(year + 1) - spring(year)

            self.assertGreater(span, 365.0)
            self.assertLess(span, MAX_SPAN_DAYS - 30.0)


# --- 4. Boundary coincidence, at exact binary64 equality -------------------


class TestBoundaryCoincidence(unittest.TestCase):
    """Controlled transition evidence.

    A naturally occurring sunset exactly equal to a requested bound cannot be
    relied upon: the bulk scan re-solves its roots under its own bracket and
    the Authority does not guarantee those bits. The transitions are therefore
    supplied directly, so the comparison under test is exercised at exactly
    the states intended and nothing else can explain the outcome.

    DT-A1: a reported sunset is now accepted only inside a certified setting
    bracket, and the real sky has none at a staged state. The certified
    structure is therefore staged consistently with the staged transition -
    one setting bracket ending at it when it lies in (LO, HI], none otherwise -
    so the ownership comparison remains the only thing under test.
    """

    LO = None
    HI = None

    def setUp(self):
        self.LO = tt_of(2027, 3, 21)
        self.HI = tt_of(2027, 3, 25)

    def _staged(self, sunset_tt):
        """Return a find_discrete stub reporting one sunset at sunset_tt."""
        times = ts.tt_jd([sunset_tt])
        events = [False]

        return mock.Mock(return_value=(times, events))

    def _staged_structure(self, sunset_tt=None):
        """Return a certified-structure stub consistent with the staging."""
        setting = ()
        if sunset_tt is not None and self.LO < sunset_tt <= self.HI:
            setting = ((sunset_tt - 0.01, sunset_tt),)

        return mock.Mock(
            return_value=astronomy_solver._SunsetStructure(setting, (), ())
        )

    def _count_with(self, sunset_tt):
        with mock.patch.object(
            astronomy_solver.almanac, "find_discrete", self._staged(sunset_tt)
        ), mock.patch.object(
            astronomy_solver, "_certified_sunset_structure",
            self._staged_structure(sunset_tt),
        ):
            return count_sunsets_in_interval(self.LO, self.HI, *NEW_YORK)

    def test_a_sunset_exactly_on_the_upper_bound_is_counted_and_reported(self):
        record = self._count_with(self.HI)

        self.assertEqual(record.count, 1)
        self.assertTrue(record.boundary_coincident)

    def test_one_ulp_below_the_bound_is_counted_and_is_not_coincidence(self):
        below = math.nextafter(self.HI, -math.inf)

        self.assertNotEqual(below, self.HI)

        record = self._count_with(below)

        self.assertEqual(record.count, 1)
        self.assertFalse(record.boundary_coincident)

    def test_one_ulp_above_the_bound_is_excluded_entirely(self):
        above = math.nextafter(self.HI, math.inf)

        self.assertNotEqual(above, self.HI)

        record = self._count_with(above)

        self.assertEqual(record.count, 0)
        self.assertFalse(record.boundary_coincident)

    def test_a_sunset_exactly_on_the_lower_bound_is_excluded(self):
        record = self._count_with(self.LO)

        self.assertEqual(record.count, 0)
        self.assertFalse(record.boundary_coincident)

    def test_one_ulp_above_the_lower_bound_is_included(self):
        above = math.nextafter(self.LO, math.inf)
        record = self._count_with(above)

        self.assertEqual(record.count, 1)
        self.assertFalse(record.boundary_coincident)

    def test_coincidence_is_not_a_failure(self):
        record = self._count_with(self.HI)

        self.assertTrue(record.complete)
        self.assertIsNone(record.truncation_reason)

    def test_a_sunrise_on_the_bound_is_never_counted(self):
        times = ts.tt_jd([self.HI])
        staged = mock.Mock(return_value=(times, [True]))

        with mock.patch.object(
            astronomy_solver.almanac, "find_discrete", staged
        ), mock.patch.object(
            astronomy_solver, "_certified_sunset_structure",
            self._staged_structure(),
        ):
            record = count_sunsets_in_interval(self.LO, self.HI, *NEW_YORK)

        self.assertEqual(record.count, 0)
        self.assertFalse(record.boundary_coincident)


# --- 5. Complete zero count and truncation ---------------------------------


class TestZeroCountAndTruncation(SunsetCountAssertions):
    def test_a_complete_interval_with_no_sunset_counts_zero(self):
        """Polar night, fully admitted. Zero is the answer, not an error."""
        record = count_sunsets_in_interval(
            tt_of(2027, 12, 5), tt_of(2027, 12, 20), 80.0, 20.0
        )

        self.assert_well_formed(record)
        self.assertEqual(record.count, 0)
        self.assertTrue(record.complete)
        self.assertIsNone(record.truncation_reason)

    def test_zero_over_a_complete_frontier_differs_from_truncation(self):
        """Same integer, different scientific claim."""
        empty = count_sunsets_in_interval(
            tt_of(2027, 12, 5), tt_of(2027, 12, 20), 80.0, 20.0
        )

        self.assertTrue(empty.complete)
        self.assertIsNone(empty.truncation_reason)

    @staticmethod
    def outermost_declared_end():
        """The TT beyond which NO pinned artifact declares coverage.

        Derived from the artifacts themselves rather than stated, because a
        single artifact's end is not a system frontier: another pinned
        artifact may declare the territory beyond it, and the substrate will
        legitimately admit the whole interval under that one.
        """
        return max(
            astronomy_solver.kernel_coverage_tt(name).tt_end
            for name in astronomy_solver.PINNED_KERNEL_PRECEDENCE
        )

    def test_one_artifact_s_end_is_not_a_system_frontier(self):
        """Crossing a coverage boundary is not automatically truncation.

        The primary artifact's declared end lies well inside another pinned
        artifact's coverage, so an interval straddling it is admitted WHOLE
        under that single artifact. This is one frontier under one artifact -
        not two searches stitched together - and it must report complete.
        """
        primary_end = astronomy_solver.kernel_coverage_tt(
            PRIMARY_ARTIFACT
        ).tt_end

        self.assertLess(primary_end, self.outermost_declared_end())

        record = count_sunsets_in_interval(
            primary_end - 30.0, primary_end + 30.0, *NEW_YORK
        )

        self.assert_well_formed(record)
        self.assertTrue(record.complete)
        self.assertIsNone(record.truncation_reason)
        self.assertNotEqual(record.kernel, PRIMARY_ARTIFACT)
        self.assertGreater(record.count, 0)

    def test_an_interval_running_past_all_coverage_truncates(self):
        """No stitching. The count is of the territory actually examined."""
        outer_end = self.outermost_declared_end()

        lo = outer_end - 30.0
        hi = outer_end + 30.0

        record = count_sunsets_in_interval(lo, hi, *NEW_YORK)

        self.assert_well_formed(record)
        self.assertFalse(record.complete)
        self.assertEqual(record.truncation_reason, REASON_COVERAGE_EXHAUSTED)
        self.assertLess(record.covered_hi, record.requested_hi)
        self.assertEqual(record.covered_lo, lo)
        self.assertEqual(record.covered_hi, outer_end)

        # A truncated frontier ends below the requested bound, so nothing
        # inside it can coincide with that bound.
        self.assertFalse(record.boundary_coincident)

        # The partial count is a real count of real territory.
        self.assertGreater(record.count, 0)

    def test_a_truncated_count_is_never_presented_as_complete(self):
        outer_end = self.outermost_declared_end()
        record = count_sunsets_in_interval(
            outer_end - 10.0, outer_end + 10.0, *NEW_YORK
        )

        self.assertFalse(record.complete)
        self.assertIsNotNone(record.truncation_reason)

    def test_a_lower_bound_outside_all_coverage_fails_closed(self):
        """The lower bound is never moved, so there is nothing to shorten."""
        outer_start = min(
            astronomy_solver.kernel_coverage_tt(name).tt_start
            for name in astronomy_solver.PINNED_KERNEL_PRECEDENCE
        )

        with self.assertRaises(ScientificEnvironmentError) as caught:
            count_sunsets_in_interval(
                outer_start - 30.0, outer_start + 30.0, *NEW_YORK
            )

        self.assertEqual(caught.exception.reason, REASON_COVERAGE_EXHAUSTED)


# --- 6. CLASS A - semantic equivalence with the certified walk -------------


class TestClassASemanticEquivalence(SunsetCountAssertions):
    """Integer counts only. No root is ever compared with another root."""

    def _assert_equivalent(self, latitude, longitude, lo, hi):
        walked = walked_sunsets(latitude, longitude, lo, hi)

        if walked is None:
            self.skipTest("successor walk does not apply to this observer")

        record = count_sunsets_in_interval(lo, hi, latitude, longitude)

        self.assert_well_formed(record)
        self.assertTrue(record.complete)
        self.assertEqual(record.count, len(walked))

        return record.count

    def test_agrees_with_the_walk_for_every_constructible_observer(self):
        lo, hi = tt_of(2027, 5, 1), tt_of(2027, 6, 10)

        for name, (latitude, longitude) in CONSTRUCTIBLE_OBSERVERS:
            with self.subTest(observer=name):
                counted = self._assert_equivalent(
                    latitude, longitude, lo, hi
                )

                # Non-vacuity: a real run of sunsets was compared.
                self.assertGreater(counted, 35)

    def test_agrees_across_a_southern_summer_interval(self):
        self._assert_equivalent(
            *USHUAIA, tt_of(2027, 12, 1), tt_of(2028, 1, 15)
        )

    def test_agrees_across_a_deep_time_interval(self):
        self._assert_equivalent(
            *NEW_YORK, tt_of(1650, 5, 1), tt_of(1650, 6, 10)
        )

    def test_reproduces_observer_specific_year_length(self):
        """The same governed year is a different length for two observers.

        Whole-year walks, which is what makes this the decisive Class-A case:
        an implementation that inferred a count from elapsed time rather than
        enumerating transitions would report the same number for both.
        """
        seasons = almanac.seasons(
            astronomy_solver.load_kernel(PRIMARY_ARTIFACT)
        )

        def spring(year):
            times, events = find_discrete(
                ts.utc(year, 1, 1), ts.utc(year, 12, 31), seasons
            )
            return next(
                float(t.tt)
                for t, event in zip(times, events)
                if int(event) == 0
            )

        lo, hi = spring(2027), spring(2028)

        northern = self._assert_equivalent(*NEW_YORK, lo, hi)
        southern = self._assert_equivalent(*SYDNEY, lo, hi)

        self.assertIn(northern, (365, 366))
        self.assertIn(southern, (365, 366))
        self.assertNotEqual(northern, southern)


# --- 7. CLASS B - sampling characterization --------------------------------


class TestClassBSamplingCharacterization(unittest.TestCase):
    """What the production grid resolves, measured against a finer one.

    THIS IS NOT A COMPLETENESS PROOF. The finer grid has the same structural
    blind spot at a smaller scale: the duration of the briefest sun-up state
    is a continuous function of latitude passing through zero, so no finite
    grid resolves every transition over a continuous observer domain. These
    tests record what IS resolved, and pin the known divergence.
    """

    def test_the_fine_oracle_is_genuinely_finer(self):
        self.assertLess(FINE_STEP_DAYS, PRODUCTION_STEP_DAYS)

        base = almanac.sunrise_sunset(
            astronomy_solver.load_kernel(PRIMARY_ARTIFACT),
            wgs84.latlon(*NEW_YORK),
        )

        # The production predicate really does carry the step this suite
        # believes it carries.
        self.assertEqual(base.step_days, PRODUCTION_STEP_DAYS)

    def test_grids_agree_inside_the_constructible_domain(self):
        lo, hi = tt_of(2027, 3, 21), tt_of(2027, 9, 21)

        for name, (latitude, longitude) in CONSTRUCTIBLE_OBSERVERS:
            with self.subTest(observer=name):
                record = count_sunsets_in_interval(
                    lo, hi, latitude, longitude
                )
                fine = sunsets_on_grid(
                    latitude, longitude, lo, hi, FINE_STEP_DAYS
                )
                fine_in_interval = [
                    tt for tt in fine if lo < tt <= hi
                ]

                self.assertTrue(record.complete)
                self.assertEqual(record.count, len(fine_in_interval))
                self.assertGreater(record.count, 150)

    def test_grids_agree_across_longitudes_at_the_domain_ceiling(self):
        lo, hi = tt_of(2027, 3, 21), tt_of(2027, 9, 21)

        for longitude in (-179.0, -74.0, 0.0, 20.0, 90.0, 179.0):
            with self.subTest(longitude=longitude):
                record = count_sunsets_in_interval(lo, hi, 65.0, longitude)
                fine = sunsets_on_grid(65.0, longitude, lo, hi, FINE_STEP_DAYS)

                self.assertEqual(
                    record.count, len([tt for tt in fine if lo < tt <= hi])
                )

    def test_polar_transition_divergence_is_recovered(self):
        """DT-A1 supersedes the pinned divergence this test used to record.

        Before DT-A1 the production count resolved FEWER transitions than the
        finer grid at these observers, because a brief day or night shorter
        than the 0.04-day grid fell between two samples, and the count still
        reported itself complete. The certified detection now recovers every
        such crossing, so the production count must be complete and must
        resolve at least every transition the finer grid resolves.

        Non-vacuity: at least one observer must still show the bare
        production-step grid missing transitions the certified count finds,
        or this test no longer exercises the correction it documents.
        """
        lo, hi = tt_of(2027, 3, 21), tt_of(2028, 3, 21)
        recovered = []

        for name, (latitude, longitude) in POLAR_TRANSITION_OBSERVERS:
            record = count_sunsets_in_interval(lo, hi, latitude, longitude)
            fine = [
                tt
                for tt in sunsets_on_grid(
                    latitude, longitude, lo, hi, FINE_STEP_DAYS
                )
                if lo < tt <= hi
            ]
            coarse = [
                tt
                for tt in sunsets_on_grid(
                    latitude, longitude, lo, hi, PRODUCTION_STEP_DAYS
                )
                if lo < tt <= hi
            ]

            self.assertTrue(record.complete)
            self.assertIsNone(record.truncation_reason)
            self.assertGreaterEqual(record.count, len(fine))

            if record.count > len(coarse):
                recovered.append((name, len(coarse), record.count))

        self.assertTrue(
            recovered,
            "no observer shows the production-step grid missing a transition; "
            "the DT-A1 characterization must be revisited",
        )


# --- 8. Structural controls ------------------------------------------------


class TestImplementationDiscipline(unittest.TestCase):
    """Source-level controls over how the count is obtained.

    Scoped to the solver's own executable source. Docstrings and comments are
    removed first: the module documents at length WHY it refuses nominal-day
    arithmetic and fixed year lengths, and that prose must never fail a test.
    """

    @staticmethod
    def executable_source(function):
        source = inspect.getsource(function)

        # Drop the docstring, then line comments.
        source = re.sub(r'"""[\s\S]*?"""', "", source, count=1)
        return "\n".join(
            line.split("#", 1)[0] for line in source.split("\n")
        )

    @property
    def solver_source(self):
        return self.executable_source(count_sunsets_in_interval)

    FORBIDDEN = (
        "86400",
        "86_400",
        "timedelta",
        "datetime",
        "strftime",
        "isoformat",
        "utc_datetime",
        "math.floor",
        "math.ceil",
        "round(",
        "yearType",
        "STANDARD_YEAR_DAYS",
        "ADJUSTMENT_YEAR_DAYS",
        "classifyYearLength",
        "step_days",
        "SUCCESSOR_SEARCH_SPAN_DAYS",
    )

    def forbidden_in(self, source):
        return [token for token in self.FORBIDDEN if token in source]

    def test_the_extractor_is_non_vacuous(self):
        source = self.solver_source

        self.assertGreater(len(source), 400)
        self.assertIn("supported_search_frontier", source)
        self.assertIn("find_discrete", source)
        self.assertIn("SunsetCount", source)

    def test_the_scanner_detects_a_planted_violation(self):
        planted = "days = round((hi - lo) / 86400)\n"

        detected = self.forbidden_in(planted)

        self.assertIn("86400", detected)
        self.assertIn("round(", detected)

    def test_the_scanner_ignores_prose(self):
        prose = (
            'def f():\n'
            '    """No 86400, no round(, no timedelta here."""\n'
            '    # and none of 86400 round( datetime either\n'
            '    return 1\n'
        )

        self.assertEqual(self.forbidden_in(self.executable_source_text(prose)), [])

    @staticmethod
    def executable_source_text(source):
        source = re.sub(r'"""[\s\S]*?"""', "", source, count=1)
        return "\n".join(
            line.split("#", 1)[0] for line in source.split("\n")
        )

    def test_no_nominal_day_gregorian_or_rounding_construct(self):
        self.assertEqual(self.forbidden_in(self.solver_source), [])

    def test_no_fixed_year_length_literal(self):
        source = self.solver_source

        for literal in ("365", "366", "367"):
            self.assertIsNone(
                re.search(r"\b%s\b" % literal, source),
                "fixed year-length literal %s in solver source" % literal,
            )

    def test_no_integer_division_infers_the_count(self):
        self.assertNotIn("//", self.solver_source)

    def test_exactly_one_search_and_one_frontier(self):
        """No stitching, no subdivision, no retry under a second artifact."""
        source = self.solver_source

        self.assertEqual(source.count("find_discrete"), 1)
        self.assertEqual(source.count("supported_search_frontier"), 1)
        self.assertEqual(source.count("sunrise_sunset"), 1)

    def test_the_count_is_incremented_from_reported_transitions(self):
        source = self.solver_source

        self.assertIn("for t, sun_is_up in zip(times, events)", source)

        # DT-A1: reported transitions are reconciled against the certified
        # structure, which keeps each of them and adds every missed crossing.
        self.assertIn("reported.append(event_tt)", source)
        self.assertIn("_certified_sunsets_in_frontier(", source)
        self.assertIn("count=len(sunsets)", source)

    def test_the_span_bound_is_compared_on_exact_states(self):
        source = self.solver_source

        self.assertIn("hi - lo > COUNT_MAX_SPAN_DAYS", source)

    def test_completeness_is_carried_from_the_substrate_not_decided(self):
        source = self.solver_source

        # DT-A1: complete means coverage complete AND detection certified.
        # Coverage failure keeps the substrate's own reason; only a covered
        # interval may report a detection reason.
        self.assertIn(
            "complete = frontier.complete and detection_reason is None", source
        )
        self.assertIn(
            "frontier.truncation_reason if not frontier.complete", source
        )

        # Nothing may assign completeness a literal.
        self.assertNotIn("complete=True", source)
        self.assertNotIn("complete=False", source)

    def test_the_result_carries_no_event_identity(self):
        """A count reports how many, never which."""
        fields = tuple(SunsetCount.__dataclass_fields__)

        self.assertEqual(fields, EXPECTED_FIELDS)

        record = count_sunsets_in_interval(
            tt_of(2027, 3, 21), tt_of(2027, 4, 21), *NEW_YORK
        )

        for name in fields:
            value = getattr(record, name)

            self.assertNotIsInstance(value, (list, tuple, dict, set))
            self.assertIsInstance(value, (int, float, bool, str, type(None)))

    def test_the_result_is_immutable(self):
        record = count_sunsets_in_interval(
            tt_of(2027, 3, 21), tt_of(2027, 4, 21), *NEW_YORK
        )

        with self.assertRaises(Exception):
            record.count = 999

    @staticmethod
    def module_imports():
        """Every module the solver module actually imports.

        Import STATEMENTS are extracted, never bare words. The module
        documents at length which neighbours it deliberately does NOT import
        and why, and that prose must never fail an architectural test.
        """
        source = inspect.getsource(astronomy_solver)
        executable = "\n".join(
            line.split("#", 1)[0] for line in source.split("\n")
        )

        return re.findall(
            r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))",
            executable,
            re.MULTILINE,
        )

    def test_the_import_extractor_is_non_vacuous(self):
        found = {name for pair in self.module_imports() for name in pair if name}

        # The real dependencies are present, which is what establishes that
        # these ARE the module's imports and not an empty parse.
        self.assertIn("scientific_environment", found)
        self.assertIn("skyfield", found)
        self.assertIn("skyfield.api", found)
        self.assertGreater(len(found), 5)

        # And a planted upward import would be seen.
        planted = re.findall(
            r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))",
            "from astronomical_event_transport import project_sunset_event\n",
            re.MULTILINE,
        )
        self.assertEqual(planted, [("astronomical_event_transport", "")])

    def test_the_block_marker_obeys_the_governed_grammar(self):
        """A terminal block must TERMINATE the block before it.

        The published source guards delimit a governed block from its own
        marker to the next one matching this grammar. A marker the grammar
        does not recognise would leave this operation silently absorbed into
        the preceding block, and that block's structural controls would then
        be asserted against code they were never written for.
        """
        governed = re.compile(
            r"^# A\d+[a-z]?(?:-[0-9a-z]+)? - ", re.MULTILINE
        )
        source = inspect.getsource(astronomy_solver)

        markers = governed.findall(source)

        # Non-vacuity: the grammar really does find the repository's blocks.
        self.assertGreater(len(markers), 3)

        own = source.find("# A4 - ABC-1 bulk observer-local sunset count.")
        self.assertNotEqual(own, -1, "ABC-1 block marker not found")
        self.assertIsNotNone(governed.match(source, own))

        # Nothing governed follows it: this is the terminal block, so the
        # symbols below the marker are this operation's and no other's.
        self.assertIsNone(governed.search(source, own + 1))

        # And the solver really is inside it.
        self.assertGreater(source.find("def count_sunsets_in_interval"), own)

    def test_no_upward_calendar_or_transport_dependency(self):
        """The solver sits beneath transport and beneath any calendar."""
        found = {name for pair in self.module_imports() for name in pair if name}

        for upward in (
            "server",
            "astronomical_event_transport",
            "exact_time_transport",
            "sunset_cursor",
        ):
            self.assertNotIn(upward, found)


if __name__ == "__main__":
    unittest.main()
