"""A2-3a verification for the supported directional search frontier.

Covers REASON_EPHEMERIS_COVERAGE_EXHAUSTED, REASON_EPHEMERIS_REACH_EXHAUSTED,
REASON_OBSERVER_OUT_OF_DOMAIN, the governed observer domains,
SupportedSearchFrontier, _governed_observer, _observation_is_evaluable,
_first_evaluable_state, _admit_bracket and supported_search_frontier in
astronomy_solver.py.

Test framework: standard-library unittest.
Scientific execution uses the repository's certified Skyfield runtime and
the real pinned Authority ephemeris artifacts.

INDEPENDENT ORACLE DISCIPLINE

Reason codes, observer domains, pinned artifact filenames, the governed
precedence ordering and the raw JD(TDB) bounds are declared as test-local
literals. They are deliberately NOT imported from astronomy_solver:
importing a constant to check that same constant would make the test agree
with a defective value instead of detecting it.

Evaluability is decided here by an independent probe that performs the real
topocentric solar computation and catches EphemerisRangeError directly. It
does not call _observation_is_evaluable, so that function cannot confirm
itself. The lower computation-reach frontier is likewise located by a
test-local bisection oracle rather than by _first_evaluable_state.

kernel_coverage_tt, load_kernel, select_kernel_containing_instant and the
shared timescale ARE taken from astronomy_solver. They are A2-1/A2-2
infrastructure here, already certified by their own focused suites, rather
than the units under test, and duplicating the loader would mmap a second
copy of three multi-gigabyte kernels for no verification benefit.

SCOPE

This file verifies the supported search frontier substrate only: which
pinned artifact can support one directional search, and over exactly what
interval. It asserts nothing about event determination, predecessor or
successor selection, ordering rules, HTTP behavior or any route. Those
belong to later governed operations and do not exist yet.

WHAT IS DELIBERATELY NOT ENCODED

No light-time second-count appears as an expected value. The lower reach is
a physical quantity that varies with the Earth-Sun distance and differs
between artifacts; pinning 489, 490, 507 or 510 seconds would convert a
measurement into a false constant. Every reach expectation here is derived
by measuring the certified computation at test time.

No extrapolation band width appears either. The scientific contract this
file protects is only that successful library evaluation is not a
substitute for declared BSP containment.

The endpoint-sufficiency invariant is verified as a property of the
certified runtime and pinned artifact set, not asserted as a timeless
astronomical theorem.
"""

import ast
import dataclasses
import inspect
import math
import re
import unittest

import numpy as np
from skyfield import almanac
from skyfield.api import wgs84
from skyfield.errors import EphemerisRangeError

import astronomy_solver
from astronomy_solver import (
    choose_kernel_name,
    kernel_coverage_tt,
    load_kernel,
    select_kernel_containing_instant,
    supported_search_frontier,
    ts,
)
from scientific_environment import ScientificEnvironmentError

# --- Test-local oracles ----------------------------------------------------

DE440 = "de440.bsp"
DE441_PART_1 = "de441_part-1.bsp"
DE441_PART_2 = "de441_part-2.bsp"

EXPECTED_PRECEDENCE = (DE440, DE441_PART_1, DE441_PART_2)

# Raw JD(TDB) bounds published inside each certified artifact.
PINNED_TDB_BOUNDS = {
    DE440: (2287184.5, 2688976.5),
    DE441_PART_1: (-3100015.5, 2440432.5),
    DE441_PART_2: (2440400.5, 8000016.5),
}

REASON_COVERAGE_EXHAUSTED = "EPHEMERIS_COVERAGE_EXHAUSTED"
REASON_REACH_EXHAUSTED = "EPHEMERIS_REACH_EXHAUSTED"
REASON_OBSERVER_OUT_OF_DOMAIN = "OBSERVER_OUT_OF_DOMAIN"
REASON_INSTANT_INVALID = "INSTANT_STATE_INVALID"

EXPECTED_LATITUDE_DOMAIN = (-90.0, 90.0)
EXPECTED_LONGITUDE_DOMAIN = (-180.0, 180.0)

EXPECTED_FRONTIER_FIELDS = (
    "kernel",
    "tt_lo",
    "tt_hi",
    "complete",
    "truncation_reason",
)

# A directional reach, declared test-locally. A2-3a holds no span constant:
# the horizon is always the caller's.
SPAN_DAYS = 3.0

# An ordinary interior anchor, well inside DE440's certified support.
INTERIOR_TT = 2460678.0

# Inside the window the frozen civil routing sends to DE440 even though
# DE440 publishes no data there.
DEFECT_WINDOW_TT = 2689118.000800741
DEFECT_WINDOW_CIVIL_YEAR = 2650

REFERENCE_LATITUDE = 40.7406
REFERENCE_LONGITUDE = -73.9

OBSERVERS = (
    (40.7406, -73.9, "new-york"),
    (0.0, 0.0, "null-island"),
    (-33.9, 151.2, "sydney"),
    (78.2, 15.6, "svalbard"),
    (0.0, 180.0, "antimeridian"),
)

# The heavier per-state sweeps run over this subset to keep the focused
# suite quick; the cheaper assertions use every observer above.
SWEEP_OBSERVERS = OBSERVERS[:3]

A2_3A_BLOCK_MARKER = "# A2-3a - supported directional search frontier."

# The shape of a governed production block header, used only to find where
# the A2-3a block ends. It matches the header FORM and nothing else.
GOVERNED_BLOCK_MARKER = re.compile(r"^# A\d+[a-z]?(?:-\d+[a-z]?)? - ",
                                   re.MULTILINE)


# --- Independent scientific helpers ----------------------------------------


def coverage(kernel_name):
    return kernel_coverage_tt(kernel_name)


def union_bounds():
    lo = min(coverage(name).tt_start for name in EXPECTED_PRECEDENCE)
    hi = max(coverage(name).tt_end for name in EXPECTED_PRECEDENCE)
    return lo, hi


def declares(kernel_name, tt_lo, tt_hi):
    """Test-local declared-containment oracle: exact, closed, inclusive."""
    record = coverage(kernel_name)
    return record.tt_start <= tt_lo and tt_hi <= record.tt_end


def predicate(kernel_name, latitude, longitude):
    """The settled HPC topocentric apparent-sunset predicate."""
    return almanac.sunrise_sunset(
        load_kernel(kernel_name), wgs84.latlon(latitude, longitude)
    )


def evaluates(kernel_name, tt, latitude, longitude):
    """Independent evaluability oracle.

    Performs the real computation. Deliberately does not call
    _observation_is_evaluable, so that function cannot confirm itself.
    """
    try:
        predicate(kernel_name, latitude, longitude)(ts.tt_jd(tt))
    except EphemerisRangeError:
        return False

    return True


def oracle_first_evaluable(kernel_name, unevaluable_tt, evaluable_tt,
                           latitude, longitude):
    """Independent bisection oracle for the lower computation-reach frontier.

    Halves until no representable binary64 value lies strictly between the
    two ends. No tolerance and no iteration limit, for the same reason the
    production helper has none: once no representable state remains between
    them there is nothing left to examine.
    """
    bad = unevaluable_tt
    good = evaluable_tt

    while True:
        mid = bad + (good - bad) / 2.0

        if mid == bad or mid == good:
            return good

        if evaluates(kernel_name, mid, latitude, longitude):
            good = mid
        else:
            bad = mid


_FLOOR_CACHE = {}


def measured_floor(kernel_name, latitude, longitude):
    """The measured first evaluable state above a declared start.

    Memoized in the TEST ONLY, so a sweep does not re-measure the same
    boundary repeatedly. A2-3a itself caches nothing.
    """
    key = (kernel_name, latitude, longitude)
    if key not in _FLOOR_CACHE:
        record = coverage(kernel_name)
        _FLOOR_CACHE[key] = oracle_first_evaluable(
            kernel_name,
            record.tt_start,
            record.tt_start + 0.02,
            latitude,
            longitude,
        )
    return _FLOOR_CACHE[key]


def sampled_states(frontier, latitude, longitude):
    """Every TT state the certified search actually evaluates."""
    inner = predicate(frontier.kernel, latitude, longitude)
    seen = []

    def spy(t):
        value = inner(t)
        seen.extend(
            float(x) for x in np.atleast_1d(np.asarray(t.tt)).ravel()
        )
        return value

    spy.step_days = inner.step_days
    almanac.find_discrete(
        ts.tt_jd(frontier.tt_lo), ts.tt_jd(frontier.tt_hi), spy
    )
    return seen


def frontier_for(anchor, horizon, latitude=REFERENCE_LATITUDE,
                 longitude=REFERENCE_LONGITUDE):
    return supported_search_frontier(anchor, horizon, latitude, longitude)


class FrontierAssertions(unittest.TestCase):
    """Invariants every returned frontier must satisfy."""

    def assert_frontier_invariants(self, frontier, anchor, latitude,
                                   longitude):
        self.assertIn(frontier.kernel, EXPECTED_PRECEDENCE)
        self.assertLess(frontier.tt_lo, frontier.tt_hi)

        # The anchor is retained exactly, and is one of the two endpoints.
        self.assertIn(anchor, (frontier.tt_lo, frontier.tt_hi))

        # One artifact declares the WHOLE returned interval.
        self.assertTrue(
            declares(frontier.kernel, frontier.tt_lo, frontier.tt_hi),
            "returned frontier is not wholly declared by its own artifact",
        )

        # Both endpoints are genuinely evaluable under that artifact.
        for endpoint in (frontier.tt_lo, frontier.tt_hi):
            self.assertTrue(
                evaluates(frontier.kernel, endpoint, latitude, longitude),
                "frontier endpoint %r is not evaluable" % endpoint,
            )

        # No second interval can exist: the record has no room for one.
        self.assertEqual(
            tuple(f.name for f in dataclasses.fields(frontier)),
            EXPECTED_FRONTIER_FIELDS,
        )

        if frontier.complete:
            self.assertIsNone(frontier.truncation_reason)
        else:
            self.assertIn(
                frontier.truncation_reason,
                (REASON_COVERAGE_EXHAUSTED, REASON_REACH_EXHAUSTED),
            )

    def assert_fails_closed(self, reason, anchor, horizon,
                            latitude=REFERENCE_LATITUDE,
                            longitude=REFERENCE_LONGITUDE):
        with self.assertRaises(ScientificEnvironmentError) as caught:
            supported_search_frontier(anchor, horizon, latitude, longitude)
        self.assertEqual(caught.exception.reason, reason)
        return caught.exception


# --- 1. Observer validation ------------------------------------------------


class TestObserverDomain(FrontierAssertions):
    def setUp(self):
        self.anchor = INTERIOR_TT
        self.horizon = INTERIOR_TT + SPAN_DAYS

    def test_domains_are_the_expected_governed_ranges(self):
        self.assertEqual(
            tuple(astronomy_solver.OBSERVER_LATITUDE_DOMAIN),
            EXPECTED_LATITUDE_DOMAIN,
        )
        self.assertEqual(
            tuple(astronomy_solver.OBSERVER_LONGITUDE_DOMAIN),
            EXPECTED_LONGITUDE_DOMAIN,
        )

    def test_inclusive_latitude_bounds_are_accepted(self):
        for latitude in EXPECTED_LATITUDE_DOMAIN:
            with self.subTest(latitude=latitude):
                frontier = frontier_for(
                    self.anchor, self.horizon, latitude, 0.0
                )
                self.assert_frontier_invariants(
                    frontier, self.anchor, latitude, 0.0
                )

    def test_inclusive_longitude_bounds_are_accepted(self):
        for longitude in EXPECTED_LONGITUDE_DOMAIN:
            with self.subTest(longitude=longitude):
                frontier = frontier_for(
                    self.anchor, self.horizon, 0.0, longitude
                )
                self.assert_frontier_invariants(
                    frontier, self.anchor, 0.0, longitude
                )

    def test_ordinary_observers_are_accepted(self):
        for latitude, longitude, label in OBSERVERS:
            with self.subTest(observer=label):
                frontier = frontier_for(
                    self.anchor, self.horizon, latitude, longitude
                )
                self.assertTrue(frontier.complete)

    def test_integer_coordinates_are_accepted(self):
        frontier = frontier_for(self.anchor, self.horizon, 40, -74)
        self.assertTrue(frontier.complete)

    def test_bool_is_rejected(self):
        for value in (True, False):
            with self.subTest(value=value):
                self.assert_fails_closed(
                    REASON_OBSERVER_OUT_OF_DOMAIN,
                    self.anchor, self.horizon, value, 0.0,
                )
                self.assert_fails_closed(
                    REASON_OBSERVER_OUT_OF_DOMAIN,
                    self.anchor, self.horizon, 0.0, value,
                )

    def test_non_numeric_is_rejected(self):
        for value in ("40.0", None, [40.0], {}, object(), complex(1, 0)):
            with self.subTest(value=repr(value)):
                self.assert_fails_closed(
                    REASON_OBSERVER_OUT_OF_DOMAIN,
                    self.anchor, self.horizon, value, 0.0,
                )
                self.assert_fails_closed(
                    REASON_OBSERVER_OUT_OF_DOMAIN,
                    self.anchor, self.horizon, 0.0, value,
                )

    def test_nan_is_rejected(self):
        self.assert_fails_closed(
            REASON_OBSERVER_OUT_OF_DOMAIN,
            self.anchor, self.horizon, float("nan"), 0.0,
        )
        self.assert_fails_closed(
            REASON_OBSERVER_OUT_OF_DOMAIN,
            self.anchor, self.horizon, 0.0, float("nan"),
        )

    def test_infinities_are_rejected(self):
        for value in (float("inf"), float("-inf")):
            with self.subTest(value=value):
                self.assert_fails_closed(
                    REASON_OBSERVER_OUT_OF_DOMAIN,
                    self.anchor, self.horizon, value, 0.0,
                )
                self.assert_fails_closed(
                    REASON_OBSERVER_OUT_OF_DOMAIN,
                    self.anchor, self.horizon, 0.0, value,
                )

    def test_integer_too_large_for_float_is_rejected(self):
        self.assert_fails_closed(
            REASON_OBSERVER_OUT_OF_DOMAIN,
            self.anchor, self.horizon, 10 ** 400, 0.0,
        )

    def test_latitude_outside_the_domain_is_rejected(self):
        for value in (90.0000001, -90.0000001, 91.0, -91.0, 1e6):
            with self.subTest(latitude=value):
                self.assert_fails_closed(
                    REASON_OBSERVER_OUT_OF_DOMAIN,
                    self.anchor, self.horizon, value, 0.0,
                )

    def test_longitude_outside_the_domain_is_rejected(self):
        for value in (180.0000001, -180.0000001, 181.0, -181.0, 1e6):
            with self.subTest(longitude=value):
                self.assert_fails_closed(
                    REASON_OBSERVER_OUT_OF_DOMAIN,
                    self.anchor, self.horizon, 0.0, value,
                )

    def test_malformed_observer_is_never_reported_as_reach_exhausted(self):
        """A NaN latitude reaches the ephemeris as an invalid cast.

        Unvalidated, it surfaces as an ephemeris range error - the exact
        signal A2-3a reads as exhausted computational reach. It must be
        reported as a malformed observer instead.
        """
        # The hazard is real: the raw computation genuinely raises.
        with self.assertRaises(EphemerisRangeError):
            predicate(DE440, float("nan"), 0.0)(ts.tt_jd(INTERIOR_TT))

        for latitude, longitude in (
            (float("nan"), 0.0),
            (0.0, float("nan")),
            (float("inf"), 0.0),
            (95.0, 0.0),
            (0.0, 400.0),
        ):
            with self.subTest(latitude=latitude, longitude=longitude):
                error = self.assert_fails_closed(
                    REASON_OBSERVER_OUT_OF_DOMAIN,
                    self.anchor, self.horizon, latitude, longitude,
                )
                self.assertNotEqual(error.reason, REASON_REACH_EXHAUSTED)
                self.assertNotEqual(error.reason, REASON_COVERAGE_EXHAUSTED)

    def test_out_of_domain_observer_is_not_silently_computed(self):
        """A latitude of 95 degrees computes silently in the raw stack."""
        self.assertTrue(evaluates(DE440, INTERIOR_TT, 95.0, 0.0))
        self.assert_fails_closed(
            REASON_OBSERVER_OUT_OF_DOMAIN,
            self.anchor, self.horizon, 95.0, 0.0,
        )

    def test_signed_zero_is_accepted_without_canonicalization(self):
        validate = astronomy_solver._governed_observer

        negative = validate(-0.0, "latitude", EXPECTED_LATITUDE_DOMAIN)
        self.assertEqual(negative, 0.0)
        self.assertEqual(
            math.copysign(1.0, negative), -1.0,
            "this layer must not canonicalize signed zero",
        )

        positive = validate(0.0, "latitude", EXPECTED_LATITUDE_DOMAIN)
        self.assertEqual(math.copysign(1.0, positive), 1.0)

        minus = frontier_for(self.anchor, self.horizon, -0.0, -0.0)
        plus = frontier_for(self.anchor, self.horizon, 0.0, 0.0)
        self.assertEqual(minus, plus)


# --- 2. Whole-bracket admission and precedence -----------------------------


class TestAdmitBracket(unittest.TestCase):
    def admit(self, tt_lo, tt_hi, latitude=REFERENCE_LATITUDE,
              longitude=REFERENCE_LONGITUDE):
        return astronomy_solver._admit_bracket(
            tt_lo, tt_hi, latitude, longitude
        )

    def test_whole_declared_containment_is_required(self):
        lo, hi = union_bounds()
        self.assertIsNone(self.admit(lo - 1.0, lo + 1.0))
        self.assertIsNone(self.admit(hi - 1.0, hi + 1.0))
        self.assertIsNone(self.admit(lo, hi))

    def test_interior_bracket_is_admitted_under_de440(self):
        self.assertEqual(
            self.admit(INTERIOR_TT, INTERIOR_TT + SPAN_DAYS), DE440
        )

    def test_de440_precedence_is_preserved_where_it_can_evaluate(self):
        record = coverage(DE440)
        span = record.tt_end - record.tt_start
        probes = [
            record.tt_start + span * fraction
            for fraction in (0.25, 0.5, 0.75)
        ]
        probes.append(coverage(DE441_PART_2).tt_start)
        probes.append(coverage(DE441_PART_1).tt_end - SPAN_DAYS)
        for tt in probes:
            with self.subTest(tt=tt):
                self.assertTrue(declares(DE440, tt, tt + SPAN_DAYS))
                self.assertEqual(self.admit(tt, tt + SPAN_DAYS), DE440)

    def test_scan_continues_past_a_declaring_but_unevaluable_artifact(self):
        """DE440's exact declared start is declared by DE440 and unevaluable.

        Stopping at the first declaring artifact would refuse a bracket that
        DE441 part 1 genuinely supports.
        """
        start = coverage(DE440).tt_start
        bracket = (start, start + SPAN_DAYS)

        # DE440 really does declare it, and really cannot evaluate it.
        self.assertTrue(declares(DE440, *bracket))
        self.assertFalse(
            evaluates(DE440, start, REFERENCE_LATITUDE, REFERENCE_LONGITUDE)
        )

        # DE441 part 1 declares the same bracket and can evaluate it.
        self.assertTrue(declares(DE441_PART_1, *bracket))
        self.assertTrue(
            evaluates(
                DE441_PART_1, start, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
            )
        )

        self.assertEqual(self.admit(*bracket), DE441_PART_1)

    def test_de441_part_2_declared_start_still_prefers_de440(self):
        start = coverage(DE441_PART_2).tt_start
        self.assertEqual(self.admit(start, start + SPAN_DAYS), DE440)

    def test_de440_upper_straddle_selects_de441_part_2(self):
        end = coverage(DE440).tt_end
        self.assertEqual(self.admit(end - 1.0, end + 1.0), DE441_PART_2)
        self.assertEqual(
            self.admit(end + 1.0, end + 1.0 + SPAN_DAYS), DE441_PART_2
        )

    def test_de440_lower_straddle_selects_de441_part_1(self):
        start = coverage(DE440).tt_start
        self.assertEqual(self.admit(start - 1.0, start + 1.0), DE441_PART_1)

    def test_defect_window_selects_de441_part_2(self):
        self.assertEqual(
            self.admit(DEFECT_WINDOW_TT, DEFECT_WINDOW_TT + SPAN_DAYS),
            DE441_PART_2,
        )

    def test_unevaluable_endpoint_alone_prevents_admission(self):
        record = coverage(DE441_PART_1)
        bracket = (record.tt_start, record.tt_start + SPAN_DAYS)
        self.assertTrue(declares(DE441_PART_1, *bracket))
        self.assertFalse(
            evaluates(
                DE441_PART_1, record.tt_start,
                REFERENCE_LATITUDE, REFERENCE_LONGITUDE,
            )
        )
        self.assertIsNone(self.admit(*bracket))

    def test_admission_is_deterministic(self):
        bracket = (INTERIOR_TT, INTERIOR_TT + SPAN_DAYS)
        results = {self.admit(*bracket) for _ in range(5)}
        self.assertEqual(len(results), 1)


# --- 3. Certified endpoint-sufficiency invariant ---------------------------


class TestEndpointSufficiencyInvariant(unittest.TestCase):
    """Endpoint probes certify whole-bracket evaluability.

    This is a property of the certified runtime and the pinned artifact
    set, verified here rather than assumed. A different runtime or artifact
    set could read differently, which is precisely why it is tested.
    """

    INTERIOR_SAMPLES = 40

    def assert_interior_is_evaluable(self, kernel_name, tt_lo, tt_hi,
                                     latitude, longitude):
        self.assertTrue(
            evaluates(kernel_name, tt_lo, latitude, longitude),
            "lower endpoint is not evaluable",
        )
        self.assertTrue(
            evaluates(kernel_name, tt_hi, latitude, longitude),
            "upper endpoint is not evaluable",
        )

        failures = [
            float(tt)
            for tt in np.linspace(tt_lo, tt_hi, self.INTERIOR_SAMPLES)
            if not evaluates(kernel_name, float(tt), latitude, longitude)
        ]
        self.assertEqual(
            failures, [],
            "interior states failed while both endpoints evaluated",
        )

    def test_lower_edge_bracket_has_no_interior_failure(self):
        for kernel_name in EXPECTED_PRECEDENCE:
            for latitude, longitude, label in SWEEP_OBSERVERS:
                with self.subTest(kernel=kernel_name, observer=label):
                    floor = measured_floor(kernel_name, latitude, longitude)
                    self.assert_interior_is_evaluable(
                        kernel_name, floor, floor + SPAN_DAYS,
                        latitude, longitude,
                    )

    def test_bracket_ending_at_the_declared_upper_bound_is_whole(self):
        for kernel_name in EXPECTED_PRECEDENCE:
            for latitude, longitude, label in SWEEP_OBSERVERS:
                with self.subTest(kernel=kernel_name, observer=label):
                    end = coverage(kernel_name).tt_end
                    self.assert_interior_is_evaluable(
                        kernel_name, end - SPAN_DAYS, end,
                        latitude, longitude,
                    )

    def test_scalar_and_vectorized_support_domains_coincide(self):
        for kernel_name in EXPECTED_PRECEDENCE:
            for latitude, longitude, label in SWEEP_OBSERVERS:
                with self.subTest(kernel=kernel_name, observer=label):
                    floor = measured_floor(kernel_name, latitude, longitude)
                    below = math.nextafter(floor, -math.inf)
                    is_sun_up = predicate(kernel_name, latitude, longitude)

                    # Scalar probe.
                    self.assertTrue(
                        evaluates(kernel_name, floor, latitude, longitude)
                    )
                    self.assertFalse(
                        evaluates(kernel_name, below, latitude, longitude)
                    )

                    # Vectorized predicate, several array shapes.
                    for count in (2, 12, 77):
                        is_sun_up(
                            ts.tt_jd(
                                np.linspace(floor, floor + SPAN_DAYS, count)
                            )
                        )
                        with self.assertRaises(EphemerisRangeError):
                            is_sun_up(
                                ts.tt_jd(
                                    np.linspace(
                                        below, below + SPAN_DAYS, count
                                    )
                                )
                            )

                    # The certified search itself.
                    almanac.find_discrete(
                        ts.tt_jd(floor),
                        ts.tt_jd(floor + SPAN_DAYS),
                        is_sun_up,
                    )
                    with self.assertRaises(EphemerisRangeError):
                        almanac.find_discrete(
                            ts.tt_jd(below),
                            ts.tt_jd(floor + SPAN_DAYS),
                            is_sun_up,
                        )


# --- 4. Exact lower computation-reach frontier -----------------------------


class TestFirstEvaluableState(unittest.TestCase):
    """The authorized lower-edge recovery, verified against an oracle."""

    KERNEL = DE441_PART_1

    def setUp(self):
        self.record = coverage(self.KERNEL)
        self.latitude = REFERENCE_LATITUDE
        self.longitude = REFERENCE_LONGITUDE
        self.is_sun_up = predicate(
            self.KERNEL, self.latitude, self.longitude
        )
        self.unevaluable = self.record.tt_start
        self.evaluable = self.record.tt_start + 1.0

    def resolve(self, evaluable_tt):
        return astronomy_solver._first_evaluable_state(
            self.is_sun_up, self.unevaluable, evaluable_tt
        )

    def test_the_declared_start_really_is_unevaluable(self):
        self.assertFalse(
            evaluates(
                self.KERNEL, self.unevaluable, self.latitude, self.longitude
            )
        )

    def test_the_upper_state_really_is_evaluable(self):
        self.assertTrue(
            evaluates(
                self.KERNEL, self.evaluable, self.latitude, self.longitude
            )
        )

    def test_returned_state_is_evaluable(self):
        resolved = self.resolve(self.evaluable)
        self.assertTrue(
            evaluates(self.KERNEL, resolved, self.latitude, self.longitude)
        )

    def test_immediate_predecessor_is_unevaluable(self):
        """Nothing evaluable is discarded: the boundary is exact."""
        resolved = self.resolve(self.evaluable)
        self.assertFalse(
            evaluates(
                self.KERNEL,
                math.nextafter(resolved, -math.inf),
                self.latitude,
                self.longitude,
            )
        )

    def test_matches_the_independent_bisection_oracle(self):
        expected = oracle_first_evaluable(
            self.KERNEL,
            self.unevaluable,
            self.evaluable,
            self.latitude,
            self.longitude,
        )
        self.assertEqual(self.resolve(self.evaluable), expected)

    def test_result_is_deterministic(self):
        results = {self.resolve(self.evaluable) for _ in range(3)}
        self.assertEqual(len(results), 1)

    def test_result_is_independent_of_the_starting_bracket(self):
        widths = (0.006, 0.02, 1.0, SPAN_DAYS)
        results = set()
        for width in widths:
            upper = self.record.tt_start + width
            self.assertTrue(
                evaluates(
                    self.KERNEL, upper, self.latitude, self.longitude
                ),
                "the probe bracket must start from an evaluable state",
            )
            results.add(self.resolve(upper))
        self.assertEqual(
            len(results), 1,
            "the boundary must be a property of the computation, not the "
            "search width",
        )

    def test_resolved_state_lies_inside_declared_coverage(self):
        resolved = self.resolve(self.evaluable)
        self.assertGreater(resolved, self.record.tt_start)
        self.assertLess(resolved, self.record.tt_end)

    def test_helper_has_no_tolerance_or_iteration_cap(self):
        body = executable_source(
            inspect.getsource(astronomy_solver._first_evaluable_state)
        )
        self.assertIn("while True:", body)
        self.assertIn("if mid == bad or mid == good:", body)
        for token in ("range(", "epsilon", "tolerance", "1e-", "abs("):
            with self.subTest(token=token):
                self.assertNotIn(token, body)


# --- 5. Complete interior frontiers ----------------------------------------


class TestCompleteFrontier(FrontierAssertions):
    def test_forward_interior_request_is_complete(self):
        anchor, horizon = INTERIOR_TT, INTERIOR_TT + SPAN_DAYS
        frontier = frontier_for(anchor, horizon)
        self.assert_frontier_invariants(
            frontier, anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.assertTrue(frontier.complete)
        self.assertIsNone(frontier.truncation_reason)
        self.assertEqual(frontier.tt_lo, anchor)
        self.assertEqual(frontier.tt_hi, horizon)
        self.assertEqual(frontier.kernel, DE440)

    def test_backward_interior_request_is_complete(self):
        anchor, horizon = INTERIOR_TT, INTERIOR_TT - SPAN_DAYS
        frontier = frontier_for(anchor, horizon)
        self.assert_frontier_invariants(
            frontier, anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.assertTrue(frontier.complete)
        self.assertEqual(frontier.tt_lo, horizon)
        self.assertEqual(frontier.tt_hi, anchor)
        self.assertEqual(frontier.kernel, DE440)

    def test_whole_requested_horizon_is_preserved(self):
        for anchor, horizon in (
            (INTERIOR_TT, INTERIOR_TT + SPAN_DAYS),
            (INTERIOR_TT, INTERIOR_TT - SPAN_DAYS),
        ):
            with self.subTest(anchor=anchor, horizon=horizon):
                frontier = frontier_for(anchor, horizon)
                self.assertEqual(
                    {frontier.tt_lo, frontier.tt_hi}, {anchor, horizon}
                )

    def test_global_observer_matrix_is_complete(self):
        anchor, horizon = INTERIOR_TT, INTERIOR_TT + SPAN_DAYS
        for latitude, longitude, label in OBSERVERS:
            with self.subTest(observer=label):
                frontier = frontier_for(anchor, horizon, latitude, longitude)
                self.assert_frontier_invariants(
                    frontier, anchor, latitude, longitude
                )
                self.assertTrue(frontier.complete)

    def test_frontier_record_is_immutable(self):
        frontier = frontier_for(INTERIOR_TT, INTERIOR_TT + SPAN_DAYS)
        with self.assertRaises(Exception):
            frontier.tt_lo = 0.0

    def test_repeated_requests_are_deterministic(self):
        results = {
            frontier_for(INTERIOR_TT, INTERIOR_TT + SPAN_DAYS)
            for _ in range(3)
        }
        self.assertEqual(len(results), 1)


# --- 6. Artifact transitions -----------------------------------------------


class TestArtifactTransitions(FrontierAssertions):
    def test_de440_lower_straddle_resolves_through_de441_part_1(self):
        start = coverage(DE440).tt_start
        for anchor, horizon in (
            (start - 1.0, start - 1.0 + SPAN_DAYS),
            (start + 1.0, start + 1.0 - SPAN_DAYS),
        ):
            with self.subTest(anchor=anchor):
                frontier = frontier_for(anchor, horizon)
                self.assert_frontier_invariants(
                    frontier, anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
                )
                self.assertTrue(frontier.complete)
                self.assertEqual(frontier.kernel, DE441_PART_1)

    def test_de440_upper_straddle_resolves_through_de441_part_2(self):
        end = coverage(DE440).tt_end
        for anchor, horizon in (
            (end - 1.0, end - 1.0 + SPAN_DAYS),
            (end + 1.0, end + 1.0 - SPAN_DAYS),
        ):
            with self.subTest(anchor=anchor):
                frontier = frontier_for(anchor, horizon)
                self.assert_frontier_invariants(
                    frontier, anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
                )
                self.assertTrue(frontier.complete)
                self.assertEqual(frontier.kernel, DE441_PART_2)

    def test_de441_part_1_upper_overlap_preserves_de440_precedence(self):
        end = coverage(DE441_PART_1).tt_end
        self.assertTrue(declares(DE440, end - 1.0, end - 1.0 + SPAN_DAYS))
        frontier = frontier_for(end - 1.0, end - 1.0 + SPAN_DAYS)
        self.assertEqual(frontier.kernel, DE440)
        self.assertTrue(frontier.complete)

    def test_de440_declared_floor_anchor_resolves_through_de441_part_1(self):
        start = coverage(DE440).tt_start
        frontier = frontier_for(start, start + SPAN_DAYS)
        self.assertEqual(frontier.kernel, DE441_PART_1)
        self.assertTrue(frontier.complete)

    def test_legacy_2650_routing_defect_is_bypassed_not_repaired(self):
        # The frozen civil routing still points at the artifact with no data.
        self.assertEqual(choose_kernel_name(DEFECT_WINDOW_CIVIL_YEAR), DE440)
        self.assertGreater(DEFECT_WINDOW_TT, coverage(DE440).tt_end)

        for horizon in (
            DEFECT_WINDOW_TT + SPAN_DAYS,
            DEFECT_WINDOW_TT - SPAN_DAYS,
        ):
            with self.subTest(horizon=horizon):
                frontier = frontier_for(DEFECT_WINDOW_TT, horizon)
                self.assert_frontier_invariants(
                    frontier, DEFECT_WINDOW_TT,
                    REFERENCE_LATITUDE, REFERENCE_LONGITUDE,
                )
                self.assertEqual(frontier.kernel, DE441_PART_2)
                self.assertTrue(frontier.complete)


# --- 7. Absolute outer edges -----------------------------------------------


class TestAbsoluteUpperEdge(FrontierAssertions):
    def test_forward_search_truncates_to_declared_coverage(self):
        ceiling = coverage(DE441_PART_2).tt_end
        anchor = ceiling - 1.0
        frontier = frontier_for(anchor, anchor + SPAN_DAYS)

        self.assert_frontier_invariants(
            frontier, anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.assertFalse(frontier.complete)
        self.assertEqual(
            frontier.truncation_reason, REASON_COVERAGE_EXHAUSTED
        )
        self.assertEqual(frontier.tt_lo, anchor, "the anchor must not move")
        self.assertEqual(frontier.tt_hi, ceiling)
        self.assertEqual(frontier.kernel, DE441_PART_2)

    def test_successful_evaluation_beyond_declared_coverage_is_refused(self):
        """Library success is not a substitute for declared containment.

        The certified stack is demonstrated to return a value beyond the
        declared upper bound. That value is nevertheless refused, because
        the declared containment test - not the absence of an error - is
        the binding upper guard. No band width is asserted here.
        """
        ceiling = coverage(DE441_PART_2).tt_end
        beyond = ceiling + 1.0

        # The stack really does evaluate out there without raising.
        self.assertTrue(
            evaluates(
                DE441_PART_2, beyond, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
            )
        )
        # And it is genuinely outside declared coverage.
        self.assertFalse(declares(DE441_PART_2, beyond, beyond))
        self.assertIsNone(select_kernel_containing_instant(beyond))

        # Admission refuses it anyway.
        self.assertIsNone(
            astronomy_solver._admit_bracket(
                ceiling - 1.0, beyond,
                REFERENCE_LATITUDE, REFERENCE_LONGITUDE,
            )
        )

        # And the frontier stops exactly at the declared bound.
        frontier = frontier_for(ceiling - 1.0, beyond)
        self.assertEqual(frontier.tt_hi, ceiling)
        self.assertFalse(frontier.complete)

    def test_anchor_exactly_at_the_ceiling_fails_closed(self):
        ceiling = coverage(DE441_PART_2).tt_end
        self.assert_fails_closed(
            REASON_COVERAGE_EXHAUSTED, ceiling, ceiling + SPAN_DAYS
        )


class TestAbsoluteLowerEdge(FrontierAssertions):
    def test_backward_search_recovers_the_exact_evaluable_floor(self):
        record = coverage(DE441_PART_1)
        anchor = record.tt_start + 1.0
        frontier = frontier_for(anchor, anchor - SPAN_DAYS)

        self.assert_frontier_invariants(
            frontier, anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.assertFalse(frontier.complete)
        self.assertEqual(frontier.truncation_reason, REASON_REACH_EXHAUSTED)
        self.assertEqual(frontier.tt_hi, anchor, "the anchor must not move")
        self.assertEqual(frontier.kernel, DE441_PART_1)

        expected = oracle_first_evaluable(
            DE441_PART_1, record.tt_start, anchor,
            REFERENCE_LATITUDE, REFERENCE_LONGITUDE,
        )
        self.assertEqual(frontier.tt_lo, expected)

    def test_no_evaluable_territory_is_discarded(self):
        record = coverage(DE441_PART_1)
        anchor = record.tt_start + 1.0
        frontier = frontier_for(anchor, anchor - SPAN_DAYS)

        self.assertTrue(
            evaluates(
                frontier.kernel, frontier.tt_lo,
                REFERENCE_LATITUDE, REFERENCE_LONGITUDE,
            )
        )
        self.assertFalse(
            evaluates(
                frontier.kernel,
                math.nextafter(frontier.tt_lo, -math.inf),
                REFERENCE_LATITUDE, REFERENCE_LONGITUDE,
            )
        )
        self.assertGreater(frontier.tt_lo, record.tt_start)

    def test_recovery_holds_for_several_observers(self):
        record = coverage(DE441_PART_1)
        anchor = record.tt_start + 1.0
        for latitude, longitude, label in SWEEP_OBSERVERS:
            with self.subTest(observer=label):
                frontier = frontier_for(
                    anchor, anchor - SPAN_DAYS, latitude, longitude
                )
                self.assertFalse(frontier.complete)
                self.assertEqual(
                    frontier.truncation_reason, REASON_REACH_EXHAUSTED
                )
                self.assertEqual(frontier.tt_hi, anchor)
                self.assertEqual(
                    frontier.tt_lo,
                    oracle_first_evaluable(
                        DE441_PART_1, record.tt_start, anchor,
                        latitude, longitude,
                    ),
                )

    def test_anchor_exactly_at_the_floor_fails_closed_backward(self):
        floor = coverage(DE441_PART_1).tt_start
        self.assert_fails_closed(
            REASON_COVERAGE_EXHAUSTED, floor, floor - SPAN_DAYS
        )

    def test_forward_anchor_inside_the_unevaluable_prefix_fails_closed(self):
        """The anchor is never moved, so there is no territory to return."""
        record = coverage(DE441_PART_1)
        anchor = record.tt_start

        self.assertTrue(declares(DE441_PART_1, anchor, anchor + SPAN_DAYS))
        self.assertFalse(
            evaluates(
                DE441_PART_1, anchor,
                REFERENCE_LATITUDE, REFERENCE_LONGITUDE,
            )
        )

        self.assert_fails_closed(
            REASON_REACH_EXHAUSTED, anchor, anchor + SPAN_DAYS
        )


# --- 8. Unsupported anchors and malformed frontiers ------------------------


class TestUnsupportedAnchors(FrontierAssertions):
    def test_anchor_below_the_certified_union_fails_closed(self):
        lo, _hi = union_bounds()
        for offset in (1e-6, 1.0, 1e6):
            with self.subTest(offset=offset):
                anchor = lo - offset
                self.assert_fails_closed(
                    REASON_COVERAGE_EXHAUSTED, anchor, anchor + SPAN_DAYS
                )

    def test_anchor_above_the_certified_union_fails_closed(self):
        _lo, hi = union_bounds()
        for offset in (1e-6, 1.0, 1e6):
            with self.subTest(offset=offset):
                anchor = hi + offset
                self.assert_fails_closed(
                    REASON_COVERAGE_EXHAUSTED, anchor, anchor - SPAN_DAYS
                )

    def test_anchor_equal_to_horizon_is_malformed(self):
        for anchor in (INTERIOR_TT, coverage(DE440).tt_start, 0.0):
            with self.subTest(anchor=anchor):
                self.assert_fails_closed(
                    REASON_INSTANT_INVALID, anchor, anchor
                )

    def test_invalid_anchor_states_are_rejected(self):
        bad_values = (
            True, False, "2461485.0", None, [2461485.0], {}, object(),
            complex(1, 0), float("nan"), float("inf"), float("-inf"),
            10 ** 400,
        )
        for value in bad_values:
            with self.subTest(value=repr(value)):
                self.assert_fails_closed(
                    REASON_INSTANT_INVALID, value, INTERIOR_TT
                )
                self.assert_fails_closed(
                    REASON_INSTANT_INVALID, INTERIOR_TT, value
                )

    def test_reason_codes_are_the_expected_stable_strings(self):
        self.assertEqual(
            astronomy_solver.REASON_EPHEMERIS_COVERAGE_EXHAUSTED,
            REASON_COVERAGE_EXHAUSTED,
        )
        self.assertEqual(
            astronomy_solver.REASON_EPHEMERIS_REACH_EXHAUSTED,
            REASON_REACH_EXHAUSTED,
        )
        self.assertEqual(
            astronomy_solver.REASON_OBSERVER_OUT_OF_DOMAIN,
            REASON_OBSERVER_OUT_OF_DOMAIN,
        )

    def test_coverage_and_reach_exhaustion_are_distinguished(self):
        """The two exhaustion reasons must not collapse into one."""
        floor = coverage(DE441_PART_1).tt_start
        coverage_error = self.assert_fails_closed(
            REASON_COVERAGE_EXHAUSTED, floor, floor - SPAN_DAYS
        )
        reach_error = self.assert_fails_closed(
            REASON_REACH_EXHAUSTED, floor, floor + SPAN_DAYS
        )
        self.assertNotEqual(coverage_error.reason, reach_error.reason)

    def test_reach_exhaustion_is_not_reported_as_inconsistent_coverage(self):
        floor = coverage(DE441_PART_1).tt_start
        error = self.assert_fails_closed(
            REASON_REACH_EXHAUSTED, floor, floor + SPAN_DAYS
        )
        self.assertNotIn("KERNEL_COVERAGE_INCONSISTENT", str(error))
        self.assertNotEqual(error.reason, "KERNEL_SEGMENTS_UNAVAILABLE")


# --- 9. Contiguous-frontier invariant --------------------------------------


def frontier_cases():
    """Every frontier-producing case this file exercises, in one place."""
    d440 = coverage(DE440)
    d441_1 = coverage(DE441_PART_1)
    d441_2 = coverage(DE441_PART_2)
    return (
        ("interior forward", INTERIOR_TT, INTERIOR_TT + SPAN_DAYS),
        ("interior backward", INTERIOR_TT, INTERIOR_TT - SPAN_DAYS),
        ("de440 lower straddle", d440.tt_start - 1.0,
         d440.tt_start - 1.0 + SPAN_DAYS),
        ("de440 lower back", d440.tt_start + 1.0,
         d440.tt_start + 1.0 - SPAN_DAYS),
        ("de440 declared floor", d440.tt_start, d440.tt_start + SPAN_DAYS),
        ("de440 upper straddle", d440.tt_end - 1.0,
         d440.tt_end - 1.0 + SPAN_DAYS),
        ("de440 upper back", d440.tt_end + 1.0,
         d440.tt_end + 1.0 - SPAN_DAYS),
        ("defect window forward", DEFECT_WINDOW_TT,
         DEFECT_WINDOW_TT + SPAN_DAYS),
        ("defect window backward", DEFECT_WINDOW_TT,
         DEFECT_WINDOW_TT - SPAN_DAYS),
        ("truncated at ceiling", d441_2.tt_end - 1.0,
         d441_2.tt_end - 1.0 + SPAN_DAYS),
        ("recovered at floor", d441_1.tt_start + 1.0,
         d441_1.tt_start + 1.0 - SPAN_DAYS),
    )


class TestContiguousFrontierInvariant(FrontierAssertions):
    def test_every_frontier_satisfies_the_invariants(self):
        for label, anchor, horizon in frontier_cases():
            with self.subTest(case=label):
                frontier = frontier_for(anchor, horizon)
                self.assert_frontier_invariants(
                    frontier, anchor,
                    REFERENCE_LATITUDE, REFERENCE_LONGITUDE,
                )

    def test_the_anchor_is_always_retained_exactly(self):
        for label, anchor, horizon in frontier_cases():
            with self.subTest(case=label):
                frontier = frontier_for(anchor, horizon)
                backward = horizon < anchor
                if backward:
                    self.assertEqual(frontier.tt_hi, anchor)
                else:
                    self.assertEqual(frontier.tt_lo, anchor)

    def test_only_the_far_endpoint_is_ever_shortened(self):
        for label, anchor, horizon in frontier_cases():
            with self.subTest(case=label):
                frontier = frontier_for(anchor, horizon)
                backward = horizon < anchor
                far = frontier.tt_lo if backward else frontier.tt_hi
                if frontier.complete:
                    self.assertEqual(far, horizon)
                elif backward:
                    self.assertGreater(far, horizon)
                else:
                    self.assertLess(far, horizon)

    def test_declared_coverage_contains_the_whole_returned_interval(self):
        for label, anchor, horizon in frontier_cases():
            with self.subTest(case=label):
                frontier = frontier_for(anchor, horizon)
                self.assertTrue(
                    declares(frontier.kernel, frontier.tt_lo, frontier.tt_hi)
                )

    def test_the_search_never_leaves_the_returned_frontier(self):
        """Instrument the real machinery: no sample escapes the frontier."""
        for label, anchor, horizon in frontier_cases():
            with self.subTest(case=label):
                frontier = frontier_for(anchor, horizon)
                seen = sampled_states(
                    frontier, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
                )
                self.assertTrue(seen, "the search sampled nothing")
                self.assertGreaterEqual(min(seen), frontier.tt_lo)
                self.assertLessEqual(max(seen), frontier.tt_hi)

    def test_no_unsupported_state_lies_inside_a_returned_frontier(self):
        for label, anchor, horizon in frontier_cases():
            with self.subTest(case=label):
                frontier = frontier_for(anchor, horizon)
                probes = np.linspace(frontier.tt_lo, frontier.tt_hi, 12)
                for tt in probes:
                    self.assertTrue(
                        evaluates(
                            frontier.kernel, float(tt),
                            REFERENCE_LATITUDE, REFERENCE_LONGITUDE,
                        ),
                        "%s: unsupported state inside the frontier" % label,
                    )


# --- 10. Structural independence -------------------------------------------


def executable_source(block):
    """Return a block's executable logic, prose excluded.

    Follows the established repository pattern, with one correction: an AST
    node's ``body`` attribute is not always a list. ``ast.IfExp`` - the
    conditional expression - carries a single expression there, and
    subscripting it raises TypeError. The A2-3a block legitimately uses
    conditional expressions to derive search direction, so the list check
    below is required rather than defensive.

    The existing A2-1 and A2-2 helpers are deliberately left alone: they are
    already certified against blocks that contain no conditional expression,
    and this operation does not modify them.
    """
    lines = block.splitlines(keepends=True)
    drop = set()
    for node in ast.walk(ast.parse(block)):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if (isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            drop.update(range(first.lineno - 1, first.end_lineno))
    kept = (l for i, l in enumerate(lines) if i not in drop)
    return "".join(l for l in kept if not l.lstrip().startswith("#"))


class TestExecutableSourceHelper(unittest.TestCase):
    """The new helper must survive the node shapes A2-3a actually uses."""

    def test_conditional_expressions_do_not_break_extraction(self):
        source = (
            'def f(a, b):\n'
            '    """Doc."""\n'
            '    return a if a < b else b\n'
        )
        extracted = executable_source(source)
        self.assertIn("return a if a < b else b", extracted)
        self.assertNotIn("Doc.", extracted)

    def test_comments_and_docstrings_are_removed(self):
        source = (
            'def f():\n'
            '    """Doc."""\n'
            '    # comment\n'
            '    return 1\n'
        )
        extracted = executable_source(source)
        self.assertNotIn("Doc.", extracted)
        self.assertNotIn("comment", extracted)
        self.assertIn("return 1", extracted)


class TestA2_3aStructuralIndependence(unittest.TestCase):
    """The A2-3a block must not inherit civil routing or legacy guards.

    Scoped to the A2-3a block's executable logic. The same tokens exist
    legitimately elsewhere in astronomy_solver.py, in frozen legacy code
    this increment does not touch, and in prose that names an excluded
    mechanism in order to state that it is excluded.
    """

    def setUp(self):
        source = inspect.getsource(astronomy_solver)
        index = source.find(A2_3A_BLOCK_MARKER)
        self.assertNotEqual(index, -1, "A2-3a block marker not found")
        following = GOVERNED_BLOCK_MARKER.search(
            source, index + len(A2_3A_BLOCK_MARKER)
        )
        self.block = source[
            index:following.start() if following else len(source)
        ]
        self.preamble = source[:index]
        self.executable = executable_source(self.block)

    def test_block_was_located_and_is_substantial(self):
        self.assertIn("def supported_search_frontier", self.block)
        self.assertIn("def _admit_bracket", self.block)
        self.assertIn("def _first_evaluable_state", self.block)
        self.assertIn("def _governed_observer", self.block)
        self.assertIn("def _observation_is_evaluable", self.block)
        self.assertIn("class SupportedSearchFrontier", self.block)
        self.assertGreater(len(self.block), 1000)

    def test_block_stops_at_the_next_governed_block(self):
        """The scan must cover A2-3a and end there, not annex what follows."""
        self.assertIsNone(
            GOVERNED_BLOCK_MARKER.search(self.block, len(A2_3A_BLOCK_MARKER)),
            "the A2-3a scan reaches into a later governed block",
        )

    def test_block_does_not_use_civil_routing_or_legacy_guards(self):
        # The oracle must not pass by inspecting nothing.
        self.assertIn("def supported_search_frontier", self.executable)
        self.assertIn("PINNED_KERNEL_PRECEDENCE", self.executable)
        self.assertIn("_first_evaluable_state", self.executable)

        # Prose really was excluded.
        self.assertNotIn("TWO QUESTIONS, KEPT SEPARATE", self.executable)
        self.assertNotIn("light time", self.executable)

        for token in (
            "choose_kernel_name",
            "PRIMARY_START_YEAR",
            "PRIMARY_END_YEAR",
            "ts.utc(",
            "3600",
            "min_gap",
            "epsilon",
            "tolerance",
            "timedelta",
            "/ 15",
            "utc_datetime",
            "isoformat",
            "lru_cache",
            "cache",
            "datetime",
            "strftime",
        ):
            with self.subTest(token=token):
                self.assertNotIn(token, self.executable)

    def test_block_encodes_no_fixed_light_time_margin(self):
        """No second-count may stand in for the measured reach."""
        for token in ("480", "489", "490", "500", "507", "510", "520"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.executable)

    def test_observer_parameter_name_is_not_prohibited(self):
        """`longitude` is a legitimate observer parameter here."""
        self.assertIn("longitude", self.executable)

    def test_block_consumes_the_certified_substrate(self):
        for token in (
            "_exact_finite_tt",
            "kernel_coverage_tt",
            "select_kernel_containing_instant",
            "select_kernel_for_interval",
            "SunsetChronologyError",
            "REASON_INSTANT_STATE_INVALID",
            "almanac.sunrise_sunset",
            "wgs84.latlon",
        ):
            with self.subTest(token=token):
                self.assertIn(token, self.executable)

    def test_frozen_legacy_code_is_still_present_ahead_of_the_block(self):
        """Guards against 'passing' by having deleted the legacy code."""
        for token in (
            "def choose_kernel_name",
            "PRIMARY_START_YEAR",
            "PRIMARY_END_YEAR",
            "min_gap_seconds",
            "def find_sunset_successor",
            "def kernel_coverage_tt",
            "def select_kernel_containing_instant",
            "def select_kernel_for_interval",
        ):
            with self.subTest(token=token):
                self.assertIn(token, self.preamble)


if __name__ == "__main__":
    unittest.main()
