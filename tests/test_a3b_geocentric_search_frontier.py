"""A3b-i verification for the geocentric supported search frontier.

Covers GeocentricSearchFrontier, _admit_geocentric_bracket and
geocentric_search_frontier in astronomy_solver.py.

Test framework: standard-library unittest.
Scientific execution uses the repository's certified Skyfield runtime and
the real pinned Authority ephemeris artifacts.

INDEPENDENT ORACLE DISCIPLINE

Reason codes, pinned artifact filenames, the record field names, the block
marker and the governed-marker grammar are declared as test-local literals.
They are deliberately NOT imported from astronomy_solver: importing a
constant to check that same constant would make the test agree with a
defective value instead of detecting it.

Evaluability is decided here by invoking almanac.seasons directly and
catching EphemerisRangeError. It does not call _observation_is_evaluable.
The first evaluable state is located by a test-local bisection oracle, not
by _first_evaluable_state. Neither production helper can therefore confirm
itself.

kernel_coverage_tt IS used, as certified A2-1 infrastructure, to obtain
declared bounds.

SCOPE

A3b-i answers one question: for a requested directional TT interval, which
pinned artifact can actually support the geocentric solar-longitude
computation, and over exactly what contiguous interval. It solves no event,
accepts no event kind, defines no horizon and serves no route, so this file
asserts nothing about any of those.

WHAT IS DELIBERATELY NOT ENCODED

No light-time second-count appears as an expected value. Every reach
expectation is measured at test time against the real computation. No
horizon constant appears: each case supplies its own horizon_tt.
"""

import ast
import dataclasses
import inspect
import math
import re
import unittest
from unittest import mock

import numpy as np
from skyfield import almanac
from skyfield.api import wgs84
from skyfield.errors import EphemerisRangeError

import astronomy_solver
from astronomy_solver import (
    choose_kernel_name,
    geocentric_search_frontier,
    kernel_coverage_tt,
    load_kernel,
    ts,
)
from scientific_environment import ScientificEnvironmentError

# --- Test-local oracles ----------------------------------------------------

DE440 = "de440.bsp"
DE441_PART_1 = "de441_part-1.bsp"
DE441_PART_2 = "de441_part-2.bsp"

PINNED_ARTIFACTS = (DE440, DE441_PART_1, DE441_PART_2)

REASON_COVERAGE_EXHAUSTED = "EPHEMERIS_COVERAGE_EXHAUSTED"
REASON_REACH_EXHAUSTED = "EPHEMERIS_REACH_EXHAUSTED"
REASON_INSTANT_INVALID = "INSTANT_STATE_INVALID"
REASON_OBSERVER_OUT_OF_DOMAIN = "OBSERVER_OUT_OF_DOMAIN"

EXPECTED_FRONTIER_FIELDS = ("kernel", "tt_lo", "tt_hi", "complete",
                            "truncation_reason")

# Horizons are supplied per case; A3b-i holds no span constant and neither
# does this file beyond these probe widths.
WIDE_HORIZON_DAYS = 380.0
NARROW_HORIZON_DAYS = 30.0

INTERIOR_TT = 2460678.0

DEFECT_WINDOW_TT = 2689118.000800741
DEFECT_WINDOW_CIVIL_YEAR = 2650

# A fixed reference observer, used ONLY to obtain topocentric floors for the
# non-substitutability comparison. It is never passed to A3b-i, which
# accepts no observer at all.
REFERENCE_LATITUDE = 40.7406
REFERENCE_LONGITUDE = -73.9

A3B_I_BLOCK_MARKER = "# A3b-i - geocentric supported search frontier."

GOVERNED_BLOCK_MARKER = re.compile(r"^# A\d+[a-z]?(?:-[0-9a-z]+)? - ",
                                   re.MULTILINE)


# --- Independent helpers ---------------------------------------------------


def coverage(kernel_name):
    return kernel_coverage_tt(kernel_name)


def union_bounds():
    lo = min(coverage(name).tt_start for name in PINNED_ARTIFACTS)
    hi = max(coverage(name).tt_end for name in PINNED_ARTIFACTS)
    return lo, hi


def season_predicate(kernel_name):
    """The real geocentric solar-longitude predicate. No observer."""
    return almanac.seasons(load_kernel(kernel_name))


def geocentric_evaluates(kernel_name, tt):
    """Independent geocentric evaluability oracle.

    Performs the real computation. Deliberately does not call
    _observation_is_evaluable, so that helper cannot confirm itself.
    """
    try:
        season_predicate(kernel_name)(ts.tt_jd(tt))
    except EphemerisRangeError:
        return False

    return True


def topocentric_evaluates(kernel_name, tt, latitude, longitude):
    """The sunset computation's evaluability, for the comparison only."""
    try:
        almanac.sunrise_sunset(
            load_kernel(kernel_name), wgs84.latlon(latitude, longitude)
        )(ts.tt_jd(tt))
    except EphemerisRangeError:
        return False

    return True


def _bisect_first_true(probe, bad, good):
    """Representation-exhaustive bisection oracle, written test-locally."""
    while True:
        mid = bad + (good - bad) / 2.0
        if mid == bad or mid == good:
            return good
        if probe(mid):
            good = mid
        else:
            bad = mid


_GEO_FLOORS = {}
_TOPO_FLOORS = {}


def geocentric_floor(kernel_name):
    """Measured first geocentrically evaluable state above declared start."""
    if kernel_name not in _GEO_FLOORS:
        record = coverage(kernel_name)
        _GEO_FLOORS[kernel_name] = _bisect_first_true(
            lambda tt: geocentric_evaluates(kernel_name, tt),
            record.tt_start,
            record.tt_start + 0.02,
        )
    return _GEO_FLOORS[kernel_name]


def topocentric_floor(kernel_name):
    """Measured first topocentrically evaluable state, reference observer."""
    if kernel_name not in _TOPO_FLOORS:
        record = coverage(kernel_name)
        _TOPO_FLOORS[kernel_name] = _bisect_first_true(
            lambda tt: topocentric_evaluates(
                kernel_name, tt, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
            ),
            record.tt_start,
            record.tt_start + 0.02,
        )
    return _TOPO_FLOORS[kernel_name]


def declares(kernel_name, tt_lo, tt_hi):
    """Test-local declared-containment oracle: exact, closed, inclusive."""
    record = coverage(kernel_name)
    return record.tt_start <= tt_lo and tt_hi <= record.tt_end


def executable_source(block):
    """Return a block's executable logic, prose excluded."""
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


class FrontierAssertions(unittest.TestCase):
    def assert_fails_closed(self, reason, anchor_tt, horizon_tt):
        with self.assertRaises(ScientificEnvironmentError) as caught:
            geocentric_search_frontier(anchor_tt, horizon_tt)
        self.assertEqual(caught.exception.reason, reason)
        return caught.exception

    def assert_frontier_invariants(self, frontier, anchor, horizon):
        self.assertIsInstance(
            frontier, astronomy_solver.GeocentricSearchFrontier
        )
        self.assertIn(frontier.kernel, PINNED_ARTIFACTS)
        self.assertLess(frontier.tt_lo, frontier.tt_hi)

        # The anchor is retained exactly and is one of the two endpoints.
        self.assertIn(anchor, (frontier.tt_lo, frontier.tt_hi))

        # One artifact declares the WHOLE returned interval.
        self.assertTrue(
            declares(frontier.kernel, frontier.tt_lo, frontier.tt_hi),
            "returned frontier is not wholly declared by its own artifact",
        )

        # Both endpoints genuinely evaluate under that artifact.
        for endpoint in (frontier.tt_lo, frontier.tt_hi):
            self.assertTrue(
                geocentric_evaluates(frontier.kernel, endpoint),
                "frontier endpoint %r is not geocentrically evaluable"
                % endpoint,
            )

        # Only the far side may have shortened.
        backward = horizon < anchor
        far = frontier.tt_lo if backward else frontier.tt_hi
        if frontier.complete:
            self.assertEqual(far, horizon)
            self.assertIsNone(frontier.truncation_reason)
        elif backward:
            self.assertGreater(far, horizon)
            self.assertIn(
                frontier.truncation_reason,
                (REASON_COVERAGE_EXHAUSTED, REASON_REACH_EXHAUSTED),
            )
        else:
            self.assertLess(far, horizon)
            self.assertIn(
                frontier.truncation_reason,
                (REASON_COVERAGE_EXHAUSTED, REASON_REACH_EXHAUSTED),
            )


# --- 1. Record contract ----------------------------------------------------


class TestRecordContract(unittest.TestCase):
    def setUp(self):
        self.frontier = geocentric_search_frontier(
            INTERIOR_TT, INTERIOR_TT + WIDE_HORIZON_DAYS
        )

    def test_fields_are_exactly_the_expected_set(self):
        self.assertEqual(
            tuple(f.name for f in dataclasses.fields(self.frontier)),
            EXPECTED_FRONTIER_FIELDS,
        )

    def test_record_is_frozen(self):
        for field in EXPECTED_FRONTIER_FIELDS:
            with self.subTest(field=field):
                with self.assertRaises(Exception):
                    setattr(self.frontier, field, None)

    def test_is_a_distinct_type_from_the_sunset_frontier(self):
        self.assertIsNot(
            astronomy_solver.GeocentricSearchFrontier,
            astronomy_solver.SupportedSearchFrontier,
        )
        self.assertNotIsInstance(
            self.frontier, astronomy_solver.SupportedSearchFrontier
        )
        sunset_frontier = astronomy_solver.supported_search_frontier(
            INTERIOR_TT, INTERIOR_TT + 3.0,
            REFERENCE_LATITUDE, REFERENCE_LONGITUDE,
        )
        self.assertNotIsInstance(
            sunset_frontier, astronomy_solver.GeocentricSearchFrontier
        )

    def test_no_observer_event_or_transport_field(self):
        names = {f.name for f in dataclasses.fields(self.frontier)}
        for forbidden in (
            "latitude", "longitude", "observer", "location",
            "kind", "event", "event_kind", "solar_longitude",
            "utc", "iso", "isoformat", "datetime", "timestamp", "unix",
            "year", "month", "day", "weekday", "season",
        ):
            with self.subTest(field=forbidden):
                self.assertNotIn(forbidden, names)
                self.assertFalse(hasattr(self.frontier, forbidden))

    def test_exactly_five_fields(self):
        self.assertEqual(len(dataclasses.fields(self.frontier)), 5)


# --- 2. Observer-free contract ---------------------------------------------


class TestObserverFreeContract(unittest.TestCase):
    def test_signature_is_exactly_anchor_and_horizon(self):
        signature = inspect.signature(geocentric_search_frontier)
        self.assertEqual(
            list(signature.parameters), ["anchor_tt", "horizon_tt"]
        )

    def test_admission_signature_takes_no_observer(self):
        signature = inspect.signature(
            astronomy_solver._admit_geocentric_bracket
        )
        self.assertEqual(list(signature.parameters), ["tt_lo", "tt_hi"])

    def test_block_contains_no_observer_machinery(self):
        block = a3b_i_block()
        executable = executable_source(block)
        for token in (
            "wgs84", "latlon",
            "_governed_observer", "OBSERVER_OUT_OF_DOMAIN",
            "sunrise_sunset", "supported_search_frontier",
        ):
            with self.subTest(token=token):
                self.assertNotIn(token, executable)

    def test_no_function_in_the_block_takes_an_observer_parameter(self):
        """Checked on the parse tree, not as text.

        A substring scan cannot distinguish an observer coordinate from the
        name of the quantity being computed: the diagnostic messages
        legitimately say "solar-longitude". The parameter names are the
        thing that matters, so they are read from the AST.
        """
        tree = ast.parse(a3b_i_block())
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            arguments = node.args
            names = [
                argument.arg
                for group in (
                    arguments.posonlyargs,
                    arguments.args,
                    arguments.kwonlyargs,
                )
                for argument in group
            ]
            for extra in (arguments.vararg, arguments.kwarg):
                if extra is not None:
                    names.append(extra.arg)

            with self.subTest(function=node.name):
                for forbidden in ("latitude", "longitude"):
                    self.assertNotIn(
                        forbidden, names,
                        "%s takes an observer parameter" % node.name,
                    )

    def test_calling_with_an_observer_is_a_type_error(self):
        with self.assertRaises(TypeError):
            geocentric_search_frontier(
                INTERIOR_TT, INTERIOR_TT + 1.0, REFERENCE_LATITUDE
            )


def a3b_i_block():
    source = inspect.getsource(astronomy_solver)
    index = source.find(A3B_I_BLOCK_MARKER)
    assert index != -1, "A3b-i block marker not found"
    following = GOVERNED_BLOCK_MARKER.search(
        source, index + len(A3B_I_BLOCK_MARKER)
    )
    return source[index:following.start() if following else len(source)]


# --- 3. Declared coverage AND actual evaluability ---------------------------


class TestConjunctionRule(FrontierAssertions):
    def admit(self, tt_lo, tt_hi):
        return astronomy_solver._admit_geocentric_bracket(tt_lo, tt_hi)

    def test_declared_containment_is_required(self):
        lo, hi = union_bounds()
        self.assertIsNone(self.admit(lo - 1.0, lo + 1.0))
        self.assertIsNone(self.admit(hi - 1.0, hi + 1.0))
        self.assertIsNone(self.admit(lo, hi))

    def test_actual_evaluability_is_required(self):
        """Declared alone is not enough at an artifact's declared start."""
        record = coverage(DE441_PART_1)
        bracket = (record.tt_start, record.tt_start + NARROW_HORIZON_DAYS)
        self.assertTrue(declares(DE441_PART_1, *bracket))
        self.assertFalse(
            geocentric_evaluates(DE441_PART_1, record.tt_start),
            "this probe requires an unevaluable declared start",
        )
        self.assertIsNone(self.admit(*bracket))

    def test_scan_continues_past_a_declaring_but_unevaluable_artifact(self):
        """DE440's declared start is declared by DE440 and unevaluable."""
        start = coverage(DE440).tt_start
        bracket = (start, start + NARROW_HORIZON_DAYS)

        self.assertTrue(declares(DE440, *bracket))
        self.assertFalse(geocentric_evaluates(DE440, start))

        self.assertTrue(declares(DE441_PART_1, *bracket))
        self.assertTrue(geocentric_evaluates(DE441_PART_1, start))

        self.assertEqual(self.admit(*bracket), DE441_PART_1)

    def test_admission_is_deterministic(self):
        bracket = (INTERIOR_TT, INTERIOR_TT + WIDE_HORIZON_DAYS)
        self.assertEqual(len({self.admit(*bracket) for _ in range(3)}), 1)


# --- 4. Endpoint-sufficiency invariant -------------------------------------


class TestEndpointSufficiency(unittest.TestCase):
    """Two endpoint probes certify the whole bracket.

    A property of the certified runtime and pinned artifact set, verified
    here rather than assumed.
    """

    INTERIOR_SAMPLES = 60

    def assert_interior_is_evaluable(self, kernel_name, tt_lo, tt_hi, label):
        self.assertTrue(
            geocentric_evaluates(kernel_name, tt_lo), "%s: lower" % label
        )
        self.assertTrue(
            geocentric_evaluates(kernel_name, tt_hi), "%s: upper" % label
        )
        failures = [
            float(tt)
            for tt in np.linspace(tt_lo, tt_hi, self.INTERIOR_SAMPLES)
            if not geocentric_evaluates(kernel_name, float(tt))
        ]
        self.assertEqual(
            failures, [],
            "%s: interior states failed while both endpoints evaluated"
            % label,
        )

    def test_brackets_at_each_first_evaluable_state(self):
        for kernel_name in PINNED_ARTIFACTS:
            with self.subTest(kernel=kernel_name):
                floor = geocentric_floor(kernel_name)
                self.assert_interior_is_evaluable(
                    kernel_name, floor, floor + WIDE_HORIZON_DAYS,
                    "%s at floor" % kernel_name,
                )

    def test_brackets_ending_at_each_declared_upper_bound(self):
        for kernel_name in PINNED_ARTIFACTS:
            with self.subTest(kernel=kernel_name):
                end = coverage(kernel_name).tt_end
                self.assert_interior_is_evaluable(
                    kernel_name, end - WIDE_HORIZON_DAYS, end,
                    "%s at upper bound" % kernel_name,
                )

    def test_brackets_in_overlap_and_precedence_territory(self):
        probes = (
            (DE440, coverage(DE441_PART_2).tt_start, "de441p2 start"),
            (DE440, coverage(DE441_PART_1).tt_end - WIDE_HORIZON_DAYS,
             "de441p1 end"),
            (DE441_PART_1, coverage(DE440).tt_start + 1.0, "de440 start"),
            (DE441_PART_2, coverage(DE440).tt_end - 1.0, "de440 end"),
        )
        for kernel_name, tt_lo, label in probes:
            with self.subTest(kernel=kernel_name, where=label):
                self.assert_interior_is_evaluable(
                    kernel_name, tt_lo, tt_lo + NARROW_HORIZON_DAYS, label
                )

    def test_admitted_frontiers_have_no_interior_failure(self):
        cases = (
            (INTERIOR_TT, INTERIOR_TT + WIDE_HORIZON_DAYS),
            (coverage(DE440).tt_start - 1.0,
             coverage(DE440).tt_start - 1.0 + WIDE_HORIZON_DAYS),
            (coverage(DE440).tt_end - 1.0,
             coverage(DE440).tt_end - 1.0 + WIDE_HORIZON_DAYS),
            (DEFECT_WINDOW_TT, DEFECT_WINDOW_TT + WIDE_HORIZON_DAYS),
        )
        for anchor, horizon in cases:
            with self.subTest(anchor=anchor):
                frontier = geocentric_search_frontier(anchor, horizon)
                self.assert_interior_is_evaluable(
                    frontier.kernel, frontier.tt_lo, frontier.tt_hi,
                    "admitted %r" % anchor,
                )


# --- 5. Computation-specific reach -----------------------------------------


class TestComputationSpecificReach(unittest.TestCase):
    """The geocentric reach is its own, and is not the sunset reach."""

    def test_each_artifact_has_a_measurable_geocentric_floor(self):
        for kernel_name in PINNED_ARTIFACTS:
            with self.subTest(kernel=kernel_name):
                record = coverage(kernel_name)
                floor = geocentric_floor(kernel_name)
                self.assertGreater(floor, record.tt_start)
                self.assertTrue(geocentric_evaluates(kernel_name, floor))
                self.assertFalse(
                    geocentric_evaluates(
                        kernel_name, math.nextafter(floor, -math.inf)
                    ),
                    "the located boundary is not exact",
                )

    def test_geocentric_and_topocentric_floors_differ(self):
        for kernel_name in PINNED_ARTIFACTS:
            with self.subTest(kernel=kernel_name):
                self.assertNotEqual(
                    geocentric_floor(kernel_name),
                    topocentric_floor(kernel_name),
                    "the two reaches must not be assumed equal",
                )

    def test_the_difference_is_not_uniformly_signed(self):
        """One reach is not simply a margin on the other.

        Measured against a fixed reference observer: for at least one
        artifact the geocentric boundary is the lower of the two, and for
        at least one it is the higher. Neither can bound the other.
        """
        lower = [
            k for k in PINNED_ARTIFACTS
            if geocentric_floor(k) < topocentric_floor(k)
        ]
        higher = [
            k for k in PINNED_ARTIFACTS
            if geocentric_floor(k) > topocentric_floor(k)
        ]
        self.assertTrue(lower, "expected an artifact with a lower geo floor")
        self.assertTrue(higher, "expected an artifact with a higher geo floor")

    def test_a_topocentrically_supported_state_can_be_geocentrically_unsupported(self):
        """Direct proof the sunset frontier cannot substitute for this one."""
        found = False
        for kernel_name in PINNED_ARTIFACTS:
            if topocentric_floor(kernel_name) < geocentric_floor(kernel_name):
                state = topocentric_floor(kernel_name)
                self.assertTrue(
                    topocentric_evaluates(
                        kernel_name, state,
                        REFERENCE_LATITUDE, REFERENCE_LONGITUDE,
                    )
                )
                self.assertFalse(
                    geocentric_evaluates(kernel_name, state),
                    "%s: state is topocentrically supported yet "
                    "geocentrically unsupported" % kernel_name,
                )
                found = True
        self.assertTrue(found, "no artifact exhibited this direction")

    def test_a_geocentrically_supported_state_can_be_topocentrically_unsupported(self):
        """And the reverse, so neither reach bounds the other."""
        found = False
        for kernel_name in PINNED_ARTIFACTS:
            if geocentric_floor(kernel_name) < topocentric_floor(kernel_name):
                state = geocentric_floor(kernel_name)
                self.assertTrue(geocentric_evaluates(kernel_name, state))
                self.assertFalse(
                    topocentric_evaluates(
                        kernel_name, state,
                        REFERENCE_LATITUDE, REFERENCE_LONGITUDE,
                    ),
                    "%s: state is geocentrically supported yet "
                    "topocentrically unsupported" % kernel_name,
                )
                found = True
        self.assertTrue(found, "no artifact exhibited this direction")

    def test_no_light_time_constant_is_encoded(self):
        executable = executable_source(a3b_i_block())
        for token in ("480", "489", "490", "500", "507", "510", "520"):
            with self.subTest(token=token):
                self.assertNotIn(token, executable)


# --- 6. Lower-reach recovery -----------------------------------------------


class TestLowerReachRecovery(FrontierAssertions):
    def setUp(self):
        self.record = coverage(DE441_PART_1)
        self.anchor = self.record.tt_start + 5.0
        self.horizon = self.anchor - WIDE_HORIZON_DAYS

    def test_the_geometry_the_recovery_requires_really_holds(self):
        self.assertFalse(
            geocentric_evaluates(DE441_PART_1, self.record.tt_start),
            "declared lower bound must be geocentrically unevaluable",
        )
        self.assertTrue(
            geocentric_evaluates(DE441_PART_1, self.anchor),
            "anchor must be evaluable",
        )
        self.assertTrue(
            declares(DE441_PART_1, self.record.tt_start, self.anchor)
        )

    def test_recovery_returns_a_shortened_reach_truncated_frontier(self):
        frontier = geocentric_search_frontier(self.anchor, self.horizon)
        self.assert_frontier_invariants(frontier, self.anchor, self.horizon)
        self.assertFalse(frontier.complete)
        self.assertEqual(
            frontier.truncation_reason, REASON_REACH_EXHAUSTED
        )
        self.assertEqual(frontier.kernel, DE441_PART_1)

    def test_anchor_is_unchanged(self):
        frontier = geocentric_search_frontier(self.anchor, self.horizon)
        self.assertEqual(frontier.tt_hi, self.anchor)

    def test_returned_lower_bound_is_the_exact_first_evaluable_state(self):
        frontier = geocentric_search_frontier(self.anchor, self.horizon)
        self.assertTrue(geocentric_evaluates(DE441_PART_1, frontier.tt_lo))
        self.assertFalse(
            geocentric_evaluates(
                DE441_PART_1, math.nextafter(frontier.tt_lo, -math.inf)
            ),
            "no evaluable state may be discarded",
        )
        self.assertEqual(frontier.tt_lo, geocentric_floor(DE441_PART_1))
        self.assertGreater(frontier.tt_lo, self.record.tt_start)

    def test_first_evaluable_state_is_actually_invoked(self):
        """Not an encoded margin: the neutral primitive does the work."""
        with mock.patch.object(
            astronomy_solver, "_first_evaluable_state",
            wraps=astronomy_solver._first_evaluable_state,
        ) as recorder:
            frontier = geocentric_search_frontier(self.anchor, self.horizon)

        self.assertEqual(recorder.call_count, 1)
        probe, unevaluable_tt, evaluable_tt = recorder.call_args.args
        self.assertLess(unevaluable_tt, evaluable_tt)
        self.assertEqual(evaluable_tt, self.anchor)
        self.assertFalse(
            geocentric_evaluates(DE441_PART_1, unevaluable_tt)
        )

    def test_recovery_is_deterministic(self):
        results = {
            geocentric_search_frontier(self.anchor, self.horizon)
            for _ in range(3)
        }
        self.assertEqual(len(results), 1)


# --- 7. Forward unevaluable anchor -----------------------------------------


class TestForwardUnevaluableAnchor(FrontierAssertions):
    def test_forward_from_an_unevaluable_anchor_fails_closed(self):
        record = coverage(DE441_PART_1)
        for offset in (0.0, 0.001, 0.004):
            anchor = record.tt_start + offset
            with self.subTest(offset=offset):
                self.assertFalse(
                    geocentric_evaluates(DE441_PART_1, anchor),
                    "this probe requires an unevaluable anchor",
                )
                self.assert_fails_closed(
                    REASON_REACH_EXHAUSTED,
                    anchor, anchor + WIDE_HORIZON_DAYS,
                )

    def test_the_anchor_is_never_moved_forward(self):
        record = coverage(DE441_PART_1)
        anchor = record.tt_start + 0.001
        with mock.patch.object(
            astronomy_solver, "_first_evaluable_state",
            wraps=astronomy_solver._first_evaluable_state,
        ) as recorder:
            with self.assertRaises(ScientificEnvironmentError):
                geocentric_search_frontier(
                    anchor, anchor + WIDE_HORIZON_DAYS
                )
        self.assertEqual(
            recorder.call_count, 0,
            "no recovery may run on a forward request",
        )


# --- 8. Coverage-edge truncation -------------------------------------------


class TestCoverageEdgeTruncation(FrontierAssertions):
    """Forward truncation, and both absolute collapse cases.

    Note on the backward direction: a backward frontier clamped to an
    artifact's declared start always lands on a geocentrically unevaluable
    state, because every pinned artifact's declared start precedes its own
    first evaluable state. Backward truncation therefore always resolves
    through the lower-reach recovery and carries REACH_EXHAUSTED, and a
    backward COVERAGE_EXHAUSTED truncation is unreachable rather than
    untested. The collapse case below is the backward coverage outcome.
    """

    def test_forward_truncation_at_the_absolute_ceiling(self):
        ceiling = coverage(DE441_PART_2).tt_end
        for offset in (1.0, 10.0):
            anchor = ceiling - offset
            with self.subTest(offset=offset):
                frontier = geocentric_search_frontier(
                    anchor, anchor + WIDE_HORIZON_DAYS
                )
                self.assert_frontier_invariants(
                    frontier, anchor, anchor + WIDE_HORIZON_DAYS
                )
                self.assertFalse(frontier.complete)
                self.assertEqual(
                    frontier.truncation_reason, REASON_COVERAGE_EXHAUSTED
                )
                self.assertEqual(frontier.tt_lo, anchor)
                self.assertEqual(frontier.tt_hi, ceiling)

    def test_forward_collapse_at_the_ceiling_fails_closed(self):
        ceiling = coverage(DE441_PART_2).tt_end
        self.assert_fails_closed(
            REASON_COVERAGE_EXHAUSTED, ceiling, ceiling + WIDE_HORIZON_DAYS
        )

    def test_backward_collapse_at_the_floor_fails_closed(self):
        floor = coverage(DE441_PART_1).tt_start
        self.assert_fails_closed(
            REASON_COVERAGE_EXHAUSTED, floor, floor - WIDE_HORIZON_DAYS
        )

    def test_backward_truncation_carries_reach_exhausted(self):
        record = coverage(DE441_PART_1)
        anchor = record.tt_start + 5.0
        frontier = geocentric_search_frontier(
            anchor, anchor - WIDE_HORIZON_DAYS
        )
        self.assertFalse(frontier.complete)
        self.assertEqual(
            frontier.truncation_reason, REASON_REACH_EXHAUSTED
        )


# --- 9. Precedence and transitions -----------------------------------------


class TestPrecedenceAndTransitions(FrontierAssertions):
    def test_de440_is_preferred_where_it_supports_the_interval(self):
        record = coverage(DE440)
        span = record.tt_end - record.tt_start
        probes = [record.tt_start + span * f for f in (0.25, 0.5, 0.75)]
        probes.append(coverage(DE441_PART_2).tt_start)
        for anchor in probes:
            with self.subTest(anchor=anchor):
                frontier = geocentric_search_frontier(
                    anchor, anchor + WIDE_HORIZON_DAYS
                )
                self.assertEqual(frontier.kernel, DE440)
                self.assertTrue(frontier.complete)

    def test_de440_lower_straddle_resolves_through_de441_part_1(self):
        anchor = coverage(DE440).tt_start - 1.0
        frontier = geocentric_search_frontier(
            anchor, anchor + WIDE_HORIZON_DAYS
        )
        self.assert_frontier_invariants(
            frontier, anchor, anchor + WIDE_HORIZON_DAYS
        )
        self.assertEqual(frontier.kernel, DE441_PART_1)
        self.assertTrue(frontier.complete)

    def test_de440_upper_straddle_resolves_through_de441_part_2(self):
        anchor = coverage(DE440).tt_end - 1.0
        frontier = geocentric_search_frontier(
            anchor, anchor + WIDE_HORIZON_DAYS
        )
        self.assert_frontier_invariants(
            frontier, anchor, anchor + WIDE_HORIZON_DAYS
        )
        self.assertEqual(frontier.kernel, DE441_PART_2)
        self.assertTrue(frontier.complete)

    def test_2650_defect_present_in_legacy_but_bypassed_here(self):
        self.assertEqual(choose_kernel_name(DEFECT_WINDOW_CIVIL_YEAR), DE440)
        self.assertGreater(DEFECT_WINDOW_TT, coverage(DE440).tt_end)

        frontier = geocentric_search_frontier(
            DEFECT_WINDOW_TT, DEFECT_WINDOW_TT + WIDE_HORIZON_DAYS
        )
        self.assert_frontier_invariants(
            frontier, DEFECT_WINDOW_TT,
            DEFECT_WINDOW_TT + WIDE_HORIZON_DAYS,
        )
        self.assertEqual(frontier.kernel, DE441_PART_2)


# --- 10. Contiguity and anchor immobility ----------------------------------


def frontier_cases():
    d440 = coverage(DE440)
    d441_1 = coverage(DE441_PART_1)
    d441_2 = coverage(DE441_PART_2)
    W = WIDE_HORIZON_DAYS
    return (
        ("interior forward", INTERIOR_TT, INTERIOR_TT + W),
        ("interior backward", INTERIOR_TT, INTERIOR_TT - W),
        ("de440 lower straddle", d440.tt_start - 1.0,
         d440.tt_start - 1.0 + W),
        ("de440 lower backward", d440.tt_start + 1.0,
         d440.tt_start + 1.0 - W),
        ("de440 upper straddle", d440.tt_end - 1.0, d440.tt_end - 1.0 + W),
        ("de440 upper backward", d440.tt_end + 1.0, d440.tt_end + 1.0 - W),
        ("defect window", DEFECT_WINDOW_TT, DEFECT_WINDOW_TT + W),
        ("ceiling truncated", d441_2.tt_end - 1.0, d441_2.tt_end - 1.0 + W),
        ("floor recovered", d441_1.tt_start + 5.0,
         d441_1.tt_start + 5.0 - W),
    )


class TestContiguityAndAnchorImmobility(FrontierAssertions):
    def test_all_invariants_hold_for_every_case(self):
        for label, anchor, horizon in frontier_cases():
            with self.subTest(case=label):
                frontier = geocentric_search_frontier(anchor, horizon)
                self.assert_frontier_invariants(frontier, anchor, horizon)

    def test_anchor_is_always_one_endpoint_exactly(self):
        for label, anchor, horizon in frontier_cases():
            with self.subTest(case=label):
                frontier = geocentric_search_frontier(anchor, horizon)
                if horizon < anchor:
                    self.assertEqual(frontier.tt_hi, anchor)
                else:
                    self.assertEqual(frontier.tt_lo, anchor)

    def test_one_artifact_declares_the_whole_returned_interval(self):
        for label, anchor, horizon in frontier_cases():
            with self.subTest(case=label):
                frontier = geocentric_search_frontier(anchor, horizon)
                self.assertTrue(
                    declares(frontier.kernel, frontier.tt_lo, frontier.tt_hi)
                )

    def test_no_unsupported_state_lies_inside_a_returned_frontier(self):
        for label, anchor, horizon in frontier_cases():
            with self.subTest(case=label):
                frontier = geocentric_search_frontier(anchor, horizon)
                for tt in np.linspace(frontier.tt_lo, frontier.tt_hi, 12):
                    self.assertTrue(
                        geocentric_evaluates(frontier.kernel, float(tt)),
                        "%s: unsupported state inside the frontier" % label,
                    )

    def test_returned_record_cannot_carry_a_second_interval(self):
        frontier = geocentric_search_frontier(
            INTERIOR_TT, INTERIOR_TT + WIDE_HORIZON_DAYS
        )
        self.assertEqual(
            tuple(f.name for f in dataclasses.fields(frontier)),
            EXPECTED_FRONTIER_FIELDS,
        )


# --- 11. Failure taxonomy --------------------------------------------------


class TestFailureTaxonomy(FrontierAssertions):
    BAD_TT = (
        True, False, "2461485.0", None, [2461485.0], {}, object(),
        complex(1, 0), float("nan"), float("inf"), float("-inf"), 10 ** 400,
    )

    def test_malformed_anchor_reports_instant_state_invalid(self):
        for value in self.BAD_TT:
            with self.subTest(value=repr(value)):
                self.assert_fails_closed(
                    REASON_INSTANT_INVALID, value, INTERIOR_TT
                )

    def test_malformed_horizon_reports_instant_state_invalid(self):
        for value in self.BAD_TT:
            with self.subTest(value=repr(value)):
                self.assert_fails_closed(
                    REASON_INSTANT_INVALID, INTERIOR_TT, value
                )

    def test_equal_anchor_and_horizon_is_malformed(self):
        for anchor in (INTERIOR_TT, coverage(DE440).tt_start, 0.0):
            with self.subTest(anchor=anchor):
                self.assert_fails_closed(
                    REASON_INSTANT_INVALID, anchor, anchor
                )

    def test_unsupported_anchor_reports_coverage_exhausted(self):
        lo, hi = union_bounds()
        for anchor in (lo - 1.0, lo - 1e6, hi + 1.0, hi + 1e6):
            with self.subTest(anchor=anchor):
                self.assert_fails_closed(
                    REASON_COVERAGE_EXHAUSTED,
                    anchor, anchor + WIDE_HORIZON_DAYS,
                )

    def test_declared_but_unevaluable_reports_reach_exhausted(self):
        floor = coverage(DE441_PART_1).tt_start
        self.assertTrue(
            declares(DE441_PART_1, floor, floor + WIDE_HORIZON_DAYS)
        )
        self.assert_fails_closed(
            REASON_REACH_EXHAUSTED, floor, floor + WIDE_HORIZON_DAYS
        )

    def test_observer_reason_is_never_produced(self):
        probes = (
            (float("nan"), INTERIOR_TT),
            (INTERIOR_TT, INTERIOR_TT),
            (union_bounds()[0] - 1.0,
             union_bounds()[0] - 1.0 + WIDE_HORIZON_DAYS),
            (coverage(DE441_PART_1).tt_start,
             coverage(DE441_PART_1).tt_start + WIDE_HORIZON_DAYS),
        )
        for anchor, horizon in probes:
            with self.subTest(anchor=anchor):
                with self.assertRaises(ScientificEnvironmentError) as caught:
                    geocentric_search_frontier(anchor, horizon)
                self.assertNotEqual(
                    caught.exception.reason, REASON_OBSERVER_OUT_OF_DOMAIN
                )
                self.assertIn(
                    caught.exception.reason,
                    (REASON_INSTANT_INVALID, REASON_COVERAGE_EXHAUSTED,
                     REASON_REACH_EXHAUSTED),
                )


# --- 12. Structural isolation ----------------------------------------------


class TestA3bIStructuralIsolation(unittest.TestCase):
    def setUp(self):
        source = inspect.getsource(astronomy_solver)
        index = source.find(A3B_I_BLOCK_MARKER)
        self.assertNotEqual(index, -1, "A3b-i block marker not found")
        following = GOVERNED_BLOCK_MARKER.search(
            source, index + len(A3B_I_BLOCK_MARKER)
        )
        self.block = source[
            index:following.start() if following else len(source)
        ]
        self.preamble = source[:index]
        self.executable = executable_source(self.block)

    def test_block_was_located_and_is_substantial(self):
        self.assertIn("class GeocentricSearchFrontier", self.block)
        self.assertIn("def _admit_geocentric_bracket", self.block)
        self.assertIn("def geocentric_search_frontier", self.block)
        self.assertGreater(len(self.block), 1000)

    def test_block_stops_at_the_next_governed_block(self):
        self.assertIsNone(
            GOVERNED_BLOCK_MARKER.search(
                self.block, len(A3B_I_BLOCK_MARKER)
            ),
            "the A3b-i scan reaches into a later governed block",
        )

    def test_intended_operations_are_present(self):
        """Anti-vacuity: the guards below must inspect real logic."""
        for token in (
            "def geocentric_search_frontier",
            "def _admit_geocentric_bracket",
            "_exact_finite_tt",
            "kernel_coverage_tt",
            "PINNED_KERNEL_PRECEDENCE",
            "select_kernel_containing_instant",
            "select_kernel_for_interval",
            "almanac.seasons",
            "load_kernel",
            "_observation_is_evaluable",
            "_first_evaluable_state",
            "GeocentricSearchFrontier(",
        ):
            with self.subTest(token=token):
                self.assertIn(token, self.executable)
        self.assertGreater(len(self.executable), 800)

        # Prose really was excluded.
        self.assertNotIn("TWO QUESTIONS, KEPT SEPARATE", self.executable)
        self.assertNotIn("light time", self.executable)

    def test_block_defines_exactly_the_authorized_symbols(self):
        tree = ast.parse(self.block)
        classes = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
        functions = [
            n.name for n in tree.body if isinstance(n, ast.FunctionDef)
        ]
        self.assertEqual(classes, ["GeocentricSearchFrontier"])
        self.assertEqual(
            functions,
            ["_admit_geocentric_bracket", "geocentric_search_frontier"],
        )

    def test_block_introduces_no_import_constant_or_extra_helper(self):
        tree = ast.parse(self.block)
        self.assertEqual(
            [n for n in tree.body if isinstance(n, ast.Assign)], []
        )
        self.assertEqual(
            [
                n for n in tree.body
                if isinstance(n, (ast.Import, ast.ImportFrom))
            ],
            [],
        )
        self.assertEqual(
            len([n for n in tree.body if isinstance(n, ast.FunctionDef)]), 2
        )
        self.assertEqual(
            len([n for n in tree.body if isinstance(n, ast.ClassDef)]), 1
        )

    def test_block_introduces_no_reason_code(self):
        for token in ("REASON_EVENT", "REASON_SOLAR", "EVENT_KIND_INVALID",
                      "SOLAR_LONGITUDE_EVENT_UNRESOLVED"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.executable)

    def test_block_performs_no_event_solving(self):
        for token in (
            "find_discrete",
            "AstronomicalEvent",
            "SOLAR_LONGITUDE_",
            "kind",
            "event",
        ):
            with self.subTest(token=token):
                self.assertNotIn(token, self.executable)

    def test_block_has_no_civil_routing_or_legacy_guards(self):
        for token in (
            "choose_kernel_name",
            "PRIMARY_START_YEAR",
            "PRIMARY_END_YEAR",
            "ts.utc(",
            "utc_datetime",
            "utc_strftime",
            "isoformat",
            "strftime",
            "datetime",
            "timestamp",
            "timedelta",
            "3600",
            "min_gap",
            "epsilon",
            "tolerance",
            "/ 15",
            "lru_cache",
            "cache",
            "HTTPException",
            "status_code",
            "weekday",
        ):
            with self.subTest(token=token):
                self.assertNotIn(token, self.executable)

    def test_block_has_no_horizon_constant(self):
        for token in ("380", "365", "90.0", "SEARCH_SPAN", "SPAN_DAYS"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.executable)

    def test_frozen_code_is_still_present_ahead_of_the_block(self):
        for token in (
            "def choose_kernel_name",
            "PRIMARY_START_YEAR",
            "PRIMARY_END_YEAR",
            "min_gap_seconds",
            "def find_equinox",
            "def find_season_events",
            "def format_skyfield_time",
            "def kernel_coverage_tt",
            "def select_kernel_containing_instant",
            "def select_kernel_for_interval",
            "def supported_search_frontier",
            "class SupportedSearchFrontier",
            "def find_sunset_predecessor",
            "def find_sunset_from_instant",
            "class SunsetBracket",
            "class AstronomicalEvent",
        ):
            with self.subTest(token=token):
                self.assertIn(token, self.preamble)


# --- 13. A3b-0 reuse -------------------------------------------------------


class TestA3b0Reuse(unittest.TestCase):
    def setUp(self):
        self.block = a3b_i_block()
        self.executable = executable_source(self.block)

    def test_both_neutral_helpers_are_referenced(self):
        self.assertIn("_observation_is_evaluable", self.executable)
        self.assertIn("_first_evaluable_state", self.executable)

    def test_neither_helper_is_redefined_in_the_block(self):
        self.assertNotIn("def _observation_is_evaluable", self.block)
        self.assertNotIn("def _first_evaluable_state", self.block)

    def test_the_helpers_live_outside_and_ahead_of_the_block(self):
        source = inspect.getsource(astronomy_solver)
        index = source.find(A3B_I_BLOCK_MARKER)
        preamble = source[:index]
        self.assertIn("def _observation_is_evaluable", preamble)
        self.assertIn("def _first_evaluable_state", preamble)

    def test_no_bisection_is_copied_into_the_block(self):
        for token in ("while True:", "mid = bad", "mid == bad",
                      "bad = mid", "good = mid"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.executable)

    def test_helpers_accept_the_geocentric_predicate_directly(self):
        """No fabricated observer is needed to reach them."""
        record = kernel_coverage_tt(DE441_PART_1)
        probe = season_predicate(DE441_PART_1)

        self.assertFalse(
            astronomy_solver._observation_is_evaluable(
                probe, record.tt_start
            )
        )
        self.assertTrue(
            astronomy_solver._observation_is_evaluable(
                probe, record.tt_start + 1.0
            )
        )
        self.assertEqual(
            astronomy_solver._first_evaluable_state(
                probe, record.tt_start, record.tt_start + 1.0
            ),
            geocentric_floor(DE441_PART_1),
        )


if __name__ == "__main__":
    unittest.main()
