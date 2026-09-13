"""A3b-ii verification for the strict solar-longitude directional solver.

Covers the governed constants, kind validation, sampling control, canonical
event identity, strict directional semantics, frontier semantics, the
fail-closed taxonomy and structural isolation of the A3b-ii block in
astronomy_solver.py.

Test framework: standard-library unittest.
Scientific execution uses the repository's certified Skyfield runtime and the
real pinned Authority ephemeris artifacts.

INVOCATION
----------
Run from the repository root with the repository interpreter:

    .\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v

INDEPENDENT ORACLE DISCIPLINE

Reason codes, pinned artifact filenames, governed event-kind strings, the
certified extrema, the production constant values and the block marker are
declared here as test-local literals. They are deliberately NOT imported from
astronomy_solver: importing a constant to check that same constant would make
the test agree with a defective value instead of detecting it.

Canonical boundaries are relocated here by a test-local bisection oracle that
never calls _canonical_quadrant_boundary, and crossings are re-enumerated here
against almanac.seasons directly rather than through
_solar_longitude_crossings. No production helper can therefore confirm itself.

kernel_coverage_tt and geocentric_search_frontier ARE used, as certified A2-1
and A3b-i infrastructure, to obtain declared bounds and the frontier a solve
is expected to have run under.

EVIDENCE DISCIPLINE

PROVED      - follows structurally from the code under test.
MEASURED    - observed against the real pinned artifacts at test time.
EMPIRICALLY CHARACTERIZED
            - observed over a representative sample; not a universal claim.

The certified extrema below are MEASURED values tied to the pinned artifact
and runtime generation. They are not timeless astronomical constants. The
opt-in companion file rederives them.

WHAT IS DELIBERATELY NOT ENCODED

No light-time second-count appears. No civil year, Gregorian date, timezone or
calendar quantity appears. No observer appears, because A3b-ii accepts none.
"""

import ast
import inspect
import math
import re
import unittest
from unittest import mock

import numpy as np
from skyfield import almanac
from skyfield.errors import EphemerisRangeError

import astronomy_solver
from astronomy_solver import (
    AstronomicalEvent,
    find_solar_longitude_event_after,
    find_solar_longitude_event_before,
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

REASON_EVENT_KIND_INVALID = "EVENT_KIND_INVALID"
REASON_SOLAR_LONGITUDE_EVENT_UNRESOLVED = "SOLAR_LONGITUDE_EVENT_UNRESOLVED"
REASON_COVERAGE_EXHAUSTED = "EPHEMERIS_COVERAGE_EXHAUSTED"
REASON_REACH_EXHAUSTED = "EPHEMERIS_REACH_EXHAUSTED"
REASON_INSTANT_INVALID = "INSTANT_STATE_INVALID"
REASON_OBSERVER_OUT_OF_DOMAIN = "OBSERVER_OUT_OF_DOMAIN"

K000 = "SOLAR_LONGITUDE_000"
K090 = "SOLAR_LONGITUDE_090"
K180 = "SOLAR_LONGITUDE_180"
K270 = "SOLAR_LONGITUDE_270"

GOVERNED_KINDS = (K000, K090, K180, K270)

# Expected production constant VALUES, as independent literals.
EXPECTED_SEARCH_SPAN_DAYS = 380.0
EXPECTED_SAMPLE_STEP_DAYS = 88.0

# Certification evidence. MEASURED across the pinned authoritative domain and
# rederived by the opt-in companion file. Never imported from production,
# because production deliberately does not carry them.
CERTIFIED_MAX_SAME_KIND_GAP_DAYS = 365.253272318
CERTIFIED_MIN_ADJACENT_QUADRANT_GAP_DAYS = 88.159551771

# The installed Skyfield default. Asserted, not assumed: the whole sampling
# argument exists because this value is unsafe for some frontier widths.
INSTALLED_SEASONS_STEP_DAYS = 90.0

# Frontier widths that are demonstrably unsafe under the installed default.
UNSAFE_SPANS_UNDER_INSTALLED_DEFAULT = (89.0, 179.0, 269.0, 355.0)

MODERN_ANCHOR = 2460678.0
DEFECT_WINDOW_TT = 2689118.000800741

A3B_II_BLOCK_MARKER = "# A3b-ii - strict solar-longitude directional solver."

GOVERNED_BLOCK_MARKER = re.compile(r"^# A\d+[a-z]?(?:-[0-9a-z]+)? - ",
                                   re.MULTILINE)


# --- Independent helpers ---------------------------------------------------


class _GridCaptured(Exception):
    """Raised by the grid spy once the initial sample vector is in hand."""


def measured_initial_grid(step_days, span, anchor=MODERN_ANCHOR):
    """Return the initial grid the INSTALLED find_discrete actually builds.

    The predicate is a spy that records the sample vector it is handed and
    then aborts, so the grid is measured from real Skyfield behavior rather
    than restated from a formula, and no ephemeris is evaluated.
    """
    box = {}

    def spy(t):
        box["jd"] = np.atleast_1d(np.array(t.tt, dtype=float))
        raise _GridCaptured

    spy.step_days = step_days

    try:
        almanac.find_discrete(ts.tt_jd(anchor), ts.tt_jd(anchor + span), spy)
    except _GridCaptured:
        pass

    return box["jd"]


def max_real_grid_gap(step_days, span, anchor=MODERN_ANCHOR):
    return float(np.diff(measured_initial_grid(step_days, span, anchor)).max())


def independent_predicate(kernel_name):
    """The governed predicate, built here rather than obtained from A3b-ii."""
    return almanac.seasons(load_kernel(kernel_name))


def independent_crossings(kernel_name, tt_lo, tt_hi, step_days):
    """Re-enumerate crossings without calling _solar_longitude_crossings."""
    season_at = independent_predicate(kernel_name)

    def wrapper(t):
        return season_at(t)

    wrapper.step_days = step_days

    times, events = almanac.find_discrete(
        ts.tt_jd(tt_lo), ts.tt_jd(tt_hi), wrapper
    )
    return [(float(t.tt), int(v)) for t, v in zip(times, events)]


def independent_canonical(season_at, tt_below, tt_at_or_above, quadrant):
    """Relocate a canonical boundary without calling the production helper.

    Terminates by exhaustion of the binary64 representation: there is no
    epsilon, no tolerance and no iteration cap here either, because the
    property under test is exactly that none is needed.
    """
    below = float(tt_below)
    at_or_above = float(tt_at_or_above)

    while True:
        mid = below + (at_or_above - below) / 2.0

        if mid == below or mid == at_or_above:
            return at_or_above

        if int(season_at(ts.tt_jd(mid))) == quadrant:
            at_or_above = mid
        else:
            below = mid


def injected_range_error(message):
    """Build a real EphemerisRangeError for fault injection.

    The certified Skyfield constructor requires message, start_time,
    end_time, time_mask and segment. Only the exception type matters to the
    production path under test, but the real constructor is used so the test
    exercises the actual exception class the solver can receive.
    """
    return EphemerisRangeError(message, None, None, None, None)


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


def supported_anchors():
    """Anchors spanning every territory A3b-ii must answer in.

    Bounds are read from the artifacts at test time rather than hard-coded,
    so a changed artifact moves the probes instead of silently invalidating
    them.
    """
    de440 = kernel_coverage_tt(DE440)
    part_1 = kernel_coverage_tt(DE441_PART_1)
    part_2 = kernel_coverage_tt(DE441_PART_2)

    return (
        ("modern interior", MODERN_ANCHOR),
        ("de440 lower transition", de440.tt_start + 200.0),
        ("de440 upper transition", de440.tt_end - 200.0),
        ("2650 defect territory", DEFECT_WINDOW_TT),
        ("deep time", 700000.0),
        ("deeper time", -1000000.0),
        ("deep-time ceiling approach", part_1.tt_end - 1000.0),
        ("far future", 5000000.0),
        ("far-future floor approach", part_2.tt_start + 1000.0),
        ("far-future ceiling approach", part_2.tt_end - 1000.0),
    )


class SolverAssertions(unittest.TestCase):
    def assert_fails_closed(self, reason, function, *args):
        with self.assertRaises(ScientificEnvironmentError) as caught:
            function(*args)
        self.assertEqual(caught.exception.reason, reason)
        return caught.exception

    def assert_is_event(self, result, kind):
        self.assertIsNotNone(result)
        self.assertIsInstance(result, AstronomicalEvent)
        self.assertIsInstance(result.tt, float)
        self.assertTrue(math.isfinite(result.tt))
        self.assertEqual(result.kind, kind)
        self.assertIn(result.kernel, PINNED_ARTIFACTS)


# --- 1. Production constants -----------------------------------------------


class TestProductionConstants(SolverAssertions):
    def test_search_span_is_the_governed_value(self):
        self.assertEqual(
            astronomy_solver.SOLAR_LONGITUDE_SEARCH_SPAN_DAYS,
            EXPECTED_SEARCH_SPAN_DAYS,
        )

    def test_sample_step_is_the_governed_value(self):
        self.assertEqual(
            astronomy_solver.SOLAR_LONGITUDE_SAMPLE_STEP_DAYS,
            EXPECTED_SAMPLE_STEP_DAYS,
        )

    def test_horizon_exceeds_certified_maximum_same_kind_gap(self):
        """Why a complete frontier must always contain a requested crossing."""
        self.assertGreater(
            EXPECTED_SEARCH_SPAN_DAYS, CERTIFIED_MAX_SAME_KIND_GAP_DAYS
        )
        self.assertGreater(
            EXPECTED_SEARCH_SPAN_DAYS - CERTIFIED_MAX_SAME_KIND_GAP_DAYS, 14.0
        )

    def test_sample_step_is_below_certified_minimum_adjacent_gap(self):
        """Why no sample interval can ever contain two crossings."""
        self.assertLess(
            EXPECTED_SAMPLE_STEP_DAYS,
            CERTIFIED_MIN_ADJACENT_QUADRANT_GAP_DAYS,
        )

    def test_the_sample_step_margin_is_not_vanishing(self):
        """The certified minimum itself was rejected as the step.

        Its binary64 margin is smaller than the precision to which that
        certification value is recorded, and a margin that thin is not a
        margin. The governed step keeps a margin far larger than the
        certified value's own last recorded digit.
        """
        margin = (CERTIFIED_MIN_ADJACENT_QUADRANT_GAP_DAYS
                  - EXPECTED_SAMPLE_STEP_DAYS)
        self.assertGreater(margin, 1e-3)

    def test_production_carries_no_certified_extremum(self):
        """The extrema are certification evidence, not production constants."""
        for name in dir(astronomy_solver):
            value = getattr(astronomy_solver, name)
            if isinstance(value, float):
                self.assertNotEqual(value, CERTIFIED_MAX_SAME_KIND_GAP_DAYS)
                self.assertNotEqual(
                    value, CERTIFIED_MIN_ADJACENT_QUADRANT_GAP_DAYS
                )


# --- 2. Sampling wrapper neutrality ----------------------------------------


class TestSamplingWrapperNeutrality(SolverAssertions):
    def setUp(self):
        self.sampled, self.underlying = (
            astronomy_solver._sampling_controlled_seasons(DE440)
        )
        self.independent = independent_predicate(DE440)

    def test_wrapper_carries_the_governed_step(self):
        self.assertEqual(self.sampled.step_days, EXPECTED_SAMPLE_STEP_DAYS)

    def test_underlying_predicate_is_not_mutated(self):
        self.assertEqual(self.underlying.step_days,
                         INSTALLED_SEASONS_STEP_DAYS)
        self.assertEqual(self.independent.step_days,
                         INSTALLED_SEASONS_STEP_DAYS)

    def test_wrapper_returns_exactly_the_governed_quadrants(self):
        probe = ts.tt_jd(
            np.linspace(MODERN_ANCHOR, MODERN_ANCHOR + 380.0, 401)
        )
        wrapped = np.asarray(self.sampled(probe))
        underlying = np.asarray(self.underlying(probe))
        independent = np.asarray(self.independent(probe))

        self.assertTrue(np.array_equal(wrapped, underlying))
        self.assertTrue(np.array_equal(wrapped, independent))
        # Anti-vacuity: the sample must actually span several quadrants.
        self.assertGreaterEqual(len(set(wrapped.tolist())), 4)

    def test_wrapper_is_neutral_under_every_artifact(self):
        for kernel_name in PINNED_ARTIFACTS:
            with self.subTest(kernel=kernel_name):
                sampled, underlying = (
                    astronomy_solver._sampling_controlled_seasons(kernel_name)
                )
                coverage = kernel_coverage_tt(kernel_name)
                base = coverage.tt_start + 5000.0
                probe = ts.tt_jd(np.linspace(base, base + 380.0, 201))
                self.assertTrue(
                    np.array_equal(
                        np.asarray(sampled(probe)),
                        np.asarray(underlying(probe)),
                    )
                )
                self.assertEqual(sampled.step_days, EXPECTED_SAMPLE_STEP_DAYS)


# --- 3. Real-grid sampling safety ------------------------------------------


class TestRealGridSafety(SolverAssertions):
    def test_installed_default_step_is_what_we_think_it_is(self):
        """Anti-vacuity: the whole mechanism exists because of this value."""
        self.assertEqual(
            independent_predicate(DE440).step_days,
            INSTALLED_SEASONS_STEP_DAYS,
        )

    def test_governed_step_is_safe_for_every_span_in_the_horizon(self):
        """MEASURED over dense spans, including exact-multiple edges."""
        spans = [float(v) for v in np.linspace(1e-6, 380.0, 2001)]
        for multiple in range(1, 6):
            exact = multiple * EXPECTED_SAMPLE_STEP_DAYS
            for delta in (-1e-9, -1e-12, 0.0, 1e-12, 1e-9):
                candidate = exact + delta
                if 0.0 < candidate <= 380.0:
                    spans.append(candidate)

        worst = 0.0
        for span in spans:
            gap = max_real_grid_gap(EXPECTED_SAMPLE_STEP_DAYS, span)
            if gap > worst:
                worst = gap

        self.assertLessEqual(worst, EXPECTED_SAMPLE_STEP_DAYS)
        self.assertLess(worst, CERTIFIED_MIN_ADJACENT_QUADRANT_GAP_DAYS)

    def test_governed_horizon_grid_is_the_certified_one(self):
        grid = measured_initial_grid(EXPECTED_SAMPLE_STEP_DAYS, 380.0)
        self.assertEqual(len(grid), 6)
        self.assertAlmostEqual(float(np.diff(grid).max()), 76.0, places=9)

    def test_installed_default_is_unsafe_for_real_truncated_spans(self):
        """Non-vacuity: the governed step is not decoration.

        These spans are reachable frontier widths, and under the installed
        default each samples at or above the certified closest approach of two
        adjacent crossings.
        """
        for span in UNSAFE_SPANS_UNDER_INSTALLED_DEFAULT:
            with self.subTest(span=span):
                unsafe = max_real_grid_gap(INSTALLED_SEASONS_STEP_DAYS, span)
                self.assertGreaterEqual(
                    unsafe, CERTIFIED_MIN_ADJACENT_QUADRANT_GAP_DAYS
                )
                safe = max_real_grid_gap(EXPECTED_SAMPLE_STEP_DAYS, span)
                self.assertLess(
                    safe, CERTIFIED_MIN_ADJACENT_QUADRANT_GAP_DAYS
                )

    def test_spacing_is_a_sawtooth_not_a_monotone_function(self):
        """Why a single certified width could not have settled the question."""
        gaps = [max_real_grid_gap(INSTALLED_SEASONS_STEP_DAYS, span)
                for span in (88.0, 89.0, 91.0, 179.0, 181.0)]
        self.assertFalse(all(a <= b for a, b in zip(gaps, gaps[1:])))


# --- 4. Governed kind validation -------------------------------------------


class TestGovernedKindValidation(SolverAssertions):
    REJECTED = (
        None, True, False, 0, 1, 90, 0.0, 90.0, b"SOLAR_LONGITUDE_000",
        "", " ", "000", "090", "SOLAR_LONGITUDE_45", "SOLAR_LONGITUDE_360",
        "solar_longitude_000", "Solar_Longitude_000",
        "SOLAR_LONGITUDE_000 ", " SOLAR_LONGITUDE_000",
        "Spring", "spring", "Summer", "Autumn", "Winter",
        "spring_equinox", "summer_solstice", "autumn_equinox",
        "winter_solstice", "equinox", "solstice",
    )

    def test_each_governed_kind_is_accepted(self):
        for kind in GOVERNED_KINDS:
            with self.subTest(kind=kind):
                self.assert_is_event(
                    find_solar_longitude_event_after(MODERN_ANCHOR, kind), kind
                )

    def test_every_other_kind_is_rejected(self):
        for kind in self.REJECTED:
            for solver in (find_solar_longitude_event_before,
                           find_solar_longitude_event_after):
                with self.subTest(kind=kind, solver=solver.__name__):
                    self.assert_fails_closed(
                        REASON_EVENT_KIND_INVALID, solver, MODERN_ANCHOR, kind
                    )

    def test_rejection_precedes_any_frontier_or_kernel_work(self):
        """An invalid kind never costs an ephemeris read."""
        with mock.patch.object(
            astronomy_solver, "geocentric_search_frontier",
            side_effect=AssertionError("frontier consulted for an invalid kind")
        ) as frontier, mock.patch.object(
            astronomy_solver, "load_kernel",
            side_effect=AssertionError("kernel loaded for an invalid kind")
        ) as kernel:
            for solver in (find_solar_longitude_event_before,
                           find_solar_longitude_event_after):
                self.assert_fails_closed(
                    REASON_EVENT_KIND_INVALID, solver, MODERN_ANCHOR, "Spring"
                )
        self.assertEqual(frontier.call_count, 0)
        self.assertEqual(kernel.call_count, 0)

    def test_malformed_anchor_reports_instant_state_invalid(self):
        for anchor in (None, "2460678.0", float("nan"), float("inf"), True,
                       object()):
            for solver in (find_solar_longitude_event_before,
                           find_solar_longitude_event_after):
                with self.subTest(anchor=repr(anchor)):
                    self.assert_fails_closed(
                        REASON_INSTANT_INVALID, solver, anchor, K000
                    )

    def test_observer_reason_is_never_produced(self):
        """A3b-ii accepts no observer, so its reason cannot arise here."""
        for kind in GOVERNED_KINDS:
            for solver in (find_solar_longitude_event_before,
                           find_solar_longitude_event_after):
                try:
                    solver(MODERN_ANCHOR, kind)
                except ScientificEnvironmentError as error:
                    self.assertNotEqual(
                        error.reason, REASON_OBSERVER_OUT_OF_DOMAIN
                    )

    def test_an_observer_argument_is_a_type_error(self):
        for solver in (find_solar_longitude_event_before,
                       find_solar_longitude_event_after):
            with self.subTest(solver=solver.__name__):
                with self.assertRaises(TypeError):
                    solver(MODERN_ANCHOR, K000, 40.7406)


# --- 5. Canonical event identity -------------------------------------------


class CanonicalSampleMixin(unittest.TestCase):
    """A representative crossing from each artifact and each governed kind."""

    @classmethod
    def setUpClass(cls):
        de440 = kernel_coverage_tt(DE440)
        part_1 = kernel_coverage_tt(DE441_PART_1)
        part_2 = kernel_coverage_tt(DE441_PART_2)

        cls.bases = (
            (DE440, MODERN_ANCHOR),
            (DE440, de440.tt_start + 50000.0),
            (DE441_PART_1, 700000.0),
            (DE441_PART_1, -1000000.0),
            (DE441_PART_2, 5000000.0),
            (DE441_PART_2, part_2.tt_start + 100000.0),
        )

        cls.sample = []
        for expected_kernel, base in cls.bases:
            for quadrant, kind in enumerate(GOVERNED_KINDS):
                event = find_solar_longitude_event_after(base, kind)
                cls.sample.append((event, quadrant, kind))


class TestCanonicalBoundary(CanonicalSampleMixin, SolverAssertions):
    def test_sample_covers_every_artifact_and_kind(self):
        self.assertEqual(
            {kind for _e, _q, kind in self.sample}, set(GOVERNED_KINDS)
        )
        self.assertEqual(
            {e.kernel for e, _q, _k in self.sample}, set(PINNED_ARTIFACTS)
        )

    def test_canonical_state_carries_the_requested_quadrant(self):
        for event, quadrant, kind in self.sample:
            with self.subTest(kind=kind, kernel=event.kernel):
                season_at = independent_predicate(event.kernel)
                self.assertEqual(
                    int(season_at(ts.tt_jd(event.tt))), quadrant
                )

    def test_immediate_predecessor_does_not(self):
        for event, quadrant, kind in self.sample:
            with self.subTest(kind=kind, kernel=event.kernel):
                season_at = independent_predicate(event.kernel)
                predecessor = math.nextafter(event.tt, -math.inf)
                self.assertNotEqual(
                    int(season_at(ts.tt_jd(predecessor))), quadrant
                )

    def test_no_chatter_around_the_transition(self):
        for event, quadrant, kind in self.sample:
            with self.subTest(kind=kind, kernel=event.kernel):
                season_at = independent_predicate(event.kernel)
                state = event.tt
                for _ in range(6):
                    state = math.nextafter(state, -math.inf)
                    self.assertNotEqual(
                        int(season_at(ts.tt_jd(state))), quadrant
                    )
                state = event.tt
                for _ in range(6):
                    self.assertEqual(
                        int(season_at(ts.tt_jd(state))), quadrant
                    )
                    state = math.nextafter(state, math.inf)

    def test_relocation_from_materially_different_lower_bounds_agrees(self):
        for event, quadrant, kind in self.sample:
            with self.subTest(kind=kind, kernel=event.kernel):
                season_at = independent_predicate(event.kernel)
                relocated = {
                    independent_canonical(
                        season_at, event.tt - back, event.tt, quadrant
                    )
                    for back in (5.0, 45.0, 90.0, 200.0, 370.0)
                }
                self.assertEqual(relocated, {event.tt})

    def test_canonical_identity_survives_sampling_and_bracketing(self):
        """Raw endpoints move; the canonical boundary does not."""
        raw_differed = 0
        comparisons = 0

        for event, quadrant, kind in self.sample:
            season_at = independent_predicate(event.kernel)
            for step_days in (90.0, 88.0, 60.0, 30.0, 5.0):
                with self.subTest(kind=kind, step_days=step_days):
                    reported = independent_crossings(
                        event.kernel, event.tt - 190.0, event.tt + 190.0,
                        step_days,
                    )
                    canonical = []
                    below = event.tt - 190.0
                    for index, (raw_tt, value) in enumerate(reported):
                        if value == quadrant:
                            canonical.append(
                                independent_canonical(
                                    season_at, below, raw_tt, quadrant
                                )
                            )
                            comparisons += 1
                            if raw_tt != event.tt:
                                raw_differed += 1
                        below = raw_tt
                    self.assertIn(event.tt, canonical)

        # Anti-vacuity: if raw endpoints never moved, the property above
        # would be trivially true and would prove nothing.
        self.assertGreater(comparisons, 0)
        self.assertGreater(raw_differed, 0)


# --- 6. Strict directional semantics ---------------------------------------


class TestStrictDirectionalSemantics(SolverAssertions):
    def test_every_territory_and_kind_brackets_the_anchor(self):
        for label, anchor in supported_anchors():
            for kind in GOVERNED_KINDS:
                with self.subTest(territory=label, kind=kind):
                    before = find_solar_longitude_event_before(anchor, kind)
                    after = find_solar_longitude_event_after(anchor, kind)

                    self.assert_is_event(before, kind)
                    self.assert_is_event(after, kind)
                    self.assertLess(before.tt, anchor)
                    self.assertGreater(after.tt, anchor)
                    self.assertLess(before.tt, after.tt)

    def test_returned_kernel_is_the_frontier_artifact(self):
        for label, anchor in supported_anchors():
            for kind in GOVERNED_KINDS:
                with self.subTest(territory=label, kind=kind):
                    backward = geocentric_search_frontier(
                        anchor, anchor - EXPECTED_SEARCH_SPAN_DAYS
                    )
                    forward = geocentric_search_frontier(
                        anchor, anchor + EXPECTED_SEARCH_SPAN_DAYS
                    )
                    self.assertEqual(
                        find_solar_longitude_event_before(anchor, kind).kernel,
                        backward.kernel,
                    )
                    self.assertEqual(
                        find_solar_longitude_event_after(anchor, kind).kernel,
                        forward.kernel,
                    )

    def test_gaps_are_astronomically_sane(self):
        for label, anchor in supported_anchors():
            for kind in GOVERNED_KINDS:
                with self.subTest(territory=label, kind=kind):
                    before = find_solar_longitude_event_before(anchor, kind)
                    after = find_solar_longitude_event_after(anchor, kind)
                    separation = after.tt - before.tt
                    self.assertGreater(separation, 0.0)
                    self.assertLess(
                        separation, 2.0 * EXPECTED_SEARCH_SPAN_DAYS
                    )

    def test_the_two_directions_agree_on_the_same_anchor(self):
        """One anchor, one exact state, both directions."""
        anchor = MODERN_ANCHOR
        for kind in GOVERNED_KINDS:
            with self.subTest(kind=kind):
                first = find_solar_longitude_event_after(anchor, kind)
                second = find_solar_longitude_event_after(anchor, kind)
                self.assertEqual(first.tt, second.tt)
                self.assertEqual(first, second)


# --- 7. Exact-root anchors -------------------------------------------------


class TestExactRootAnchors(CanonicalSampleMixin, SolverAssertions):
    def test_after_a_canonical_root_returns_the_next_same_kind_crossing(self):
        for event, _quadrant, kind in self.sample:
            with self.subTest(kind=kind, kernel=event.kernel):
                following = find_solar_longitude_event_after(event.tt, kind)
                self.assert_is_event(following, kind)
                self.assertNotEqual(following.tt, event.tt)
                self.assertGreater(following.tt, event.tt)
                self.assertGreater(following.tt - event.tt, 360.0)
                self.assertLess(
                    following.tt - event.tt, CERTIFIED_MAX_SAME_KIND_GAP_DAYS
                )

    def test_before_a_canonical_root_returns_the_previous_crossing(self):
        for event, _quadrant, kind in self.sample:
            with self.subTest(kind=kind, kernel=event.kernel):
                preceding = find_solar_longitude_event_before(event.tt, kind)
                self.assert_is_event(preceding, kind)
                self.assertNotEqual(preceding.tt, event.tt)
                self.assertLess(preceding.tt, event.tt)
                self.assertGreater(event.tt - preceding.tt, 360.0)
                self.assertLess(
                    event.tt - preceding.tt, CERTIFIED_MAX_SAME_KIND_GAP_DAYS
                )

    def test_the_anchor_crossing_is_returned_by_neither_direction(self):
        for event, _quadrant, kind in self.sample:
            with self.subTest(kind=kind, kernel=event.kernel):
                self.assertNotIn(
                    event.tt,
                    (find_solar_longitude_event_before(event.tt, kind).tt,
                     find_solar_longitude_event_after(event.tt, kind).tt),
                )

    def test_the_anchor_is_bracketed_by_its_own_neighbours(self):
        for event, _quadrant, kind in self.sample:
            with self.subTest(kind=kind, kernel=event.kernel):
                preceding = find_solar_longitude_event_before(event.tt, kind)
                following = find_solar_longitude_event_after(event.tt, kind)
                self.assertLess(preceding.tt, event.tt)
                self.assertLess(event.tt, following.tt)


# --- 8. Incomplete-frontier semantics --------------------------------------


class TestIncompleteFrontierSemantics(SolverAssertions):
    """A truncated frontier is a shortened answer, never an absence."""

    @classmethod
    def setUpClass(cls):
        part_1 = kernel_coverage_tt(DE441_PART_1)
        part_2 = kernel_coverage_tt(DE441_PART_2)
        cls.forward_anchor = part_2.tt_end - 200.0
        cls.backward_anchor = part_1.tt_start + 200.0

    def _truncated(self, anchor, signed_span):
        frontier = geocentric_search_frontier(anchor, anchor + signed_span)
        self.assertFalse(
            frontier.complete,
            "probe anchor no longer produces a truncated frontier",
        )
        self.assertIn(
            frontier.truncation_reason,
            (REASON_COVERAGE_EXHAUSTED, REASON_REACH_EXHAUSTED),
        )
        present = {
            value for _tt, value in independent_crossings(
                frontier.kernel, frontier.tt_lo, frontier.tt_hi,
                EXPECTED_SAMPLE_STEP_DAYS,
            )
        }
        return frontier, present

    def test_forward_truncated_frontier_is_genuinely_truncated(self):
        frontier, present = self._truncated(
            self.forward_anchor, EXPECTED_SEARCH_SPAN_DAYS
        )
        self.assertLess(
            frontier.tt_hi - frontier.tt_lo, EXPECTED_SEARCH_SPAN_DAYS
        )
        # Anti-vacuity: the window must be partial, not empty and not whole.
        self.assertTrue(0 < len(present) < 4)

    def test_forward_truncated_with_a_crossing_returns_it(self):
        frontier, present = self._truncated(
            self.forward_anchor, EXPECTED_SEARCH_SPAN_DAYS
        )
        for quadrant in sorted(present):
            kind = GOVERNED_KINDS[quadrant]
            with self.subTest(kind=kind):
                event = find_solar_longitude_event_after(
                    self.forward_anchor, kind
                )
                self.assert_is_event(event, kind)
                self.assertGreater(event.tt, self.forward_anchor)
                self.assertLessEqual(event.tt, frontier.tt_hi)
                self.assertEqual(event.kernel, frontier.kernel)

    def test_forward_truncated_without_a_crossing_propagates_exhaustion(self):
        frontier, present = self._truncated(
            self.forward_anchor, EXPECTED_SEARCH_SPAN_DAYS
        )
        absent = [q for q in range(4) if q not in present]
        self.assertTrue(absent, "probe window contains every kind")
        for quadrant in absent:
            kind = GOVERNED_KINDS[quadrant]
            with self.subTest(kind=kind):
                error = self.assert_fails_closed(
                    frontier.truncation_reason,
                    find_solar_longitude_event_after,
                    self.forward_anchor, kind,
                )
                self.assertNotEqual(
                    error.reason, REASON_SOLAR_LONGITUDE_EVENT_UNRESOLVED
                )

    def test_backward_truncated_frontier_is_genuinely_truncated(self):
        frontier, present = self._truncated(
            self.backward_anchor, -EXPECTED_SEARCH_SPAN_DAYS
        )
        self.assertLess(
            frontier.tt_hi - frontier.tt_lo, EXPECTED_SEARCH_SPAN_DAYS
        )
        self.assertTrue(0 < len(present) < 4)

    def test_backward_truncated_with_a_crossing_returns_it(self):
        frontier, present = self._truncated(
            self.backward_anchor, -EXPECTED_SEARCH_SPAN_DAYS
        )
        for quadrant in sorted(present):
            kind = GOVERNED_KINDS[quadrant]
            with self.subTest(kind=kind):
                event = find_solar_longitude_event_before(
                    self.backward_anchor, kind
                )
                self.assert_is_event(event, kind)
                self.assertLess(event.tt, self.backward_anchor)
                self.assertGreaterEqual(event.tt, frontier.tt_lo)
                self.assertEqual(event.kernel, frontier.kernel)

    def test_backward_truncated_without_a_crossing_propagates_exhaustion(self):
        frontier, present = self._truncated(
            self.backward_anchor, -EXPECTED_SEARCH_SPAN_DAYS
        )
        absent = [q for q in range(4) if q not in present]
        self.assertTrue(absent, "probe window contains every kind")
        for quadrant in absent:
            kind = GOVERNED_KINDS[quadrant]
            with self.subTest(kind=kind):
                error = self.assert_fails_closed(
                    frontier.truncation_reason,
                    find_solar_longitude_event_before,
                    self.backward_anchor, kind,
                )
                self.assertNotEqual(
                    error.reason, REASON_SOLAR_LONGITUDE_EVENT_UNRESOLVED
                )

    def test_truncated_and_empty_never_returns_none(self):
        for anchor, signed, solver in (
            (self.forward_anchor, EXPECTED_SEARCH_SPAN_DAYS,
             find_solar_longitude_event_after),
            (self.backward_anchor, -EXPECTED_SEARCH_SPAN_DAYS,
             find_solar_longitude_event_before),
        ):
            _frontier, present = self._truncated(anchor, signed)
            for quadrant in range(4):
                kind = GOVERNED_KINDS[quadrant]
                with self.subTest(kind=kind, solver=solver.__name__):
                    try:
                        self.assertIsNotNone(solver(anchor, kind))
                    except ScientificEnvironmentError:
                        pass


# --- 9. Complete-frontier unresolved failure -------------------------------


class TestCompleteFrontierUnresolved(SolverAssertions):
    """A complete horizon with no crossing contradicts the certification."""

    def test_after_reports_unresolved_not_exhaustion(self):
        frontier = geocentric_search_frontier(
            MODERN_ANCHOR, MODERN_ANCHOR + EXPECTED_SEARCH_SPAN_DAYS
        )
        self.assertTrue(frontier.complete)
        self.assertIsNone(frontier.truncation_reason)

        with mock.patch.object(
            astronomy_solver, "_solar_longitude_crossings", return_value=()
        ):
            error = self.assert_fails_closed(
                REASON_SOLAR_LONGITUDE_EVENT_UNRESOLVED,
                find_solar_longitude_event_after, MODERN_ANCHOR, K000,
            )
        self.assertNotIn(
            error.reason, (REASON_COVERAGE_EXHAUSTED, REASON_REACH_EXHAUSTED)
        )

    def test_before_reports_unresolved_not_exhaustion(self):
        frontier = geocentric_search_frontier(
            MODERN_ANCHOR, MODERN_ANCHOR - EXPECTED_SEARCH_SPAN_DAYS
        )
        self.assertTrue(frontier.complete)

        with mock.patch.object(
            astronomy_solver, "_solar_longitude_crossings", return_value=()
        ):
            error = self.assert_fails_closed(
                REASON_SOLAR_LONGITUDE_EVENT_UNRESOLVED,
                find_solar_longitude_event_before, MODERN_ANCHOR, K270,
            )
        self.assertNotIn(
            error.reason, (REASON_COVERAGE_EXHAUSTED, REASON_REACH_EXHAUSTED)
        )

    def test_unresolved_never_returns_none(self):
        with mock.patch.object(
            astronomy_solver, "_solar_longitude_crossings", return_value=()
        ):
            for solver in (find_solar_longitude_event_before,
                           find_solar_longitude_event_after):
                for kind in GOVERNED_KINDS:
                    with self.subTest(solver=solver.__name__, kind=kind):
                        with self.assertRaises(ScientificEnvironmentError):
                            solver(MODERN_ANCHOR, kind)

    def test_a_crossing_of_another_kind_does_not_satisfy_the_request(self):
        """Only the requested kind counts toward the horizon invariant."""
        frontier = geocentric_search_frontier(
            MODERN_ANCHOR, MODERN_ANCHOR + EXPECTED_SEARCH_SPAN_DAYS
        )
        reported = independent_crossings(
            frontier.kernel, frontier.tt_lo, frontier.tt_hi,
            EXPECTED_SAMPLE_STEP_DAYS,
        )
        self.assertEqual({v for _tt, v in reported}, {0, 1, 2, 3})


# --- 10. EphemerisRangeError protection ------------------------------------


class TestEphemerisRangeErrorProtection(SolverAssertions):
    """One governed boundary covers enumeration AND canonicalization."""

    def test_enumeration_failure_becomes_reach_exhausted(self):
        injected = injected_range_error("injected enumeration failure")

        with mock.patch.object(
            astronomy_solver.almanac, "find_discrete", side_effect=injected
        ) as find_discrete:
            error = self.assert_fails_closed(
                REASON_REACH_EXHAUSTED,
                find_solar_longitude_event_after, MODERN_ANCHOR, K000,
            )

        self.assertIs(error.__cause__, injected)
        self.assertEqual(find_discrete.call_count, 1, "the search was retried")

    def test_canonicalization_failure_becomes_reach_exhausted(self):
        """The failure is injected into the SECOND half of the operation.

        find_discrete runs against the sampling wrapper and succeeds; the
        canonicalization predicate is the one that fails. A handler protecting
        only the enumeration would let the raw error escape.
        """
        injected = injected_range_error("injected canonicalization failure")
        real = astronomy_solver._sampling_controlled_seasons
        calls = {"factory": 0, "predicate": 0}

        def failing_factory(kernel_name):
            calls["factory"] += 1
            sampled, _underlying = real(kernel_name)

            def failing_predicate(t):
                calls["predicate"] += 1
                raise injected

            return sampled, failing_predicate

        with mock.patch.object(
            astronomy_solver, "_sampling_controlled_seasons", failing_factory
        ):
            error = self.assert_fails_closed(
                REASON_REACH_EXHAUSTED,
                find_solar_longitude_event_after, MODERN_ANCHOR, K000,
            )

        self.assertIs(error.__cause__, injected)
        self.assertGreater(calls["predicate"], 0, "canonicalization never ran")
        self.assertEqual(calls["factory"], 1, "a second artifact was opened")

    def test_both_directions_are_protected(self):
        injected = injected_range_error("injected")
        for solver in (find_solar_longitude_event_before,
                       find_solar_longitude_event_after):
            with self.subTest(solver=solver.__name__):
                with mock.patch.object(
                    astronomy_solver.almanac, "find_discrete",
                    side_effect=injected,
                ):
                    error = self.assert_fails_closed(
                        REASON_REACH_EXHAUSTED, solver, MODERN_ANCHOR, K180
                    )
                self.assertIs(error.__cause__, injected)

    def test_no_second_kernel_is_opened_after_a_failure(self):
        injected = injected_range_error("injected")
        with mock.patch.object(
            astronomy_solver.almanac, "find_discrete", side_effect=injected
        ), mock.patch.object(
            astronomy_solver, "_sampling_controlled_seasons",
            wraps=astronomy_solver._sampling_controlled_seasons,
        ) as factory:
            self.assert_fails_closed(
                REASON_REACH_EXHAUSTED,
                find_solar_longitude_event_after, MODERN_ANCHOR, K000,
            )
        self.assertEqual(factory.call_count, 1)

    def test_only_ephemeris_range_error_is_translated(self):
        """A different failure is not a statement about exhausted reach."""
        for injected in (ValueError("not a reach statement"),
                         RuntimeError("not a reach statement"),
                         MemoryError()):
            with self.subTest(error=type(injected).__name__):
                with mock.patch.object(
                    astronomy_solver.almanac, "find_discrete",
                    side_effect=injected,
                ):
                    with self.assertRaises(type(injected)):
                        find_solar_longitude_event_after(MODERN_ANCHOR, K000)

    def test_the_translation_happens_exactly_once(self):
        injected = injected_range_error("injected")
        with mock.patch.object(
            astronomy_solver.almanac, "find_discrete", side_effect=injected
        ):
            with self.assertRaises(ScientificEnvironmentError) as caught:
                find_solar_longitude_event_after(MODERN_ANCHOR, K000)

        # One governed error wrapping the raw error, not a chain of them.
        self.assertIs(caught.exception.__cause__, injected)
        self.assertIsNone(injected.__cause__)


# --- 11. Kernel provenance -------------------------------------------------


class TestKernelProvenance(SolverAssertions):
    def test_solve_runs_under_exactly_the_frontier_artifact(self):
        anchor = MODERN_ANCHOR
        frontier = geocentric_search_frontier(
            anchor, anchor + EXPECTED_SEARCH_SPAN_DAYS
        )
        opened = []
        real = astronomy_solver.load_kernel

        def recording(kernel_name):
            opened.append(kernel_name)
            return real(kernel_name)

        with mock.patch.object(astronomy_solver, "load_kernel", recording):
            event = find_solar_longitude_event_after(anchor, K000)

        self.assertEqual(event.kernel, frontier.kernel)
        self.assertTrue(opened)
        self.assertEqual(set(opened), {frontier.kernel})

    def test_civil_routing_is_never_consulted(self):
        with mock.patch.object(
            astronomy_solver, "choose_kernel_name",
            side_effect=AssertionError("civil routing consulted"),
        ) as routing:
            for kind in GOVERNED_KINDS:
                find_solar_longitude_event_before(MODERN_ANCHOR, kind)
                find_solar_longitude_event_after(MODERN_ANCHOR, kind)
        self.assertEqual(routing.call_count, 0)

    def test_de440_is_used_where_it_supports_the_search(self):
        event = find_solar_longitude_event_after(MODERN_ANCHOR, K000)
        self.assertEqual(event.kernel, DE440)

    def test_deep_time_and_far_future_use_the_de441_parts(self):
        self.assertEqual(
            find_solar_longitude_event_after(700000.0, K000).kernel,
            DE441_PART_1,
        )
        self.assertEqual(
            find_solar_longitude_event_after(5000000.0, K000).kernel,
            DE441_PART_2,
        )

    def test_provenance_is_a_single_artifact_per_event(self):
        for label, anchor in supported_anchors():
            with self.subTest(territory=label):
                event = find_solar_longitude_event_after(anchor, K090)
                self.assertIn(event.kernel, PINNED_ARTIFACTS)
                self.assertIsInstance(event.kernel, str)


# --- 12. No-None contract --------------------------------------------------


class TestNoNoneContract(SolverAssertions):
    def test_every_solve_returns_an_event_or_fails_closed(self):
        coverage = [kernel_coverage_tt(name) for name in PINNED_ARTIFACTS]
        anchors = [anchor for _label, anchor in supported_anchors()]
        anchors += [
            coverage[0].tt_start, coverage[0].tt_end,
            coverage[1].tt_start, coverage[1].tt_end,
            coverage[2].tt_start, coverage[2].tt_end,
            9.9e6, -9.9e6, 0.0, 1.0,
        ]

        returned = failed = 0
        for anchor in anchors:
            for kind in GOVERNED_KINDS:
                for solver in (find_solar_longitude_event_before,
                               find_solar_longitude_event_after):
                    with self.subTest(anchor=anchor, kind=kind,
                                      solver=solver.__name__):
                        try:
                            result = solver(anchor, kind)
                        except ScientificEnvironmentError:
                            failed += 1
                            continue
                        self.assertIsNotNone(result)
                        self.assertIsInstance(result, AstronomicalEvent)
                        returned += 1

        # Anti-vacuity: both outcomes must actually occur in this matrix.
        self.assertGreater(returned, 0)
        self.assertGreater(failed, 0)

    def test_no_solver_declares_an_optional_return(self):
        for solver in (find_solar_longitude_event_before,
                       find_solar_longitude_event_after):
            with self.subTest(solver=solver.__name__):
                source = inspect.getsource(solver)
                self.assertNotIn("return None", source)
                self.assertNotIn("Optional", source)


# --- 13. Structural isolation ----------------------------------------------


class TestA3bIIStructuralIsolation(unittest.TestCase):
    def setUp(self):
        source = inspect.getsource(astronomy_solver)
        index = source.find(A3B_II_BLOCK_MARKER)
        self.assertNotEqual(index, -1, "A3b-ii block marker not found")
        following = GOVERNED_BLOCK_MARKER.search(
            source, index + len(A3B_II_BLOCK_MARKER)
        )
        self.block = source[
            index:following.start() if following else len(source)
        ]
        self.preamble = source[:index]
        self.executable = executable_source(self.block)
        self.tree = ast.parse(self.block)

    def test_marker_obeys_the_governed_grammar(self):
        self.assertTrue(GOVERNED_BLOCK_MARKER.match(A3B_II_BLOCK_MARKER))

    def test_block_was_located_and_is_substantial(self):
        self.assertIn("def find_solar_longitude_event_before", self.block)
        self.assertIn("def find_solar_longitude_event_after", self.block)
        self.assertGreater(len(self.block), 1000)

    def test_block_stops_at_the_next_governed_block(self):
        self.assertIsNone(
            GOVERNED_BLOCK_MARKER.search(
                self.block, len(A3B_II_BLOCK_MARKER)
            ),
            "the A3b-ii scan reaches into a later governed block",
        )

    def test_intended_operations_are_present(self):
        """Anti-vacuity: the guards below must inspect real logic."""
        for token in (
            "_exact_finite_tt",
            "geocentric_search_frontier",
            "almanac.seasons",
            "almanac.find_discrete",
            "load_kernel",
            "AstronomicalEvent(",
            "SunsetChronologyError",
            "EphemerisRangeError",
            "ts.tt_jd",
        ):
            with self.subTest(token=token):
                self.assertIn(token, self.executable)
        self.assertGreater(len(self.executable), 800)
        self.assertNotIn("CANONICAL EVENT IDENTITY", self.executable)
        self.assertNotIn("load-bearing", self.executable)

    def test_block_defines_exactly_the_authorized_symbols(self):
        self.assertEqual(
            [n.name for n in self.tree.body if isinstance(n, ast.ClassDef)],
            [],
        )
        self.assertEqual(
            [n.name for n in self.tree.body
             if isinstance(n, ast.FunctionDef)],
            [
                "_governed_solar_longitude_quadrant",
                "_sampling_controlled_seasons",
                "_canonical_quadrant_boundary",
                "_solar_longitude_crossings",
                "find_solar_longitude_event_before",
                "find_solar_longitude_event_after",
            ],
        )

    def test_block_declares_exactly_the_authorized_constants(self):
        self.assertEqual(
            [n.targets[0].id for n in self.tree.body
             if isinstance(n, ast.Assign)],
            [
                "REASON_EVENT_KIND_INVALID",
                "REASON_SOLAR_LONGITUDE_EVENT_UNRESOLVED",
                "SOLAR_LONGITUDE_SEARCH_SPAN_DAYS",
                "SOLAR_LONGITUDE_SAMPLE_STEP_DAYS",
                "GOVERNED_SOLAR_LONGITUDE_KINDS",
            ],
        )

    def test_block_introduces_no_import(self):
        self.assertEqual(
            [n for n in self.tree.body
             if isinstance(n, (ast.Import, ast.ImportFrom))],
            [],
        )

    def test_no_function_in_the_block_takes_an_observer(self):
        """AST-checked, so solar-longitude prose cannot mask an observer."""
        forbidden = {"latitude", "longitude", "lat", "lon", "observer",
                     "location", "site", "year", "date", "when", "tz",
                     "timezone", "hemisphere"}
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            names = {a.arg for a in node.args.args}
            names |= {a.arg for a in node.args.kwonlyargs}
            names |= {a.arg for a in node.args.posonlyargs}
            with self.subTest(function=node.name):
                self.assertEqual(names & forbidden, set())

    def test_public_solvers_take_exactly_an_anchor_and_a_kind(self):
        for solver in (find_solar_longitude_event_before,
                       find_solar_longitude_event_after):
            with self.subTest(solver=solver.__name__):
                self.assertEqual(
                    list(inspect.signature(solver).parameters), ["tt", "kind"]
                )

    def test_block_has_no_observer_or_topocentric_machinery(self):
        for token in ("wgs84", "latlon", "subpoint", "sunrise_sunset",
                      "OBSERVER_LATITUDE_DOMAIN", "OBSERVER_LONGITUDE_DOMAIN",
                      "_governed_observer", "_admit_bracket",
                      "supported_search_frontier", "SunsetEvent",
                      "SunsetBracket", "SunsetSuccessor"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.executable)

    def test_block_has_no_civil_gregorian_or_legacy_routing(self):
        for token in ("choose_kernel_name", "PRIMARY_START_YEAR",
                      "PRIMARY_END_YEAR", "get_eph_for_year",
                      "get_eph_for_datetime", "find_equinox",
                      "find_season_events", "select_kernel_containing_instant",
                      "select_kernel_for_interval", "ts.utc(", "utc_datetime",
                      "utc_strftime", "isoformat", "strftime", "datetime",
                      "timedelta", "timestamp", "timezone", "hemisphere",
                      "weekday", "month", "/ 15", "HTTPException",
                      "status_code"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.executable)

    def test_block_has_no_public_season_terminology(self):
        for token in ("Spring", "Summer", "Autumn", "Winter", "spring",
                      "summer", "autumn", "winter", "equinox", "solstice",
                      "season_map"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.executable)

    def test_block_has_no_tolerance_or_artificial_shift(self):
        for token in ("epsilon", "EPSILON", "tolerance", "atol", "rtol",
                      "isclose", "round(", "nextafter", "min_gap",
                      "3600", "1e-", "abs("):
            with self.subTest(token=token):
                self.assertNotIn(token, self.executable)

    def test_block_uses_no_private_skyfield_or_subdivision(self):
        for token in ("_find_discrete", "searchlib", "linspace", "chunk",
                      "subdivide", "stitch", "while a <", "for chunk"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.executable)

    def test_block_has_no_year_length_assumption(self):
        for token in ("365", "366", "tropical", "year_length"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.executable)

    def test_governed_numbers_appear_exactly_once_each(self):
        self.assertEqual(self.executable.count("380.0"), 1)
        self.assertEqual(self.executable.count("88.0"), 1)
        self.assertEqual(self.executable.count("90.0"), 0)

    def test_exactly_one_search_is_performed(self):
        self.assertEqual(self.executable.count("almanac.find_discrete"), 1)
        self.assertEqual(self.executable.count("almanac.seasons"), 1)

    def test_exactly_one_protection_boundary_exists(self):
        self.assertEqual(self.executable.count("except EphemerisRangeError"), 1)
        self.assertEqual(self.executable.count("from error"), 1)
        self.assertEqual(self.executable.count("try:"), 1)

    def test_frozen_code_is_still_present_ahead_of_the_block(self):
        for token in (
            "def choose_kernel_name",
            "PRIMARY_START_YEAR",
            "def _exact_finite_tt",
            "def kernel_coverage_tt",
            "def select_kernel_containing_instant",
            "def supported_search_frontier",
            "def _first_evaluable_state",
            "def _observation_is_evaluable",
            "def find_sunset_predecessor",
            "def find_sunset_from_instant",
            "def find_sunset_bracket",
            "class SupportedSearchFrontier",
            "class SunsetEvent",
            "class SunsetBracket",
            "class AstronomicalEvent",
            "class GeocentricSearchFrontier",
            "def geocentric_search_frontier",
        ):
            with self.subTest(token=token):
                self.assertIn(token, self.preamble)

    def test_published_bisection_helper_is_not_redefined(self):
        """A3b-ii introduces its own; it must not touch the published one."""
        self.assertNotIn("def _first_evaluable_state", self.block)
        self.assertNotIn("def _observation_is_evaluable", self.block)
        self.assertIn("def _first_evaluable_state", self.preamble)

    def test_the_two_bisections_remain_distinct_operations(self):
        self.assertNotIn("_first_evaluable_state", self.executable)
        self.assertNotIn("_observation_is_evaluable", self.executable)


if __name__ == "__main__":
    unittest.main()
