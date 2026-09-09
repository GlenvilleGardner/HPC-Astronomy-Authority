"""A2-2 verification for data-authoritative kernel selection.

Covers PINNED_KERNEL_PRECEDENCE, select_kernel_containing_instant and
select_kernel_for_interval in astronomy_solver.py.

Test framework: standard-library unittest.
Scientific execution uses the repository's certified Skyfield runtime and
the real pinned Authority ephemeris artifacts.

INDEPENDENT ORACLE DISCIPLINE

The expected precedence ordering, the pinned artifact filenames, the raw
JD(TDB) bounds and the stable reason code are declared as test-local
literals. They are deliberately NOT imported from astronomy_solver:
importing PINNED_KERNEL_PRECEDENCE to check PINNED_KERNEL_PRECEDENCE would
make the test agree with a defective constant instead of detecting it.

kernel_coverage_tt IS used to obtain boundary probes. It is A2-1, already
certified by tests/test_a2_kernel_coverage.py, and is repository
infrastructure here rather than the unit under test.

SCOPE

This file verifies selection only: which pinned artifact has governed
precedence for a TT state or interval under its declared certified
coverage. It asserts nothing about interval solving, sunset determination,
HTTP behavior or any route, none of which exist yet.

It also asserts nothing about computation-specific ephemeris reach beyond
the single contract-distinction test below, which proves only that
selection reports declared coverage and is not a guarantee of
evaluability. No light-time value is encoded as an A2-2 expectation; the
actual reach problem belongs to A2-3.
"""

import ast
import inspect
import unittest

import astronomy_solver
from astronomy_solver import (
    choose_kernel_name,
    kernel_coverage_tt,
    load_kernel,
    select_kernel_containing_instant,
    select_kernel_for_interval,
    ts,
)
from scientific_environment import ScientificEnvironmentError

# --- Test-local oracles ----------------------------------------------------

DE440 = "de440.bsp"
DE441_PART_1 = "de441_part-1.bsp"
DE441_PART_2 = "de441_part-2.bsp"

# The governed, order-sensitive scientific precedence.
EXPECTED_PRECEDENCE = (DE440, DE441_PART_1, DE441_PART_2)

# Raw JD(TDB) bounds published inside each certified artifact.
PINNED_TDB_BOUNDS = {
    DE440: (2287184.5, 2688976.5),
    DE441_PART_1: (-3100015.5, 2440432.5),
    DE441_PART_2: (2440400.5, 8000016.5),
}

REASON_INSTANT_INVALID = "INSTANT_STATE_INVALID"

# 2650-06-15T12:00:00Z, inside the window the frozen civil routing sends to
# DE440 even though DE440 publishes no data there.
DEFECT_WINDOW_TT = 2689118.000800741
DEFECT_WINDOW_CIVIL_YEAR = 2650

# The directional search reach A2-3 will use. Declared here because the
# single-artifact coverability of a three-day interval is an A2 design
# invariant this file must protect, not a value A2-2 itself reads.
BRACKET_SPAN_DAYS = 3.0

A2_BLOCK_MARKER = "# A2-2 - data-authoritative kernel selection."


def coverage(kernel_name):
    return kernel_coverage_tt(kernel_name)


def holders(tt):
    """Every pinned artifact whose declared coverage contains tt."""
    return [
        name
        for name in EXPECTED_PRECEDENCE
        if coverage(name).tt_start <= tt <= coverage(name).tt_end
    ]


def union_bounds():
    lo = min(coverage(n).tt_start for n in EXPECTED_PRECEDENCE)
    hi = max(coverage(n).tt_end for n in EXPECTED_PRECEDENCE)
    return lo, hi


# --- 1. Governed precedence ------------------------------------------------


class TestGovernedPrecedence(unittest.TestCase):
    def test_precedence_is_the_expected_fixed_ordering(self):
        self.assertEqual(
            tuple(astronomy_solver.PINNED_KERNEL_PRECEDENCE), EXPECTED_PRECEDENCE
        )

    def test_precedence_entries_are_distinct(self):
        entries = tuple(astronomy_solver.PINNED_KERNEL_PRECEDENCE)
        self.assertEqual(len(set(entries)), len(entries))

    def test_selection_is_order_sensitive_where_artifacts_overlap(self):
        """Where several artifacts contain the state, the FIRST one wins."""
        probes = (
            coverage(DE440).tt_start,
            coverage(DE441_PART_2).tt_start,
            coverage(DE441_PART_1).tt_end,
            coverage(DE440).tt_end,
        )
        for tt in probes:
            with self.subTest(tt=tt):
                containing = holders(tt)
                self.assertGreater(
                    len(containing), 1, "probe must be a genuine overlap"
                )
                self.assertEqual(select_kernel_containing_instant(tt), containing[0])

    def test_selection_is_deterministic(self):
        for tt in (coverage(DE440).tt_start, DEFECT_WINDOW_TT, -1000000.0):
            with self.subTest(tt=tt):
                results = {select_kernel_containing_instant(tt) for _ in range(5)}
                self.assertEqual(len(results), 1)


# --- 2. Instant selection --------------------------------------------------


class TestInstantSelection(unittest.TestCase):
    def test_below_de440_support_selects_de441_part_1(self):
        tt = coverage(DE440).tt_start - 1.0
        self.assertEqual(select_kernel_containing_instant(tt), DE441_PART_1)

    def test_above_de440_support_selects_de441_part_2(self):
        tt = coverage(DE440).tt_end + 1.0
        self.assertEqual(select_kernel_containing_instant(tt), DE441_PART_2)

    def test_de440_wins_throughout_its_declared_support(self):
        c = coverage(DE440)
        span = c.tt_end - c.tt_start
        probes = [c.tt_start + span * f for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
        probes.append(coverage(DE441_PART_1).tt_end)
        probes.append(coverage(DE441_PART_2).tt_start)
        for tt in probes:
            with self.subTest(tt=tt):
                self.assertEqual(select_kernel_containing_instant(tt), DE440)

    def test_exact_boundaries_are_closed_inclusive(self):
        for name in EXPECTED_PRECEDENCE:
            c = coverage(name)
            for label, tt in (("start", c.tt_start), ("end", c.tt_end)):
                with self.subTest(kernel=name, bound=label):
                    self.assertIsNotNone(select_kernel_containing_instant(tt))
                    self.assertIn(name, holders(tt))

    def test_absolute_outer_boundaries_have_unique_holders(self):
        lo, hi = union_bounds()
        self.assertEqual(holders(lo), [DE441_PART_1])
        self.assertEqual(holders(hi), [DE441_PART_2])
        self.assertEqual(select_kernel_containing_instant(lo), DE441_PART_1)
        self.assertEqual(select_kernel_containing_instant(hi), DE441_PART_2)

    def test_outside_the_certified_union_returns_none(self):
        lo, hi = union_bounds()
        for tt in (lo - 1e-6, lo - 1.0, lo - 1e6, hi + 1e-6, hi + 1.0, hi + 1e6):
            with self.subTest(tt=tt):
                self.assertIsNone(select_kernel_containing_instant(tt))


# --- 3. Interval selection -------------------------------------------------


class TestIntervalSelection(unittest.TestCase):
    def test_whole_interval_must_fit_one_artifact(self):
        lo, hi = union_bounds()
        self.assertIsNone(select_kernel_for_interval(lo, hi))

    def test_interval_crossing_de440_lower_bound_selects_de441_part_1(self):
        start = coverage(DE440).tt_start
        self.assertEqual(
            select_kernel_for_interval(start - 1.0, start + 1.0), DE441_PART_1
        )

    def test_interval_crossing_de440_upper_bound_selects_de441_part_2(self):
        end = coverage(DE440).tt_end
        self.assertEqual(
            select_kernel_for_interval(end - 1.0, end + 1.0), DE441_PART_2
        )

    def test_interval_inside_de440_selects_de440(self):
        c = coverage(DE440)
        mid = c.tt_start + (c.tt_end - c.tt_start) / 2.0
        self.assertEqual(
            select_kernel_for_interval(mid, mid + BRACKET_SPAN_DAYS), DE440
        )

    def test_zero_width_interval_agrees_with_instant_selection(self):
        lo, hi = union_bounds()
        probes = [
            lo,
            hi,
            coverage(DE440).tt_start,
            coverage(DE440).tt_end,
            DEFECT_WINDOW_TT,
            lo - 1.0,
            hi + 1.0,
        ]
        for tt in probes:
            with self.subTest(tt=tt):
                self.assertEqual(
                    select_kernel_for_interval(tt, tt),
                    select_kernel_containing_instant(tt),
                )

    def test_reversed_interval_fails_closed(self):
        c = coverage(DE440)
        with self.assertRaises(ScientificEnvironmentError) as caught:
            select_kernel_for_interval(c.tt_end, c.tt_start)
        self.assertEqual(caught.exception.reason, REASON_INSTANT_INVALID)

    def test_reversed_interval_by_one_ulp_fails_closed(self):
        tt = coverage(DE440).tt_start
        with self.assertRaises(ScientificEnvironmentError) as caught:
            select_kernel_for_interval(tt, tt - 1e-9)
        self.assertEqual(caught.exception.reason, REASON_INSTANT_INVALID)

    def test_interval_overrunning_the_union_returns_none(self):
        lo, hi = union_bounds()
        self.assertIsNone(select_kernel_for_interval(hi - 1.0, hi + 2.0))
        self.assertIsNone(select_kernel_for_interval(lo - 2.0, lo + 1.0))

    def test_interval_ending_exactly_at_the_ceiling_is_selected(self):
        _, hi = union_bounds()
        self.assertEqual(
            select_kernel_for_interval(hi - BRACKET_SPAN_DAYS, hi), DE441_PART_2
        )

    def test_interval_starting_exactly_at_the_floor_is_selected(self):
        lo, _ = union_bounds()
        self.assertEqual(
            select_kernel_for_interval(lo, lo + BRACKET_SPAN_DAYS), DE441_PART_1
        )


# --- 4. Known 2650 legacy routing defect -----------------------------------


class TestLegacyDefectNotInherited(unittest.TestCase):
    def test_a2_bypasses_the_defect_while_legacy_routing_is_unchanged(self):
        # The instant really is inside the DE440 coverage gap.
        self.assertGreater(DEFECT_WINDOW_TT, coverage(DE440).tt_end)
        self.assertLessEqual(DEFECT_WINDOW_TT, coverage(DE441_PART_2).tt_end)

        # Frozen civil routing still points at the artifact with no data.
        self.assertEqual(choose_kernel_name(DEFECT_WINDOW_CIVIL_YEAR), DE440)

        # The data-authoritative selector does not inherit that defect.
        self.assertEqual(
            select_kernel_containing_instant(DEFECT_WINDOW_TT), DE441_PART_2
        )
        self.assertEqual(
            select_kernel_for_interval(
                DEFECT_WINDOW_TT, DEFECT_WINDOW_TT + BRACKET_SPAN_DAYS
            ),
            DE441_PART_2,
        )


# --- 5. Invalid TT ---------------------------------------------------------


class TestInvalidTT(unittest.TestCase):
    BAD_VALUES = (
        True,
        False,
        "2461485.0",
        None,
        [2461485.0],
        {},
        object(),
        complex(1, 0),
        float("nan"),
        float("inf"),
        float("-inf"),
        10 ** 400,
    )

    def test_instant_selector_rejects_invalid_values(self):
        for value in self.BAD_VALUES:
            with self.subTest(value=repr(value)):
                with self.assertRaises(ScientificEnvironmentError) as caught:
                    select_kernel_containing_instant(value)
                self.assertEqual(caught.exception.reason, REASON_INSTANT_INVALID)

    def test_interval_selector_rejects_invalid_low_bound(self):
        for value in self.BAD_VALUES:
            with self.subTest(value=repr(value)):
                with self.assertRaises(ScientificEnvironmentError) as caught:
                    select_kernel_for_interval(value, 2461485.0)
                self.assertEqual(caught.exception.reason, REASON_INSTANT_INVALID)

    def test_interval_selector_rejects_invalid_high_bound(self):
        for value in self.BAD_VALUES:
            with self.subTest(value=repr(value)):
                with self.assertRaises(ScientificEnvironmentError) as caught:
                    select_kernel_for_interval(2461485.0, value)
                self.assertEqual(caught.exception.reason, REASON_INSTANT_INVALID)


# --- 6. Artifact and coverage invariants -----------------------------------


class TestArtifactInvariants(unittest.TestCase):
    """If the certified artifact set changes, these must fail loudly.

    Each assertion here is a premise the A2 selection design rests on. A
    future artifact that breaks one of them changes scientific meaning, and
    must not do so silently.
    """

    def test_raw_tdb_bounds_are_the_pinned_ones(self):
        for name, (start, end) in PINNED_TDB_BOUNDS.items():
            with self.subTest(kernel=name):
                c = coverage(name)
                self.assertEqual(c.tdb_start, start)
                self.assertEqual(c.tdb_end, end)

    def test_certified_union_is_contiguous(self):
        spans = sorted(
            (coverage(n).tt_start, coverage(n).tt_end) for n in EXPECTED_PRECEDENCE
        )
        merged_end = spans[0][1]
        for start, end in spans[1:]:
            self.assertLessEqual(
                start, merged_end, "a gap opened in the certified union"
            )
            merged_end = max(merged_end, end)

    def test_de440_contains_the_de441_part_overlap(self):
        p1, p2, d440 = (
            coverage(DE441_PART_1),
            coverage(DE441_PART_2),
            coverage(DE440),
        )
        overlap_lo = max(p1.tt_start, p2.tt_start)
        overlap_hi = min(p1.tt_end, p2.tt_end)
        self.assertLess(overlap_lo, overlap_hi, "the DE441 parts must overlap")
        self.assertLessEqual(d440.tt_start, overlap_lo)
        self.assertLessEqual(overlap_hi, d440.tt_end)
        # Therefore a DE441-vs-DE441 precedence comparison is unreachable.
        for tt in (overlap_lo, (overlap_lo + overlap_hi) / 2.0, overlap_hi):
            with self.subTest(tt=tt):
                self.assertEqual(select_kernel_containing_instant(tt), DE440)

    def test_interior_three_day_intervals_are_single_artifact_coverable(self):
        lo, hi = union_bounds()
        probe = lo + BRACKET_SPAN_DAYS
        step = (hi - lo) / 400.0
        checked = 0
        while probe < hi - BRACKET_SPAN_DAYS:
            forward = select_kernel_for_interval(probe, probe + BRACKET_SPAN_DAYS)
            backward = select_kernel_for_interval(probe - BRACKET_SPAN_DAYS, probe)
            self.assertIsNotNone(forward, "forward interval lost coverage at %r" % probe)
            self.assertIsNotNone(
                backward, "backward interval lost coverage at %r" % probe
            )
            checked += 1
            probe += step
        self.assertGreater(checked, 100)


# --- 7. Declared coverage is not a computation guarantee -------------------


class TestDeclaredCoverageIsNotEvaluability(unittest.TestCase):
    """Selection reports declared coverage, not computation reach.

    A2-2's contract is that the returned artifact DECLARES coverage of the
    state. It is deliberately not a promise that a later computation can
    evaluate there: an observation that resolves light time reads the
    ephemeris before the instant it is asked about.

    This test pins the contract distinction only. It encodes no margin and
    no fixed lookback; the magnitude of that reach, and the handling of it,
    belong to A2-3.
    """

    def test_selection_succeeds_at_a_declared_lower_endpoint(self):
        for name in EXPECTED_PRECEDENCE:
            with self.subTest(kernel=name):
                self.assertIsNotNone(
                    select_kernel_containing_instant(coverage(name).tt_start)
                )

    def test_declared_coverage_does_not_imply_observation_is_evaluable(self):
        name = select_kernel_containing_instant(coverage(DE440).tt_start)
        self.assertIsNotNone(name)

        eph = load_kernel(name)
        t = ts.tt_jd(coverage(name).tt_start)
        with self.assertRaises(Exception):
            eph["earth"].at(t).observe(eph["sun"]).apparent()


# --- 8. Structural independence --------------------------------------------


class TestA2SelectionStructuralIndependence(unittest.TestCase):
    """The A2-2 block must not inherit civil routing or legacy guards.

    Scoped to the A2-2 block's executable logic. The same tokens exist
    legitimately elsewhere in astronomy_solver.py, in frozen legacy code
    this increment does not touch, and in prose that names an excluded
    mechanism in order to state that it is excluded.
    """

    @staticmethod
    def _executable_source(block):
        """Return the block's executable logic, prose excluded.

        Follows the A2-1 house pattern from
        tests/test_a2_kernel_coverage.py.
        """
        lines = block.splitlines(keepends=True)
        drop = set()
        for node in ast.walk(ast.parse(block)):
            body = getattr(node, "body", None)
            if not body:
                continue
            first = body[0]
            if (isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                drop.update(range(first.lineno - 1, first.end_lineno))
        kept = (l for i, l in enumerate(lines) if i not in drop)
        return "".join(l for l in kept if not l.lstrip().startswith("#"))

    def setUp(self):
        source = inspect.getsource(astronomy_solver)
        index = source.find(A2_BLOCK_MARKER)
        self.assertNotEqual(index, -1, "A2-2 block marker not found")
        self.block = source[index:]
        self.preamble = source[:index]
        self.executable = self._executable_source(self.block)

    def test_block_was_located_and_is_substantial(self):
        self.assertIn("def select_kernel_containing_instant", self.block)
        self.assertIn("def select_kernel_for_interval", self.block)
        self.assertGreater(len(self.block), 1000)

    def test_block_does_not_use_civil_routing_or_legacy_guards(self):
        # The oracle must not pass by inspecting nothing.
        self.assertIn("def select_kernel_containing_instant", self.executable)
        self.assertIn("PINNED_KERNEL_PRECEDENCE", self.executable)

        # Prose really was excluded.
        self.assertNotIn("SCOPE OF THE CLAIM", self.executable)
        self.assertNotIn("light time", self.executable)

        for token in (
            "choose_kernel_name",
            "PRIMARY_START_YEAR",
            "PRIMARY_END_YEAR",
            "ts.utc(",
            "3600",
            "min_gap",
            "timedelta",
            "longitude",
            "/ 15",
            "utc_datetime",
            "isoformat",
        ):
            with self.subTest(token=token):
                self.assertNotIn(token, self.executable)

    def test_frozen_legacy_code_is_still_present_ahead_of_the_block(self):
        """Guards against 'passing' by having deleted the legacy code."""
        for token in (
            "def choose_kernel_name",
            "PRIMARY_START_YEAR",
            "PRIMARY_END_YEAR",
            "min_gap_seconds",
            "def find_sunset_successor",
            "def kernel_coverage_tt",
        ):
            with self.subTest(token=token):
                self.assertIn(token, self.preamble)


if __name__ == "__main__":
    unittest.main()
