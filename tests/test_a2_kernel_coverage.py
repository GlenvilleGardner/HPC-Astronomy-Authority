"""A2-1 verification for the certified kernel coverage substrate.

Covers kernel_coverage_tt, KernelCoverage, SunsetChronologyError,
REQUIRED_SOLAR_SEGMENT_PAIRS and _exact_finite_tt in astronomy_solver.py.

Test framework: standard-library unittest.
Scientific execution uses the repository's certified Skyfield runtime and
the real pinned Authority ephemeris artifacts.

INDEPENDENT ORACLE DISCIPLINE

Kernel filenames, raw JD(TDB) segment bounds, the required center/target
pairs and the fail-closed reason code are declared as test-local literals.
They are deliberately NOT imported from astronomy_solver: importing them
would make the tests agree with a defective constant instead of detecting
it.

The TT oracle is obtained by calling segment.time_range(ts) directly on the
certified Skyfield segment objects, independently of the function under
test, so a defective conversion inside kernel_coverage_tt cannot confirm
itself.

The shared timescale and the kernel loader ARE taken from astronomy_solver.
They are repository infrastructure rather than the unit under test, and
duplicating them would mmap a second copy of three multi-gigabyte kernels
for no verification benefit.

SCOPE

This file verifies the A2-1 coverage substrate only. It asserts nothing
about kernel selection, precedence, interval solving, sunset determination,
HTTP behavior, or any route. Those belong to later governed operations and
do not exist yet.

Fail-closed invariants that cannot be provoked with the certified artifact
set - a missing required pair, a duplicated required pair, and required
segments sharing no interval - are exercised against stub kernels injected
in place of load_kernel. The stubs model only the segment metadata surface
that kernel_coverage_tt reads; no astronomy is stubbed, because none is
performed by the unit under test.
"""

import ast
import inspect
import math
import re
import unittest
from unittest import mock

import astronomy_solver
from astronomy_solver import kernel_coverage_tt, load_kernel, ts
from scientific_environment import ScientificEnvironmentError

# --- Test-local oracles ----------------------------------------------------

PINNED_KERNELS = ("de440.bsp", "de441_part-1.bsp", "de441_part-2.bsp")

REQUIRED_PAIRS = ((0, 3), (3, 399), (0, 10))

# Raw JD(TDB) segment bounds published inside each certified artifact.
PINNED_TDB_BOUNDS = {
    "de440.bsp": (2287184.5, 2688976.5),
    "de441_part-1.bsp": (-3100015.5, 2440432.5),
    "de441_part-2.bsp": (2440400.5, 8000016.5),
}

REASON_SEGMENTS_UNAVAILABLE = "KERNEL_SEGMENTS_UNAVAILABLE"
REASON_INSTANT_INVALID = "INSTANT_STATE_INVALID"

A2_BLOCK_MARKER = "# A2-1 - certified kernel coverage substrate."

# The shape of a governed production block header, used only to find where
# THIS block ends. It matches the header FORM and nothing else, so A2-1
# certification governs the A2-1 block and stops.
GOVERNED_BLOCK_MARKER = re.compile(r"^# A\d+[a-z]?(?:-\d+[a-z]?)? - ",
                                   re.MULTILINE)


# --- Stub kernel surface ---------------------------------------------------


class _StubSpkSegment(object):
    """The raw jplephem descriptor surface kernel_coverage_tt reads."""

    def __init__(self, start_jd, end_jd):
        self.start_jd = start_jd
        self.end_jd = end_jd


class _StubSegment(object):
    """The Skyfield segment surface kernel_coverage_tt reads."""

    def __init__(self, center, target, start_jd, end_jd):
        self.center = center
        self.target = target
        self.spk_segment = _StubSpkSegment(start_jd, end_jd)

    def time_range(self, timescale):
        return (
            timescale.tdb_jd(self.spk_segment.start_jd),
            timescale.tdb_jd(self.spk_segment.end_jd),
        )


class _StubKernel(object):
    def __init__(self, segments):
        self.segments = list(segments)


def _stub_kernel(pairs_and_bounds):
    return _StubKernel(
        _StubSegment(center, target, start, end)
        for (center, target), (start, end) in pairs_and_bounds
    )


class _CoverageCacheIsolation(unittest.TestCase):
    """Every coverage test starts and ends with an empty coverage cache.

    kernel_coverage_tt is memoized. Clearing it keeps stub-injected cases
    from leaking into artifact cases and vice versa, and lets the caching
    behavior itself be asserted from a known state.
    """

    def setUp(self):
        kernel_coverage_tt.cache_clear()

    def tearDown(self):
        kernel_coverage_tt.cache_clear()


# --- Certified artifact coverage -------------------------------------------


class TestCertifiedArtifactCoverage(_CoverageCacheIsolation):
    def test_required_pairs_present_exactly_once(self):
        for kernel_name in PINNED_KERNELS:
            eph = load_kernel(kernel_name)
            for pair in REQUIRED_PAIRS:
                with self.subTest(kernel=kernel_name, pair=pair):
                    matches = [
                        s for s in eph.segments if (s.center, s.target) == pair
                    ]
                    self.assertEqual(len(matches), 1)

    def test_raw_tdb_bounds_equal_published_segment_metadata(self):
        for kernel_name in PINNED_KERNELS:
            with self.subTest(kernel=kernel_name):
                coverage = kernel_coverage_tt(kernel_name)
                expected_start, expected_end = PINNED_TDB_BOUNDS[kernel_name]
                self.assertEqual(coverage.tdb_start, expected_start)
                self.assertEqual(coverage.tdb_end, expected_end)

    def test_tt_bounds_equal_certified_time_range(self):
        for kernel_name in PINNED_KERNELS:
            eph = load_kernel(kernel_name)
            oracle_start = None
            oracle_end = None
            for pair in REQUIRED_PAIRS:
                segment = next(
                    s for s in eph.segments if (s.center, s.target) == pair
                )
                t_start, t_end = segment.time_range(ts)
                start = float(t_start.tt)
                end = float(t_end.tt)
                oracle_start = start if oracle_start is None else max(
                    oracle_start, start
                )
                oracle_end = end if oracle_end is None else min(oracle_end, end)

            with self.subTest(kernel=kernel_name):
                coverage = kernel_coverage_tt(kernel_name)
                self.assertEqual(coverage.tt_start, oracle_start)
                self.assertEqual(coverage.tt_end, oracle_end)

    def test_kernel_name_is_reported_truthfully(self):
        for kernel_name in PINNED_KERNELS:
            with self.subTest(kernel=kernel_name):
                self.assertEqual(kernel_coverage_tt(kernel_name).kernel, kernel_name)

    def test_bounds_are_finite_and_ordered(self):
        for kernel_name in PINNED_KERNELS:
            with self.subTest(kernel=kernel_name):
                coverage = kernel_coverage_tt(kernel_name)
                for value in (
                    coverage.tdb_start,
                    coverage.tdb_end,
                    coverage.tt_start,
                    coverage.tt_end,
                ):
                    self.assertIsInstance(value, float)
                    self.assertTrue(math.isfinite(value))
                self.assertLess(coverage.tdb_start, coverage.tdb_end)
                self.assertLess(coverage.tt_start, coverage.tt_end)

    def test_coverage_record_is_immutable(self):
        coverage = kernel_coverage_tt(PINNED_KERNELS[0])
        with self.assertRaises(Exception):
            coverage.tt_start = 0.0

    def test_returned_coverage_is_the_intersection(self):
        """Distinct per-pair bounds must intersect, not union or first-win."""
        stub = _stub_kernel(
            (
                ((0, 3), (2287184.5, 2688976.5)),
                ((3, 399), (2300000.5, 2600000.5)),
                ((0, 10), (2290000.5, 2650000.5)),
            )
        )
        with mock.patch.object(astronomy_solver, "load_kernel", return_value=stub):
            coverage = kernel_coverage_tt("stub-intersection.bsp")

        self.assertEqual(coverage.tdb_start, 2300000.5)
        self.assertEqual(coverage.tdb_end, 2600000.5)
        self.assertEqual(coverage.tt_start, float(ts.tdb_jd(2300000.5).tt))
        self.assertEqual(coverage.tt_end, float(ts.tdb_jd(2600000.5).tt))


# --- Skyfield chain equivalence --------------------------------------------


class TestSkyfieldChainEquivalence(_CoverageCacheIsolation):
    """REQUIRED_SOLAR_SEGMENT_PAIRS must not become a stale HPC assumption.

    The pairs are a hard-coded model of the vector chain a topocentric
    sunset consumes. Skyfield resolves that chain itself, by target, at
    lookup time. If a future certified artifact or a future library
    version resolved a different chain, the coverage this module reports
    would describe segments the solver does not actually evaluate.
    """

    @staticmethod
    def _chain_pairs(vector_function):
        parts = getattr(vector_function, "vector_functions", None)
        if parts is None:
            parts = (vector_function,)
        return {(part.center, part.target) for part in parts}

    def test_resolved_earth_and_sun_chain_matches_required_pairs(self):
        for kernel_name in PINNED_KERNELS:
            with self.subTest(kernel=kernel_name):
                eph = load_kernel(kernel_name)
                resolved = self._chain_pairs(eph["earth"]) | self._chain_pairs(
                    eph["sun"]
                )
                self.assertEqual(resolved, set(REQUIRED_PAIRS))

    def test_module_constant_matches_the_resolved_chain(self):
        self.assertEqual(
            set(astronomy_solver.REQUIRED_SOLAR_SEGMENT_PAIRS), set(REQUIRED_PAIRS)
        )


# --- Fail-closed invariants ------------------------------------------------


class TestFailClosedInvariants(_CoverageCacheIsolation):
    def assert_segments_unavailable(self, stub, kernel_name):
        with mock.patch.object(astronomy_solver, "load_kernel", return_value=stub):
            with self.assertRaises(ScientificEnvironmentError) as caught:
                kernel_coverage_tt(kernel_name)

        self.assertEqual(caught.exception.reason, REASON_SEGMENTS_UNAVAILABLE)
        return caught.exception

    def test_missing_required_pair_fails_closed(self):
        stub = _stub_kernel(
            (
                ((0, 3), (2287184.5, 2688976.5)),
                ((0, 10), (2287184.5, 2688976.5)),
            )
        )
        error = self.assert_segments_unavailable(stub, "stub-missing.bsp")
        self.assertIn("(3, 399)", str(error))

    def test_duplicated_required_pair_fails_closed(self):
        stub = _stub_kernel(
            (
                ((0, 3), (2287184.5, 2688976.5)),
                ((0, 3), (2300000.5, 2600000.5)),
                ((3, 399), (2287184.5, 2688976.5)),
                ((0, 10), (2287184.5, 2688976.5)),
            )
        )
        error = self.assert_segments_unavailable(stub, "stub-duplicate.bsp")
        self.assertIn("more than once", str(error))

    def test_required_segments_without_common_interval_fail_closed(self):
        stub = _stub_kernel(
            (
                ((0, 3), (2287184.5, 2300000.5)),
                ((3, 399), (2400000.5, 2500000.5)),
                ((0, 10), (2287184.5, 2500000.5)),
            )
        )
        error = self.assert_segments_unavailable(stub, "stub-disjoint.bsp")
        self.assertIn("non-empty interval", str(error))

    def test_failures_are_not_cached(self):
        stub = _stub_kernel((((0, 3), (2287184.5, 2688976.5)),))
        loader = mock.Mock(return_value=stub)

        with mock.patch.object(astronomy_solver, "load_kernel", loader):
            for _ in range(2):
                with self.assertRaises(ScientificEnvironmentError):
                    kernel_coverage_tt("stub-uncached-failure.bsp")

        self.assertEqual(loader.call_count, 2)

    def test_failure_does_not_poison_a_later_success(self):
        broken = _stub_kernel((((0, 3), (2287184.5, 2688976.5)),))
        whole = _stub_kernel(
            (
                ((0, 3), (2287184.5, 2688976.5)),
                ((3, 399), (2287184.5, 2688976.5)),
                ((0, 10), (2287184.5, 2688976.5)),
            )
        )

        with mock.patch.object(astronomy_solver, "load_kernel", return_value=broken):
            with self.assertRaises(ScientificEnvironmentError):
                kernel_coverage_tt("stub-recovering.bsp")

        with mock.patch.object(astronomy_solver, "load_kernel", return_value=whole):
            coverage = kernel_coverage_tt("stub-recovering.bsp")

        self.assertEqual(coverage.tdb_start, 2287184.5)

    def test_successful_coverage_is_cached(self):
        kernel_name = PINNED_KERNELS[0]
        first = kernel_coverage_tt(kernel_name)
        hits_before = kernel_coverage_tt.cache_info().hits
        second = kernel_coverage_tt(kernel_name)

        self.assertIs(first, second)
        self.assertEqual(kernel_coverage_tt.cache_info().hits, hits_before + 1)

    def test_cached_success_does_not_reload_the_kernel(self):
        stub = _stub_kernel(
            (
                ((0, 3), (2287184.5, 2688976.5)),
                ((3, 399), (2287184.5, 2688976.5)),
                ((0, 10), (2287184.5, 2688976.5)),
            )
        )
        loader = mock.Mock(return_value=stub)

        with mock.patch.object(astronomy_solver, "load_kernel", loader):
            kernel_coverage_tt("stub-cached-success.bsp")
            kernel_coverage_tt("stub-cached-success.bsp")

        self.assertEqual(loader.call_count, 1)

    def test_error_is_a_scientific_environment_error(self):
        stub = _stub_kernel((((0, 3), (2287184.5, 2688976.5)),))
        with mock.patch.object(astronomy_solver, "load_kernel", return_value=stub):
            with self.assertRaises(ScientificEnvironmentError):
                kernel_coverage_tt("stub-error-type.bsp")


# --- Exact TT validation ---------------------------------------------------


class TestExactFiniteTT(unittest.TestCase):
    def setUp(self):
        self.validate = astronomy_solver._exact_finite_tt

    def assert_rejected(self, value):
        with self.assertRaises(ScientificEnvironmentError) as caught:
            self.validate(value, "tt")
        self.assertEqual(caught.exception.reason, REASON_INSTANT_INVALID)
        return caught.exception

    def test_accepts_float_and_returns_it(self):
        result = self.validate(2461485.4645259595, "tt")
        self.assertIsInstance(result, float)
        self.assertEqual(result, 2461485.4645259595)

    def test_accepts_int_and_returns_float(self):
        result = self.validate(2461485, "tt")
        self.assertIsInstance(result, float)
        self.assertEqual(result, 2461485.0)

    def test_accepts_negative_and_zero(self):
        self.assertEqual(self.validate(-3100015.5, "tt"), -3100015.5)
        self.assertEqual(self.validate(0, "tt"), 0.0)

    def test_rejects_bool(self):
        for value in (True, False):
            with self.subTest(value=value):
                self.assert_rejected(value)

    def test_rejects_non_numeric(self):
        for value in ("2461485.0", None, [2461485.0], {}, object(), complex(1, 0)):
            with self.subTest(value=repr(value)):
                self.assert_rejected(value)

    def test_rejects_nan(self):
        self.assert_rejected(float("nan"))

    def test_rejects_infinities(self):
        for value in (float("inf"), float("-inf")):
            with self.subTest(value=value):
                self.assert_rejected(value)

    def test_rejects_integer_too_large_for_float(self):
        self.assert_rejected(10 ** 400)

    def test_reason_code_is_stable(self):
        error = self.assert_rejected(float("nan"))
        self.assertEqual(error.reason, "INSTANT_STATE_INVALID")

    def test_field_name_appears_in_the_diagnostic(self):
        with self.assertRaises(ScientificEnvironmentError) as caught:
            self.validate(float("nan"), "anchor tt")
        self.assertIn("anchor tt", str(caught.exception))


# --- Structural independence -----------------------------------------------


class TestA2BlockStructuralIndependence(unittest.TestCase):
    """The A2-1 block must not inherit civil routing or legacy guards.

    The assertion is scoped to the A2-1 block only. The same tokens exist
    legitimately elsewhere in astronomy_solver.py, in frozen legacy code
    that this increment does not touch and must not disturb.
    """

    @staticmethod
    def _executable_source(block):
        """Return the block's executable logic, prose excluded.

        Follows the established repository pattern in
        tests/test_sunset_successor.py: the prohibition governs executable
        logic, not prose. The block's header comment legitimately names
        the legacy routing it refuses to use, and matching raw text would
        read that promise as a violation.
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
        self.assertNotEqual(index, -1, "A2-1 block marker not found")
        following = GOVERNED_BLOCK_MARKER.search(
            source, index + len(A2_BLOCK_MARKER))
        self.block = source[index:following.start() if following else len(source)]
        self.preamble = source[:index]
        self.executable = self._executable_source(self.block)

    def test_block_was_located_and_is_substantial(self):
        self.assertIn("def kernel_coverage_tt", self.block)
        self.assertIn("def _exact_finite_tt", self.block)
        self.assertGreater(len(self.block), 1000)

    def test_block_stops_at_the_next_governed_block(self):
        """The scan must cover A2-1 and end there, not annex what follows."""
        self.assertIsNone(
            GOVERNED_BLOCK_MARKER.search(self.block, len(A2_BLOCK_MARKER)),
            "the A2-1 scan reaches into a later governed block",
        )

    def test_block_does_not_use_civil_routing_or_legacy_guards(self):
        # The oracle must not pass by inspecting nothing.
        self.assertIn("def kernel_coverage_tt", self.executable)
        self.assertIn("REQUIRED_SOLAR_SEGMENT_PAIRS", self.executable)

        # Prose really was excluded.
        self.assertNotIn("SINGLE-SEGMENT INVARIANT", self.executable)
        self.assertNotIn("legacy compatibility routing", self.executable)

        for token in (
            "choose_kernel_name",
            "PRIMARY_START_YEAR",
            "PRIMARY_END_YEAR",
            "3600",
            "min_gap",
            "timedelta",
            "ts.utc(",
            "/ 15",
            "longitude",
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
        ):
            with self.subTest(token=token):
                self.assertIn(token, self.preamble)


if __name__ == "__main__":
    unittest.main()
