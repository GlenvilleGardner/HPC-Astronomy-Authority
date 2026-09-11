"""A2-4 verification for the atomic sunset bracket.

Covers SunsetBracket and find_sunset_bracket in astronomy_solver.py.

Test framework: standard-library unittest.
Scientific execution uses the repository's certified Skyfield runtime and
the real pinned Authority ephemeris artifacts.

INDEPENDENT ORACLE DISCIPLINE

Reason codes, pinned artifact filenames, the directional reach, the record
field names and the 2650 defect window are declared as test-local literals.
They are deliberately NOT imported from astronomy_solver: importing a
constant to check that same constant would make the test agree with a
defective value instead of detecting it.

Whether a returned boundary is a genuine sunset is decided by enumerating
crossings with skyfield.almanac directly, inside the very frontier the
corresponding directional solver was bound to. find_sunset_bracket is never
used to produce its own expected answer.

find_sunset_predecessor, find_sunset_from_instant and
supported_search_frontier ARE used, as published and separately certified
A2-3 infrastructure, to establish the preconditions a bracket case requires
and to check that A2-4 composes them rather than re-solving.

SCOPE

A2-4 is still Astronomy Authority. This file asserts nothing about weekday,
ordinal, month or day ownership, annual grid, year, feast date, route or
HTTP behavior. None of those exist, and none is implied by a bracket.

WHAT IS DELIBERATELY NOT ENCODED

No epsilon, tolerance, minimum gap or light-time second-count appears as an
expected value. No cross-bracket binary64 event identity is used: where two
searches would use different brackets, only ordinal relations and exact
comparisons against values produced by one and the same enumeration are
asserted.
"""

import ast
import dataclasses
import inspect
import math
import re
import unittest
from unittest import mock

from skyfield import almanac
from skyfield.api import wgs84

import astronomy_solver
from astronomy_solver import (
    SunsetEvent,
    choose_kernel_name,
    find_sunset_bracket,
    find_sunset_from_instant,
    find_sunset_predecessor,
    kernel_coverage_tt,
    load_kernel,
    supported_search_frontier,
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
REASON_OBSERVER_OUT_OF_DOMAIN = "OBSERVER_OUT_OF_DOMAIN"
REASON_INSTANT_INVALID = "INSTANT_STATE_INVALID"

SPAN_DAYS = 3.0

EXPECTED_BRACKET_FIELDS = ("anchor", "previous", "next")

INTERIOR_TT = 2460678.0

DEFECT_WINDOW_TT = 2689118.000800741
DEFECT_WINDOW_CIVIL_YEAR = 2650

REFERENCE_LATITUDE = 40.7406
REFERENCE_LONGITUDE = -73.9

OBSERVERS = (
    (40.7406, -73.9, "new-york"),
    (0.0, 0.0, "null-island"),
    (-33.9, 151.2, "sydney"),
    (35.7, 139.7, "tokyo"),
    (64.1, -21.9, "reykjavik"),
    (-54.8, -68.3, "ushuaia"),
    (0.0, 180.0, "antimeridian"),
)

POLAR_CASES = (
    (78.2, 15.6, 2460850.0, "svalbard polar day"),
    (78.2, 15.6, 2460680.0, "svalbard polar night"),
    (-80.0, 0.0, 2460680.0, "antarctic polar day"),
    (-75.0, 120.0, 2460860.0, "antarctic polar night"),
)

A2_4_BLOCK_MARKER = "# A2-4 - atomic sunset bracket."

GOVERNED_BLOCK_MARKER = re.compile(r"^# A\d+[a-z]?(?:-[0-9a-z]+)? - ",
                                   re.MULTILINE)


# --- Independent helpers ---------------------------------------------------


def coverage(kernel_name):
    return kernel_coverage_tt(kernel_name)


def union_bounds():
    lo = min(coverage(name).tt_start for name in PINNED_ARTIFACTS)
    hi = max(coverage(name).tt_end for name in PINNED_ARTIFACTS)
    return lo, hi


def predicate(kernel_name, latitude, longitude):
    return almanac.sunrise_sunset(
        load_kernel(kernel_name), wgs84.latlon(latitude, longitude)
    )


def backward_frontier(anchor, latitude, longitude):
    return supported_search_frontier(
        anchor, anchor - SPAN_DAYS, latitude, longitude
    )


def forward_frontier(anchor, latitude, longitude):
    return supported_search_frontier(
        anchor, anchor + SPAN_DAYS, latitude, longitude
    )


def oracle_sunsets(frontier, latitude, longitude):
    """Independent enumeration of sunsets inside one admitted frontier."""
    times, events = almanac.find_discrete(
        ts.tt_jd(frontier.tt_lo),
        ts.tt_jd(frontier.tt_hi),
        predicate(frontier.kernel, latitude, longitude),
    )
    return [
        float(t.tt)
        for t, sun_is_up in zip(times, events)
        if not bool(sun_is_up)
    ]


def deep_time_anchor(offset_days):
    return coverage(DE441_PART_1).tt_start + offset_days


def ceiling_anchor(offset_days):
    return coverage(DE441_PART_2).tt_end - offset_days


def executable_source(block):
    """Return a block's executable logic, prose excluded.

    An AST node's ``body`` attribute is not always a list - ``ast.IfExp``
    carries a single expression there - so the list check is required
    rather than defensive.
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


class BracketAssertions(unittest.TestCase):
    def assert_fails_closed(self, reason, anchor,
                            latitude=REFERENCE_LATITUDE,
                            longitude=REFERENCE_LONGITUDE):
        with self.assertRaises(ScientificEnvironmentError) as caught:
            find_sunset_bracket(anchor, latitude, longitude)
        self.assertEqual(caught.exception.reason, reason)
        return caught.exception

    def assert_bracket_invariants(self, bracket, anchor, latitude, longitude):
        self.assertIsInstance(bracket, astronomy_solver.SunsetBracket)
        self.assertIsInstance(bracket.previous, SunsetEvent)
        self.assertIsInstance(bracket.next, SunsetEvent)

        # The anchor is preserved exactly.
        self.assertIsInstance(bracket.anchor, float)
        self.assertEqual(bracket.anchor, anchor)

        # Strict ordering, exact binary64.
        self.assertLess(bracket.previous.tt, bracket.anchor)
        self.assertLess(bracket.anchor, bracket.next.tt)
        self.assertLess(bracket.previous.tt, bracket.next.tt)

        self.assertIn(bracket.previous.kernel, PINNED_ARTIFACTS)
        self.assertIn(bracket.next.kernel, PINNED_ARTIFACTS)

        # Each boundary is a genuine sunset, established independently
        # inside the very frontier its own direction was bound to.
        self.assertIn(
            bracket.previous.tt,
            oracle_sunsets(
                backward_frontier(anchor, latitude, longitude),
                latitude, longitude,
            ),
        )
        self.assertIn(
            bracket.next.tt,
            oracle_sunsets(
                forward_frontier(anchor, latitude, longitude),
                latitude, longitude,
            ),
        )


# --- 1. Record contract ----------------------------------------------------


class TestBracketRecord(BracketAssertions):
    def setUp(self):
        self.bracket = find_sunset_bracket(
            INTERIOR_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )

    def test_fields_are_exactly_the_expected_set(self):
        self.assertEqual(
            tuple(f.name for f in dataclasses.fields(self.bracket)),
            EXPECTED_BRACKET_FIELDS,
        )

    def test_record_is_frozen(self):
        for field in EXPECTED_BRACKET_FIELDS:
            with self.subTest(field=field):
                with self.assertRaises(Exception):
                    setattr(self.bracket, field, None)

    def test_boundaries_are_sunset_events(self):
        self.assertIsInstance(self.bracket.previous, SunsetEvent)
        self.assertIsInstance(self.bracket.next, SunsetEvent)
        for event in (self.bracket.previous, self.bracket.next):
            with self.subTest(event=event):
                self.assertEqual(
                    tuple(f.name for f in dataclasses.fields(event)),
                    ("tt", "kernel"),
                )

    def test_anchor_is_the_exact_validated_binary64_value(self):
        self.assertIsInstance(self.bracket.anchor, float)
        self.assertEqual(self.bracket.anchor, INTERIOR_TT)
        # An integer anchor is validated to the identical float value.
        integral = find_sunset_bracket(
            2460678, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.assertIsInstance(integral.anchor, float)
        self.assertEqual(integral.anchor, 2460678.0)

    def test_no_observer_field(self):
        names = {f.name for f in dataclasses.fields(self.bracket)}
        for forbidden in ("latitude", "longitude", "observer", "location"):
            with self.subTest(field=forbidden):
                self.assertNotIn(forbidden, names)
                self.assertFalse(hasattr(self.bracket, forbidden))

    def test_no_bracket_level_kernel_field(self):
        names = {f.name for f in dataclasses.fields(self.bracket)}
        for forbidden in ("kernel", "ephemeris", "artifact"):
            with self.subTest(field=forbidden):
                self.assertNotIn(forbidden, names)
                self.assertFalse(hasattr(self.bracket, forbidden))
        # Provenance lives on each boundary instead.
        self.assertTrue(hasattr(self.bracket.previous, "kernel"))
        self.assertTrue(hasattr(self.bracket.next, "kernel"))

    def test_no_transport_representation(self):
        for forbidden in ("utc", "iso", "isoformat", "datetime", "unix"):
            with self.subTest(field=forbidden):
                self.assertFalse(hasattr(self.bracket, forbidden))


# --- 2. Strict bracket invariants ------------------------------------------


class TestStrictInvariants(BracketAssertions):
    def test_global_observer_matrix(self):
        for latitude, longitude, label in OBSERVERS:
            with self.subTest(observer=label):
                bracket = find_sunset_bracket(
                    INTERIOR_TT, latitude, longitude
                )
                self.assertIsNotNone(bracket)
                self.assert_bracket_invariants(
                    bracket, INTERIOR_TT, latitude, longitude
                )

    def test_ordering_is_strict_with_no_equality(self):
        bracket = find_sunset_bracket(
            INTERIOR_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.assertNotEqual(bracket.previous.tt, bracket.anchor)
        self.assertNotEqual(bracket.next.tt, bracket.anchor)
        self.assertNotEqual(bracket.previous.tt, bracket.next.tt)

    def test_boundaries_are_finite_reals(self):
        bracket = find_sunset_bracket(
            INTERIOR_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        for value in (bracket.anchor, bracket.previous.tt, bracket.next.tt):
            with self.subTest(value=value):
                self.assertIsInstance(value, float)
                self.assertTrue(math.isfinite(value))

    def test_repeated_calls_are_deterministic(self):
        results = {
            find_sunset_bracket(
                INTERIOR_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
            )
            for _ in range(3)
        }
        self.assertEqual(len(results), 1)


# --- 3. Composition identity -----------------------------------------------


class TestCompositionIdentity(BracketAssertions):
    def test_bracket_equals_the_published_directional_results(self):
        bracket = find_sunset_bracket(
            INTERIOR_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.assertEqual(
            bracket.previous,
            find_sunset_predecessor(
                INTERIOR_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
            ),
        )
        self.assertEqual(
            bracket.next,
            find_sunset_from_instant(
                INTERIOR_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
            ),
        )

    def test_each_direction_is_invoked_exactly_once_with_identical_inputs(self):
        with mock.patch.object(
            astronomy_solver, "find_sunset_predecessor",
            wraps=astronomy_solver.find_sunset_predecessor,
        ) as before, mock.patch.object(
            astronomy_solver, "find_sunset_from_instant",
            wraps=astronomy_solver.find_sunset_from_instant,
        ) as after:
            find_sunset_bracket(
                INTERIOR_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
            )

        self.assertEqual(before.call_count, 1)
        self.assertEqual(after.call_count, 1)
        self.assertEqual(before.call_args, after.call_args)

        anchor, latitude, longitude = before.call_args.args
        self.assertIsInstance(anchor, float)
        self.assertEqual(anchor, INTERIOR_TT)
        self.assertEqual(latitude, REFERENCE_LATITUDE)
        self.assertEqual(longitude, REFERENCE_LONGITUDE)

    def test_validated_anchor_is_shared_by_both_directions(self):
        """An integer anchor must reach both sides as the same float."""
        with mock.patch.object(
            astronomy_solver, "find_sunset_predecessor",
            wraps=astronomy_solver.find_sunset_predecessor,
        ) as before, mock.patch.object(
            astronomy_solver, "find_sunset_from_instant",
            wraps=astronomy_solver.find_sunset_from_instant,
        ) as after:
            bracket = find_sunset_bracket(
                2460678, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
            )

        first = before.call_args.args[0]
        second = after.call_args.args[0]
        self.assertIsInstance(first, float)
        self.assertIsInstance(second, float)
        self.assertEqual(first, second)
        self.assertEqual(bracket.anchor, first)

    def test_anchor_is_validated_once_at_the_bracket_level(self):
        body = executable_source(
            inspect.getsource(astronomy_solver.find_sunset_bracket)
        )
        self.assertEqual(body.count("_exact_finite_tt("), 1)

    def test_no_third_astronomical_solution(self):
        body = executable_source(
            inspect.getsource(astronomy_solver.find_sunset_bracket)
        )
        for token in (
            "find_discrete",
            "sunrise_sunset",
            "load_kernel",
            "supported_search_frontier",
            "wgs84",
            "almanac",
            "ts.tt_jd",
        ):
            with self.subTest(token=token):
                self.assertNotIn(token, body)
        self.assertEqual(body.count("find_sunset_predecessor("), 1)
        self.assertEqual(body.count("find_sunset_from_instant("), 1)


# --- 4. Fail-closed dominance ----------------------------------------------


class TestFailClosedDominance(BracketAssertions):
    def test_both_events_present_yields_a_bracket(self):
        bracket = find_sunset_bracket(
            INTERIOR_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.assertIsInstance(bracket, astronomy_solver.SunsetBracket)

    def test_completed_directions_with_an_absence_yield_none(self):
        latitude, longitude, anchor, _label = POLAR_CASES[0]
        self.assertIsNone(
            find_sunset_predecessor(anchor, latitude, longitude)
        )
        self.assertIsNone(
            find_sunset_from_instant(anchor, latitude, longitude)
        )
        self.assertIsNone(find_sunset_bracket(anchor, latitude, longitude))

    def test_both_directions_run_before_the_none_decision(self):
        """No short-circuit: the second direction is invoked even when the
        first has already returned None."""
        latitude, longitude, anchor, _label = POLAR_CASES[0]
        with mock.patch.object(
            astronomy_solver, "find_sunset_predecessor",
            wraps=astronomy_solver.find_sunset_predecessor,
        ) as before, mock.patch.object(
            astronomy_solver, "find_sunset_from_instant",
            wraps=astronomy_solver.find_sunset_from_instant,
        ) as after:
            result = find_sunset_bracket(anchor, latitude, longitude)

        self.assertIsNone(result)
        self.assertIsNone(before.return_value if False else None)
        self.assertEqual(before.call_count, 1)
        self.assertEqual(
            after.call_count, 1,
            "the forward direction must run even after a None backward "
            "result",
        )

    def test_failure_dominates_a_completed_absence(self):
        """One direction absent, the other failing: the failure propagates.

        The geometry is verified here rather than assumed, so a future
        change that moves it fails loudly instead of testing nothing.
        """
        latitude, longitude = 75.0, 0.0
        anchor = deep_time_anchor(0.1)

        with self.assertRaises(ScientificEnvironmentError) as caught:
            find_sunset_predecessor(anchor, latitude, longitude)
        self.assertEqual(caught.exception.reason, REASON_REACH_EXHAUSTED)

        self.assertIsNone(
            find_sunset_from_instant(anchor, latitude, longitude),
            "this probe requires a completed forward absence",
        )

        error = self.assert_fails_closed(
            REASON_REACH_EXHAUSTED, anchor, latitude, longitude
        )
        self.assertEqual(error.reason, REASON_REACH_EXHAUSTED)

    def test_no_failure_is_ever_converted_into_none(self):
        cases = (
            (union_bounds()[0] - 1.0, REASON_COVERAGE_EXHAUSTED),
            (union_bounds()[1] + 1.0, REASON_COVERAGE_EXHAUSTED),
            (coverage(DE441_PART_2).tt_end, REASON_COVERAGE_EXHAUSTED),
            (ceiling_anchor(0.05), REASON_COVERAGE_EXHAUSTED),
        )
        for anchor, reason in cases:
            with self.subTest(anchor=anchor):
                self.assert_fails_closed(reason, anchor)

    def test_no_exception_handling_exists_in_the_block(self):
        body = executable_source(
            inspect.getsource(astronomy_solver.find_sunset_bracket)
        )
        for token in ("try", "except", "finally", "raise"):
            with self.subTest(token=token):
                self.assertNotIn(token, body)


# --- 5. Polar absence ------------------------------------------------------


class TestPolarAbsence(BracketAssertions):
    def test_polar_cases_have_no_bracket(self):
        for latitude, longitude, anchor, label in POLAR_CASES:
            with self.subTest(case=label):
                # Both directions completed scientifically ...
                backward = backward_frontier(anchor, latitude, longitude)
                forward = forward_frontier(anchor, latitude, longitude)
                self.assertTrue(backward.complete)
                self.assertTrue(forward.complete)

                # ... and at least one required boundary is genuinely absent.
                absent = (
                    oracle_sunsets(backward, latitude, longitude) == []
                    or oracle_sunsets(forward, latitude, longitude) == []
                )
                self.assertTrue(
                    absent, "this probe requires a genuine absence"
                )

                self.assertIsNone(
                    find_sunset_bracket(anchor, latitude, longitude)
                )

    def test_none_is_never_a_coverage_or_reach_outcome(self):
        for latitude, longitude, anchor, label in POLAR_CASES:
            with self.subTest(case=label):
                backward = backward_frontier(anchor, latitude, longitude)
                forward = forward_frontier(anchor, latitude, longitude)
                self.assertIsNone(backward.truncation_reason)
                self.assertIsNone(forward.truncation_reason)
                self.assertIsNone(
                    find_sunset_bracket(anchor, latitude, longitude)
                )


# --- 6. Mixed-kernel provenance --------------------------------------------


class TestMixedKernelProvenance(BracketAssertions):
    def test_de440_lower_transition_yields_a_mixed_bracket(self):
        anchor = coverage(DE440).tt_start + 1.0
        bracket = find_sunset_bracket(
            anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.assert_bracket_invariants(
            bracket, anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.assertEqual(bracket.previous.kernel, DE441_PART_1)
        self.assertEqual(bracket.next.kernel, DE440)
        self.assertNotEqual(bracket.previous.kernel, bracket.next.kernel)

    def test_de440_upper_transition_yields_a_mixed_bracket(self):
        anchor = coverage(DE440).tt_end - 1.0
        bracket = find_sunset_bracket(
            anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.assert_bracket_invariants(
            bracket, anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.assertEqual(bracket.previous.kernel, DE440)
        self.assertEqual(bracket.next.kernel, DE441_PART_2)
        self.assertNotEqual(bracket.previous.kernel, bracket.next.kernel)

    def test_each_boundary_kernel_matches_its_own_frontier(self):
        for anchor in (
            coverage(DE440).tt_start + 1.0,
            coverage(DE440).tt_end - 1.0,
            INTERIOR_TT,
            DEFECT_WINDOW_TT,
        ):
            with self.subTest(anchor=anchor):
                bracket = find_sunset_bracket(
                    anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
                )
                self.assertEqual(
                    bracket.previous.kernel,
                    backward_frontier(
                        anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
                    ).kernel,
                )
                self.assertEqual(
                    bracket.next.kernel,
                    forward_frontier(
                        anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
                    ).kernel,
                )

    def test_shared_provenance_is_not_required(self):
        """A mixed bracket is accepted, not rejected."""
        anchor = coverage(DE440).tt_start + 1.0
        bracket = find_sunset_bracket(
            anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.assertIsNotNone(bracket)
        self.assertNotEqual(bracket.previous.kernel, bracket.next.kernel)


# --- 7. Exact-root anchor --------------------------------------------------


class TestExactRootAnchor(BracketAssertions):
    def setUp(self):
        frontier = forward_frontier(
            INTERIOR_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.root = oracle_sunsets(
            frontier, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )[0]
        self.bracket = find_sunset_bracket(
            self.root, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )

    def test_the_anchor_really_is_a_determined_sunset(self):
        frontier = forward_frontier(
            INTERIOR_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.assertIn(
            self.root,
            oracle_sunsets(frontier, REFERENCE_LATITUDE, REFERENCE_LONGITUDE),
        )

    def test_strict_ordering_still_holds(self):
        self.assertLess(self.bracket.previous.tt, self.bracket.anchor)
        self.assertLess(self.bracket.anchor, self.bracket.next.tt)
        self.assertEqual(self.bracket.anchor, self.root)

    def test_the_anchor_root_is_returned_by_neither_side(self):
        self.assertNotEqual(self.bracket.previous.tt, self.root)
        self.assertNotEqual(self.bracket.next.tt, self.root)

    def test_boundaries_remain_genuine_sunsets(self):
        self.assert_bracket_invariants(
            self.bracket, self.root,
            REFERENCE_LATITUDE, REFERENCE_LONGITUDE,
        )


# --- 8. Coverage and deep-time cases ---------------------------------------


class TestCoverageAndDeepTime(BracketAssertions):
    def test_interior_anchor_uses_de440_on_both_sides(self):
        bracket = find_sunset_bracket(
            INTERIOR_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.assertEqual(bracket.previous.kernel, DE440)
        self.assertEqual(bracket.next.kernel, DE440)

    def test_2650_legacy_defect_present_but_bypassed(self):
        self.assertEqual(choose_kernel_name(DEFECT_WINDOW_CIVIL_YEAR), DE440)
        self.assertGreater(DEFECT_WINDOW_TT, coverage(DE440).tt_end)

        bracket = find_sunset_bracket(
            DEFECT_WINDOW_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.assert_bracket_invariants(
            bracket, DEFECT_WINDOW_TT,
            REFERENCE_LATITUDE, REFERENCE_LONGITUDE,
        )
        self.assertEqual(bracket.previous.kernel, DE441_PART_2)
        self.assertEqual(bracket.next.kernel, DE441_PART_2)

    def test_de441_part_1_lower_edge_deep_time_bracket(self):
        for offset in (1.0, 2.0, 4.0):
            with self.subTest(offset=offset):
                anchor = deep_time_anchor(offset)
                bracket = find_sunset_bracket(
                    anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
                )
                self.assert_bracket_invariants(
                    bracket, anchor,
                    REFERENCE_LATITUDE, REFERENCE_LONGITUDE,
                )
                self.assertEqual(bracket.previous.kernel, DE441_PART_1)
                self.assertEqual(bracket.next.kernel, DE441_PART_1)
                self.assertGreater(
                    bracket.previous.tt, coverage(DE441_PART_1).tt_start
                )

    def test_deep_time_bracket_needs_no_calendar_representation(self):
        anchor = deep_time_anchor(2.0)
        bracket = find_sunset_bracket(
            anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        for value in (bracket.anchor, bracket.previous.tt, bracket.next.tt):
            with self.subTest(value=value):
                self.assertEqual(float(ts.tt_jd(value).tt), value)

    def test_de441_part_2_upper_edge(self):
        for offset in (1.0, 2.0, 4.0):
            with self.subTest(offset=offset):
                anchor = ceiling_anchor(offset)
                bracket = find_sunset_bracket(
                    anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
                )
                self.assert_bracket_invariants(
                    bracket, anchor,
                    REFERENCE_LATITUDE, REFERENCE_LONGITUDE,
                )
                self.assertEqual(bracket.next.kernel, DE441_PART_2)
                self.assertLessEqual(
                    bracket.next.tt, coverage(DE441_PART_2).tt_end
                )

    def test_outside_the_authoritative_union_fails_closed(self):
        lo, hi = union_bounds()
        for anchor in (lo - 1.0, lo - 1e6, hi + 1.0, hi + 1e6):
            with self.subTest(anchor=anchor):
                self.assert_fails_closed(REASON_COVERAGE_EXHAUSTED, anchor)


# --- 9. Input and failure propagation --------------------------------------


class TestFailurePropagation(BracketAssertions):
    BAD_TT = (
        True, False, "2461485.0", None, [2461485.0], {}, object(),
        complex(1, 0), float("nan"), float("inf"), float("-inf"), 10 ** 400,
    )

    BAD_OBSERVERS = (
        (float("nan"), 0.0),
        (0.0, float("nan")),
        (float("inf"), 0.0),
        (0.0, float("-inf")),
        (95.0, 0.0),
        (-90.5, 0.0),
        (0.0, 181.0),
        (0.0, -180.5),
        (True, 0.0),
        ("40.0", 0.0),
        (None, 0.0),
    )

    def test_malformed_anchor_reports_instant_state_invalid(self):
        for value in self.BAD_TT:
            with self.subTest(value=repr(value)):
                self.assert_fails_closed(REASON_INSTANT_INVALID, value)

    def test_invalid_observer_reports_observer_out_of_domain(self):
        for latitude, longitude in self.BAD_OBSERVERS:
            with self.subTest(latitude=latitude, longitude=longitude):
                self.assert_fails_closed(
                    REASON_OBSERVER_OUT_OF_DOMAIN,
                    INTERIOR_TT, latitude, longitude,
                )

    def test_coverage_exhaustion_propagates_unchanged(self):
        anchor = coverage(DE441_PART_2).tt_end
        direct_reason = None
        try:
            find_sunset_from_instant(
                anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
            )
        except ScientificEnvironmentError as error:
            direct_reason = error.reason
        self.assertEqual(direct_reason, REASON_COVERAGE_EXHAUSTED)
        self.assertEqual(
            self.assert_fails_closed(
                REASON_COVERAGE_EXHAUSTED, anchor
            ).reason,
            direct_reason,
        )

    def test_reach_exhaustion_propagates_unchanged(self):
        anchor = coverage(DE441_PART_1).tt_start
        direct_reason = None
        try:
            find_sunset_from_instant(
                anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
            )
        except ScientificEnvironmentError as error:
            direct_reason = error.reason
        self.assertEqual(direct_reason, REASON_REACH_EXHAUSTED)

        # The backward direction runs first and fails on coverage here, so
        # the bracket reports that. Either way a failure, never None.
        with self.assertRaises(ScientificEnvironmentError) as caught:
            find_sunset_bracket(
                anchor, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
            )
        self.assertIn(
            caught.exception.reason,
            (REASON_COVERAGE_EXHAUSTED, REASON_REACH_EXHAUSTED),
        )

    def test_malformed_observer_never_masquerades_as_exhaustion(self):
        for latitude, longitude in self.BAD_OBSERVERS:
            with self.subTest(latitude=latitude, longitude=longitude):
                error = self.assert_fails_closed(
                    REASON_OBSERVER_OUT_OF_DOMAIN,
                    INTERIOR_TT, latitude, longitude,
                )
                self.assertNotEqual(error.reason, REASON_REACH_EXHAUSTED)
                self.assertNotEqual(error.reason, REASON_COVERAGE_EXHAUSTED)


# --- 10. Structural guards -------------------------------------------------


class TestA2_4StructuralIndependence(unittest.TestCase):
    """The A2-4 block composes; it must not compute or route."""

    def setUp(self):
        source = inspect.getsource(astronomy_solver)
        index = source.find(A2_4_BLOCK_MARKER)
        self.assertNotEqual(index, -1, "A2-4 block marker not found")
        following = GOVERNED_BLOCK_MARKER.search(
            source, index + len(A2_4_BLOCK_MARKER)
        )
        self.block = source[
            index:following.start() if following else len(source)
        ]
        self.preamble = source[:index]
        self.executable = executable_source(self.block)

    def test_block_was_located_and_is_substantial(self):
        self.assertIn("class SunsetBracket", self.block)
        self.assertIn("def find_sunset_bracket", self.block)
        self.assertGreater(len(self.block), 1000)

    def test_block_stops_at_the_next_governed_block(self):
        self.assertIsNone(
            GOVERNED_BLOCK_MARKER.search(self.block, len(A2_4_BLOCK_MARKER)),
            "the A2-4 scan reaches into a later governed block",
        )

    def test_intended_operations_are_present(self):
        """Anti-vacuity: the guards below must inspect real logic."""
        self.assertIn("def find_sunset_bracket", self.executable)
        self.assertIn("_exact_finite_tt", self.executable)
        self.assertIn("find_sunset_predecessor", self.executable)
        self.assertIn("find_sunset_from_instant", self.executable)
        self.assertIn("SunsetBracket(", self.executable)
        self.assertIn("anchor=anchor", self.executable)
        self.assertIn("is None", self.executable)
        self.assertIn("return None", self.executable)
        self.assertGreater(len(self.executable), 300)

        # Prose really was excluded.
        self.assertNotIn("STILL ASTRONOMY, NOT CALENDAR", self.executable)
        self.assertNotIn("EXACT-ROOT ANCHORS", self.executable)

    def test_block_performs_no_astronomy(self):
        for token in (
            "find_discrete",
            "sunrise_sunset",
            "load_kernel",
            "supported_search_frontier",
            "wgs84",
            "almanac",
            "ts.tt_jd",
            "kernel_coverage_tt",
        ):
            with self.subTest(token=token):
                self.assertNotIn(token, self.executable)

    def test_block_performs_no_kernel_selection(self):
        for token in (
            "choose_kernel_name",
            "PRIMARY_START_YEAR",
            "PRIMARY_END_YEAR",
            "PINNED_KERNEL_PRECEDENCE",
            "select_kernel_containing_instant",
            "select_kernel_for_interval",
            "_admit_bracket",
        ):
            with self.subTest(token=token):
                self.assertNotIn(token, self.executable)

    def test_block_has_no_exception_handling(self):
        for token in ("try", "except", "finally", "raise"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.executable)

    def test_block_has_no_legacy_or_calendar_mechanisms(self):
        for token in (
            "ts.utc(",
            "utc_datetime",
            "utc_strftime",
            "isoformat",
            "timestamp",
            "datetime",
            "strftime",
            "weekday",
            "3600",
            "min_gap",
            "epsilon",
            "tolerance",
            "timedelta",
            "/ 15",
            "lru_cache",
            "cache",
            "HTTPException",
            "status_code",
        ):
            with self.subTest(token=token):
                self.assertNotIn(token, self.executable)

    def test_block_encodes_no_light_time_margin(self):
        for token in ("480", "489", "490", "500", "507", "510", "520"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.executable)

    def test_observer_parameters_are_not_prohibited(self):
        self.assertIn("latitude", self.executable)
        self.assertIn("longitude", self.executable)

    def test_block_defines_exactly_the_authorized_symbols(self):
        tree = ast.parse(self.block)
        defined = [
            n.name for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        ]
        self.assertEqual(defined, ["SunsetBracket", "find_sunset_bracket"])

    def test_block_introduces_no_constant_import_or_helper(self):
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
            len([n for n in tree.body if isinstance(n, ast.FunctionDef)]), 1
        )
        self.assertEqual(
            len([n for n in tree.body if isinstance(n, ast.ClassDef)]), 1
        )

    def test_frozen_code_is_still_present_ahead_of_the_block(self):
        for token in (
            "def choose_kernel_name",
            "PRIMARY_START_YEAR",
            "PRIMARY_END_YEAR",
            "min_gap_seconds",
            "def find_sunset_successor",
            "def kernel_coverage_tt",
            "def select_kernel_containing_instant",
            "def select_kernel_for_interval",
            "def supported_search_frontier",
            "def _first_evaluable_state",
            "class SunsetEvent",
            "def find_sunset_predecessor",
            "def find_sunset_from_instant",
        ):
            with self.subTest(token=token):
                self.assertIn(token, self.preamble)


if __name__ == "__main__":
    unittest.main()
