"""A1b verification for find_sunset_successor in astronomy_solver.py.

Test framework: standard-library unittest.
Scientific execution uses the repository's installed Skyfield runtime and
real Authority ephemeris data.

INDEPENDENT ORACLE DISCIPLINE

Golden instants, golden Terrestrial Time values, observer coordinates,
kernel names and fail-closed reason codes are declared as test-local
literals. They are deliberately NOT imported from astronomy_solver:
importing them would make the tests agree with a defective constant instead
of detecting it.

The sunset predicate used as the oracle is built here directly from
skyfield.almanac, not obtained from the function under test, so a defective
predicate cannot confirm itself.

The shared timescale and the kernel loader ARE taken from astronomy_solver.
They are repository infrastructure rather than the unit under test, and
duplicating them would load a second copy of a 119 MB kernel and a second
Delta-T table for no verification benefit.

SCOPE

Real Authority astronomy. The real de440.bsp kernel is loaded and real
crossings are computed; nothing is mocked, stubbed or approximated.

Observer compatibility is NOT tested here. Binding a continuation witness
to an observer, and rejecting a nearby-but-different observer, is owned by
A1a and verified in tests/test_sunset_cursor.py.

This file asserts nothing about HTTP behavior, about any /sunset-successor
route, about cross-kernel routing boundaries, or about general astronomical
correctness beyond the instants exercised below.
"""

import ast
import inspect
import unittest
from datetime import datetime, timezone

from skyfield import almanac
from skyfield.api import wgs84

import astronomy_solver
from astronomy_solver import (
    SunsetSuccessorError,
    find_next_sunset_after_utc,
    find_sunset_successor,
    find_sunset_utc,
    load_kernel,
    ts,
)
from scientific_environment import ScientificEnvironmentError

# --- Test-local oracles ----------------------------------------------------

REFERENCE_LATITUDE = 40.7406
REFERENCE_LONGITUDE = -73.9586

REFERENCE_KERNEL = "de440.bsp"

# The sunset that closes 2027-03-20 for the reference observer, and the
# exact binary64 Terrestrial Time solver state it was found at.
CURSOR_SUNSET_UTC = "2027-03-20T23:07:45.858897+00:00"
CURSOR_TT = 2461485.4645259595

# The next distinct crossing, continued from CURSOR_TT.
SUCCESSOR_SUNSET_UTC = "2027-03-21T23:08:49.679673+00:00"
SUCCESSOR_TT = 2461486.465264626

# Two further crossings, used to prove that continuation never stalls on a
# root it has already returned.
CHAIN_SUNSET_UTC = (
    "2027-03-21T23:08:49.679673+00:00",
    "2027-03-22T23:09:53.362048+00:00",
    "2027-03-23T23:10:56.922998+00:00",
)

ONE_SECOND_IN_DAYS = 1.0 / 86400.0

# Longyearbyen, Svalbard, during polar night: the Sun is down, and it does
# not rise or set anywhere inside the search horizon.
POLAR_LATITUDE = 78.2232
POLAR_LONGITUDE = 15.6267
POLAR_TT = 2461754.500800741

REASON_CURSOR_STATE_INVALID = "CURSOR_STATE_INVALID"
REASON_CURSOR_NOT_POST_TRANSITION = "CURSOR_NOT_POST_TRANSITION"

SEARCH_SPAN_DAYS = 3.0


def reference_predicate(latitude=REFERENCE_LATITUDE, longitude=REFERENCE_LONGITUDE):
    """Build the sunset predicate independently of the code under test."""
    return almanac.sunrise_sunset(
        load_kernel(REFERENCE_KERNEL), wgs84.latlon(latitude, longitude)
    )


def sun_is_up_at(tt, latitude=REFERENCE_LATITUDE, longitude=REFERENCE_LONGITUDE):
    """Evaluate the independent predicate at an exact TT state."""
    return bool(reference_predicate(latitude, longitude)(ts.tt_jd(tt)))


def independent_crossings(tt, span_days=SEARCH_SPAN_DAYS,
                          latitude=REFERENCE_LATITUDE,
                          longitude=REFERENCE_LONGITUDE):
    """Return (tt, sun_is_up) for every crossing in an independent search."""
    times, events = almanac.find_discrete(
        ts.tt_jd(tt),
        ts.tt_jd(tt + span_days),
        reference_predicate(latitude, longitude),
    )
    return [(float(t.tt), bool(event)) for t, event in zip(times, events)]


def successor_body_source():
    """Return find_sunset_successor's executable body, docstring excluded.

    The prohibition governs executable logic, not prose: the docstring
    legitimately names the representations the search must never use, and
    matching against it would read a promise as a violation.
    """
    source = inspect.getsource(astronomy_solver.find_sunset_successor)
    lines = source.splitlines(keepends=True)

    function = ast.parse(source).body[0]
    first = function.body[0]

    if (isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)):
        del lines[first.lineno - 1:first.end_lineno]

    return "".join(lines)


# --- 1. Post-transition validation and fail-closed input -------------------


class TestPostTransitionValidation(unittest.TestCase):

    def assert_fails_closed(self, tt_value, reason, latitude=REFERENCE_LATITUDE,
                            longitude=REFERENCE_LONGITUDE):
        with self.assertRaises(SunsetSuccessorError) as caught:
            find_sunset_successor(tt_value, latitude, longitude)

        self.assertEqual(caught.exception.reason, reason)
        return caught.exception

    def test_known_cursor_is_post_transition(self):
        # The published cursor state must sit on the sun-down side of the
        # predicate, confirmed by the independent oracle rather than by the
        # function under test.
        self.assertFalse(sun_is_up_at(CURSOR_TT))

        # One second earlier the Sun is still up, so the state really is
        # the far side of a crossing and not merely night in general.
        self.assertTrue(sun_is_up_at(CURSOR_TT - ONE_SECOND_IN_DAYS))

        # A state proven post-transition is accepted for continuation.
        self.assertIsNotNone(
            find_sunset_successor(
                CURSOR_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
            )
        )

    def test_pre_transition_cursor_fails_closed(self):
        pre_transition_tt = CURSOR_TT - ONE_SECOND_IN_DAYS

        self.assertTrue(sun_is_up_at(pre_transition_tt))

        self.assert_fails_closed(
            pre_transition_tt, REASON_CURSOR_NOT_POST_TRANSITION
        )

    def test_sun_up_cursor_fails_closed(self):
        daylight_tt = float(ts.utc(2027, 3, 20, 17, 0, 0).tt)

        self.assertTrue(sun_is_up_at(daylight_tt))

        self.assert_fails_closed(
            daylight_tt, REASON_CURSOR_NOT_POST_TRANSITION
        )

    def test_non_finite_state_fails_closed(self):
        for tt_value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(tt_value=tt_value):
                self.assert_fails_closed(tt_value, REASON_CURSOR_STATE_INVALID)

        for tt_value in (None, "2461485.4645259595", True, complex(1, 0), []):
            with self.subTest(tt_value=tt_value):
                self.assert_fails_closed(tt_value, REASON_CURSOR_STATE_INVALID)

    def test_huge_numeric_state_fails_closed(self):
        huge = 10 ** 400

        with self.assertRaises(SunsetSuccessorError) as caught:
            find_sunset_successor(huge, REFERENCE_LATITUDE, REFERENCE_LONGITUDE)

        error = caught.exception

        # The declared fail-closed reason, not an incidental classification.
        self.assertEqual(error.reason, REASON_CURSOR_STATE_INVALID)

        # The raw OverflowError must not escape the boundary, and it must
        # not be silently discarded either: it is preserved as the cause.
        self.assertIsInstance(error.__cause__, OverflowError)

        # Independent confirmation that this value really does overflow the
        # float conversion, so the test is exercising the intended path
        # rather than passing for an unrelated reason.
        with self.assertRaises(OverflowError):
            float(huge)

    def test_error_is_scientific_environment_error(self):
        self.assertTrue(
            issubclass(SunsetSuccessorError, ScientificEnvironmentError)
        )

        error = self.assert_fails_closed(
            float("nan"), REASON_CURSOR_STATE_INVALID
        )
        self.assertIsInstance(error, ScientificEnvironmentError)


# --- 2. Successor continuation from an exact TT state ----------------------


class TestSuccessorFromExactState(unittest.TestCase):

    def setUp(self):
        self.successor = find_sunset_successor(
            CURSOR_TT, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )

    def test_successor_utc_matches_golden(self):
        self.assertEqual(self.successor.utc.isoformat(), SUCCESSOR_SUNSET_UTC)
        self.assertEqual(self.successor.utc.tzinfo, timezone.utc)

    def test_successor_tt_matches_golden(self):
        self.assertEqual(self.successor.tt, SUCCESSOR_TT)

    def test_successor_is_distinct_crossing(self):
        self.assertNotEqual(self.successor.tt, CURSOR_TT)
        self.assertGreater(self.successor.tt, CURSOR_TT)
        self.assertNotEqual(self.successor.utc.isoformat(), CURSOR_SUNSET_UTC)

    def test_no_same_root_rediscovery(self):
        # The originating root is the sunset that closes 2027-03-20 for
        # this observer, produced by the existing solver.
        origin_utc, _kernel = find_sunset_utc(
            datetime(2027, 3, 20, tzinfo=timezone.utc),
            REFERENCE_LATITUDE,
            REFERENCE_LONGITUDE,
        )
        self.assertEqual(origin_utc.isoformat(), CURSOR_SUNSET_UTC)
        self.assertNotEqual(self.successor.utc, origin_utc)

        # An independent search over the same bracket never reports the
        # continuation state itself as a crossing: starting from a proven
        # sun-down sample, the originating root presents no sign change.
        crossings = independent_crossings(CURSOR_TT)
        self.assertNotIn(CURSOR_TT, [tt for tt, _up in crossings])

        first_sunset = next(tt for tt, up in crossings if not up)
        self.assertEqual(first_sunset, self.successor.tt)

    def test_successor_state_is_post_transition(self):
        # The returned continuation state is itself a valid continuation
        # state, so continuation composes without special-casing.
        self.assertFalse(sun_is_up_at(self.successor.tt))
        self.assertTrue(
            sun_is_up_at(self.successor.tt - ONE_SECOND_IN_DAYS)
        )

    def test_supplied_state_round_trips(self):
        # The value the solver searches from is bit-identical to the value
        # supplied: reconstruction through ts.tt_jd is exact.
        self.assertEqual(float(ts.tt_jd(CURSOR_TT).tt), CURSOR_TT)

    def test_returned_state_round_trips(self):
        self.assertEqual(
            float(ts.tt_jd(self.successor.tt).tt), self.successor.tt
        )

    def test_kernel_provenance(self):
        self.assertEqual(self.successor.kernel, REFERENCE_KERNEL)
        self.assertEqual(
            self.successor._fields, ("utc", "tt", "kernel")
        )


# --- 3. Repeated continuation ----------------------------------------------


class TestSuccessorChain(unittest.TestCase):

    def test_three_step_chain_is_strictly_forward(self):
        tt = CURSOR_TT
        seen = []

        for expected_utc in CHAIN_SUNSET_UTC:
            successor = find_sunset_successor(
                tt, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
            )

            self.assertIsNotNone(successor)
            self.assertEqual(successor.utc.isoformat(), expected_utc)
            self.assertGreater(successor.tt, tt)

            seen.append(successor.tt)
            tt = successor.tt

        self.assertEqual(len(set(seen)), len(CHAIN_SUNSET_UTC))
        self.assertEqual(seen, sorted(seen))


# --- 4. Absence of any event-identity tolerance ----------------------------


class TestNoArbitraryTolerance(unittest.TestCase):

    def test_no_gap_constant_in_successor_path(self):
        body = successor_body_source()

        # The oracle must not pass by inspecting nothing.
        self.assertIn("return SunsetSuccessor(", body)
        self.assertNotIn("Terrestrial Time Julian Date", body)

        for forbidden in (
            "3600",
            "min_gap",
            "timestamp(",
            "fromtimestamp",
            "seconds=",
            "milliseconds",
            "EPSILON",
            "abs(",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, body)

    def test_search_bracket_is_built_from_tt(self):
        body = successor_body_source()

        # The bracket is constructed from exact Terrestrial Time.
        self.assertIn("ts.tt_jd(tt)", body)
        self.assertIn("ts.tt_jd(tt + SUCCESSOR_SEARCH_SPAN_DAYS)", body)

        # It is never routed through a calendar or offset representation.
        for forbidden in ("ts.from_datetime", "timedelta", "isoformat"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, body)

        # utc_datetime appears exactly once: producing the returned answer,
        # never the search state.
        self.assertEqual(body.count("utc_datetime()"), 1)


# --- 5. Existing golden behavior is untouched ------------------------------


class TestSunsetAfterGoldenPreserved(unittest.TestCase):

    def test_guard_still_present(self):
        source = inspect.getsource(find_next_sunset_after_utc)

        self.assertIn("min_gap_seconds = 3600", source)
        self.assertIn("after_timestamp + min_gap_seconds", source)

    def test_sunset_after_from_cursor_instant(self):
        sunset_dt, kernel = find_next_sunset_after_utc(
            datetime(2027, 3, 20, 23, 7, 45, 858897, tzinfo=timezone.utc),
            REFERENCE_LATITUDE,
            REFERENCE_LONGITUDE,
        )

        self.assertEqual(sunset_dt.isoformat(), SUCCESSOR_SUNSET_UTC)
        self.assertEqual(kernel, REFERENCE_KERNEL)

    def test_sunset_after_from_midday(self):
        sunset_dt, kernel = find_next_sunset_after_utc(
            datetime(2027, 3, 20, 12, 0, 0, tzinfo=timezone.utc),
            REFERENCE_LATITUDE,
            REFERENCE_LONGITUDE,
        )

        self.assertEqual(sunset_dt.isoformat(), CURSOR_SUNSET_UTC)
        self.assertEqual(kernel, REFERENCE_KERNEL)

    def test_sunset_golden_unchanged(self):
        sunset_dt, kernel = find_sunset_utc(
            datetime(2027, 3, 20, tzinfo=timezone.utc),
            REFERENCE_LATITUDE,
            REFERENCE_LONGITUDE,
        )

        self.assertEqual(sunset_dt.isoformat(), CURSOR_SUNSET_UTC)
        self.assertEqual(kernel, REFERENCE_KERNEL)


# --- 6. Polar absence ------------------------------------------------------


class TestPolarNoSuccessor(unittest.TestCase):

    def test_polar_state_is_sun_down(self):
        # Isolates the branch under test: this state passes post-transition
        # validation, so a None result means "no successor found", not
        # "rejected before searching".
        self.assertFalse(
            sun_is_up_at(POLAR_TT, POLAR_LATITUDE, POLAR_LONGITUDE)
        )

        self.assertEqual(
            independent_crossings(
                POLAR_TT,
                latitude=POLAR_LATITUDE,
                longitude=POLAR_LONGITUDE,
            ),
            [],
        )

    def test_polar_yields_no_successor(self):
        successor = find_sunset_successor(
            POLAR_TT, POLAR_LATITUDE, POLAR_LONGITUDE
        )

        # No successor is reported rather than a fabricated crossing.
        self.assertIsNone(successor)


if __name__ == "__main__":
    unittest.main(verbosity=2)
