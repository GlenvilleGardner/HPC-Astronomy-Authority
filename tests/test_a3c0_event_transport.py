"""A3c-0 verification for astronomical_event_transport.py.

Covers utc_reference, project_exact_instant, project_astronomical_event and
reason_detail.

Test framework: standard-library unittest.
Scientific execution uses the repository's certified Skyfield runtime and the
real pinned Authority ephemeris artifacts.

INVOCATION
----------
Run from the repository root with the repository interpreter:

    .\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v

INDEPENDENT ORACLE DISCIPLINE

Expected key sets, the governed event kinds, the pinned artifact filenames
and the measured calendar-representability boundaries are declared here as
test-local literals. The bit-level invariant is checked with struct.pack
directly rather than by calling the codec under test to confirm itself.

EVIDENCE DISCIPLINE

MEASURED - the calendar-representability boundaries below were located by
bisecting the real certified timescale. They are properties of the certified
runtime and of Python's datetime, not timeless astronomical facts.
"""

import ast
import inspect
import math
import struct
import unittest
from unittest import mock

import astronomical_event_transport
import astronomy_solver
from astronomical_event_transport import (
    project_astronomical_event,
    project_exact_instant,
    reason_detail,
    utc_reference,
)
from astronomy_solver import (
    find_solar_longitude_event_after,
    find_sunset_bracket,
    ts,
)

# --- Test-local oracles ----------------------------------------------------

K000 = "SOLAR_LONGITUDE_000"
K090 = "SOLAR_LONGITUDE_090"
K180 = "SOLAR_LONGITUDE_180"
K270 = "SOLAR_LONGITUDE_270"
GOVERNED_KINDS = (K000, K090, K180, K270)

DE440 = "de440.bsp"
DE441_PART_1 = "de441_part-1.bsp"
DE441_PART_2 = "de441_part-2.bsp"
PINNED_ARTIFACTS = (DE440, DE441_PART_1, DE441_PART_2)

EXPECTED_INSTANT_KEYS = ("ttBits", "tt", "utc")
EXPECTED_EVENT_KEYS = ("ttBits", "tt", "utc", "kind", "kernel")
EXPECTED_REASON_KEYS = ("reason", "message")

# MEASURED: the closed span over which a calendar datetime can be formed.
UTC_REPRESENTABLE_LOW_TT = 1721425.500489      # 0001-01-01T00:00:00.047837
UTC_REPRESENTABLE_HIGH_TT = 5373484.5008       # 9999-12-31T23:59:59.942642
UTC_UNREPRESENTABLE_LOW_TT = 1721425.500488    # year 0 is out of range
UTC_UNREPRESENTABLE_HIGH_TT = 5373484.500801   # year 10000 is out of range

MODERN_ANCHOR = 2460678.0
DEEP_TIME_ANCHOR = 700000.0
DEEPER_TIME_ANCHOR = -1000000.0
FAR_FUTURE_ANCHOR = 5000000.0

REFERENCE_LATITUDE = 40.7406
REFERENCE_LONGITUDE = -73.9


def bits(value):
    return struct.pack(">d", value)


def executable_source(text):
    """Return a module's executable logic, prose excluded.

    A governed module names the mechanisms it excludes, in order to state
    that it excludes them. Scanning raw source for a prohibited token would
    therefore convict a module of the very thing its documentation promises
    it does not do. The established repository mechanism strips docstrings
    through the AST and drops comment-only lines, so a structural guard
    inspects logic rather than prose.
    """
    lines = text.splitlines(keepends=True)
    drop = set()
    for node in ast.walk(ast.parse(text)):
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


class ProjectionAssertions(unittest.TestCase):
    def assert_instant_invariant(self, projection):
        """The one invariant a consumer may rely on."""
        from exact_time_transport import decode_tt_bits

        self.assertEqual(bits(decode_tt_bits(projection["ttBits"])),
                         bits(projection["tt"]))


# --- 1. Exact instant projection -------------------------------------------


class TestProjectExactInstant(ProjectionAssertions):
    SAMPLES = (
        2460754.8768300507, 2460666.3976841737, 700287.9296739673,
        -1000000.0, 5000100.674300475, 0.0, -0.0, 1.0,
    )

    def test_keys_are_exactly_the_expected_set_in_order(self):
        projection = project_exact_instant(MODERN_ANCHOR)
        self.assertEqual(tuple(projection), EXPECTED_INSTANT_KEYS)

    def test_bit_invariant_holds_for_every_sample(self):
        for value in self.SAMPLES:
            with self.subTest(value=value):
                self.assert_instant_invariant(project_exact_instant(value))

    def test_tt_is_bit_identical_to_the_supplied_state(self):
        for value in self.SAMPLES:
            with self.subTest(value=value):
                self.assertEqual(bits(project_exact_instant(value)["tt"]),
                                 bits(value))

    def test_tt_is_a_float_not_a_string(self):
        projection = project_exact_instant(MODERN_ANCHOR)
        self.assertIsInstance(projection["tt"], float)
        self.assertIsInstance(projection["ttBits"], str)

    def test_signed_zero_projects_distinctly(self):
        positive = project_exact_instant(0.0)
        negative = project_exact_instant(-0.0)
        self.assertNotEqual(positive["ttBits"], negative["ttBits"])
        self.assertEqual(bits(negative["tt"]), bits(-0.0))

    def test_utc_key_is_always_present(self):
        for value in self.SAMPLES + (UTC_UNREPRESENTABLE_LOW_TT,
                                     UTC_UNREPRESENTABLE_HIGH_TT):
            with self.subTest(value=value):
                self.assertIn("utc", project_exact_instant(value))

    def test_malformed_state_fails_closed_through_the_codec(self):
        from exact_time_transport import ExactTimeTransportError

        for value in (float("nan"), float("inf"), None, True, "x"):
            with self.subTest(value=repr(value)):
                with self.assertRaises(ExactTimeTransportError):
                    project_exact_instant(value)


# --- 2. UTC reference projection -------------------------------------------


class TestUtcReference(ProjectionAssertions):
    def test_representable_instants_render(self):
        rendered = utc_reference(2460754.8768300507)
        self.assertIsInstance(rendered, str)
        self.assertTrue(rendered.startswith("2025-03-20T"))
        self.assertTrue(rendered.endswith("+00:00"))

    def test_exactly_six_fractional_digits_always(self):
        for value in (2460754.8768300507, 2460666.3976841737,
                      5000100.674300475, 2460678.0, 2451545.0,
                      UTC_REPRESENTABLE_LOW_TT, UTC_REPRESENTABLE_HIGH_TT):
            with self.subTest(value=value):
                rendered = utc_reference(value)
                self.assertIsNotNone(rendered)
                fraction = rendered.split("+")[0].split(".")[1]
                self.assertEqual(len(fraction), 6, rendered)
                self.assertTrue(fraction.isdigit())

    def test_deep_time_is_none(self):
        for value in (700287.9296739673, -1000000.0, -3100015.5, 0.0):
            with self.subTest(value=value):
                self.assertIsNone(utc_reference(value))

    def test_far_future_beyond_datetime_is_none(self):
        for value in (8000016.5, 6000000.0, UTC_UNREPRESENTABLE_HIGH_TT):
            with self.subTest(value=value):
                self.assertIsNone(utc_reference(value))

    def test_the_measured_boundaries_are_still_where_they_were(self):
        self.assertIsNotNone(utc_reference(UTC_REPRESENTABLE_LOW_TT))
        self.assertIsNone(utc_reference(UTC_UNREPRESENTABLE_LOW_TT))
        self.assertIsNotNone(utc_reference(UTC_REPRESENTABLE_HIGH_TT))
        self.assertIsNone(utc_reference(UTC_UNREPRESENTABLE_HIGH_TT))

    def test_no_whole_second_or_millisecond_normalization(self):
        """The two losses that disqualified a rendering as identity."""
        source = executable_source(
            inspect.getsource(astronomical_event_transport)
        )
        for token in ("replace(microsecond", "round(", "timespec=\"seconds\"",
                      "timespec=\"milliseconds\"", "strftime", "//"):
            with self.subTest(token=token):
                self.assertNotIn(token, source)

    def test_only_value_error_becomes_none(self):
        """A non-ValueError is not a statement about representability."""
        for injected in (RuntimeError("not a representability statement"),
                         TypeError("nor is this"),
                         MemoryError()):
            with self.subTest(error=type(injected).__name__):
                with mock.patch.object(
                    astronomical_event_transport, "ts"
                ) as patched:
                    patched.tt_jd.side_effect = injected
                    with self.assertRaises(type(injected)):
                        utc_reference(MODERN_ANCHOR)

    def test_value_error_is_converted(self):
        with mock.patch.object(astronomical_event_transport, "ts") as patched:
            patched.tt_jd.side_effect = ValueError("year 0 is out of range")
            self.assertIsNone(utc_reference(MODERN_ANCHOR))

    def test_utc_never_feeds_back_into_the_exact_state(self):
        for value in (2460754.8768300507, 700287.9296739673):
            with self.subTest(value=value):
                projection = project_exact_instant(value)
                self.assertEqual(bits(projection["tt"]), bits(value))


# --- 3. AstronomicalEvent projection ---------------------------------------


class TestProjectAstronomicalEvent(ProjectionAssertions):
    @classmethod
    def setUpClass(cls):
        cls.events = []
        for anchor in (MODERN_ANCHOR, DEEP_TIME_ANCHOR, DEEPER_TIME_ANCHOR,
                       FAR_FUTURE_ANCHOR):
            for kind in GOVERNED_KINDS:
                cls.events.append(
                    find_solar_longitude_event_after(anchor, kind)
                )

    def test_keys_are_exactly_the_expected_set_in_order(self):
        projection = project_astronomical_event(self.events[0])
        self.assertEqual(tuple(projection), EXPECTED_EVENT_KEYS)

    def test_all_four_governed_kinds_project(self):
        seen = set()
        for event in self.events:
            projection = project_astronomical_event(event)
            self.assertEqual(projection["kind"], event.kind)
            seen.add(projection["kind"])
        self.assertEqual(seen, set(GOVERNED_KINDS))

    def test_all_three_artifacts_survive_projection(self):
        seen = {project_astronomical_event(e)["kernel"] for e in self.events}
        self.assertEqual(seen, set(PINNED_ARTIFACTS))

    def test_bit_invariant_holds_for_every_event(self):
        for event in self.events:
            with self.subTest(kind=event.kind, kernel=event.kernel):
                projection = project_astronomical_event(event)
                self.assert_instant_invariant(projection)
                self.assertEqual(bits(projection["tt"]), bits(event.tt))

    def test_deep_time_events_project_with_null_utc(self):
        deep = [e for e in self.events if e.kernel == DE441_PART_1]
        self.assertTrue(deep)
        for event in deep:
            with self.subTest(kind=event.kind):
                projection = project_astronomical_event(event)
                self.assertIn("utc", projection)
                self.assertIsNone(projection["utc"])
                self.assertIsNotNone(projection["ttBits"])

    def test_representable_events_project_with_a_reference(self):
        modern = [e for e in self.events if e.kernel == DE440]
        self.assertTrue(modern)
        for event in modern:
            with self.subTest(kind=event.kind):
                self.assertIsInstance(
                    project_astronomical_event(event)["utc"], str
                )

    def test_the_record_itself_is_not_modified(self):
        event = self.events[0]
        before = (event.tt, event.kind, event.kernel)
        project_astronomical_event(event)
        self.assertEqual((event.tt, event.kind, event.kernel), before)
        self.assertEqual(
            [f.name for f in __import__("dataclasses").fields(event)],
            ["tt", "kind", "kernel"],
        )

    def test_no_calendar_or_season_terminology_is_produced(self):
        for event in self.events:
            projection = project_astronomical_event(event)
            rendered = repr(projection)
            for token in ("Spring", "Summer", "Autumn", "Winter",
                          "equinox", "solstice"):
                with self.subTest(kind=event.kind, token=token):
                    self.assertNotIn(token, rendered)

    def test_sunset_events_need_no_separate_codec(self):
        """A2 events already project through the same exact primitive."""
        bracket = find_sunset_bracket(
            MODERN_ANCHOR, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.assertIsNotNone(bracket)
        for member in (bracket.previous, bracket.next):
            with self.subTest(kernel=member.kernel):
                projection = project_exact_instant(member.tt)
                self.assert_instant_invariant(projection)
                self.assertEqual(bits(projection["tt"]), bits(member.tt))


# --- 4. Reason detail ------------------------------------------------------


class TestReasonDetail(unittest.TestCase):
    def test_shape_is_exactly_reason_and_message(self):
        detail = reason_detail("EVENT_KIND_INVALID", "a message")
        self.assertEqual(tuple(detail), EXPECTED_REASON_KEYS)
        self.assertEqual(detail, {"reason": "EVENT_KIND_INVALID",
                                  "message": "a message"})

    def test_values_are_carried_verbatim(self):
        for reason in (astronomy_solver.REASON_EVENT_KIND_INVALID,
                       astronomy_solver.REASON_SOLAR_LONGITUDE_EVENT_UNRESOLVED,
                       astronomy_solver.REASON_EPHEMERIS_COVERAGE_EXHAUSTED,
                       astronomy_solver.REASON_EPHEMERIS_REACH_EXHAUSTED,
                       astronomy_solver.REASON_INSTANT_STATE_INVALID):
            with self.subTest(reason=reason):
                detail = reason_detail(reason, "text")
                self.assertEqual(detail["reason"], reason)
                self.assertEqual(detail["message"], "text")

    def test_no_http_status_is_carried(self):
        detail = reason_detail("X", "y")
        for token in ("status", "statusCode", "status_code", "code", "http"):
            with self.subTest(token=token):
                self.assertNotIn(token, detail)


# --- 5. Structural isolation -----------------------------------------------


class TestTransportIsolation(unittest.TestCase):
    def setUp(self):
        self.source = executable_source(
            inspect.getsource(astronomical_event_transport)
        )

    def test_module_decides_no_http_semantics(self):
        for token in ("HTTPException", "status_code", "fastapi", "starlette",
                      "@app.", "Response", "404", "400", "500"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.source)

    def test_module_performs_no_astronomy(self):
        for token in ("almanac", "find_discrete", "seasons", "load_kernel",
                      "geocentric_search_frontier", "wgs84", "latlon",
                      "observer", "latitude", "longitude", "kernel_coverage"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.source)

    def test_module_has_no_civil_or_gregorian_routing(self):
        for token in ("choose_kernel_name", "PRIMARY_START_YEAR",
                      "PRIMARY_END_YEAR", "ts.utc(", "civil", "timestamp()",
                      "timedelta", "fromisoformat"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.source)

    def test_module_defines_exactly_the_authorized_symbols(self):
        tree = ast.parse(self.source)
        self.assertEqual(
            [n.name for n in tree.body if isinstance(n, ast.FunctionDef)],
            ["utc_reference", "project_exact_instant",
             "project_astronomical_event", "project_sunset_event",
             "project_sunset_bracket", "reason_detail"],
        )
        self.assertEqual(
            [n.name for n in tree.body if isinstance(n, ast.ClassDef)], []
        )
        self.assertEqual(
            [n for n in tree.body if isinstance(n, ast.Assign)], []
        )

    def test_module_imports_only_what_it_needs(self):
        tree = ast.parse(self.source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
        self.assertEqual(imported, {"astronomy_solver", "exact_time_transport"})

    def test_it_reuses_the_canonical_codec_rather_than_a_second_one(self):
        self.assertIn("from exact_time_transport import", self.source)
        self.assertNotIn("struct", self.source)
        self.assertNotIn("hex()", self.source)

    def test_tt_is_derived_from_the_bits_not_from_the_argument(self):
        """Anti-drift: the two fields cannot disagree by construction."""
        self.assertIn("exact = decode_tt_bits(tt_bits)", self.source)
        self.assertIn('"tt": exact', self.source)


if __name__ == "__main__":
    unittest.main()
