"""A3a verification for the exact astronomical event record.

Covers AstronomicalEvent and the four governed solar-longitude identities
in astronomy_solver.py.

Test framework: standard-library unittest.
Scientific execution uses the repository's certified Skyfield runtime.

INDEPENDENT ORACLE DISCIPLINE

The identity literals, the record field names and the prohibited vocabulary
are declared as test-local literals. They are deliberately NOT imported from
astronomy_solver and then compared against themselves: importing a constant
to check that same constant would make the test agree with a defective value
instead of detecting it. Every expectation below is written out here and
compared against what the module actually publishes.

The shared timescale IS taken from astronomy_solver. It is certified
repository infrastructure rather than the unit under test, and building a
second one would load a second Delta-T table for no verification benefit.

SCOPE

A3a is substrate only: a record and four identities. Nothing is computed,
nothing is validated, nothing is served, and no existing caller is migrated.
This file therefore asserts nothing about event computation, season or
equinox behavior, kernel selection, transport, routes or HTTP, and nothing
about calendars.

WHAT IS DELIBERATELY NOT ENCODED

Python's datetime limits are not made part of the astronomical contract.
Where they appear it is only as supporting evidence that a transport field
could not have been mandatory, never as an assertion about astronomy.

Claims about Terrestrial Time being sufficient are scoped to the certified
runtime and timescale that this suite actually exercises.
"""

import ast
import dataclasses
import inspect
import math
import re
import unittest

import astronomy_solver
from astronomy_solver import AstronomicalEvent, ts

# --- Test-local oracles ----------------------------------------------------

EXPECTED_EVENT_FIELDS = ("tt", "kind", "kernel")

# The governed identities, written out here rather than imported.
EXPECTED_IDENTITY_NAMES = (
    "SOLAR_LONGITUDE_000",
    "SOLAR_LONGITUDE_090",
    "SOLAR_LONGITUDE_180",
    "SOLAR_LONGITUDE_270",
)

EXPECTED_IDENTITY_VALUES = {
    "SOLAR_LONGITUDE_000": "SOLAR_LONGITUDE_000",
    "SOLAR_LONGITUDE_090": "SOLAR_LONGITUDE_090",
    "SOLAR_LONGITUDE_180": "SOLAR_LONGITUDE_180",
    "SOLAR_LONGITUDE_270": "SOLAR_LONGITUDE_270",
}

HEMISPHERE_VOCABULARY = (
    "spring", "summer", "autumn", "fall", "winter",
    "vernal", "autumnal", "equinox", "solstice",
    "north", "south", "hemisphere",
)

GREGORIAN_MONTH_VOCABULARY = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)

# Representative pinned artifact name, used only as a provenance string.
DE440 = "de440.bsp"
DE441_PART_1 = "de441_part-1.bsp"
DE441_PART_2 = "de441_part-2.bsp"

# Representative exact binary64 Terrestrial Time states.
MODERN_TT = 2460754.8768300507          # 2025 solar-longitude 0 crossing
DE440_ERA_TT = 2287184.4999999991       # DE440 declared lower bound
DEEP_TIME_TT = -3100014.9943311699      # DE441 part 1, deep negative era
FAR_FUTURE_TT = 8000016.3607090004      # DE441 part 2, far future

REPRESENTATIVE_TT = (
    MODERN_TT,
    DE440_ERA_TT,
    DEEP_TIME_TT,
    FAR_FUTURE_TT,
)

A3A_BLOCK_MARKER = "# A3a - exact astronomical event record."

GOVERNED_BLOCK_MARKER = re.compile(r"^# A\d+[a-z]?(?:-\d+[a-z]?)? - ",
                                   re.MULTILINE)


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


# --- 1. Record contract ----------------------------------------------------


class TestRecordContract(unittest.TestCase):
    def setUp(self):
        self.event = AstronomicalEvent(
            tt=MODERN_TT, kind="SOLAR_LONGITUDE_000", kernel=DE440
        )

    def test_fields_are_exactly_the_expected_set(self):
        self.assertEqual(
            tuple(f.name for f in dataclasses.fields(self.event)),
            EXPECTED_EVENT_FIELDS,
        )

    def test_record_is_frozen(self):
        for field in EXPECTED_EVENT_FIELDS:
            with self.subTest(field=field):
                with self.assertRaises(Exception):
                    setattr(self.event, field, None)

    def test_is_a_named_dataclass_not_a_tuple(self):
        self.assertTrue(dataclasses.is_dataclass(self.event))
        self.assertNotIsInstance(self.event, tuple)
        with self.assertRaises(TypeError):
            iter(self.event)
        with self.assertRaises(TypeError):
            self.event[0]

    def test_values_are_carried_verbatim(self):
        self.assertEqual(self.event.tt, MODERN_TT)
        self.assertEqual(self.event.kind, "SOLAR_LONGITUDE_000")
        self.assertEqual(self.event.kernel, DE440)

    def test_equality_and_hashing_use_all_three_fields(self):
        same = AstronomicalEvent(
            tt=MODERN_TT, kind="SOLAR_LONGITUDE_000", kernel=DE440
        )
        self.assertEqual(self.event, same)
        self.assertEqual(hash(self.event), hash(same))

        for altered in (
            AstronomicalEvent(
                tt=MODERN_TT + 1.0, kind="SOLAR_LONGITUDE_000", kernel=DE440
            ),
            AstronomicalEvent(
                tt=MODERN_TT, kind="SOLAR_LONGITUDE_090", kernel=DE440
            ),
            AstronomicalEvent(
                tt=MODERN_TT, kind="SOLAR_LONGITUDE_000",
                kernel=DE441_PART_2,
            ),
        ):
            with self.subTest(altered=altered):
                self.assertNotEqual(self.event, altered)

    def test_no_observer_field(self):
        names = {f.name for f in dataclasses.fields(self.event)}
        for forbidden in (
            "latitude", "longitude", "observer", "location", "elevation",
        ):
            with self.subTest(field=forbidden):
                self.assertNotIn(forbidden, names)
                self.assertFalse(hasattr(self.event, forbidden))

    def test_no_transport_or_calendar_field(self):
        names = {f.name for f in dataclasses.fields(self.event)}
        for forbidden in (
            "utc", "utcISO", "iso", "isoformat", "datetime", "date", "time",
            "timestamp", "unix", "millis", "milliseconds", "epoch",
            "year", "month", "day", "weekday", "hemisphere", "season",
            "route", "status", "detail",
        ):
            with self.subTest(field=forbidden):
                self.assertNotIn(forbidden, names)
                self.assertFalse(hasattr(self.event, forbidden))

    def test_no_tdb_field(self):
        names = {f.name for f in dataclasses.fields(self.event)}
        for forbidden in ("tdb", "tdb_start", "tdb_end", "barycentric"):
            with self.subTest(field=forbidden):
                self.assertNotIn(forbidden, names)
                self.assertFalse(hasattr(self.event, forbidden))

    def test_exactly_three_fields(self):
        self.assertEqual(len(dataclasses.fields(self.event)), 3)


# --- 2. Governed identities ------------------------------------------------


class TestSolarLongitudeIdentities(unittest.TestCase):
    def published(self):
        return {
            name: getattr(astronomy_solver, name)
            for name in EXPECTED_IDENTITY_NAMES
        }

    def test_all_four_identities_are_published(self):
        for name in EXPECTED_IDENTITY_NAMES:
            with self.subTest(name=name):
                self.assertTrue(
                    hasattr(astronomy_solver, name),
                    "%s is not published" % name,
                )

    def test_literal_values_are_exactly_as_expected(self):
        for name, expected in EXPECTED_IDENTITY_VALUES.items():
            with self.subTest(name=name):
                self.assertEqual(getattr(astronomy_solver, name), expected)

    def test_values_are_non_empty_strings(self):
        for name, value in self.published().items():
            with self.subTest(name=name):
                self.assertIsInstance(value, str)
                self.assertTrue(value)
                self.assertEqual(value, value.strip())

    def test_all_four_are_distinct(self):
        values = list(self.published().values())
        self.assertEqual(len(set(values)), 4)
        self.assertEqual(len(set(EXPECTED_IDENTITY_NAMES)), 4)

    def test_no_hemisphere_or_season_vocabulary(self):
        for name, value in self.published().items():
            for word in HEMISPHERE_VOCABULARY:
                with self.subTest(name=name, word=word):
                    self.assertNotIn(word, value.lower())
                    self.assertNotIn(word, name.lower())

    def test_no_gregorian_month_vocabulary(self):
        for name, value in self.published().items():
            for word in GREGORIAN_MONTH_VOCABULARY:
                with self.subTest(name=name, word=word):
                    self.assertNotIn(word, value.lower())
                    self.assertNotIn(word, name.lower())

    def test_each_identity_names_its_defining_longitude(self):
        for name, degrees in (
            ("SOLAR_LONGITUDE_000", "000"),
            ("SOLAR_LONGITUDE_090", "090"),
            ("SOLAR_LONGITUDE_180", "180"),
            ("SOLAR_LONGITUDE_270", "270"),
        ):
            with self.subTest(name=name):
                self.assertTrue(getattr(astronomy_solver, name).endswith(degrees))

    def test_identities_are_usable_as_the_record_kind(self):
        for name, value in self.published().items():
            with self.subTest(name=name):
                event = AstronomicalEvent(
                    tt=MODERN_TT, kind=value, kernel=DE440
                )
                self.assertEqual(event.kind, value)


# --- 3. Exact TT preservation ----------------------------------------------


class TestExactTTPreservation(unittest.TestCase):
    """TT is carried unchanged, under this certified runtime and timescale."""

    def test_representative_values_are_carried_verbatim(self):
        for value in REPRESENTATIVE_TT:
            with self.subTest(tt=value):
                event = AstronomicalEvent(
                    tt=value, kind="SOLAR_LONGITUDE_000", kernel=DE440
                )
                self.assertEqual(event.tt, value)
                self.assertIsInstance(event.tt, float)
                self.assertTrue(math.isfinite(event.tt))

    def test_timescale_round_trip_is_exact(self):
        for value in REPRESENTATIVE_TT:
            with self.subTest(tt=value):
                event = AstronomicalEvent(
                    tt=value, kind="SOLAR_LONGITUDE_000", kernel=DE440
                )
                self.assertEqual(
                    float(ts.tt_jd(event.tt).tt), event.tt,
                    "TT must survive the certified timescale unchanged",
                )

    def test_modern_era_value(self):
        event = AstronomicalEvent(
            tt=MODERN_TT, kind="SOLAR_LONGITUDE_000", kernel=DE440
        )
        self.assertEqual(event.tt, 2460754.8768300507)
        self.assertEqual(float(ts.tt_jd(event.tt).tt), 2460754.8768300507)

    def test_de440_era_value(self):
        event = AstronomicalEvent(
            tt=DE440_ERA_TT, kind="SOLAR_LONGITUDE_090", kernel=DE440
        )
        self.assertEqual(event.tt, 2287184.4999999991)
        self.assertEqual(float(ts.tt_jd(event.tt).tt), 2287184.4999999991)

    def test_deep_time_negative_era_value(self):
        event = AstronomicalEvent(
            tt=DEEP_TIME_TT, kind="SOLAR_LONGITUDE_180",
            kernel=DE441_PART_1,
        )
        self.assertLess(event.tt, 0.0)
        self.assertEqual(event.tt, -3100014.9943311699)
        self.assertEqual(float(ts.tt_jd(event.tt).tt), -3100014.9943311699)

    def test_far_future_value(self):
        event = AstronomicalEvent(
            tt=FAR_FUTURE_TT, kind="SOLAR_LONGITUDE_270",
            kernel=DE441_PART_2,
        )
        self.assertEqual(event.tt, 8000016.3607090004)
        self.assertEqual(float(ts.tt_jd(event.tt).tt), 8000016.3607090004)

    def test_integer_input_is_carried_as_given_without_coercion(self):
        """The record stores what it is handed; it is not a validator."""
        event = AstronomicalEvent(
            tt=2460754, kind="SOLAR_LONGITUDE_000", kernel=DE440
        )
        self.assertEqual(event.tt, 2460754)
        self.assertEqual(float(event.tt), 2460754.0)
        self.assertEqual(float(ts.tt_jd(float(event.tt)).tt), 2460754.0)

    def test_neighbouring_binary64_states_stay_distinct(self):
        lower = math.nextafter(MODERN_TT, -math.inf)
        a = AstronomicalEvent(
            tt=MODERN_TT, kind="SOLAR_LONGITUDE_000", kernel=DE440
        )
        b = AstronomicalEvent(
            tt=lower, kind="SOLAR_LONGITUDE_000", kernel=DE440
        )
        self.assertNotEqual(a.tt, b.tt)
        self.assertNotEqual(a, b)


# --- 4. Deep-time admissibility --------------------------------------------


class TestDeepTimeAdmissibility(unittest.TestCase):
    """A record outside Python datetime's range is still fully valid.

    Python's limits appear here only as supporting evidence that a transport
    field could not have been made mandatory. They are not asserted as an
    astronomical property.
    """

    def setUp(self):
        self.event = AstronomicalEvent(
            tt=DEEP_TIME_TT, kind="SOLAR_LONGITUDE_180",
            kernel=DE441_PART_1,
        )

    def test_record_is_constructible_and_complete(self):
        self.assertEqual(self.event.tt, DEEP_TIME_TT)
        self.assertEqual(self.event.kind, "SOLAR_LONGITUDE_180")
        self.assertEqual(self.event.kernel, DE441_PART_1)

    def test_state_is_reconstructable_through_the_certified_timescale(self):
        self.assertEqual(float(ts.tt_jd(self.event.tt).tt), self.event.tt)

    def test_record_validity_is_independent_of_calendar_renderability(self):
        t = ts.tt_jd(self.event.tt)
        try:
            t.utc_datetime()
            renderable = True
        except (ValueError, OverflowError):
            renderable = False

        # Whatever a presentation layer can or cannot do, the record is
        # unchanged and still satisfies every claim made about it.
        self.assertIsInstance(renderable, bool)
        self.assertEqual(self.event.tt, DEEP_TIME_TT)
        self.assertEqual(float(ts.tt_jd(self.event.tt).tt), self.event.tt)
        self.assertEqual(
            tuple(f.name for f in dataclasses.fields(self.event)),
            EXPECTED_EVENT_FIELDS,
        )

    def test_supporting_evidence_a_transport_field_could_not_be_mandatory(self):
        """Evidence only: some determined states have no datetime rendering."""
        t = ts.tt_jd(DEEP_TIME_TT)
        with self.assertRaises((ValueError, OverflowError)):
            t.utc_datetime()


# --- 5. Structural guards --------------------------------------------------


class TestA3aStructuralIndependence(unittest.TestCase):
    """A3a is substrate: it must not compute, validate, route or convert."""

    def setUp(self):
        source = inspect.getsource(astronomy_solver)
        index = source.find(A3A_BLOCK_MARKER)
        self.assertNotEqual(index, -1, "A3a block marker not found")
        following = GOVERNED_BLOCK_MARKER.search(
            source, index + len(A3A_BLOCK_MARKER)
        )
        self.block = source[
            index:following.start() if following else len(source)
        ]
        self.preamble = source[:index]
        self.executable = executable_source(self.block)

    def test_block_was_located_and_is_substantial(self):
        self.assertIn("class AstronomicalEvent", self.block)
        self.assertIn("SOLAR_LONGITUDE_000", self.block)
        self.assertGreater(len(self.block), 1000)

    def test_block_stops_at_the_next_governed_block(self):
        self.assertIsNone(
            GOVERNED_BLOCK_MARKER.search(self.block, len(A3A_BLOCK_MARKER)),
            "the A3a scan reaches into a later governed block",
        )

    def test_intended_content_is_present(self):
        """Anti-vacuity: the guards below must inspect real logic."""
        self.assertIn("class AstronomicalEvent", self.executable)
        self.assertIn("@dataclass(frozen=True)", self.executable)
        self.assertIn("tt: float", self.executable)
        self.assertIn("kind: str", self.executable)
        self.assertIn("kernel: str", self.executable)
        for name in EXPECTED_IDENTITY_NAMES:
            with self.subTest(name=name):
                self.assertIn(name, self.executable)
        self.assertGreater(len(self.executable), 150)

        # Prose really was excluded.
        self.assertNotIn("WHY A RECORD AT ALL", self.executable)
        self.assertNotIn("TT IS SUFFICIENT", self.executable)

    def test_block_performs_no_computation(self):
        for token in (
            "find_discrete",
            "almanac",
            "seasons",
            "sunrise_sunset",
            "load_kernel",
            "supported_search_frontier",
            "wgs84",
            "ts.tt_jd",
            "kernel_coverage_tt",
            "_exact_finite_tt",
            "_governed_observer",
            "find_sunset",
            "def ",
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

    def test_block_has_no_civil_or_transport_authority(self):
        for token in (
            "ts.utc(",
            "utc_strftime",
            "utc_datetime",
            "utc_iso",
            "isoformat",
            "strftime",
            "datetime",
            "timestamp",
            "timedelta",
            "format_skyfield_time",
            "1000",
            "3600",
            "/ 15",
            "lru_cache",
            "cache",
            "HTTPException",
            "status_code",
        ):
            with self.subTest(token=token):
                self.assertNotIn(token, self.executable)

    def test_block_has_no_hemisphere_or_month_vocabulary(self):
        lowered = self.executable.lower()
        for word in HEMISPHERE_VOCABULARY + GREGORIAN_MONTH_VOCABULARY:
            with self.subTest(word=word):
                self.assertNotIn(word, lowered)

    def test_block_defines_exactly_the_authorized_symbols(self):
        tree = ast.parse(self.block)

        classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        assigns = [n for n in tree.body if isinstance(n, ast.Assign)]
        imports = [
            n for n in tree.body
            if isinstance(n, (ast.Import, ast.ImportFrom))
        ]

        self.assertEqual([c.name for c in classes], ["AstronomicalEvent"])
        self.assertEqual(functions, [], "A3a defines no helper function")
        self.assertEqual(imports, [], "A3a introduces no import")
        self.assertEqual(
            [t.id for n in assigns for t in n.targets],
            list(EXPECTED_IDENTITY_NAMES),
        )

    def test_block_introduces_no_reason_code(self):
        for token in ("REASON_", "SunsetChronologyError", "raise "):
            with self.subTest(token=token):
                self.assertNotIn(token, self.executable)

    def test_frozen_code_is_still_present_ahead_of_the_block(self):
        for token in (
            "def choose_kernel_name",
            "PRIMARY_START_YEAR",
            "PRIMARY_END_YEAR",
            "min_gap_seconds",
            "def format_skyfield_time",
            "def find_equinox",
            "def find_season_events",
            "def find_sunset_utc",
            "def find_next_sunset_after_utc",
            "def find_sunset_successor",
            "def kernel_coverage_tt",
            "def select_kernel_containing_instant",
            "def select_kernel_for_interval",
            "def supported_search_frontier",
            "def _first_evaluable_state",
            "class SunsetEvent",
            "def find_sunset_predecessor",
            "def find_sunset_from_instant",
            "class SunsetBracket",
            "def find_sunset_bracket",
        ):
            with self.subTest(token=token):
                self.assertIn(token, self.preamble)

    def test_legacy_season_paths_are_not_migrated(self):
        """A3a must not have rewired the existing equinox/season callers."""
        for name in ("find_equinox", "find_season_events"):
            with self.subTest(name=name):
                body = inspect.getsource(getattr(astronomy_solver, name))
                self.assertNotIn("AstronomicalEvent", body)
                self.assertNotIn("SOLAR_LONGITUDE_", body)
                self.assertIn("choose_kernel_name", body)


if __name__ == "__main__":
    unittest.main()
