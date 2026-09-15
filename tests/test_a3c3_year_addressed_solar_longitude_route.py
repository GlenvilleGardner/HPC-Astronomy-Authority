"""A3c-3 verification for the year-addressed exact solar-longitude event.

Covers the A3c-3 block in astronomy_solver.py
(_year_addressing_anchor, find_solar_longitude_event_in_year) and
GET /solar-longitude-event-in-year in server.py: the request contract, year
numbering, year validation, the success projection, canonical event identity,
artifact provenance, the artifact-derived domain, addressing uniqueness, the
governed failure envelope, relationship to the frozen legacy year path, and
structural isolation of the production block.

Test framework: standard-library unittest.
Scientific execution uses the repository's certified Skyfield runtime and the
real pinned Authority ephemeris artifacts.

INVOCATION
----------
Run from the repository root with the repository interpreter:

    .\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v

ROUTE SEAM

The route function is invoked directly, following A1c, A1d, A3c-1 and A3c-2.
httpx is not installed, so starlette's TestClient is unavailable, and
installing a dependency in order to test is not permitted. Direct invocation
exercises the real route body, including the HTTPException mapping, which is
asserted through the exception object itself.

Direct invocation also means FastAPI performs no parameter coercion, so the
production year guard is exercised by the values a caller actually supplies
rather than by whatever a framework would have converted them into first.

INDEPENDENT ORACLE DISCIPLINE

Reason codes, pinned artifact filenames, governed event-kind strings, the
expected key sets, the governed search horizon, the route path and the
measured domain extrema are declared here as test-local literals. They are
deliberately NOT imported from astronomy_solver: importing a constant to
check that same constant would make the test agree with a defective value
instead of detecting it.

Canonical boundaries are relocated here by a test-local bisection oracle that
never calls _canonical_quadrant_boundary, and the quadrant mapping is
re-declared here rather than read from production.

The published projection IS called, as certified A3c-0 infrastructure, to
assert that the route returns exactly it and hand-builds nothing.

EVIDENCE DISCIPLINE

PROVED    - follows structurally from the code under test.
MEASURED  - observed against the real pinned artifacts and certified runtime
            at test time.
SAMPLED   - observed over a representative sample spanning the domain; not a
            universal claim.

The domain extrema and the addressing-uniqueness results below are MEASURED
and SAMPLED properties tied to the pinned artifact generation and the
certified runtime. They are NOT timeless astronomical constants, and they are
cross-checked against the artifacts' own declared coverage so that an
artifact change fails certification rather than silently changing what the
operation answers.

WHAT IS DELIBERATELY NOT ENCODED

No epsilon, tolerance or gap governs event identity anywhere in this file.
Where two determinations are compared, they are compared bit for bit. Where a
whole-second legacy RENDERING is compared against an exact instant, the
comparison is a discrimination argument over a separation of order one year,
explicitly labelled as such, and never a definition of identity.

No HPC/SCE year, epoch, weekday, month, Telma, year classification or sunset
chronology appears. A3c-3 is Astronomy Authority; none of those exist here
and none is implied by an addressed year.
"""

import ast
import datetime
import inspect
import math
import re
import struct
import unittest
from unittest import mock

from fastapi import HTTPException
from skyfield import almanac

import astronomy_solver
import server
from astronomy_solver import (
    SunsetChronologyError,
    find_equinox,
    find_solar_longitude_event_after,
    find_solar_longitude_event_in_year,
    kernel_coverage_tt,
    load_kernel,
    ts,
)
from astronomical_event_transport import project_astronomical_event
from exact_time_transport import decode_tt_bits

# --- Test-local oracles ----------------------------------------------------

EVENT_IN_YEAR_PATH = "/solar-longitude-event-in-year"

DE440 = "de440.bsp"
DE441_PART_1 = "de441_part-1.bsp"
DE441_PART_2 = "de441_part-2.bsp"
PINNED_ARTIFACTS = (DE440, DE441_PART_1, DE441_PART_2)

# The four governed identities in Skyfield quadrant order, re-declared here.
KIND_000 = "SOLAR_LONGITUDE_000"
KIND_090 = "SOLAR_LONGITUDE_090"
KIND_180 = "SOLAR_LONGITUDE_180"
KIND_270 = "SOLAR_LONGITUDE_270"
GOVERNED_KINDS = (KIND_000, KIND_090, KIND_180, KIND_270)
QUADRANT_OF = {KIND_000: 0, KIND_090: 1, KIND_180: 2, KIND_270: 3}

EXPECTED_EVENT_KEYS = ("ttBits", "tt", "utc", "kind", "kernel")
EXPECTED_QUERY_PARAMETERS = ["year", "kind"]

REASON_EVENT_YEAR_INVALID = "EVENT_YEAR_INVALID"
REASON_EVENT_KIND_INVALID = "EVENT_KIND_INVALID"
REASON_COVERAGE_EXHAUSTED = "EPHEMERIS_COVERAGE_EXHAUSTED"

# The governed directional reach of one solar-longitude search, in TT days.
# Re-declared rather than imported, and used only to predict which years the
# artifacts can support.
SEARCH_SPAN_DAYS = 380.0

MODERN_YEAR = 2020

# MEASURED domain extrema for the currently pinned artifact generation.
# These are evidence about these artifacts, not an eternal claim, and
# TestArtifactDomain cross-checks every one of them against the artifacts'
# own declared coverage.
UNSUPPORTED_LOW_YEAR = -13200
SUPPORTED_LOW_YEAR = -13199
SUPPORTED_HIGH_YEAR = 17190
UNSUPPORTED_HIGH_YEAR = 17191

# The deep-time capability the downstream research system requires of this
# Authority. It is an astronomical year and carries no HPC/SCE meaning here.
RESEARCH_DEEP_TIME_YEAR = -4018

# SAMPLED years spanning the supported domain, for addressing uniqueness.
UNIQUENESS_SAMPLE_YEARS = (
    SUPPORTED_LOW_YEAR, -9000, -4018, -1, 0, 1, MODERN_YEAR, 9999,
    10000, SUPPORTED_HIGH_YEAR - 1,
)

# SAMPLED positive years inside the range the frozen legacy equinox
# computation can actually answer. 2650 is excluded: the legacy path raises a
# raw ephemeris range error there, which is a known frozen-legacy defect and
# is not this increment's business.
LEGACY_AGREEMENT_YEARS = (1550, 1601, 1776, 1900, 2020, 2023, 2048, 2649)

# Years at or below zero, where A3c-3 and the frozen legacy path intentionally
# diverge because they use different numbering.
LEGACY_DIVERGENCE_YEARS = (0, -1, -2, -100)

MALFORMED_YEARS = (
    ("float", 2020.5),
    ("integral float", 2020.0),
    ("bool true", True),
    ("bool false", False),
    ("string", "2020"),
    ("none", None),
    ("complex", complex(2020, 0)),
)

A3C3_BLOCK_MARKER = "# A3c-3 - year-addressed exact solar-longitude event."
A3C3_ROUTE_BLOCK_MARKER = (
    "# A3c-3 - year-addressed exact solar-longitude event route."
)

GOVERNED_BLOCK_MARKER = re.compile(r"^# A\d+[a-z]?(?:-[0-9a-z]+)? - ",
                                   re.MULTILINE)

# Rendered by the frozen legacy path as -BBBB for BCE year BBBB.
LEGACY_YEAR_LABEL = re.compile(r"^(-?)(\d{4})-")


# --- Independent helpers ---------------------------------------------------


def bits(value):
    """The exact binary64 pattern of a float, as an independent oracle."""
    return struct.pack(">d", value)


def event_response(year, kind):
    return server.solar_longitude_event_in_year(year=year, kind=kind)


def registered_routes():
    found = {}
    for route in server.app.routes:
        endpoint = getattr(route, "endpoint", None)
        if endpoint is None:
            continue
        found[route.path] = (
            set(route.methods),
            endpoint.__name__,
            list(inspect.signature(endpoint).parameters),
        )
    return found


def executable_source(text):
    """Return a block's executable logic, prose excluded."""
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


def year_anchor(year):
    """The start of an astronomical year on the certified timescale."""
    return float(ts.utc(year, 1, 1).tt)


def astronomical_year_of(tt):
    """The astronomical year a TT state falls in, read from the timescale."""
    return int(ts.tt_jd(tt).tt_calendar()[0])


def legacy_label_to_astronomical_year(rendered):
    """Convert a frozen-legacy rendered year label to astronomical numbering.

    The legacy renderer writes a BCE year as -BBBB, where BBBB counts BCE
    years from 1. Astronomical numbering has a year zero, so BCE year B is
    astronomical year 1 - B. A positive label is already astronomical.
    """
    match = LEGACY_YEAR_LABEL.match(rendered)
    assert match is not None, "unparseable legacy rendering %r" % rendered
    sign, digits = match.groups()
    return 1 - int(digits) if sign else int(digits)


def oracle_canonical_boundary(kernel_name, tt_below, tt_at_or_above, quadrant):
    """Relocate a canonical quadrant boundary without production helpers.

    Bisects the governed predicate itself until no representable binary64
    state remains between the two ends. There is no epsilon and no
    convergence threshold: termination is exhaustion of the representation.

    Never calls _canonical_quadrant_boundary or _solar_longitude_crossings.
    """
    season_at = almanac.seasons(load_kernel(kernel_name))
    below = tt_below
    at_or_above = tt_at_or_above
    while True:
        mid = below + (at_or_above - below) / 2.0
        if mid == below or mid == at_or_above:
            return at_or_above
        if int(season_at(ts.tt_jd(mid))) == quadrant:
            at_or_above = mid
        else:
            below = mid


def declared_coverage_supports(anchor):
    """Whether some pinned artifact declares the whole governed horizon.

    Read from the artifacts' own segment metadata via A2-1. This is the
    independent prediction the measured domain extrema are checked against.
    """
    for name in PINNED_ARTIFACTS:
        coverage = kernel_coverage_tt(name)
        if (coverage.tt_start <= anchor
                and anchor + SEARCH_SPAN_DAYS <= coverage.tt_end):
            return True
    return False


class EventAssertions(unittest.TestCase):
    def assert_rejects(self, year, kind, reason, status=400):
        with self.assertRaises(HTTPException) as caught:
            event_response(year, kind)
        error = caught.exception
        self.assertEqual(error.status_code, status)
        self.assertIsInstance(error.detail, dict)
        self.assertEqual(tuple(error.detail), ("reason", "message"))
        self.assertEqual(error.detail["reason"], reason)
        self.assertIsInstance(error.detail["message"], str)
        return error

    def assert_is_event(self, response, kind):
        self.assertIsInstance(response, dict)
        self.assertEqual(tuple(response), EXPECTED_EVENT_KEYS)
        self.assertEqual(response["kind"], kind)
        self.assertIn(response["kernel"], PINNED_ARTIFACTS)
        self.assertIsInstance(response["tt"], float)
        self.assertTrue(math.isfinite(response["tt"]))
        self.assertIsInstance(response["ttBits"], str)
        # ttBits is the identity; tt is produced from those same bits.
        self.assertEqual(bits(decode_tt_bits(response["ttBits"])),
                         bits(response["tt"]))
        self.assertTrue(response["utc"] is None
                        or isinstance(response["utc"], str))
        return response


# --- 1. Request contract and route surface ---------------------------------


class TestRouteSurface(EventAssertions):
    def test_path_is_registered(self):
        self.assertIn(EVENT_IN_YEAR_PATH, registered_routes())

    def test_route_is_get_only(self):
        methods = registered_routes()[EVENT_IN_YEAR_PATH][0]
        self.assertEqual(methods - {"HEAD"}, {"GET"})

    def test_endpoint_name(self):
        self.assertEqual(registered_routes()[EVENT_IN_YEAR_PATH][1],
                         "solar_longitude_event_in_year")

    def test_parameters_are_exactly_year_and_kind(self):
        self.assertEqual(registered_routes()[EVENT_IN_YEAR_PATH][2],
                         EXPECTED_QUERY_PARAMETERS)

    def test_no_observer_instant_or_artifact_parameter(self):
        """No client authority over observer, instant, calendar or artifact."""
        forbidden = {
            "ttBits", "tt", "utc", "date", "afterUTC", "instant",
            "latitude", "longitude", "observer", "elevation",
            "kernel", "ephemeris", "artifact",
            "month", "day", "hour", "minute", "second",
            "timezone", "tz", "offset", "direction", "cursor",
            "hemisphere", "season", "calendar", "era", "bceYear",
        }
        self.assertEqual(
            set(registered_routes()[EVENT_IN_YEAR_PATH][2]) & forbidden, set()
        )

    def test_year_is_the_only_temporal_parameter(self):
        parameters = registered_routes()[EVENT_IN_YEAR_PATH][2]
        self.assertEqual([p for p in parameters if p != "kind"], ["year"])


# --- 2. Success projection --------------------------------------------------


class TestSuccessProjection(EventAssertions):
    def test_response_is_exactly_the_published_projection(self):
        for kind in GOVERNED_KINDS:
            with self.subTest(kind=kind):
                expected = project_astronomical_event(
                    find_solar_longitude_event_in_year(MODERN_YEAR, kind)
                )
                self.assertEqual(event_response(MODERN_YEAR, kind), expected)

    def test_response_shape_and_bit_consistency(self):
        for kind in GOVERNED_KINDS:
            with self.subTest(kind=kind):
                self.assert_is_event(event_response(MODERN_YEAR, kind), kind)

    def test_addressing_year_is_not_echoed_back(self):
        """The address is not the answer and is never reported as one."""
        response = event_response(MODERN_YEAR, KIND_000)
        self.assertNotIn("year", response)
        self.assertEqual(tuple(response), EXPECTED_EVENT_KEYS)

    def test_kind_is_carried_through_unaltered(self):
        for kind in GOVERNED_KINDS:
            with self.subTest(kind=kind):
                self.assertEqual(event_response(MODERN_YEAR, kind)["kind"],
                                 kind)

    def test_no_hemisphere_or_season_terminology_is_produced(self):
        blob = repr(
            [event_response(MODERN_YEAR, k) for k in GOVERNED_KINDS]
        ).lower()
        for token in ("spring", "summer", "autumn", "fall", "winter",
                      "equinox", "solstice", "hemisphere", "abib", "telma"):
            with self.subTest(token=token):
                self.assertNotIn(token, blob)

    def test_repeated_calls_are_deterministic(self):
        first = event_response(MODERN_YEAR, KIND_000)
        second = event_response(MODERN_YEAR, KIND_000)
        self.assertEqual(first, second)
        self.assertEqual(bits(first["tt"]), bits(second["tt"]))


# --- 3. All four governed kinds --------------------------------------------


class TestAllFourGovernedKinds(EventAssertions):
    def test_every_governed_kind_resolves(self):
        seen = {}
        for kind in GOVERNED_KINDS:
            with self.subTest(kind=kind):
                response = self.assert_is_event(
                    event_response(MODERN_YEAR, kind), kind
                )
                seen[kind] = response["tt"]
        self.assertEqual(len(set(seen.values())), 4)

    def test_kinds_resolve_in_quadrant_order_within_the_year(self):
        resolved = [event_response(MODERN_YEAR, k)["tt"]
                    for k in GOVERNED_KINDS]
        self.assertEqual(resolved, sorted(resolved))

    def test_ungoverned_kinds_are_refused(self):
        for bad in ("solar_longitude_000", "SOLAR_LONGITUDE_0",
                    "SOLAR_LONGITUDE_000 ", " SOLAR_LONGITUDE_000",
                    "000", "0", "Spring", "", "SOLAR_LONGITUDE_360",
                    None, 0, 0.0, True):
            with self.subTest(kind=bad):
                self.assert_rejects(MODERN_YEAR, bad, REASON_EVENT_KIND_INVALID)


# --- 4. Year numbering semantics -------------------------------------------


class TestSkyfieldYearSemantics(unittest.TestCase):
    """Certification C. The addressing convention the operation relies on.

    MEASURED against the certified runtime. If a runtime change altered any
    of these, A3c-3's addressing would silently shift, so they are pinned
    here rather than assumed.
    """

    def test_timescale_has_no_julian_calendar_cutoff(self):
        """Proleptic Gregorian throughout, deep time included."""
        self.assertIsNone(ts.julian_calendar_cutoff)

    def test_supplied_astronomical_year_is_used_unshifted(self):
        for year in (2, 1, 0, -1, -2, -100, -4018):
            with self.subTest(year=year):
                self.assertEqual(int(ts.utc(year, 1, 1).tt_calendar()[0]),
                                 year)

    def test_year_zero_exists_and_is_a_leap_year(self):
        """The defining signature of astronomical year numbering."""
        self.assertEqual(round(year_anchor(1) - year_anchor(0)), 366)
        self.assertEqual(round(year_anchor(0) - year_anchor(-1)), 365)
        self.assertEqual(round(year_anchor(2) - year_anchor(1)), 365)

    def test_year_numbering_is_contiguous_across_the_epoch_boundary(self):
        """No gap at zero: 1 = 1 CE, 0 = 1 BCE, -1 = 2 BCE."""
        anchors = [year_anchor(y) for y in (-2, -1, 0, 1, 2)]
        self.assertEqual(anchors, sorted(anchors))
        for lower, upper in zip(anchors, anchors[1:]):
            self.assertIn(round(upper - lower), (365, 366))


class TestYearAddressingOrigin(EventAssertions):
    def test_origin_is_the_start_of_the_addressed_year(self):
        """PROVED by observation of the state handed to the substrate."""
        captured = []
        real = astronomy_solver.find_solar_longitude_event_after

        def recorder(tt, kind):
            captured.append((tt, kind))
            return real(tt, kind)

        with mock.patch.object(
            astronomy_solver, "find_solar_longitude_event_after", recorder
        ):
            find_solar_longitude_event_in_year(MODERN_YEAR, KIND_000)

        self.assertEqual(len(captured), 1)
        self.assertEqual(bits(captured[0][0]), bits(year_anchor(MODERN_YEAR)))
        self.assertEqual(captured[0][1], KIND_000)

    def test_origin_is_never_the_reported_instant(self):
        for year in (MODERN_YEAR, 1, 0, RESEARCH_DEEP_TIME_YEAR):
            for kind in GOVERNED_KINDS:
                with self.subTest(year=year, kind=kind):
                    response = event_response(year, kind)
                    self.assertNotEqual(bits(response["tt"]),
                                        bits(year_anchor(year)))
                    self.assertGreater(response["tt"], year_anchor(year))


# --- 5. Year validation -----------------------------------------------------


class TestYearValidation(EventAssertions):
    def test_non_integer_years_are_refused(self):
        for label, bad in MALFORMED_YEARS:
            with self.subTest(case=label):
                self.assert_rejects(bad, KIND_000, REASON_EVENT_YEAR_INVALID)

    def test_bool_is_refused_although_it_is_an_int_subclass(self):
        """PROVED: bool would otherwise silently address year 0 or 1."""
        self.assert_rejects(True, KIND_000, REASON_EVENT_YEAR_INVALID)
        self.assert_rejects(False, KIND_000, REASON_EVENT_YEAR_INVALID)

    def test_unrepresentable_magnitudes_are_refused(self):
        """Both guards are load-bearing.

        MEASURED: 10**308 yields a non-finite state without raising, and
        10**309 raises while converting. Each reaches a different guard and
        both report the same year reason.
        """
        for magnitude in (10 ** 308, 10 ** 309, -(10 ** 309), 10 ** 400):
            with self.subTest(magnitude=len(str(abs(magnitude)))):
                self.assert_rejects(magnitude, KIND_000,
                                    REASON_EVENT_YEAR_INVALID)

    def test_year_failure_message_is_bounded(self):
        """A rejected year may carry unbounded digits; the message may not."""
        error = self.assert_rejects(10 ** 400, KIND_000,
                                    REASON_EVENT_YEAR_INVALID)
        message = error.detail["message"]
        self.assertLess(len(message), 400)
        self.assertNotIn("0" * 50, message)

    def test_year_is_validated_before_kind(self):
        """A question cannot be asked from an origin that does not exist."""
        self.assert_rejects(2020.5, "not-a-kind", REASON_EVENT_YEAR_INVALID)

    def test_a_valid_year_is_not_refused(self):
        self.assert_is_event(event_response(MODERN_YEAR, KIND_000), KIND_000)


# --- 6. Canonical event identity -------------------------------------------


class TestCanonicalIdentity(EventAssertions):
    """Certification 8. Identity is bracket-independent, by delegation.

    No epsilon appears. Determinations are compared bit for bit.
    """

    def test_year_addressing_matches_an_arbitrary_distant_anchor(self):
        """Same crossing, different bracket, bit-identical result."""
        for kind in GOVERNED_KINDS:
            with self.subTest(kind=kind):
                addressed = find_solar_longitude_event_in_year(
                    MODERN_YEAR, kind
                )
                # A deliberately different bracket: 200 days closer in.
                elsewhere = find_solar_longitude_event_after(
                    addressed.tt - 200.0, kind
                )
                self.assertEqual(bits(addressed.tt), bits(elsewhere.tt))
                self.assertEqual(addressed.kind, elsewhere.kind)

    def test_result_matches_an_independent_bisection_oracle(self):
        """The canonical boundary, relocated without production helpers."""
        for kind in GOVERNED_KINDS:
            with self.subTest(kind=kind):
                event = find_solar_longitude_event_in_year(MODERN_YEAR, kind)
                expected = oracle_canonical_boundary(
                    event.kernel, event.tt - 1.0, event.tt, QUADRANT_OF[kind]
                )
                self.assertEqual(bits(event.tt), bits(expected))

    def test_the_boundary_is_exact_at_binary64_resolution(self):
        """The reported state carries the quadrant; its predecessor does not."""
        for kind in GOVERNED_KINDS:
            with self.subTest(kind=kind):
                event = find_solar_longitude_event_in_year(MODERN_YEAR, kind)
                season_at = almanac.seasons(load_kernel(event.kernel))
                previous = math.nextafter(event.tt, -math.inf)
                self.assertEqual(int(season_at(ts.tt_jd(event.tt))),
                                 QUADRANT_OF[kind])
                self.assertNotEqual(int(season_at(ts.tt_jd(previous))),
                                    QUADRANT_OF[kind])


# --- 7. Artifact provenance -------------------------------------------------


class TestArtifactProvenance(EventAssertions):
    def test_kernel_is_a_pinned_artifact_and_is_reported_per_event(self):
        for year in (SUPPORTED_LOW_YEAR, RESEARCH_DEEP_TIME_YEAR, 1,
                     MODERN_YEAR, 9999, SUPPORTED_HIGH_YEAR):
            with self.subTest(year=year):
                response = event_response(year, KIND_000)
                self.assertIn(response["kernel"], PINNED_ARTIFACTS)

    def test_all_three_artifacts_are_exercised_across_the_domain(self):
        seen = {
            event_response(year, KIND_000)["kernel"]
            for year in (RESEARCH_DEEP_TIME_YEAR, MODERN_YEAR,
                         SUPPORTED_HIGH_YEAR)
        }
        self.assertEqual(seen, set(PINNED_ARTIFACTS))

    def test_provenance_matches_declared_coverage(self):
        """The reported artifact really declares the governed horizon."""
        for year in (RESEARCH_DEEP_TIME_YEAR, MODERN_YEAR,
                     SUPPORTED_HIGH_YEAR):
            with self.subTest(year=year):
                response = event_response(year, KIND_000)
                coverage = kernel_coverage_tt(response["kernel"])
                anchor = year_anchor(year)
                self.assertLessEqual(coverage.tt_start, anchor)
                self.assertLessEqual(anchor + SEARCH_SPAN_DAYS,
                                     coverage.tt_end)

    def test_caller_cannot_influence_artifact_selection(self):
        """Provider neutrality: no artifact parameter exists at all."""
        self.assertEqual(registered_routes()[EVENT_IN_YEAR_PATH][2],
                         EXPECTED_QUERY_PARAMETERS)
        signature = inspect.signature(find_solar_longitude_event_in_year)
        self.assertEqual(list(signature.parameters), ["year", "kind"])


# --- 8. Artifact-derived domain --------------------------------------------


class TestArtifactDomain(EventAssertions):
    """Certification D. MEASURED extrema, cross-checked against coverage."""

    def test_supported_edges_resolve_for_every_governed_kind(self):
        for year in (SUPPORTED_LOW_YEAR, SUPPORTED_HIGH_YEAR):
            for kind in GOVERNED_KINDS:
                with self.subTest(year=year, kind=kind):
                    self.assert_is_event(event_response(year, kind), kind)

    def test_unsupported_edges_fail_closed_for_every_governed_kind(self):
        for year in (UNSUPPORTED_LOW_YEAR, UNSUPPORTED_HIGH_YEAR):
            for kind in GOVERNED_KINDS:
                with self.subTest(year=year, kind=kind):
                    self.assert_rejects(year, kind, REASON_COVERAGE_EXHAUSTED)

    def test_measured_extrema_agree_with_declared_artifact_coverage(self):
        """Artifact drift must fail certification, not change the answer.

        The extrema above are MEASURED. Here they are predicted independently
        from the artifacts' own segment metadata, so a future artifact
        generation that moves coverage breaks this test instead of silently
        redefining the operation's domain.
        """
        self.assertTrue(declared_coverage_supports(
            year_anchor(SUPPORTED_LOW_YEAR)))
        self.assertTrue(declared_coverage_supports(
            year_anchor(SUPPORTED_HIGH_YEAR)))
        self.assertFalse(declared_coverage_supports(
            year_anchor(UNSUPPORTED_LOW_YEAR)))
        self.assertFalse(declared_coverage_supports(
            year_anchor(UNSUPPORTED_HIGH_YEAR)))

    def test_the_supported_edges_are_genuinely_adjacent_to_the_unsupported(self):
        self.assertEqual(SUPPORTED_LOW_YEAR - 1, UNSUPPORTED_LOW_YEAR)
        self.assertEqual(SUPPORTED_HIGH_YEAR + 1, UNSUPPORTED_HIGH_YEAR)

    def test_far_outside_coverage_also_fails_closed(self):
        for year in (-50000, 50000):
            with self.subTest(year=year):
                self.assert_rejects(year, KIND_000, REASON_COVERAGE_EXHAUSTED)

    def test_absence_is_never_an_empty_success(self):
        """The operation never returns None and never 404s."""
        with self.assertRaises(HTTPException) as caught:
            event_response(UNSUPPORTED_HIGH_YEAR, KIND_000)
        self.assertEqual(caught.exception.status_code, 400)


# --- 9. Addressing uniqueness ----------------------------------------------


class TestAddressingUniqueness(EventAssertions):
    """Certification E. SAMPLED across the supported domain.

    Certification only. No production algorithm depends on this being
    re-derived at run time.
    """

    def test_each_kind_resolves_inside_its_addressed_year(self):
        for year in UNIQUENESS_SAMPLE_YEARS:
            lower = year_anchor(year)
            upper = year_anchor(year + 1)
            for kind in GOVERNED_KINDS:
                with self.subTest(year=year, kind=kind):
                    event = find_solar_longitude_event_in_year(year, kind)
                    self.assertLess(lower, event.tt)
                    self.assertLess(event.tt, upper)

    def test_consecutive_years_resolve_to_distinct_crossings(self):
        for year in (SUPPORTED_LOW_YEAR, RESEARCH_DEEP_TIME_YEAR, 0,
                     MODERN_YEAR):
            with self.subTest(year=year):
                this = find_solar_longitude_event_in_year(year, KIND_000)
                following = find_solar_longitude_event_in_year(
                    year + 1, KIND_000
                )
                self.assertNotEqual(bits(this.tt), bits(following.tt))
                self.assertLess(this.tt, following.tt)
                # SAMPLED: same-kind crossings are about a year apart, which
                # is why a year names one crossing unambiguously.
                self.assertGreater(following.tt - this.tt, 300.0)

    def test_addressed_year_matches_the_timescale_year_of_the_result(self):
        for year in UNIQUENESS_SAMPLE_YEARS:
            for kind in GOVERNED_KINDS:
                with self.subTest(year=year, kind=kind):
                    event = find_solar_longitude_event_in_year(year, kind)
                    self.assertEqual(astronomical_year_of(event.tt), year)


# --- 10. Nullable UTC reference --------------------------------------------


class TestUtcReference(EventAssertions):
    """Certification 7. utc is reference information, never identity."""

    def test_utc_is_present_inside_the_representable_span(self):
        for year in (1, MODERN_YEAR, 9999):
            with self.subTest(year=year):
                self.assertIsInstance(event_response(year, KIND_000)["utc"],
                                      str)

    def test_utc_is_null_outside_the_representable_span(self):
        for year in (SUPPORTED_LOW_YEAR, RESEARCH_DEEP_TIME_YEAR, 0, 10000,
                     SUPPORTED_HIGH_YEAR):
            with self.subTest(year=year):
                self.assertIsNone(event_response(year, KIND_000)["utc"])

    def test_the_key_is_always_present(self):
        for year in (RESEARCH_DEEP_TIME_YEAR, MODERN_YEAR):
            with self.subTest(year=year):
                self.assertIn("utc", event_response(year, KIND_000))

    def test_a_null_utc_never_degrades_the_exact_state(self):
        """Ancient civil-time limits are not ephemeris uncertainty."""
        response = event_response(RESEARCH_DEEP_TIME_YEAR, KIND_000)
        self.assertIsNone(response["utc"])
        self.assertTrue(math.isfinite(response["tt"]))
        self.assertEqual(bits(decode_tt_bits(response["ttBits"])),
                         bits(response["tt"]))


# --- 11. Frozen legacy relationship ----------------------------------------


class TestLegacyPositiveYearAgreement(unittest.TestCase):
    """Certification A. Above zero the two conventions coincide.

    Exact TT is authoritative. No timestamp-string identity is asserted.

    The legacy rendering is whole-second, so it cannot be compared bit for
    bit against an exact instant. What is asserted instead is a
    DISCRIMINATION argument: the two agree to well under a minute, while the
    nearest other crossing of the same kind is about a year away. That
    separation - order 10**7 - identifies the crossing unambiguously. It is
    not a definition of event identity and is never used as one.
    """

    def test_same_physical_crossing_for_sampled_positive_years(self):
        for year in LEGACY_AGREEMENT_YEARS:
            with self.subTest(year=year):
                rendered, _kernel = find_equinox(year)
                self.assertIsNotNone(rendered)
                event = find_solar_longitude_event_in_year(year, KIND_000)

                # Same astronomical year, read from the timescale itself.
                self.assertEqual(
                    legacy_label_to_astronomical_year(rendered),
                    astronomical_year_of(event.tt),
                )

                legacy_dt = datetime.datetime.fromisoformat(rendered)
                exact_dt = ts.tt_jd(event.tt).utc_datetime()
                separation = abs((legacy_dt - exact_dt).total_seconds())
                self.assertLess(separation, 60.0)

    def test_the_nearest_alternative_crossing_is_about_a_year_away(self):
        """Why sub-minute agreement identifies the crossing unambiguously."""
        for year in (LEGACY_AGREEMENT_YEARS[0], MODERN_YEAR):
            with self.subTest(year=year):
                this = find_solar_longitude_event_in_year(year, KIND_000)
                previous = find_solar_longitude_event_in_year(
                    year - 1, KIND_000
                )
                following = find_solar_longitude_event_in_year(
                    year + 1, KIND_000
                )
                self.assertGreater((this.tt - previous.tt) * 86400.0, 3.0e7)
                self.assertGreater((following.tt - this.tt) * 86400.0, 3.0e7)

    def test_exact_tt_is_authoritative_not_the_rendering(self):
        """The legacy rendering is lossy; the exact state is not."""
        event = find_solar_longitude_event_in_year(MODERN_YEAR, KIND_000)
        rendered, _kernel = find_equinox(MODERN_YEAR)
        self.assertEqual(len(rendered), len("0000-00-00T00:00:00+00:00"))
        exact_dt = ts.tt_jd(event.tt).utc_datetime()
        self.assertNotEqual(exact_dt.microsecond, 0)


class TestLegacyNonPositiveDivergence(unittest.TestCase):
    """Certification B. At or below zero the two intentionally diverge.

    The frozen legacy path shifts years at or below zero by +1 before
    addressing the timescale. A3c-3 does not, and must not.

    This is certified POSITIVELY rather than left to silence: the divergence
    is pinned to exactly one year, so a later maintainer cannot "repair"
    A3c-3 into reproducing the legacy shift without failing certification.

    The legacy behavior itself is frozen and is neither modified nor repaired
    by this increment.
    """

    def test_legacy_addresses_the_year_after_the_one_a3c3_addresses(self):
        for year in LEGACY_DIVERGENCE_YEARS:
            with self.subTest(year=year):
                rendered, _kernel = find_equinox(year)
                self.assertIsNotNone(rendered)
                legacy_year = legacy_label_to_astronomical_year(rendered)
                addressed = find_solar_longitude_event_in_year(year, KIND_000)

                self.assertEqual(astronomical_year_of(addressed.tt), year)
                self.assertEqual(legacy_year, year + 1)

    def test_a3c3_does_not_reproduce_the_legacy_shift(self):
        for year in LEGACY_DIVERGENCE_YEARS:
            with self.subTest(year=year):
                rendered, _kernel = find_equinox(year)
                shifted = find_solar_longitude_event_in_year(
                    year + 1, KIND_000
                )
                # Legacy(year) names the crossing A3c-3 calls year + 1.
                self.assertEqual(
                    legacy_label_to_astronomical_year(rendered),
                    astronomical_year_of(shifted.tt),
                )
                # And A3c-3(year) is a different crossing entirely.
                addressed = find_solar_longitude_event_in_year(year, KIND_000)
                self.assertNotEqual(bits(addressed.tt), bits(shifted.tt))
                self.assertLess(addressed.tt, shifted.tt)

    def test_astronomical_year_zero_is_addressable_and_distinct(self):
        """Year 0 is reachable here; the frozen legacy path collapses it.

        This is the clearest witness that the two numbering systems differ,
        and that A3c-3's addressing coordinate is its own.
        """
        zero = find_solar_longitude_event_in_year(0, KIND_000)
        one = find_solar_longitude_event_in_year(1, KIND_000)
        self.assertEqual(astronomical_year_of(zero.tt), 0)
        self.assertEqual(astronomical_year_of(one.tt), 1)
        self.assertNotEqual(bits(zero.tt), bits(one.tt))

        # MEASURED: the frozen legacy path returns the same rendering for 0
        # and 1, so astronomical year 0 has no legacy address at all.
        self.assertEqual(find_equinox(0)[0], find_equinox(1)[0])

    def test_legacy_routes_are_untouched_by_this_increment(self):
        source = inspect.getsource(astronomy_solver)
        block_start = source.find(A3C3_BLOCK_MARKER)
        self.assertNotEqual(block_start, -1)
        preamble = source[:block_start]
        # The frozen shift is still exactly where it was, three times over.
        self.assertEqual(
            preamble.count("sky_year = year if year > 0 else year + 1"), 3
        )


# --- 12. Deep-time research capability -------------------------------------


class TestDeepTimeResearchCapability(EventAssertions):
    """The deep-time reach the downstream research system requires.

    RESEARCH_DEEP_TIME_YEAR is an ASTRONOMICAL year. It carries no HPC/SCE
    meaning inside the Authority, and nothing here computes an HPC month,
    weekday, Telma, year type or sunset chronology. This class establishes
    scientific capability and nothing else.
    """

    def test_every_governed_kind_resolves_in_the_deep_time_year(self):
        for kind in GOVERNED_KINDS:
            with self.subTest(kind=kind):
                self.assert_is_event(
                    event_response(RESEARCH_DEEP_TIME_YEAR, kind), kind
                )

    def test_deep_time_events_fall_inside_the_addressed_year(self):
        lower = year_anchor(RESEARCH_DEEP_TIME_YEAR)
        upper = year_anchor(RESEARCH_DEEP_TIME_YEAR + 1)
        for kind in GOVERNED_KINDS:
            with self.subTest(kind=kind):
                event = find_solar_longitude_event_in_year(
                    RESEARCH_DEEP_TIME_YEAR, kind
                )
                self.assertLess(lower, event.tt)
                self.assertLess(event.tt, upper)
                self.assertEqual(astronomical_year_of(event.tt),
                                 RESEARCH_DEEP_TIME_YEAR)

    def test_deep_time_provenance_is_a_real_pinned_artifact(self):
        for kind in GOVERNED_KINDS:
            with self.subTest(kind=kind):
                response = event_response(RESEARCH_DEEP_TIME_YEAR, kind)
                self.assertIn(response["kernel"], PINNED_ARTIFACTS)

    def test_deep_time_event_is_exact_despite_having_no_utc_rendering(self):
        for kind in GOVERNED_KINDS:
            with self.subTest(kind=kind):
                response = event_response(RESEARCH_DEEP_TIME_YEAR, kind)
                self.assertIsNone(response["utc"])
                self.assertEqual(bits(decode_tt_bits(response["ttBits"])),
                                 bits(response["tt"]))

    def test_no_hpc_calendar_quantity_is_produced(self):
        response = event_response(RESEARCH_DEEP_TIME_YEAR, KIND_000)
        self.assertEqual(tuple(response), EXPECTED_EVENT_KEYS)


# --- 13. Governed failure envelope -----------------------------------------


class TestGovernedFailureEnvelope(EventAssertions):
    def test_every_governed_reason_crosses_unchanged(self):
        cases = (
            (2020.5, KIND_000, REASON_EVENT_YEAR_INVALID),
            (MODERN_YEAR, "nope", REASON_EVENT_KIND_INVALID),
            (UNSUPPORTED_HIGH_YEAR, KIND_000, REASON_COVERAGE_EXHAUSTED),
        )
        for year, kind, reason in cases:
            with self.subTest(reason=reason):
                self.assert_rejects(year, kind, reason)

    def test_reason_is_a_field_not_parsed_from_prose(self):
        error = self.assert_rejects(MODERN_YEAR, "nope",
                                    REASON_EVENT_KIND_INVALID)
        self.assertEqual(tuple(error.detail), ("reason", "message"))
        self.assertEqual(error.detail["reason"], REASON_EVENT_KIND_INVALID)

    def test_original_exception_is_preserved_as_cause(self):
        with self.assertRaises(HTTPException) as caught:
            event_response(2020.5, KIND_000)
        self.assertIsInstance(caught.exception.__cause__, SunsetChronologyError)

    def test_unexpected_failures_propagate_unchanged(self):
        """TAXONOMY: injected, because no ordinary input witness exists."""
        boom = RuntimeError("not a governed rejection")
        with mock.patch.object(
            server, "find_solar_longitude_event_in_year", side_effect=boom
        ):
            with self.assertRaises(RuntimeError):
                event_response(MODERN_YEAR, KIND_000)

    def test_route_calls_the_operation_exactly_once(self):
        real = server.find_solar_longitude_event_in_year
        calls = []

        def recorder(year, kind):
            calls.append((year, kind))
            return real(year, kind)

        with mock.patch.object(
            server, "find_solar_longitude_event_in_year", recorder
        ):
            event_response(MODERN_YEAR, KIND_000)
        self.assertEqual(calls, [(MODERN_YEAR, KIND_000)])


# --- 14. Production source guard -------------------------------------------


class TestProductionSourceGuard(unittest.TestCase):
    """Certification F. The block delegates astronomy and owns no domain."""

    def setUp(self):
        source = inspect.getsource(astronomy_solver)
        index = source.find(A3C3_BLOCK_MARKER)
        self.assertNotEqual(index, -1, "A3c-3 block marker not found")
        following = GOVERNED_BLOCK_MARKER.search(
            source, index + len(A3C3_BLOCK_MARKER)
        )
        self.block = source[
            index:following.start() if following else len(source)
        ]
        self.executable = executable_source(self.block)
        self.tree = ast.parse(self.block)

    def test_marker_obeys_the_governed_grammar(self):
        self.assertTrue(GOVERNED_BLOCK_MARKER.match(A3C3_BLOCK_MARKER))

    def test_block_was_located_and_is_substantial(self):
        self.assertIn("def find_solar_longitude_event_in_year", self.block)
        self.assertIn("def _year_addressing_anchor", self.block)
        self.assertGreater(len(self.block), 1000)

    def test_block_stops_at_the_next_governed_block(self):
        self.assertIsNone(
            GOVERNED_BLOCK_MARKER.search(self.block, len(A3C3_BLOCK_MARKER)),
            "the A3c-3 scan reaches into a later governed block",
        )

    def test_block_defines_exactly_the_authorized_symbols(self):
        self.assertEqual(
            [n.name for n in self.tree.body if isinstance(n, ast.ClassDef)], []
        )
        self.assertEqual(
            [n.name for n in self.tree.body
             if isinstance(n, ast.FunctionDef)],
            ["_year_addressing_anchor", "find_solar_longitude_event_in_year"],
        )

    def test_block_declares_exactly_the_authorized_constants(self):
        self.assertEqual(
            [n.targets[0].id for n in self.tree.body
             if isinstance(n, ast.Assign)],
            [
                "REASON_EVENT_YEAR_INVALID",
                "YEAR_ADDRESSING_MONTH",
                "YEAR_ADDRESSING_DAY",
            ],
        )

    def test_block_introduces_no_import(self):
        self.assertEqual(
            [n for n in self.tree.body
             if isinstance(n, (ast.Import, ast.ImportFrom))],
            [],
        )

    def test_block_performs_no_astronomy(self):
        for token in ("almanac", "find_discrete", "seasons", "wgs84",
                      "sunrise_sunset", "observe", "apparent",
                      "ecliptic_latlon", "subpoint", "earth", "sun"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.executable)

    def test_block_performs_no_artifact_selection(self):
        for token in ("load_kernel", "choose_kernel_name",
                      "select_kernel_containing_instant",
                      "select_kernel_for_interval", "kernel_coverage_tt",
                      "get_eph_for_year", "get_eph_for_datetime",
                      "PINNED_KERNEL_PRECEDENCE"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.executable)

    def test_block_contains_no_kernel_filename(self):
        for token in ("de440", "de441", ".bsp", "ephemeris"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.executable.lower())

    def test_block_declares_no_coverage_constant_or_year_bound(self):
        for token in ("tt_start", "tt_end", "coverage", "frontier",
                      "PRIMARY_START_YEAR", "PRIMARY_END_YEAR",
                      "SEARCH_SPAN", "13199", "13200", "17190", "17191",
                      "1550", "2650"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.executable)

    def test_block_uses_no_alternate_solver(self):
        called = {
            node.func.id
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("find_solar_longitude_event_after", called)
        for forbidden in ("find_solar_longitude_event_before", "find_equinox",
                          "find_season_events", "solar_longitude",
                          "geocentric_search_frontier",
                          "supported_search_frontier"):
            with self.subTest(callee=forbidden):
                self.assertNotIn(forbidden, called)

    def test_exactly_one_delegation_to_the_certified_substrate(self):
        delegations = [
            node for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "find_solar_longitude_event_after"
        ]
        self.assertEqual(len(delegations), 1)

    def test_block_catches_only_overflow_error(self):
        handlers = [n for n in ast.walk(self.tree) if isinstance(n, ast.Try)]
        self.assertEqual(len(handlers), 1)
        caught = [h.type.id for h in handlers[0].handlers]
        self.assertEqual(caught, ["OverflowError"])

    def test_block_has_no_legacy_shift_or_civil_instant_machinery(self):
        for token in ("sky_year", "datetime", "fromisoformat", "isoformat",
                      "strftime", "timestamp", "timezone", "timedelta",
                      "utc_datetime", "ut1", "Gregorian", "bce", "BCE"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.executable)

    def test_block_encodes_no_observer_or_hemisphere(self):
        """Substring scan, for tokens that cannot occur legitimately.

        "longitude" is deliberately absent from this list. It is a substring
        of the governed geocentric identity this operation exists to serve -
        find_solar_longitude_event_in_year, the delegation to
        find_solar_longitude_event_after, and the SOLAR LONGITUDE reason
        prose - so a substring scan cannot tell an observer coordinate from
        the quantity that defines the event.

        "latitude" remains, and is decisive on its own: an observer requires
        a coordinate pair, and "latitude" shares no substring with any
        governed identity. The identifier-level guard below certifies the
        absence of both coordinates without that ambiguity.
        """
        for token in ("latitude", "observer", "hemisphere",
                      "spring", "summer", "autumn", "winter"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.executable.lower())

    def test_block_declares_no_observer_parameter(self):
        """Identifier-level, so a scientific term cannot be mistaken for one.

        Stronger than a substring scan in both directions: it cannot false
        positive on "solar longitude", and it cannot be defeated by an
        observer coordinate that a prose scan happened not to look for.
        """
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                arguments = node.args
                declared = [
                    a.arg for a in (
                        list(arguments.posonlyargs)
                        + list(arguments.args)
                        + list(arguments.kwonlyargs)
                        + [arguments.vararg, arguments.kwarg]
                    ) if a is not None
                ]
                with self.subTest(function=node.name):
                    self.assertNotIn("latitude", declared)
                    self.assertNotIn("longitude", declared)

    def test_block_references_no_observer_identifier(self):
        referenced = {
            node.id for node in ast.walk(self.tree)
            if isinstance(node, ast.Name)
        }
        self.assertNotIn("latitude", referenced)
        self.assertNotIn("longitude", referenced)
        # The only longitude-bearing name is the certified substrate itself.
        self.assertEqual(
            {name for name in referenced if "longitude" in name.lower()},
            {"find_solar_longitude_event_after"},
        )

    def test_block_accesses_no_observer_attribute(self):
        attributes = {
            node.attr for node in ast.walk(self.tree)
            if isinstance(node, ast.Attribute)
        }
        self.assertNotIn("latitude", attributes)
        self.assertNotIn("longitude", attributes)

    def test_block_introduces_no_epsilon_or_tolerance(self):
        for token in ("epsilon", "tolerance", "atol", "rtol", "isclose",
                      "round("):
            with self.subTest(token=token):
                self.assertNotIn(token, self.executable.lower())

    def test_addressing_origin_is_the_start_of_the_year(self):
        constants = {
            n.targets[0].id: n.value.value for n in self.tree.body
            if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
        }
        self.assertEqual(constants["YEAR_ADDRESSING_MONTH"], 1)
        self.assertEqual(constants["YEAR_ADDRESSING_DAY"], 1)

    def test_frozen_code_is_still_present_ahead_of_the_block(self):
        source = inspect.getsource(astronomy_solver)
        preamble = source[:source.find(A3C3_BLOCK_MARKER)]
        for token in ("def find_solar_longitude_event_after",
                      "def find_solar_longitude_event_before",
                      "def find_sunset_bracket",
                      "def geocentric_search_frontier",
                      "def _canonical_quadrant_boundary",
                      "def find_equinox"):
            with self.subTest(token=token):
                self.assertIn(token, preamble)


class TestRouteSourceGuard(unittest.TestCase):
    def setUp(self):
        source = inspect.getsource(server)
        index = source.find(A3C3_ROUTE_BLOCK_MARKER)
        self.assertNotEqual(index, -1, "A3c-3 route block marker not found")
        following = GOVERNED_BLOCK_MARKER.search(
            source, index + len(A3C3_ROUTE_BLOCK_MARKER)
        )
        self.block = source[
            index:following.start() if following else len(source)
        ]
        self.executable = executable_source(
            inspect.getsource(server.solar_longitude_event_in_year)
        )

    def test_route_performs_no_astronomy_or_artifact_selection(self):
        for token in ("almanac", "find_discrete", "ts.", "wgs84",
                      "load_kernel", "choose_kernel_name", "seasons",
                      "kernel_coverage_tt"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.executable)

    def test_route_builds_no_response_and_duplicates_no_codec(self):
        for token in ("ttBits", "decode_tt_bits", "encode_tt_bits",
                      "struct", "return {"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.executable)
        self.assertIn("project_astronomical_event", self.executable)

    def test_route_catches_exactly_one_governed_type(self):
        tree = ast.parse(
            inspect.getsource(server.solar_longitude_event_in_year).strip()
        )
        handlers = [n for n in ast.walk(tree) if isinstance(n, ast.Try)]
        self.assertEqual(len(handlers), 1)
        self.assertEqual([h.type.id for h in handlers[0].handlers],
                         ["SunsetChronologyError"])

    def test_route_has_no_broad_exception_handler(self):
        for token in ("except Exception", "except BaseException", "except:"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.executable)

    def test_route_has_no_civil_or_hpc_machinery(self):
        for token in ("datetime", "parse_utc_datetime", "isoformat",
                      "timezone", "sky_year", "hpc", "sce", "telma",
                      "weekday", "equinox"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.executable.lower())

    def test_route_emits_no_404(self):
        self.assertNotIn("404", self.executable)


# --- 15. Existing contracts unchanged --------------------------------------


class TestExistingContractsUnchanged(unittest.TestCase):
    """A3c-1 and A3c-2 are additive neighbours, not dependencies."""

    A3C1_ROUTES = (
        ("/solar-longitude-event-before", "solar_longitude_event_before",
         ["ttBits", "kind"]),
        ("/solar-longitude-event-after", "solar_longitude_event_after",
         ["ttBits", "kind"]),
    )
    A3C2_ROUTES = (
        ("/sunset-bracket", "sunset_bracket",
         ["ttBits", "latitude", "longitude"]),
    )
    LEGACY_ROUTES = (
        ("/health", "health", []),
        ("/delta-t/{year}", "delta_t", ["year"]),
        ("/delta-t-bce/{year}", "delta_t_bce", ["year"]),
        ("/equinox/{year}", "equinox", ["year"]),
        ("/equinox-bce/{year}", "equinox_bce", ["year"]),
        ("/season-events", "season_events", ["year"]),
        ("/season-events-bce/{year}", "season_events_bce", ["year"]),
        ("/solar_longitude", "solar", ["date"]),
        ("/subsolar-point", "subsolar", ["date"]),
        ("/sunset", "sunset", ["date", "latitude", "longitude"]),
        ("/sunset-after", "sunset_after",
         ["afterUTC", "latitude", "longitude"]),
        ("/sunset-successor", "sunset_successor",
         ["cursor", "latitude", "longitude"]),
    )
    FRAMEWORK_ROUTES = frozenset(
        {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}
    )

    def test_every_preexisting_route_is_registered_unchanged(self):
        found = registered_routes()
        for path, name, parameters in (
            self.LEGACY_ROUTES + self.A3C1_ROUTES + self.A3C2_ROUTES
        ):
            with self.subTest(path=path):
                self.assertIn(path, found)
                methods, endpoint_name, endpoint_parameters = found[path]
                self.assertEqual(methods - {"HEAD"}, {"GET"})
                self.assertEqual(endpoint_name, name)
                self.assertEqual(endpoint_parameters, parameters)

    def test_a3c3_added_exactly_one_application_route(self):
        found = set(registered_routes()) - self.FRAMEWORK_ROUTES
        expected = {
            path for path, _n, _p in
            self.LEGACY_ROUTES + self.A3C1_ROUTES + self.A3C2_ROUTES
        } | {EVENT_IN_YEAR_PATH}
        self.assertEqual(found, expected)
        self.assertEqual(len(found), 16)

    def test_a3c1_routes_still_reject_a_year_parameter(self):
        for path, _name, parameters in self.A3C1_ROUTES:
            with self.subTest(path=path):
                self.assertNotIn("year", parameters)

    def test_a3c2_route_still_rejects_a_year_parameter(self):
        self.assertNotIn("year", registered_routes()["/sunset-bracket"][2])

    def test_a3c1_structured_detail_is_unchanged(self):
        with self.assertRaises(HTTPException) as caught:
            server.solar_longitude_event_after(ttBits="nope", kind="x")
        self.assertIsInstance(caught.exception.detail, dict)
        self.assertEqual(tuple(caught.exception.detail), ("reason", "message"))

    def test_legacy_failure_still_carries_string_detail(self):
        with self.assertRaises(HTTPException) as caught:
            server.equinox("not-a-year")
        self.assertEqual(caught.exception.status_code, 400)
        self.assertIsInstance(caught.exception.detail, str)

    def test_legacy_health_is_unchanged(self):
        response = server.health()
        self.assertEqual(set(response), {"status", "defaultKernel"})
        self.assertEqual(response["status"], "ok")

    def test_runtime_gate_still_precedes_every_astronomy_import(self):
        source = inspect.getsource(server)
        gate = source.find("verify_runtime_scientific_components()")
        self.assertNotEqual(gate, -1)
        self.assertLess(gate, source.find("from astronomy_solver import"))


if __name__ == "__main__":
    unittest.main()
