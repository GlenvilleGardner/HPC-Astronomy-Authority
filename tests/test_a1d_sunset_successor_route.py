"""A1d verification: the additive /sunset-successor route.

Test framework: standard-library unittest.

SCOPE

A1c joined real astronomy, the continuation witness and the two existing
sunset routes. A1d adds the route that consumes a witness. This file
therefore owns exactly one new surface - the HTTP contract of
/sunset-successor - and asserts that everything A1a, A1b and A1c already
certified continues to hold across it.

INDEPENDENT ORACLE DISCIPLINE

Golden instants, golden Terrestrial Time values, observer coordinates,
kernel names, reason codes and the digest prefix are declared here as
test-local literals. They are deliberately NOT imported from server,
astronomy_solver or sunset_cursor: importing them would make these tests
agree with a defective constant instead of detecting it.

The sunset predicate used as an oracle is built here directly from
skyfield.almanac. Tampered cursors are constructed from independently
implemented canonical JSON, SHA-256 and Base64URL, never from the
production encoder.

The shared timescale and kernel loader ARE imported from
astronomy_solver. They are repository infrastructure, and duplicating
them would load a second copy of a 119 MB kernel for no benefit.

ROUTE SEAM

The route functions are invoked directly, following A1c. httpx is not
installed, so starlette's TestClient is unavailable, and installing a
dependency in order to test is not permitted. Direct invocation
exercises the real route body, including the HTTPException mapping,
which is asserted through the exception object itself.

HANDOFF SEAM

The A1d -> A1b handoff is certified by observation, not by inference
from the answer. server.find_sunset_successor is temporarily replaced by
a recorder that DELEGATES to the real A1b solver, so the astronomical
result remains real and the exact call arguments become visible. No
astronomical result is fabricated, and the structural source-level
prohibition on datetime/ISO/Unix reconstruction is retained alongside
it as a complementary guard.

INDEPENDENT NEXT-DISTINCT CERTIFICATION

Golden equality proves the answer is the expected one. It does not by
itself prove the answer is the FIRST physical sunset transition after
the seed. That claim is established here against a crossing enumeration
built directly from skyfield.almanac, which never calls
find_sunset_successor and never derives its expectation from the
production successor implementation.

WHAT THIS FILE DOES NOT CLAIM

Nothing about Engine migration having occurred, about removal or
correction of the legacy one-hour /sunset-after guard, about DE440/DE441
routing boundaries, about BSP integrity, about authentication or
authorization, or about astronomical correctness beyond the instants and
observers exercised below.
"""

import ast
import base64
import hashlib
import inspect
import json
import math
import struct
import unittest
from unittest import mock

from fastapi import HTTPException
from skyfield import almanac
from skyfield.api import wgs84

import server
from astronomy_solver import (
    SunsetSuccessorError,
    find_next_sunset_after_utc,
    find_sunset_successor,
    load_kernel,
    ts,
)
from sunset_cursor import decode_sunset_cursor, encode_sunset_cursor


# --- Test-local oracles ----------------------------------------------------

REFERENCE_LATITUDE = 40.7406
REFERENCE_LONGITUDE = -73.9586
REFERENCE_KERNEL = "de440.bsp"

REFERENCE_DATE = "2027-03-20"
REFERENCE_MIDDAY_UTC = "2027-03-20T12:00:00+00:00"

# MEASURED - the sunset that closes 2027-03-20 for the reference
# observer, and the exact binary64 Terrestrial Time state it was found
# at. The same values are pinned independently by A1a, A1b and A1c.
SEED_SUNSET_UTC = "2027-03-20T23:07:45.858897+00:00"
SEED_TT = 2461485.4645259595

# MEASURED - the next three distinct crossings, continued from SEED_TT.
CHAIN_SUNSET_UTC = (
    "2027-03-21T23:08:49.679673+00:00",
    "2027-03-22T23:09:53.362048+00:00",
    "2027-03-23T23:10:56.922998+00:00",
)

# Yonkers - roughly 21 km away, close enough that an interval sanity
# check would not notice the substitution. Observer binding must.
ADJACENT_LATITUDE = 40.9312
ADJACENT_LONGITUDE = -73.8988

# Longyearbyen, Svalbard, during polar night: the Sun is down, and it
# neither rises nor sets inside the successor horizon.
POLAR_LATITUDE = 78.2232
POLAR_LONGITUDE = 15.6267
POLAR_TT = 2461754.500800741

REASON_MALFORMED_CURSOR = "MALFORMED_CURSOR"
REASON_CHECKSUM_MISMATCH = "CHECKSUM_MISMATCH"
REASON_INCOMPATIBLE_ENVIRONMENT = "INCOMPATIBLE_ENVIRONMENT"
REASON_INCOMPATIBLE_OBSERVER = "INCOMPATIBLE_OBSERVER"
REASON_CURSOR_STATE_INVALID = "CURSOR_STATE_INVALID"
REASON_CURSOR_NOT_POST_TRANSITION = "CURSOR_NOT_POST_TRANSITION"

DIGEST_PREFIX = "sha256:"

MALFORMED_CURSOR_TEXT = "not-a-cursor!"

CHAIN_STEPS = len(CHAIN_SUNSET_UTC)

SECONDS_PER_DAY = 86400.0

# Test-local mirror of the certified successor horizon. Declared as a
# literal rather than imported, so a defective production constant is
# detected instead of agreed with.
SEARCH_SPAN_DAYS = 3.0

# The independent enumeration deliberately uses a DIFFERENT bracket from
# the production search: it opens before the seed state and closes after
# the returned successor. A bracket that merely restated the production
# one could not detect a wrong bracket choice, and opening earlier also
# exposes the originating crossing, so "the seed's own root was not
# re-reported" becomes an observation rather than an assumption.
ORACLE_BACKSTEP_DAYS = 0.25
ORACLE_OVERSHOOT_DAYS = 0.25

# MEASURED, and used ONLY to compare two independent computations of the
# same root that converge from different brackets.
#
# This is NOT an event-identity rule. Nothing here is supplied to any
# solver, no production code consults it, find_sunset_successor still
# carries no gap constant, and the legacy 3600-second guard is untouched.
# It is justified by the separation asserted alongside it: consecutive
# sunsets for this observer are more than 20 hours apart, so agreement
# below one second cannot be a different crossing. Bit equality is
# asserted separately, where the brackets genuinely coincide.
MAX_ORACLE_DEVIATION_SECONDS = 1.0
MIN_SUNSET_SEPARATION_HOURS = 20.0

# Signed zero: the encoder canonicalizes -0.0 to +0.0, so the value the
# decoder returns is distinguishable from the raw request value by sign
# bit alone. This is the case where "bound observer" and "request
# observer" differ observably.
NEGATIVE_ZERO_LONGITUDE = -0.0


def reference_predicate(latitude=REFERENCE_LATITUDE,
                        longitude=REFERENCE_LONGITUDE):
    """Build the sunset predicate independently of the code under test."""
    return almanac.sunrise_sunset(
        load_kernel(REFERENCE_KERNEL), wgs84.latlon(latitude, longitude)
    )


def sun_is_up_at(tt, latitude=REFERENCE_LATITUDE,
                 longitude=REFERENCE_LONGITUDE):
    """Evaluate the independent predicate at an exact TT state."""
    return bool(reference_predicate(latitude, longitude)(ts.tt_jd(tt)))


def independent_crossings(start_tt, end_tt, latitude=REFERENCE_LATITUDE,
                          longitude=REFERENCE_LONGITUDE):
    """Enumerate (tt, sun_is_up) for every crossing in an oracle search.

    Built from skyfield.almanac directly. find_sunset_successor is never
    called, and no expectation is taken from it.
    """
    times, events = almanac.find_discrete(
        ts.tt_jd(start_tt),
        ts.tt_jd(end_tt),
        reference_predicate(latitude, longitude),
    )
    return [(float(t.tt), bool(event)) for t, event in zip(times, events)]


def sunset_crossings(crossings):
    """Keep only sun-down transitions - the sunsets."""
    return [tt for tt, sun_is_up in crossings if not sun_is_up]


def sunrise_crossings(crossings):
    return [tt for tt, sun_is_up in crossings if sun_is_up]


def bits(value):
    """IEEE-754 binary64 bit pattern.

    Equality on the bit pattern is stricter than float equality: it also
    separates -0.0 from +0.0, which is exactly the distinction the
    observer-canonicalization claim rests on.
    """
    return struct.pack(">d", value)


class RecordingSolver:
    """Record the exact call, then delegate to the real A1b solver.

    The astronomical answer is produced by production code. Only the
    arguments crossing the A1d -> A1b seam are captured, so normal
    successor behavior remains available and nothing is fabricated.
    """

    def __init__(self):
        self.calls = []

    def __call__(self, tt_value, latitude, longitude):
        self.calls.append((tt_value, latitude, longitude))
        return find_sunset_successor(tt_value, latitude, longitude)


def independent_canonical_bytes(obj):
    """Re-implement the canonical encoding, independently of production."""
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def independent_digest(obj):
    return DIGEST_PREFIX + hashlib.sha256(
        independent_canonical_bytes(obj)
    ).hexdigest()


def independent_encode_bytes(raw):
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def envelope_of(cursor):
    padded = cursor + ("=" * (-len(cursor) % 4))
    return json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))


def rebuilt_cursor(envelope):
    """Re-encode a modified envelope canonically, bypassing production.

    Canonical transport is preserved deliberately, so the decoder reaches
    the classification under test instead of rejecting the cursor as
    malformed transport.
    """
    return independent_encode_bytes(independent_canonical_bytes(envelope))


def tampered_cursor(field, value, recompute_checksum):
    envelope = envelope_of(seed_cursor())
    envelope["payload"][field] = value
    if recompute_checksum:
        envelope["checksum"] = independent_digest(envelope["payload"])
    return rebuilt_cursor(envelope)


def seed_cursor():
    return server.sunset(
        date=REFERENCE_DATE,
        latitude=REFERENCE_LATITUDE,
        longitude=REFERENCE_LONGITUDE,
    )["cursor"]


def sunset_after_cursor():
    return server.sunset_after(
        afterUTC=REFERENCE_MIDDAY_UTC,
        latitude=REFERENCE_LATITUDE,
        longitude=REFERENCE_LONGITUDE,
    )["cursor"]


def successor_response(cursor, latitude=REFERENCE_LATITUDE,
                       longitude=REFERENCE_LONGITUDE):
    return server.sunset_successor(
        cursor=cursor, latitude=latitude, longitude=longitude
    )


def decode_reference(cursor):
    return decode_sunset_cursor(
        cursor, latitude=REFERENCE_LATITUDE, longitude=REFERENCE_LONGITUDE
    )


def route_body_source():
    """Return the route's executable body, docstring excluded.

    The prohibition governs executable logic, not prose: the docstring
    legitimately names the representations the route must never use, and
    matching against it would read a promise as a violation. This mirrors
    the A1b body oracle.
    """
    source = inspect.getsource(server.sunset_successor)
    lines = source.splitlines(keepends=True)

    function = ast.parse(source).body[0]
    first = function.body[0]

    if (isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)):
        del lines[first.lineno - 1:first.end_lineno]

    return "".join(lines)


class RouteRejectionMixin:

    def assert_route_rejects(self, cursor, status, reason,
                             latitude=REFERENCE_LATITUDE,
                             longitude=REFERENCE_LONGITUDE):
        with self.assertRaises(HTTPException) as caught:
            successor_response(cursor, latitude, longitude)

        error = caught.exception
        self.assertEqual(error.status_code, status)
        self.assertIsInstance(error.detail, str)
        self.assertIn(reason, error.detail)
        return error


# --- A. /sunset cursor seeds /sunset-successor -----------------------------


class TestSunsetSeedsSuccessorRoute(unittest.TestCase):

    def setUp(self):
        self.seed = server.sunset(
            date=REFERENCE_DATE,
            latitude=REFERENCE_LATITUDE,
            longitude=REFERENCE_LONGITUDE,
        )
        self.response = successor_response(self.seed["cursor"])

    def test_seed_route_emits_a_usable_cursor(self):
        self.assertEqual(self.seed["sunsetUTC"], SEED_SUNSET_UTC)
        self.assertIsInstance(self.seed["cursor"], str)

    def test_successor_matches_golden(self):
        self.assertEqual(self.response["sunsetUTC"], CHAIN_SUNSET_UTC[0])
        self.assertEqual(self.response["kernel"], REFERENCE_KERNEL)

    def test_response_reports_the_bound_observer(self):
        self.assertEqual(self.response["latitude"], REFERENCE_LATITUDE)
        self.assertEqual(self.response["longitude"], REFERENCE_LONGITUDE)

    def test_successor_is_strictly_forward_and_distinct(self):
        self.assertGreater(
            self.response["sunsetUTC"], self.seed["sunsetUTC"]
        )
        self.assertNotEqual(
            self.response["sunsetUTC"], self.seed["sunsetUTC"]
        )


# --- B. /sunset-after cursor seeds /sunset-successor -----------------------


class TestSunsetAfterSeedsSuccessorRoute(unittest.TestCase):

    def setUp(self):
        self.seed = server.sunset_after(
            afterUTC=REFERENCE_MIDDAY_UTC,
            latitude=REFERENCE_LATITUDE,
            longitude=REFERENCE_LONGITUDE,
        )
        self.response = successor_response(self.seed["cursor"])

    def test_successor_matches_golden(self):
        self.assertEqual(self.seed["sunsetUTC"], SEED_SUNSET_UTC)
        self.assertEqual(self.response["sunsetUTC"], CHAIN_SUNSET_UTC[0])
        self.assertEqual(self.response["kernel"], REFERENCE_KERNEL)

    def test_successor_is_strictly_forward_and_distinct(self):
        decoded_seed = decode_reference(self.seed["cursor"])
        decoded_step = decode_reference(self.response["cursor"])

        self.assertGreater(decoded_step.tt, decoded_seed.tt)
        self.assertNotEqual(
            self.response["sunsetUTC"], self.seed["sunsetUTC"]
        )


# --- C. The emitted cursor chains ------------------------------------------


class TestSuccessorCursorChains(unittest.TestCase):
    """The chain is advanced ONLY by the cursor the route emits.

    No UTC datetime, ISO string, Unix timestamp or millisecond value is
    ever fed back into the search.
    """

    @classmethod
    def setUpClass(cls):
        cls.steps = []
        cursor = seed_cursor()
        for _ in range(CHAIN_STEPS):
            response = successor_response(cursor)
            cls.steps.append(response)
            cursor = response["cursor"]

    def test_every_step_emits_a_continuation_cursor(self):
        for index, response in enumerate(self.steps, 1):
            with self.subTest(step=index):
                self.assertIsInstance(response["cursor"], str)
                self.assertNotIn("cursorUnavailableReason", response)

    def test_chain_matches_golden_sequence(self):
        self.assertEqual(
            [response["sunsetUTC"] for response in self.steps],
            list(CHAIN_SUNSET_UTC),
        )

    def test_chain_is_strictly_forward_and_distinct(self):
        states = [
            decode_reference(response["cursor"]).tt
            for response in self.steps
        ]

        self.assertEqual(states, sorted(states))
        self.assertEqual(len(set(states)), CHAIN_STEPS)
        self.assertNotIn(SEED_TT, states)


# --- D. Exact TT survives the HTTP boundary --------------------------------


class TestExactStateSurvivesTheRouteBoundary(unittest.TestCase):

    def setUp(self):
        self.seed = seed_cursor()
        self.decoded_seed = decode_reference(self.seed)
        self.direct = find_sunset_successor(
            self.decoded_seed.tt, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )
        self.response = successor_response(self.seed)
        self.decoded_step = decode_reference(self.response["cursor"])

    def test_emitted_cursor_carries_the_exact_successor_state(self):
        # Binary64 equality, not tolerance. The route must hand the
        # solver's own state back out, unrounded.
        self.assertEqual(self.decoded_seed.tt, SEED_TT)
        self.assertEqual(self.decoded_step.tt, self.direct.tt)

    def test_emitted_cursor_reproduces_the_emitted_sunset_utc(self):
        self.assertEqual(
            ts.tt_jd(self.decoded_step.tt).utc_datetime().isoformat(),
            self.response["sunsetUTC"],
        )

    def test_route_never_reconstructs_time_from_utc(self):
        # Structural guard, complementary to the observed handoff in
        # TestExactStateHandoffSeam. That class proves what the route
        # passed; this one proves the route has no machinery capable of
        # deriving it from a calendar representation in the first place.
        body = route_body_source()

        # The oracle must not pass by inspecting nothing.
        self.assertIn("find_sunset_successor(", body)
        self.assertIn("decode_sunset_cursor(", body)

        for forbidden in (
            "parse_utc_datetime",
            "fromisoformat",
            "fromtimestamp",
            "timestamp(",
            "timedelta",
            "datetime(",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, body)


# --- D2. The exact-TT handoff, observed at the seam -------------------------


class TestExactStateHandoffSeam(unittest.TestCase):
    """Certify the call itself, not merely its consequences.

    server.sunset_successor resolves find_sunset_successor as a module
    global of server, so patching server.find_sunset_successor
    intercepts exactly that call. astronomy_solver is not modified and
    production exception handling is not broadened.
    """

    def record(self, cursor, latitude=REFERENCE_LATITUDE,
               longitude=REFERENCE_LONGITUDE):
        recorder = RecordingSolver()
        with mock.patch.object(
            server, "find_sunset_successor", recorder
        ):
            response = successor_response(cursor, latitude, longitude)
        return recorder, response

    def test_solver_receives_the_decoded_tt_exactly(self):
        cursor = seed_cursor()
        decoded = decode_reference(cursor)
        recorder, _response = self.record(cursor)

        self.assertEqual(len(recorder.calls), 1)
        tt_argument = recorder.calls[0][0]

        # Exact binary64 equality, not tolerance, asserted twice: on the
        # value and on its bit pattern.
        self.assertEqual(tt_argument, decoded.tt)
        self.assertEqual(bits(tt_argument), bits(decoded.tt))
        self.assertEqual(tt_argument, SEED_TT)

    def test_solver_receives_the_bound_observer(self):
        cursor = seed_cursor()
        decoded = decode_reference(cursor)
        recorder, _response = self.record(cursor)

        _tt, latitude, longitude = recorder.calls[0]

        self.assertEqual(bits(latitude), bits(decoded.latitude))
        self.assertEqual(bits(longitude), bits(decoded.longitude))

    def test_decoder_canonicalization_governs_the_observer(self):
        # The witness binds longitude +0.0; the request presents -0.0.
        # The two are equal under ==, so only the sign bit distinguishes
        # the decoder's canonical value from the raw request value.
        cursor = encode_sunset_cursor(
            tt=SEED_TT,
            latitude=REFERENCE_LATITUDE,
            longitude=NEGATIVE_ZERO_LONGITUDE,
        )
        decoded = decode_sunset_cursor(
            cursor,
            latitude=REFERENCE_LATITUDE,
            longitude=NEGATIVE_ZERO_LONGITUDE,
        )
        self.assertEqual(math.copysign(1.0, decoded.longitude), 1.0)

        recorder = RecordingSolver()
        with mock.patch.object(
            server, "find_sunset_successor", recorder
        ):
            # This case certifies ARGUMENT CONSTRUCTION only. Whether
            # this observer has a successor inside the horizon is a
            # separate astronomical question, certified elsewhere, so
            # either outcome of the delegated call is acceptable here -
            # the recording has already happened either way.
            try:
                successor_response(
                    cursor,
                    latitude=REFERENCE_LATITUDE,
                    longitude=NEGATIVE_ZERO_LONGITUDE,
                )
            except HTTPException:
                pass

        self.assertEqual(len(recorder.calls), 1)
        _tt, _latitude, longitude = recorder.calls[0]

        # The bound +0.0 was forwarded, not the request's -0.0.
        self.assertEqual(math.copysign(1.0, longitude), 1.0)
        self.assertEqual(bits(longitude), bits(decoded.longitude))
        self.assertNotEqual(bits(longitude), bits(NEGATIVE_ZERO_LONGITUDE))

    def test_delegation_preserves_the_real_successor(self):
        # The recorder wraps rather than replaces: the answer is still
        # the real astronomical one.
        _recorder, response = self.record(seed_cursor())

        self.assertEqual(response["sunsetUTC"], CHAIN_SUNSET_UTC[0])
        self.assertEqual(response["kernel"], REFERENCE_KERNEL)
        self.assertIsInstance(response["cursor"], str)

    def test_production_solver_is_restored(self):
        self.record(seed_cursor())
        self.assertIs(server.find_sunset_successor, find_sunset_successor)


# --- E. A1a fail-closed rejections map to 400 ------------------------------


class TestCursorFailClosedMapping(RouteRejectionMixin, unittest.TestCase):

    def test_observer_mismatch_is_400(self):
        self.assert_route_rejects(
            seed_cursor(),
            400,
            REASON_INCOMPATIBLE_OBSERVER,
            latitude=ADJACENT_LATITUDE,
            longitude=ADJACENT_LONGITUDE,
        )

    def test_malformed_cursor_is_400(self):
        self.assert_route_rejects(
            MALFORMED_CURSOR_TEXT, 400, REASON_MALFORMED_CURSOR
        )

    def test_checksum_mismatch_is_400(self):
        # The payload is altered and the recorded checksum is left stale,
        # so the cursor is well formed but no longer self-consistent.
        cursor = tampered_cursor(
            "ttBits", "4142C796BB75962F", recompute_checksum=False
        )
        self.assertNotEqual(cursor, seed_cursor())
        self.assert_route_rejects(cursor, 400, REASON_CHECKSUM_MISMATCH)

    def test_checksum_mismatch_is_not_an_authorization_outcome(self):
        # The checksum detects accidental corruption. It is not
        # authentication, authorization or proof of origin, so it must
        # never be reported as one.
        cursor = tampered_cursor(
            "ttBits", "4142C796BB75962F", recompute_checksum=False
        )
        with self.assertRaises(HTTPException) as caught:
            successor_response(cursor)

        self.assertNotIn(caught.exception.status_code, (401, 403))

    def test_incompatible_environment_is_400(self):
        # The checksum is recomputed, so the cursor is internally
        # consistent and fails only on the environment binding.
        cursor = tampered_cursor(
            "skyfieldVersion", "0.0", recompute_checksum=True
        )
        self.assert_route_rejects(
            cursor, 400, REASON_INCOMPATIBLE_ENVIRONMENT
        )


# --- E2. Independent next-distinct physical certification ------------------


class TestIndependentNextDistinctCertification(unittest.TestCase):
    """Establish that the route returns the FIRST sunset after the seed.

    Every expectation here is computed from skyfield.almanac directly.
    find_sunset_successor is not called, and no expected event is taken
    from the production successor implementation.
    """

    @classmethod
    def setUpClass(cls):
        cls.seed_cursor = seed_cursor()
        cls.seed_tt = decode_reference(cls.seed_cursor).tt
        cls.response = successor_response(cls.seed_cursor)
        cls.result_tt = decode_reference(cls.response["cursor"]).tt

        # Bracket 1 - opens at the seed, matching the certified horizon.
        cls.aligned = independent_crossings(
            cls.seed_tt, cls.seed_tt + SEARCH_SPAN_DAYS
        )

        # Bracket 2 - deliberately different: opens before the seed and
        # closes after the result.
        cls.wide = independent_crossings(
            cls.seed_tt - ORACLE_BACKSTEP_DAYS,
            cls.result_tt + ORACLE_OVERSHOOT_DAYS,
        )

    def test_seed_state_is_post_transition(self):
        # Isolates what follows: the seed really is the far side of a
        # sunset, confirmed by the independent predicate.
        self.assertFalse(sun_is_up_at(self.seed_tt))
        self.assertTrue(
            sun_is_up_at(self.seed_tt - 1.0 / SECONDS_PER_DAY)
        )

    def test_result_is_the_first_sunset_in_the_aligned_oracle(self):
        sunsets = sunset_crossings(self.aligned)

        self.assertTrue(sunsets)
        # The brackets coincide here, so bit equality is the correct
        # standard and no numerical allowance is used.
        self.assertEqual(sunsets[0], self.result_tt)
        self.assertEqual(bits(sunsets[0]), bits(self.result_tt))
        self.assertEqual(
            ts.tt_jd(sunsets[0]).utc_datetime().isoformat(),
            self.response["sunsetUTC"],
        )

    def test_no_sunset_is_skipped_between_seed_and_result(self):
        # Window: from the seed up to the route's answer, extended by
        # the comparison bound so that the wide bracket's own
        # recomputation of that same root falls inside it rather than
        # being counted as a separate crossing.
        window_end = self.result_tt + (
            MAX_ORACLE_DEVIATION_SECONDS / SECONDS_PER_DAY
        )
        within = [
            tt
            for tt in sunset_crossings(self.wide)
            if self.seed_tt < tt <= window_end
        ]

        # Exactly one sunset lies in that window, and it is the one the
        # route returned. A skipped crossing would make this two.
        self.assertEqual(len(within), 1)
        self.assertLess(
            abs(within[0] - self.result_tt) * SECONDS_PER_DAY,
            MAX_ORACLE_DEVIATION_SECONDS,
        )

    def test_the_two_crossings_are_separated_by_real_daylight(self):
        # Justifies the comparison above: the earliest sunset after the
        # seed is a genuinely distinct crossing, with a sunrise between
        # them and more than twenty hours of separation.
        between = [
            tt
            for tt in sunrise_crossings(self.wide)
            if self.seed_tt < tt < self.result_tt
        ]
        self.assertTrue(between)

        separation = (self.result_tt - self.seed_tt) * 24.0
        self.assertGreater(separation, MIN_SUNSET_SEPARATION_HOURS)

    def test_the_originating_crossing_is_not_rediscovered(self):
        # The wide bracket opens before the seed, so the originating
        # crossing is visible to the oracle and can be shown NOT to be
        # what the route returned.
        origin = [
            tt for tt in sunset_crossings(self.wide) if tt < self.seed_tt
        ]
        self.assertTrue(origin)

        self.assertGreater(self.result_tt, self.seed_tt)
        self.assertNotEqual(bits(self.result_tt), bits(self.seed_tt))

        # And the seed state itself is never reported as a crossing by a
        # search that starts from it: a proven sun-down start presents no
        # sign change at its own root.
        self.assertNotIn(self.seed_tt, [tt for tt, _up in self.aligned])


# --- F. A1b fail-closed rejections map to 400 ------------------------------


class TestSuccessorFailClosedMapping(RouteRejectionMixin,
                                     unittest.TestCase):

    def test_not_post_transition_is_400(self):
        daylight_tt = float(ts.utc(2027, 3, 20, 17, 0, 0).tt)

        # Confirmed against the independent predicate, not against the
        # gate inside find_sunset_successor.
        self.assertTrue(sun_is_up_at(daylight_tt))

        cursor = encode_sunset_cursor(
            tt=daylight_tt,
            latitude=REFERENCE_LATITUDE,
            longitude=REFERENCE_LONGITUDE,
        )
        self.assert_route_rejects(
            cursor, 400, REASON_CURSOR_NOT_POST_TRANSITION
        )

    def test_cursor_state_invalid_is_400(self):
        # A validated witness cannot carry a non-finite or non-numeric
        # state, so this reason is not reachable through the route with
        # real inputs. The mapping is still asserted, by patching the
        # seam the route actually calls. sunset_cursor and
        # astronomy_solver are not modified, and production exception
        # handling is not broadened.
        with mock.patch.object(
            server,
            "find_sunset_successor",
            side_effect=SunsetSuccessorError(
                REASON_CURSOR_STATE_INVALID, "induced invalid state"
            ),
        ):
            self.assert_route_rejects(
                seed_cursor(), 400, REASON_CURSOR_STATE_INVALID
            )


# --- G. No event inside the horizon is 404, not a rejection ----------------


class TestPolarObserverYieldsNotFound(unittest.TestCase):

    def polar_cursor(self):
        return encode_sunset_cursor(
            tt=POLAR_TT, latitude=POLAR_LATITUDE, longitude=POLAR_LONGITUDE
        )

    def test_polar_state_is_post_transition(self):
        # Isolates the branch under test: this state passes
        # post-transition validation, so a 404 means "no event inside the
        # horizon", not "rejected before searching".
        self.assertFalse(
            sun_is_up_at(POLAR_TT, POLAR_LATITUDE, POLAR_LONGITUDE)
        )

    def test_polar_cursor_yields_404(self):
        with self.assertRaises(HTTPException) as caught:
            server.sunset_successor(
                cursor=self.polar_cursor(),
                latitude=POLAR_LATITUDE,
                longitude=POLAR_LONGITUDE,
            )

        self.assertEqual(caught.exception.status_code, 404)


# --- H. Existing behavior is untouched -------------------------------------


class TestExistingBehaviorPreserved(unittest.TestCase):

    def test_sunset_route_unchanged(self):
        response = server.sunset(
            date=REFERENCE_DATE,
            latitude=REFERENCE_LATITUDE,
            longitude=REFERENCE_LONGITUDE,
        )

        self.assertEqual(response["date"], REFERENCE_DATE)
        self.assertEqual(response["sunsetUTC"], SEED_SUNSET_UTC)
        self.assertEqual(response["kernel"], REFERENCE_KERNEL)
        self.assertEqual(response["latitude"], REFERENCE_LATITUDE)
        self.assertEqual(response["longitude"], REFERENCE_LONGITUDE)

    def test_sunset_after_route_unchanged(self):
        response = server.sunset_after(
            afterUTC=REFERENCE_MIDDAY_UTC,
            latitude=REFERENCE_LATITUDE,
            longitude=REFERENCE_LONGITUDE,
        )

        self.assertEqual(response["afterUTC"], REFERENCE_MIDDAY_UTC)
        self.assertEqual(response["sunsetUTC"], SEED_SUNSET_UTC)
        self.assertEqual(response["kernel"], REFERENCE_KERNEL)

    def test_sunset_route_still_emits_a_cursor(self):
        # A1c non-regression at the surface A1d depends on. The full A1c
        # suite is rerun separately and is not duplicated here.
        response = server.sunset(
            date=REFERENCE_DATE,
            latitude=REFERENCE_LATITUDE,
            longitude=REFERENCE_LONGITUDE,
        )

        self.assertIsInstance(response["cursor"], str)
        self.assertNotIn("cursorUnavailableReason", response)

    def test_sunset_after_route_still_emits_a_cursor(self):
        response = server.sunset_after(
            afterUTC=REFERENCE_MIDDAY_UTC,
            latitude=REFERENCE_LATITUDE,
            longitude=REFERENCE_LONGITUDE,
        )

        self.assertIsInstance(response["cursor"], str)
        self.assertNotIn("cursorUnavailableReason", response)

    def test_legacy_one_hour_guard_still_present(self):
        source = inspect.getsource(find_next_sunset_after_utc)

        self.assertIn("min_gap_seconds = 3600", source)
        self.assertIn("after_timestamp + min_gap_seconds", source)

    def test_successor_route_introduces_no_gap_constant(self):
        body = route_body_source()

        self.assertIn("status_code=404", body)

        for forbidden in (
            "3600",
            "min_gap",
            "EPSILON",
            "tolerance",
            "seconds=",
            "abs(",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
