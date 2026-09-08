"""A1e verification: the published A1a->A1d continuation architecture as
one composed Authority capability.

Test framework: standard-library unittest.

SCOPE

A1a certified the continuation witness, A1b the successor solver, A1c
cursor emission from the two existing sunset routes, and A1d the
additive /sunset-successor route. Each was certified as a component or
as a single seam. This file certifies them COMPOSED: a client seeds a
chain from a published route, then advances it repeatedly using only
the cursors the Authority emits.

The distinction matters. A1d certified its first hop against an
independent astronomical oracle; its later hops were compared against
pinned golden instants. A golden is a recorded past measurement, not
evidence recomputable at test time. Because a migrating consumer will
depend on the CHAIN, every hop here is certified against astronomy
enumerated during the test run.

This file therefore introduces NO pinned sunset instant and NO pinned
Terrestrial Time literal. The only astronomical literals are observer
coordinates and the expected kernel routing name.

INDEPENDENT ORACLE DISCIPLINE

The sunset predicate is built here directly from skyfield.almanac.
Tampered cursors are built from independently implemented canonical
JSON, SHA-256 and Base64URL. The production successor solver is never
used as its own oracle, and no expectation is derived from it.

The A1d oracle METHODOLOGY is reused deliberately - two brackets, one
aligned with the production horizon permitting bit equality, one
opening earlier and closing later to detect a wrong bracket - but the
A1d test module is NOT imported. Cross-importing test modules is not a
convention here, and sharing literals would defeat the independence the
oracle exists to provide.

The shared timescale and kernel loader ARE imported from
astronomy_solver. They are repository infrastructure, and duplicating
them would load a second copy of a 119 MB kernel for no benefit.

ROUTE SEAM

Route functions are invoked directly, following A1c and A1d. httpx is
not installed, so starlette's TestClient is unavailable, and installing
a dependency in order to test is not permitted. HTTP status behavior is
asserted through the raised HTTPException.

MAX_ORACLE_DEVIATION_SECONDS

A test-side numerical comparison bound between two INDEPENDENTLY
RECOMPUTED roots that converge from different brackets. It is not a
solver tolerance, not an event-identity definition, not a statement of
astronomical uncertainty, and not a production constant. It is never
passed to any solver. Every use of it sits inside, or beside, the
assertions of physical separation (>20 hours) and intervening sunrise
which together make a same-crossing reading the only possible one.
Where the brackets coincide, bit equality is required instead.

CERTIFICATION SCOPE - READ BEFORE RELYING ON THIS FILE

A1e certifies COMPOSITION, not universality. The evidence here is
bounded to the observers and the epoch actually exercised below: one
mid-latitude northern observer, one signed-zero-longitude observer at
the same latitude, and a chain of consecutive sunsets in 2027-03 under
a single ephemeris kernel.

Nothing here certifies geographic generality, temporal generality,
ephemeris-routing-boundary behavior, or deep-time behavior.

WHAT THIS FILE DOES NOT CLAIM

Nothing about Engine migration; about any semantic correction to
/sunset-after or the removal or alteration of its legacy 3600-second
guard; about Zimrah Health integration readiness; about coordinate-
domain policy; about Infinity/NaN policy; about DE440/DE441
routing-boundary behavior; about A0.4 BSP integrity; about EC2
deployment reproducibility; about global certification; about deep-time
certification; or about Delta-T / IERS uncertainty certification.
"""

import ast
import base64
import hashlib
import inspect
import json
import math
import struct
import unittest
from collections import namedtuple

from fastapi import HTTPException
from skyfield import almanac
from skyfield.api import wgs84

import server
from astronomy_solver import find_next_sunset_after_utc, load_kernel, ts
from sunset_cursor import decode_sunset_cursor


# --- Test-local oracles ----------------------------------------------------

REFERENCE_LATITUDE = 40.7406
REFERENCE_LONGITUDE = -73.9586
REFERENCE_KERNEL = "de440.bsp"

REFERENCE_DATE = "2027-03-20"
REFERENCE_MIDDAY_UTC = "2027-03-20T12:00:00+00:00"

# Yonkers - roughly 21 km away, close enough that an interval sanity
# check would not notice the substitution. Observer binding must.
ADJACENT_LATITUDE = 40.9312
ADJACENT_LONGITUDE = -73.8988

# Signed zero: the encoder canonicalizes -0.0 to +0.0, so the bound
# value is distinguishable from the raw request value by sign bit alone.
NEGATIVE_ZERO_LONGITUDE = -0.0

# Four hops exceeds the three-hop chain A1d certified, so this gate is
# not a restatement of A1d, and every hop is independently certified
# rather than compared to a recorded instant.
CHAIN_HOPS = 4
SUNSET_AFTER_HOPS = 3

SECONDS_PER_DAY = 86400.0

# Test-local mirror of the certified successor horizon. Declared as a
# literal rather than imported, so a defective production constant is
# detected instead of agreed with.
SEARCH_SPAN_DAYS = 3.0

# The wide enumeration deliberately uses a DIFFERENT bracket from the
# production search: it opens before the hop's predecessor and closes
# after the hop. Opening earlier also exposes the originating crossing,
# so "the predecessor's own root was not re-reported" becomes an
# observation rather than an assumption.
ORACLE_BACKSTEP_DAYS = 0.25
ORACLE_OVERSHOOT_DAYS = 0.25

# TEST-SIDE COMPARISON BOUND ONLY. See the module docstring.
#
# Used solely to decide whether two independent computations of ONE root,
# converging from different bracket starts, found the same root. It is
# never supplied to a solver, no production code consults it, and it
# defines no event identity. Every use is accompanied by the separation
# and intervening-sunrise assertions that make a same-crossing reading
# the only available one.
MAX_ORACLE_DEVIATION_SECONDS = 1.0
MIN_SUNSET_SEPARATION_HOURS = 20.0

REASON_MALFORMED_CURSOR = "MALFORMED_CURSOR"
REASON_CHECKSUM_MISMATCH = "CHECKSUM_MISMATCH"
REASON_INCOMPATIBLE_ENVIRONMENT = "INCOMPATIBLE_ENVIRONMENT"
REASON_INCOMPATIBLE_OBSERVER = "INCOMPATIBLE_OBSERVER"

DIGEST_PREFIX = "sha256:"

MALFORMED_CURSOR_TEXT = "not-a-cursor!"

# One hop of a composed chain: the state it continued FROM, the route
# response, the cursor that response emitted, and the exact binary64 TT
# that cursor carries.
Hop = namedtuple("Hop", ("previous_tt", "response", "cursor", "tt"))


# --- Independent astronomical oracle ---------------------------------------


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


def crossings(start_tt, end_tt, latitude=REFERENCE_LATITUDE,
              longitude=REFERENCE_LONGITUDE):
    """Enumerate (tt, sun_is_up) for every crossing in an oracle search.

    Built from skyfield.almanac directly. The production successor
    solver is never called, and no expectation is taken from it.
    """
    times, events = almanac.find_discrete(
        ts.tt_jd(start_tt),
        ts.tt_jd(end_tt),
        reference_predicate(latitude, longitude),
    )
    return [(float(t.tt), bool(event)) for t, event in zip(times, events)]


def sunsets_in(crossing_list):
    """Keep only sun-down transitions - the sunsets."""
    return [tt for tt, sun_is_up in crossing_list if not sun_is_up]


def sunrises_in(crossing_list):
    return [tt for tt, sun_is_up in crossing_list if sun_is_up]


def bits(value):
    """IEEE-754 binary64 bit pattern.

    Stricter than float equality: it also separates -0.0 from +0.0,
    which is what the observer-canonicalization claim rests on.
    """
    return struct.pack(">d", value)


# --- Independent cursor construction (never the production encoder) --------


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


def tampered(cursor, field, value, recompute_checksum):
    """Alter one payload field of an EMITTED cursor and re-encode.

    Canonical transport is preserved deliberately, so the decoder
    reaches the classification under test rather than rejecting the
    cursor as malformed transport.
    """
    envelope = envelope_of(cursor)
    envelope["payload"][field] = value
    if recompute_checksum:
        envelope["checksum"] = independent_digest(envelope["payload"])
    return independent_encode_bytes(independent_canonical_bytes(envelope))


# --- Composed-chain helpers ------------------------------------------------


def sunset_seed(latitude=REFERENCE_LATITUDE, longitude=REFERENCE_LONGITUDE):
    return server.sunset(
        date=REFERENCE_DATE, latitude=latitude, longitude=longitude
    )


def sunset_after_seed(latitude=REFERENCE_LATITUDE,
                      longitude=REFERENCE_LONGITUDE):
    return server.sunset_after(
        afterUTC=REFERENCE_MIDDAY_UTC, latitude=latitude, longitude=longitude
    )


def successor_response(cursor, latitude=REFERENCE_LATITUDE,
                       longitude=REFERENCE_LONGITUDE):
    return server.sunset_successor(
        cursor=cursor, latitude=latitude, longitude=longitude
    )


def decode_for(cursor, latitude=REFERENCE_LATITUDE,
               longitude=REFERENCE_LONGITUDE):
    return decode_sunset_cursor(
        cursor, latitude=latitude, longitude=longitude
    )


def walk_chain(seed_response, hops, latitude=REFERENCE_LATITUDE,
               longitude=REFERENCE_LONGITUDE):
    """Advance a chain using ONLY the cursors the Authority emits.

    No UTC datetime, ISO string, Unix timestamp or millisecond value is
    ever fed back into the search: each hop is entered with the previous
    hop's opaque cursor and nothing else.
    """
    cursor = seed_response["cursor"]
    previous_tt = decode_for(cursor, latitude, longitude).tt

    walked = []
    for _ in range(hops):
        response = successor_response(cursor, latitude, longitude)
        cursor = response["cursor"]
        current_tt = decode_for(cursor, latitude, longitude).tt

        walked.append(
            Hop(
                previous_tt=previous_tt,
                response=response,
                cursor=cursor,
                tt=current_tt,
            )
        )
        previous_tt = current_tt

    return walked


def function_body_source(function):
    """Return a function's executable body, docstring excluded.

    The prohibition governs executable logic, not prose: these
    docstrings legitimately name the representations the code must never
    use, and matching against them would read a promise as a violation.
    This mirrors the A1b and A1d body oracles.
    """
    source = inspect.getsource(function)
    lines = source.splitlines(keepends=True)

    parsed = ast.parse(source).body[0]
    first = parsed.body[0]

    if (isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)):
        del lines[first.lineno - 1:first.end_lineno]

    return "".join(lines)


class HopOracleMixin:
    """Per-hop independent certification, reused by several classes."""

    def assert_hop_is_the_only_sunset(self, hop,
                                      latitude=REFERENCE_LATITUDE,
                                      longitude=REFERENCE_LONGITUDE):
        """Certify a hop as the FIRST genuine sunset after its predecessor.

        The claim is a CARDINALITY claim, not a comparison to a recorded
        instant: exactly one sunset lies in the window from the
        predecessor through the hop, so no sunset was skipped. A skipped
        crossing would make this two.

        Both window edges are set by the comparison bound, symmetrically.
        The wide bracket enters each root from a different start than the
        production search did, so its recomputation of an ALREADY
        ESTABLISHED crossing can land tens of microseconds to either
        side of it - later as readily as earlier. Without the exclusion,
        such a recomputation of the predecessor's own crossing would
        masquerade as an intervening sunset, and a recomputation of the
        hop's own crossing would fall outside the count.

        The exclusion is a numerical-recomputation allowance and nothing
        more. It does not define physical event identity, it is not a
        production or solver tolerance, it states no astronomical
        uncertainty, and it asserts no minimum physical separation.

        Genuine-event distinction is established independently, by the
        two assertions below: distinct sunsets for this observer are
        more than twenty hours apart, and daylight separates them.
        Those live HERE, beside the bound, so the evidence can never
        drift away from the use.
        """
        wide = crossings(
            hop.previous_tt - ORACLE_BACKSTEP_DAYS,
            hop.tt + ORACLE_OVERSHOOT_DAYS,
            latitude,
            longitude,
        )

        window_start = hop.previous_tt + (
            MAX_ORACLE_DEVIATION_SECONDS / SECONDS_PER_DAY
        )
        window_end = hop.tt + (
            MAX_ORACLE_DEVIATION_SECONDS / SECONDS_PER_DAY
        )

        within = [
            tt for tt in sunsets_in(wide)
            if window_start < tt <= window_end
        ]

        self.assertEqual(len(within), 1)
        self.assertLess(
            abs(within[0] - hop.tt) * SECONDS_PER_DAY,
            MAX_ORACLE_DEVIATION_SECONDS,
        )

        # Justification for the bound above - asserted, not assumed.
        self.assertGreater(
            (hop.tt - hop.previous_tt) * 24.0, MIN_SUNSET_SEPARATION_HOURS
        )
        self.assertTrue(
            [tt for tt in sunrises_in(wide)
             if hop.previous_tt < tt < hop.tt]
        )

    def assert_hop_bit_equals_aligned_root(self, hop,
                                           latitude=REFERENCE_LATITUDE,
                                           longitude=REFERENCE_LONGITUDE):
        """Where the brackets coincide, require BIT equality.

        This bracket matches the certified production horizon, so no
        numerical allowance is appropriate or used.
        """
        aligned = sunsets_in(
            crossings(
                hop.previous_tt,
                hop.previous_tt + SEARCH_SPAN_DAYS,
                latitude,
                longitude,
            )
        )

        self.assertTrue(aligned)
        self.assertEqual(bits(aligned[0]), bits(hop.tt))


# --- A. Four-hop composed chain from /sunset -------------------------------


class TestComposedChainFromSunsetSeed(HopOracleMixin, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.seed = sunset_seed()
        cls.chain = walk_chain(cls.seed, CHAIN_HOPS)

    def test_every_hop_emits_a_bound_continuation_cursor(self):
        self.assertEqual(len(self.chain), CHAIN_HOPS)

        for index, hop in enumerate(self.chain, 1):
            with self.subTest(hop=index):
                self.assertIsInstance(hop.cursor, str)
                self.assertIsNotNone(hop.response["cursor"])
                self.assertNotIn("cursorUnavailableReason", hop.response)

    def test_every_hop_is_the_only_sunset_in_its_window(self):
        for index, hop in enumerate(self.chain, 1):
            with self.subTest(hop=index):
                self.assert_hop_is_the_only_sunset(hop)

    def test_every_hop_bit_equals_the_aligned_oracle_root(self):
        for index, hop in enumerate(self.chain, 1):
            with self.subTest(hop=index):
                self.assert_hop_bit_equals_aligned_root(hop)

    def test_chain_is_strictly_chronological(self):
        states = [hop.tt for hop in self.chain]

        self.assertEqual(states, sorted(states))
        self.assertEqual(len(set(states)), CHAIN_HOPS)

        emitted = [hop.response["sunsetUTC"] for hop in self.chain]
        self.assertEqual(emitted, sorted(emitted))
        self.assertEqual(len(set(emitted)), CHAIN_HOPS)

    def test_no_hop_rediscovers_its_originating_sunset(self):
        for index, hop in enumerate(self.chain, 1):
            with self.subTest(hop=index):
                self.assertGreater(hop.tt, hop.previous_tt)
                self.assertNotEqual(bits(hop.tt), bits(hop.previous_tt))

                # A search starting from a proven sun-down state never
                # reports that state itself as a crossing.
                self.assertFalse(sun_is_up_at(hop.previous_tt))
                aligned = crossings(
                    hop.previous_tt, hop.previous_tt + SEARCH_SPAN_DAYS
                )
                self.assertNotIn(
                    hop.previous_tt, [tt for tt, _up in aligned]
                )

    def test_kernel_provenance_is_truthful_across_this_reference_chain(self):
        # BOUNDED CLAIM. Kernel stability is asserted for THIS 2027
        # reference chain only. It is NOT a universal successor-contract
        # invariant: a legitimate future chain crossing a governed
        # ephemeris-routing boundary may report different kernels across
        # hops, and doing so would be truthful rather than defective.
        # DE440/DE441 routing-boundary behavior is not certified here.
        for index, hop in enumerate(self.chain, 1):
            with self.subTest(hop=index):
                self.assertEqual(hop.response["kernel"], REFERENCE_KERNEL)

        self.assertEqual(self.seed["kernel"], REFERENCE_KERNEL)


# --- B. /sunset-after seed composition -------------------------------------


class TestSunsetAfterSeedComposition(HopOracleMixin, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.after_seed = sunset_after_seed()
        cls.after_chain = walk_chain(cls.after_seed, SUNSET_AFTER_HOPS)

        cls.sunset_seed = sunset_seed()
        cls.sunset_chain = walk_chain(cls.sunset_seed, SUNSET_AFTER_HOPS)

    def test_three_hop_chain_completes_from_a_sunset_after_cursor(self):
        self.assertEqual(len(self.after_chain), SUNSET_AFTER_HOPS)

        for index, hop in enumerate(self.after_chain, 1):
            with self.subTest(hop=index):
                self.assertIsInstance(hop.cursor, str)
                self.assert_hop_is_the_only_sunset(hop)
                self.assert_hop_bit_equals_aligned_root(hop)

    def test_both_seeds_converge_on_one_independently_certified_chain(self):
        # The two seeds refer to the same starting physical sunset, but
        # they reach it from different search brackets, so their exact
        # binary64 states are NOT required to agree bit for bit. What is
        # required is that they identify the same physical events.
        #
        # BOTH members of every compared pair are independently
        # certified here, so the comparison is between two hops each
        # already shown to be the sole sunset in its own window. The
        # bound therefore compares two independent computations of one
        # root; it never decides which physical event either hop is.
        self.assertEqual(
            len(self.after_chain), len(self.sunset_chain)
        )

        pairs = zip(self.after_chain, self.sunset_chain)
        for index, (after_hop, sunset_hop) in enumerate(pairs, 1):
            with self.subTest(hop=index):
                self.assert_hop_is_the_only_sunset(after_hop)
                self.assert_hop_is_the_only_sunset(sunset_hop)

                deviation = (
                    abs(after_hop.tt - sunset_hop.tt) * SECONDS_PER_DAY
                )
                self.assertLess(deviation, MAX_ORACLE_DEVIATION_SECONDS)


# --- C. Observer binding survives every hop --------------------------------


class TestObserverBindingSurvivesEveryHop(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.chain = walk_chain(sunset_seed(), CHAIN_HOPS)

    def test_bound_observer_survives_every_hop(self):
        for index, hop in enumerate(self.chain, 1):
            with self.subTest(hop=index):
                decoded = decode_for(hop.cursor)

                self.assertEqual(
                    bits(decoded.latitude), bits(REFERENCE_LATITUDE)
                )
                self.assertEqual(
                    bits(decoded.longitude), bits(REFERENCE_LONGITUDE)
                )
                self.assertEqual(
                    hop.response["latitude"], REFERENCE_LATITUDE
                )
                self.assertEqual(
                    hop.response["longitude"], REFERENCE_LONGITUDE
                )

    def test_adjacent_observer_is_rejected_at_every_hop(self):
        for index, hop in enumerate(self.chain, 1):
            with self.subTest(hop=index):
                with self.assertRaises(HTTPException) as caught:
                    successor_response(
                        hop.cursor,
                        latitude=ADJACENT_LATITUDE,
                        longitude=ADJACENT_LONGITUDE,
                    )

                error = caught.exception
                self.assertEqual(error.status_code, 400)
                self.assertIsInstance(error.detail, str)
                self.assertIn(REASON_INCOMPATIBLE_OBSERVER, error.detail)


# --- D. Chain-emitted cursors remain fail-closed ---------------------------


class TestChainEmittedCursorFailsClosed(unittest.TestCase):
    """The corrupted cursors here are produced BY the successor route,
    not by a seed route. A1d certified fail-closed behavior on seed
    cursors only.
    """

    @classmethod
    def setUpClass(cls):
        cls.emitted = walk_chain(sunset_seed(), 1)[0].cursor

    def assert_rejected(self, cursor, reason):
        with self.assertRaises(HTTPException) as caught:
            successor_response(cursor)

        error = caught.exception
        self.assertEqual(error.status_code, 400)
        self.assertIsInstance(error.detail, str)
        self.assertIn(reason, error.detail)
        self.assertNotIn(error.status_code, (401, 403))

    def test_tampered_successor_cursor_is_checksum_rejected(self):
        # The payload is altered and the recorded checksum left stale,
        # so the cursor is well formed but no longer self-consistent.
        original = envelope_of(self.emitted)["payload"]["ttBits"]
        flipped = original[:-1] + ("0" if original[-1] != "0" else "1")
        self.assertNotEqual(flipped, original)

        cursor = tampered(
            self.emitted, "ttBits", flipped, recompute_checksum=False
        )
        self.assertNotEqual(cursor, self.emitted)
        self.assert_rejected(cursor, REASON_CHECKSUM_MISMATCH)

    def test_environment_drifted_successor_cursor_is_rejected(self):
        # The checksum is recomputed, so the cursor is internally
        # consistent and fails only on the environment binding.
        cursor = tampered(
            self.emitted, "skyfieldVersion", "0.0", recompute_checksum=True
        )
        self.assert_rejected(cursor, REASON_INCOMPATIBLE_ENVIRONMENT)

    def test_malformed_replacement_is_rejected(self):
        self.assert_rejected(
            MALFORMED_CURSOR_TEXT, REASON_MALFORMED_CURSOR
        )


# --- E. Signed-zero observer composes end to end ---------------------------


class TestSignedZeroObserverComposition(HopOracleMixin, unittest.TestCase):

    def test_signed_zero_observer_completes_a_certified_successor(self):
        # The seed is requested at longitude -0.0. The encoder
        # canonicalizes the binding to +0.0, and the decoder
        # canonicalizes the request the same way, so the route must
        # complete rather than fail closed on a spurious mismatch.
        #
        # No pinned astronomical golden is introduced: the returned
        # sunset is certified against the independent oracle built for
        # this observer.
        seed = server.sunset(
            date=REFERENCE_DATE,
            latitude=REFERENCE_LATITUDE,
            longitude=NEGATIVE_ZERO_LONGITUDE,
        )
        self.assertIsInstance(seed["cursor"], str)

        chain = walk_chain(
            seed,
            1,
            latitude=REFERENCE_LATITUDE,
            longitude=NEGATIVE_ZERO_LONGITUDE,
        )
        hop = chain[0]

        # A complete successful response, not a fail-closed outcome.
        self.assertIsInstance(hop.response["sunsetUTC"], str)
        self.assertEqual(hop.response["kernel"], REFERENCE_KERNEL)
        self.assertIsInstance(hop.response["cursor"], str)

        # The canonical +0.0 is what the Authority reports and binds -
        # plain equality would be vacuous, since -0.0 == 0.0.
        self.assertEqual(
            math.copysign(1.0, hop.response["longitude"]), 1.0
        )
        self.assertEqual(
            bits(decode_for(
                hop.cursor,
                latitude=REFERENCE_LATITUDE,
                longitude=NEGATIVE_ZERO_LONGITUDE,
            ).longitude),
            bits(0.0),
        )

        # The physical event is certified against the independent
        # oracle for this observer, at longitude +0.0.
        self.assert_hop_is_the_only_sunset(
            hop, latitude=REFERENCE_LATITUDE, longitude=0.0
        )
        self.assert_hop_bit_equals_aligned_root(
            hop, latitude=REFERENCE_LATITUDE, longitude=0.0
        )


# --- F. The composed production path carries no tolerance ------------------


class TestComposedPathCarriesNoTolerance(unittest.TestCase):

    def test_no_gap_or_tolerance_constant_in_the_composed_path(self):
        # Every production function the continuation state passes
        # through, inspected together rather than one at a time.
        bodies = {
            "sunset_successor": function_body_source(
                server.sunset_successor
            ),
            "cursor_fields": function_body_source(server.cursor_fields),
            "find_sunset_successor": function_body_source(
                server.find_sunset_successor
            ),
        }

        for name, body in bodies.items():
            # The oracle must not pass by inspecting nothing.
            with self.subTest(function=name, check="non-empty"):
                self.assertTrue(body.strip())

            for forbidden in (
                "3600",
                "min_gap",
                "EPSILON",
                "tolerance",
                "timedelta",
                "timestamp(",
                "fromisoformat",
            ):
                with self.subTest(function=name, forbidden=forbidden):
                    self.assertNotIn(forbidden, body)

    def test_legacy_guard_remains_confined_to_the_legacy_path(self):
        legacy = inspect.getsource(find_next_sunset_after_utc)

        # Present and unchanged where it belongs.
        self.assertIn("min_gap_seconds = 3600", legacy)
        self.assertIn("after_timestamp + min_gap_seconds", legacy)

        # A1e neither removes nor alters it. Whether the guard is
        # scientifically correct is a separate governed question that
        # this file does not answer.


if __name__ == "__main__":
    unittest.main(verbosity=2)
