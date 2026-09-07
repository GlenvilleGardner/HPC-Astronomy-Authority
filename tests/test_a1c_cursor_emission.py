"""A1c verification: exact-TT continuation-witness emission from the two
existing Authority sunset endpoints.

Test framework: standard-library unittest.

SCOPE

This is the first increment that joins three surfaces which A1a and A1b
deliberately kept apart: real astronomy, the continuation witness, and the
HTTP route contract. It therefore lives in its own file rather than in
tests/test_sunset_cursor.py, which is astronomy-free by declaration, or
tests/test_sunset_successor.py, which asserts nothing about routes.

INDEPENDENT ORACLE DISCIPLINE

Golden instants, golden Terrestrial Time values, observer coordinates,
reason codes and domain bounds are declared here as test-local literals.
They are deliberately NOT imported from server or sunset_cursor: importing
them would make these tests agree with a defective constant instead of
detecting it.

The sunset predicate used as an oracle is built here directly from
skyfield.almanac, not obtained from the code under test.

The shared timescale and kernel loader ARE imported from astronomy_solver.
They are repository infrastructure, and duplicating them would load a
second copy of a 119 MB kernel for no verification benefit.

The cursor is never decoded, reconstructed or inspected by test-local
machinery. Every round-trip goes through the real A1a decoder.

ROUTE SEAM

The route functions are invoked directly. httpx is not installed, so
starlette's TestClient is unavailable, and installing a dependency to test
is not permitted. Direct invocation exercises the real route body,
including the cursor field merge. HTTP status behavior is unchanged by
A1c and is not re-asserted here.

WHAT THIS FILE DOES NOT CLAIM

Nothing about Engine migration having occurred, about any
/sunset-successor route, about removal or correction of the legacy
one-hour /sunset-after guard, about DE440/DE441 routing boundaries, about
BSP integrity, or about astronomical correctness beyond the instants and
observers exercised below.
"""

import math
import unittest
from datetime import datetime, timezone
from unittest import mock

from skyfield import almanac
from skyfield.api import wgs84

import server
from astronomy_solver import (
    find_next_sunset_after_utc,
    find_sunset_successor,
    find_sunset_utc,
    load_kernel,
    ts,
)
from sunset_cursor import (
    SunsetCursorError,
    decode_sunset_cursor,
    encode_sunset_cursor,
)


REFERENCE_LATITUDE = 40.7406
REFERENCE_LONGITUDE = -73.9586
REFERENCE_KERNEL = "de440.bsp"

REFERENCE_DATE = "2027-03-20"
REFERENCE_MIDDAY_UTC = "2027-03-20T12:00:00+00:00"

# MEASURED - the sunset that closes 2027-03-20 for the reference observer,
# and the exact binary64 Terrestrial Time state it was found at. The same
# TT value is pinned independently by A1a and A1b.
SEED_SUNSET_UTC = "2027-03-20T23:07:45.858897+00:00"
SEED_TT = 2461485.4645259595

# MEASURED - the next distinct crossing.
SUCCESSOR_SUNSET_UTC = "2027-03-21T23:08:49.679673+00:00"

# Finite, outside the cursor observer domain, and already measured to
# return an ordinary successful sunset. Used only to characterize the
# additive cursor contract; no coordinate validation is exercised or
# changed by this file.
OUT_OF_DOMAIN_LATITUDE = 95.0

REASON_OBSERVER_OUT_OF_DOMAIN = "OBSERVER_OUT_OF_CURSOR_DOMAIN"
REASON_ENCODING_FAILED = "CURSOR_ENCODING_FAILED"

INCLUSIVE_BOUNDARY_OBSERVERS = (
    (90.0, 0.0),
    (-90.0, 0.0),
    (0.0, 180.0),
    (0.0, -180.0),
)

OUTSIDE_BOUNDARY_OBSERVERS = (
    (90.0000001, 0.0),
    (-90.0000001, 0.0),
    (0.0, 180.0000001),
    (0.0, -180.0000001),
)

# Four steps: step 1 reaches A1b's golden successor, steps 2 and 3 reach
# A1b's two pinned chain values, and step 4 extends one step beyond
# existing A1b coverage so this gate is not a restatement of A1b.
CHAIN_STEPS = 4

# MEASURED - the largest observed disagreement between the legacy walk and
# the successor walk over eight steps was 1.21e-4 s, while consecutive
# sunsets for this observer are separated by more than 24 hours. One second
# is therefore many orders of magnitude below the smallest possible
# distance between distinct crossings.
MAX_PATH_DEVIATION_SECONDS = 1.0
MIN_SUNSET_SEPARATION_HOURS = 20.0


def reference_predicate(latitude=REFERENCE_LATITUDE, longitude=REFERENCE_LONGITUDE):
    """Build the sunset predicate independently of the code under test."""
    return almanac.sunrise_sunset(
        load_kernel(REFERENCE_KERNEL), wgs84.latlon(latitude, longitude)
    )


def sun_is_up_at(tt, latitude=REFERENCE_LATITUDE, longitude=REFERENCE_LONGITUDE):
    """Evaluate the independent predicate at an exact TT state."""
    return bool(reference_predicate(latitude, longitude)(ts.tt_jd(tt)))


def reference_sunset_response():
    return server.sunset(
        date=REFERENCE_DATE,
        latitude=REFERENCE_LATITUDE,
        longitude=REFERENCE_LONGITUDE,
    )


def reference_sunset_after_response():
    return server.sunset_after(
        afterUTC=REFERENCE_MIDDAY_UTC,
        latitude=REFERENCE_LATITUDE,
        longitude=REFERENCE_LONGITUDE,
    )


def decode_reference(cursor):
    return decode_sunset_cursor(
        cursor, latitude=REFERENCE_LATITUDE, longitude=REFERENCE_LONGITUDE
    )


# --- A. /sunset exact-TT cursor round-trip ---------------------------------


class TestSunsetSeedCursorRoundTrip(unittest.TestCase):

    def setUp(self):
        self.determination = find_sunset_utc(
            datetime(2027, 3, 20, tzinfo=timezone.utc),
            REFERENCE_LATITUDE,
            REFERENCE_LONGITUDE,
        )
        self.response = reference_sunset_response()
        self.decoded = decode_reference(self.response["cursor"])

    def test_route_sunset_utc_matches_golden(self):
        self.assertEqual(self.response["sunsetUTC"], SEED_SUNSET_UTC)
        self.assertEqual(self.response["kernel"], REFERENCE_KERNEL)

    def test_determination_tt_matches_measured_golden(self):
        self.assertEqual(self.determination.tt, SEED_TT)

    def test_cursor_is_present_and_non_null(self):
        self.assertIn("cursor", self.response)
        self.assertIsInstance(self.response["cursor"], str)

    def test_decoded_observer_equals_bound_observer(self):
        self.assertEqual(self.decoded.latitude, REFERENCE_LATITUDE)
        self.assertEqual(self.decoded.longitude, REFERENCE_LONGITUDE)

    def test_decoded_tt_is_exactly_the_determination_state(self):
        # Binary64 equality, not tolerance.
        self.assertEqual(self.decoded.tt, self.determination.tt)
        self.assertEqual(self.decoded.tt, SEED_TT)

    def test_decoded_tt_reproduces_the_emitted_sunset_utc(self):
        # The witness is converted forward through the governed timescale.
        # TT is never reconstructed from the UTC string.
        self.assertEqual(
            ts.tt_jd(self.decoded.tt).utc_datetime().isoformat(),
            self.response["sunsetUTC"],
        )


# --- B. /sunset-after exact-TT cursor round-trip ---------------------------


class TestSunsetAfterCursorRoundTrip(unittest.TestCase):

    def setUp(self):
        self.determination = find_next_sunset_after_utc(
            datetime(2027, 3, 20, 12, 0, 0, tzinfo=timezone.utc),
            REFERENCE_LATITUDE,
            REFERENCE_LONGITUDE,
        )
        self.response = reference_sunset_after_response()
        self.decoded = decode_reference(self.response["cursor"])

    def test_route_sunset_utc_matches_golden(self):
        self.assertEqual(self.response["sunsetUTC"], SEED_SUNSET_UTC)
        self.assertEqual(self.response["kernel"], REFERENCE_KERNEL)

    def test_cursor_is_present_and_non_null(self):
        self.assertIsInstance(self.response["cursor"], str)

    def test_decoded_observer_equals_bound_observer(self):
        self.assertEqual(self.decoded.latitude, REFERENCE_LATITUDE)
        self.assertEqual(self.decoded.longitude, REFERENCE_LONGITUDE)

    def test_decoded_tt_is_exactly_the_selected_root(self):
        self.assertEqual(self.decoded.tt, self.determination.tt)

    def test_decoded_tt_reproduces_the_emitted_sunset_utc(self):
        self.assertEqual(
            ts.tt_jd(self.decoded.tt).utc_datetime().isoformat(),
            self.response["sunsetUTC"],
        )


# --- C. Seed cursor drives the A1b successor -------------------------------


class TestSeedCursorDrivesSuccessor(unittest.TestCase):

    def setUp(self):
        self.response = reference_sunset_response()
        self.decoded = decode_reference(self.response["cursor"])
        self.successor = find_sunset_successor(
            self.decoded.tt, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )

    def test_seed_state_is_post_transition(self):
        # Asserted against the independent predicate, not against the gate
        # inside find_sunset_successor.
        self.assertFalse(sun_is_up_at(self.decoded.tt))

    def test_successor_is_strictly_forward(self):
        self.assertGreater(self.successor.tt, self.decoded.tt)
        self.assertGreater(
            self.successor.utc.isoformat(), self.response["sunsetUTC"]
        )

    def test_originating_root_is_not_rediscovered(self):
        self.assertNotEqual(self.successor.tt, self.decoded.tt)
        self.assertNotEqual(
            self.successor.utc.isoformat(), self.response["sunsetUTC"]
        )

    def test_successor_matches_golden(self):
        self.assertEqual(self.successor.utc.isoformat(), SUCCESSOR_SUNSET_UTC)


# --- D. /sunset-after cursor drives the A1b successor ----------------------


class TestSunsetAfterCursorDrivesSuccessor(unittest.TestCase):

    def setUp(self):
        self.response = reference_sunset_after_response()
        self.decoded = decode_reference(self.response["cursor"])
        self.successor = find_sunset_successor(
            self.decoded.tt, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        )

    def test_state_is_post_transition(self):
        self.assertFalse(sun_is_up_at(self.decoded.tt))

    def test_successor_is_strictly_forward_and_distinct(self):
        self.assertGreater(self.successor.tt, self.decoded.tt)
        self.assertNotEqual(
            self.successor.utc.isoformat(), self.response["sunsetUTC"]
        )

    def test_successor_matches_golden(self):
        self.assertEqual(self.successor.utc.isoformat(), SUCCESSOR_SUNSET_UTC)


# --- E. Migration-equivalence composition - PRIMARY A1c ACCEPTANCE GATE ----


class TestMigrationEquivalenceComposition(unittest.TestCase):
    """Establish that the emitted seed witness can drive the same physical
    sunset sequence the guarded legacy walk produces.

    The successor sequence is advanced ONLY by exact binary64 TT taken from
    the emitted continuation witness. No UTC datetime, ISO string, Unix
    timestamp or millisecond value is ever fed back into the search.
    """

    @classmethod
    def setUpClass(cls):
        determination = find_sunset_utc(
            datetime(2027, 3, 20, tzinfo=timezone.utc),
            REFERENCE_LATITUDE,
            REFERENCE_LONGITUDE,
        )
        decoded = decode_reference(reference_sunset_response()["cursor"])

        cls.seed_utc = determination.utc

        cls.legacy = []
        current = determination.utc
        for _ in range(CHAIN_STEPS):
            step = find_next_sunset_after_utc(
                current, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
            )
            cls.legacy.append(step.utc)
            current = step.utc

        cls.successor = []
        tt = decoded.tt
        for _ in range(CHAIN_STEPS):
            step = find_sunset_successor(
                tt, REFERENCE_LATITUDE, REFERENCE_LONGITUDE
            )
            cls.successor.append(step.utc)
            tt = step.tt

    def test_both_paths_yield_the_full_sequence(self):
        self.assertEqual(len(self.legacy), CHAIN_STEPS)
        self.assertEqual(len(self.successor), CHAIN_STEPS)

    def test_each_step_is_the_same_physical_crossing(self):
        # The two paths converge on each root from different brackets, so
        # they are not required to agree bit-for-bit. They are required to
        # identify the SAME physical crossing.
        #
        # This bound is NOT an identity tolerance supplied to any solver:
        # nothing here is passed to find_discrete, find_sunset_successor
        # still uses no gap constant, and the legacy 3600-second guard is
        # untouched. It is a comparison between two independent
        # computations, and it is justified by the separation asserted
        # below it - distinct sunsets for this observer are more than 20
        # hours apart, so agreement below one second cannot be a different
        # crossing.
        for index, (legacy, successor) in enumerate(
            zip(self.legacy, self.successor), 1
        ):
            with self.subTest(step=index):
                deviation = abs((legacy - successor).total_seconds())
                self.assertLess(deviation, MAX_PATH_DEVIATION_SECONDS)
                self.assertEqual(legacy.date(), successor.date())

    def test_distinct_crossings_are_far_apart(self):
        for index in range(CHAIN_STEPS - 1):
            separation = (
                self.legacy[index + 1] - self.legacy[index]
            ).total_seconds() / 3600.0
            with self.subTest(gap=index):
                self.assertGreater(separation, MIN_SUNSET_SEPARATION_HOURS)

    def test_successor_sequence_is_strictly_forward(self):
        previous = self.seed_utc
        for index, moment in enumerate(self.successor, 1):
            with self.subTest(step=index):
                self.assertGreater(moment, previous)
            previous = moment

    def test_seed_root_is_never_rediscovered(self):
        self.assertNotIn(self.seed_utc, self.successor)


# --- F. Cursor-presence contract -------------------------------------------


class TestCursorPresenceContract(unittest.TestCase):

    def test_sunset_in_domain_emits_a_cursor(self):
        response = reference_sunset_response()
        self.assertIsInstance(response["cursor"], str)
        self.assertNotIn("cursorUnavailableReason", response)

    def test_sunset_after_in_domain_emits_a_cursor(self):
        response = reference_sunset_after_response()
        self.assertIsInstance(response["cursor"], str)
        self.assertNotIn("cursorUnavailableReason", response)

    def test_out_of_domain_observer_preserves_existing_fields(self):
        response = server.sunset(
            date=REFERENCE_DATE,
            latitude=OUT_OF_DOMAIN_LATITUDE,
            longitude=REFERENCE_LONGITUDE,
        )
        for field in ("date", "latitude", "longitude", "kernel", "sunsetUTC"):
            with self.subTest(field=field):
                self.assertIn(field, response)
        self.assertEqual(response["kernel"], REFERENCE_KERNEL)

    def test_out_of_domain_observer_represents_unavailability(self):
        response = server.sunset(
            date=REFERENCE_DATE,
            latitude=OUT_OF_DOMAIN_LATITUDE,
            longitude=REFERENCE_LONGITUDE,
        )
        self.assertIn("cursor", response)
        self.assertIsNone(response["cursor"])
        self.assertEqual(
            response["cursorUnavailableReason"], REASON_OBSERVER_OUT_OF_DOMAIN
        )


# --- G. Cursor observer-domain boundaries ----------------------------------


class TestCursorDomainBoundaries(unittest.TestCase):
    """Astronomy-free.

    These cases assert only which observers the cursor can represent. They
    deliberately do NOT assert that a physical sunset exists at any of
    these coordinates on any date: cursor-domain semantics and
    astronomical event availability are separate questions, and a polar
    observer legitimately has no sunset to witness.
    """

    def encoder_accepts(self, latitude, longitude):
        try:
            encode_sunset_cursor(
                tt=SEED_TT, latitude=latitude, longitude=longitude
            )
        except SunsetCursorError:
            return False
        return True

    def test_inclusive_boundaries_are_representable(self):
        for latitude, longitude in INCLUSIVE_BOUNDARY_OBSERVERS:
            with self.subTest(latitude=latitude, longitude=longitude):
                self.assertTrue(self.encoder_accepts(latitude, longitude))
                self.assertTrue(
                    server.observer_is_cursor_representable(latitude, longitude)
                )

    def test_just_outside_boundaries_is_not_representable(self):
        for latitude, longitude in OUTSIDE_BOUNDARY_OBSERVERS:
            with self.subTest(latitude=latitude, longitude=longitude):
                self.assertFalse(self.encoder_accepts(latitude, longitude))
                self.assertFalse(
                    server.observer_is_cursor_representable(latitude, longitude)
                )

    def test_negative_zero_longitude_canonicalizes_to_positive_zero(self):
        cursor = encode_sunset_cursor(
            tt=SEED_TT, latitude=REFERENCE_LATITUDE, longitude=-0.0
        )
        decoded = decode_sunset_cursor(
            cursor, latitude=REFERENCE_LATITUDE, longitude=-0.0
        )
        # Plain equality would be vacuous: -0.0 == 0.0 in Python. The
        # canonicalization claim is about the sign bit.
        self.assertEqual(math.copysign(1.0, decoded.longitude), 1.0)


# --- H. Cursor-encoding failure contract -----------------------------------


class TestCursorEncodingFailureContract(unittest.TestCase):
    """The encoding-failure branch is reached by patching the encoder at
    the seam the production helper actually calls.

    server.cursor_fields resolves encode_sunset_cursor as a module global
    of server, so patching server.encode_sunset_cursor intercepts exactly
    that call. sunset_cursor is not modified, the real encoder is not
    corrupted, and production exception handling is not broadened.

    An out-of-domain observer cannot reach this branch: it is intercepted
    earlier by the representability pre-check and yields the other reason
    code. Patching is therefore the only way to exercise it without
    damaging production machinery.
    """

    def failing_response(self):
        with mock.patch.object(
            server,
            "encode_sunset_cursor",
            side_effect=SunsetCursorError("TEST", "induced encoding failure"),
        ):
            return server.sunset(
                date=REFERENCE_DATE,
                latitude=REFERENCE_LATITUDE,
                longitude=REFERENCE_LONGITUDE,
            )

    def test_encoding_failure_is_represented_not_raised(self):
        response = self.failing_response()
        self.assertIsNone(response["cursor"])
        self.assertEqual(
            response["cursorUnavailableReason"], REASON_ENCODING_FAILED
        )

    def test_existing_fields_survive_an_encoding_failure(self):
        response = self.failing_response()
        self.assertEqual(response["sunsetUTC"], SEED_SUNSET_UTC)
        self.assertEqual(response["kernel"], REFERENCE_KERNEL)

    def test_production_encoder_is_restored(self):
        self.failing_response()
        self.assertIs(server.encode_sunset_cursor, encode_sunset_cursor)


if __name__ == "__main__":
    unittest.main()
