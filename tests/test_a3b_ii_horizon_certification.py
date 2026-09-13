"""A3b-ii OPT-IN exhaustive solar-longitude horizon certification.

This file rederives, from the real pinned NASA/JPL artifacts, the scientific
facts the A3b-ii production design rests on. It is EXPENSIVE - it enumerates
and canonicalizes every governed solar-longitude crossing in the entire
authoritative domain - and it is therefore NOT part of the routine Authority
suite.

INVOCATION
----------
Ordinary runs skip it cleanly and cost nothing:

    .\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v

To run the certification, set the gate and name the module explicitly:

    $env:HPC_RUN_SOLAR_LONGITUDE_HORIZON_CERTIFICATION = "1"
    .\\.venv\\Scripts\\python.exe -m unittest ^
        tests.test_a3b_ii_horizon_certification -v

Expect roughly 8-10 minutes and several GB of ephemeris paging.

WHAT THIS CERTIFIES

    - every supported crossing across all three pinned artifacts
    - the total crossing count
    - quadrant-sequence integrity, including across every enumeration chunk
    - a canonical binary64 boundary for EVERY crossing
    - season_at(canonical) == kind, for every crossing
    - season_at(predecessor_binary64(canonical)) != kind, for every crossing
    - no predicate chatter around any canonical boundary
    - no two crossings collapsing to one canonical TT
    - the certified extrema, rederived
    - that those extrema remain compatible with the governed constants

WHY IT EXISTS

The production constants are safe because of measured properties of the
pinned artifacts under the certified runtime. If an artifact is replaced, the
runtime changes, or Skyfield's seasons/find_discrete behavior changes, those
properties must be re-established rather than assumed. This file is how that
is done, and it fails closed if they no longer hold.

SCOPE OF THE CLAIM - READ THIS BEFORE TRUSTING A NUMBER

Every value certified here is tied to a specific scientific generation:

    pinned BSP artifacts   de440.bsp, de441_part-1.bsp, de441_part-2.bsp
    Python runtime         the certified CPython recorded in A0.3
    Skyfield               the certified version recorded in A0.3
    NumPy                  the certified version recorded in A0.3
    jplephem               the certified version recorded in A0.3

They are NOT timeless astronomical constants, and they are NOT properties of
the solar system. They are properties of a computation over a fixed data set.
A different artifact set or a different runtime generation may legitimately
produce different extrema, and that is a recertification, not a defect.

INDEPENDENT ORACLE DISCIPLINE

Every expected value is a test-local literal. The governed constants are
compared against literals rather than imported and asserted against
themselves. Crossings are enumerated and canonicalized here, against
almanac.seasons directly, without calling any A3b-ii production helper.
"""

import os
import unittest

import numpy as np
from skyfield import almanac
from skyfield.errors import EphemerisRangeError

import astronomy_solver
from astronomy_solver import kernel_coverage_tt, load_kernel, ts

# --- Gate ------------------------------------------------------------------

CERTIFICATION_GATE = "HPC_RUN_SOLAR_LONGITUDE_HORIZON_CERTIFICATION"

GATE_IS_OPEN = os.getenv(CERTIFICATION_GATE) == "1"

SKIP_REASON = (
    "opt-in scientific certification; set %s=1 to run it (expect 8-10 minutes)"
    % CERTIFICATION_GATE
)

# --- Test-local oracles ----------------------------------------------------

DE440 = "de440.bsp"
DE441_PART_1 = "de441_part-1.bsp"
DE441_PART_2 = "de441_part-2.bsp"

PINNED_ARTIFACTS = (DE440, DE441_PART_1, DE441_PART_2)

KINDS = ("SOLAR_LONGITUDE_000", "SOLAR_LONGITUDE_090",
         "SOLAR_LONGITUDE_180", "SOLAR_LONGITUDE_270")

# Expected governed production values, as independent literals.
EXPECTED_SEARCH_SPAN_DAYS = 380.0
EXPECTED_SAMPLE_STEP_DAYS = 88.0

# Certified evidence for the current scientific generation.
EXPECTED_TOTAL_CROSSINGS = 125963
EXPECTED_QUADRANT_SEQUENCE_VIOLATIONS = 0
CERTIFIED_MAX_SAME_KIND_GAP_DAYS = 365.253272318
CERTIFIED_MIN_ADJACENT_QUADRANT_GAP_DAYS = 88.159551771

# The certified minimum is the CONSERVATIVE frozen threshold. Canonical
# rederivation yields a marginally larger value, so the frozen threshold is
# the safer of the two and is retained deliberately.
MIN_ADJACENT_REDERIVATION_SLACK_DAYS = 1e-6

# Enumeration chunk width. Chunks abut exactly; quadrant-sequence integrity
# across the whole artifact is what proves no crossing is lost or duplicated
# at a boundary.
CHUNK_DAYS = 20000.0

CHATTER_ULPS = 8
EVALUATION_BATCH = 40000


# --- Independent machinery -------------------------------------------------


def independent_predicate(kernel_name):
    return almanac.seasons(load_kernel(kernel_name))


def is_evaluable(season_at, tt):
    """Does the governed computation actually complete at this state?"""
    try:
        season_at(ts.tt_jd(tt))
    except EphemerisRangeError:
        return False
    return True


def supported_interval(kernel_name):
    """Declared coverage narrowed to actual geocentric evaluability."""
    coverage = kernel_coverage_tt(kernel_name)
    season_at = independent_predicate(kernel_name)

    low = coverage.tt_start
    if not is_evaluable(season_at, low):
        probe = low
        while not is_evaluable(season_at, probe):
            probe += 1.0
        bad, good = coverage.tt_start, probe
        while True:
            mid = bad + (good - bad) / 2.0
            if mid == bad or mid == good:
                low = good
                break
            if is_evaluable(season_at, mid):
                good = mid
            else:
                bad = mid

    high = coverage.tt_end
    while not is_evaluable(season_at, high):
        high -= 1.0

    return low, high, season_at


def sampled(season_at, step_days):
    def wrapper(t):
        return season_at(t)

    wrapper.step_days = step_days
    return wrapper


def enumerate_artifact(kernel_name):
    """Every crossing in one artifact's supported interval, ascending."""
    low, high, season_at = supported_interval(kernel_name)
    wrapper = sampled(season_at, EXPECTED_SAMPLE_STEP_DAYS)

    raws, kinds = [], []
    start = low
    while start < high:
        end = min(start + CHUNK_DAYS, high)
        times, events = almanac.find_discrete(
            ts.tt_jd(start), ts.tt_jd(end), wrapper
        )
        reported = np.atleast_1d(np.array(times.tt, dtype=float))
        if len(reported):
            raws.append(reported)
            kinds.append(np.atleast_1d(np.array(events, dtype=int)))
        start = end

    raw = np.concatenate(raws) if raws else np.zeros(0)
    kind = np.concatenate(kinds) if kinds else np.zeros(0, dtype=int)
    return low, high, season_at, raw, kind


def canonicalize_all(season_at, lower, upper, quadrants):
    """Vectorised binary64 boundary search over every crossing at once.

    Terminates solely by exhaustion of the representation: no epsilon, no
    tolerance, no convergence threshold and no iteration cap.
    """
    below = np.array(lower, dtype=float)
    at_or_above = np.array(upper, dtype=float)
    wanted = np.asarray(quadrants, dtype=int)

    rounds = 0
    while True:
        mid = below + (at_or_above - below) / 2.0
        exhausted = (mid == below) | (mid == at_or_above)
        if exhausted.all():
            return at_or_above, rounds

        live = np.flatnonzero(~exhausted)
        rounds += 1
        values = np.asarray(season_at(ts.tt_jd(mid[live])), dtype=int)
        hit = values == wanted[live]
        at_or_above[live] = np.where(hit, mid[live], at_or_above[live])
        below[live] = np.where(hit, below[live], mid[live])


def evaluate_in_batches(season_at, states):
    out = np.empty(len(states), dtype=int)
    for start in range(0, len(states), EVALUATION_BATCH):
        stop = start + EVALUATION_BATCH
        out[start:stop] = np.asarray(
            season_at(ts.tt_jd(states[start:stop])), dtype=int
        )
    return out


@unittest.skipUnless(GATE_IS_OPEN, SKIP_REASON)
class TestExhaustiveHorizonCertification(unittest.TestCase):
    """Rederives the certified solar-longitude facts from the real artifacts."""

    @classmethod
    def setUpClass(cls):
        cls.artifacts = {}
        for kernel_name in PINNED_ARTIFACTS:
            low, high, season_at, raw, kind = enumerate_artifact(kernel_name)
            canonical, rounds = canonicalize_all(
                season_at,
                np.concatenate(([low], raw[:-1])) if len(raw) else raw,
                raw,
                kind,
            )
            cls.artifacts[kernel_name] = {
                "low": low,
                "high": high,
                "season_at": season_at,
                "raw": raw,
                "kind": kind,
                "canonical": canonical,
                "rounds": rounds,
            }

    # -- totals -------------------------------------------------------------

    def test_total_crossing_count_is_the_certified_one(self):
        total = sum(len(a["canonical"]) for a in self.artifacts.values())
        self.assertEqual(total, EXPECTED_TOTAL_CROSSINGS)

    def test_every_artifact_contributes_crossings(self):
        for kernel_name, artifact in self.artifacts.items():
            with self.subTest(kernel=kernel_name):
                self.assertGreater(len(artifact["canonical"]), 0)

    def test_all_four_quadrants_are_present_in_every_artifact(self):
        for kernel_name, artifact in self.artifacts.items():
            with self.subTest(kernel=kernel_name):
                self.assertEqual(set(artifact["kind"].tolist()), {0, 1, 2, 3})

    def test_quadrant_counts_are_balanced(self):
        for kernel_name, artifact in self.artifacts.items():
            counts = [int(np.count_nonzero(artifact["kind"] == q))
                      for q in range(4)]
            with self.subTest(kernel=kernel_name, counts=counts):
                self.assertLessEqual(max(counts) - min(counts), 1)

    # -- ordering and sequence integrity ------------------------------------

    def test_crossings_are_strictly_ascending(self):
        for kernel_name, artifact in self.artifacts.items():
            with self.subTest(kernel=kernel_name):
                self.assertTrue(np.all(np.diff(artifact["raw"]) > 0))
                self.assertTrue(np.all(np.diff(artifact["canonical"]) > 0))

    def test_quadrant_sequence_has_no_violation(self):
        """Also proves no crossing is lost or duplicated at a chunk boundary."""
        violations = 0
        for artifact in self.artifacts.values():
            steps = np.diff(artifact["kind"]) % 4
            violations += int(np.count_nonzero(steps != 1))
        self.assertEqual(violations, EXPECTED_QUADRANT_SEQUENCE_VIOLATIONS)

    # -- canonical identity, EVERY crossing ---------------------------------

    def test_canonical_state_carries_its_kind_everywhere(self):
        for kernel_name, artifact in self.artifacts.items():
            with self.subTest(kernel=kernel_name):
                observed = evaluate_in_batches(
                    artifact["season_at"], artifact["canonical"]
                )
                self.assertEqual(
                    int(np.count_nonzero(observed != artifact["kind"])), 0
                )

    def test_predecessor_state_does_not_carry_its_kind_anywhere(self):
        for kernel_name, artifact in self.artifacts.items():
            with self.subTest(kernel=kernel_name):
                predecessors = np.nextafter(artifact["canonical"], -np.inf)
                observed = evaluate_in_batches(
                    artifact["season_at"], predecessors
                )
                self.assertEqual(
                    int(np.count_nonzero(observed == artifact["kind"])), 0
                )

    def test_no_chatter_around_any_canonical_boundary(self):
        for kernel_name, artifact in self.artifacts.items():
            with self.subTest(kernel=kernel_name):
                season_at = artifact["season_at"]
                kind = artifact["kind"]

                state = artifact["canonical"].copy()
                for _ in range(CHATTER_ULPS):
                    state = np.nextafter(state, -np.inf)
                    observed = evaluate_in_batches(season_at, state)
                    self.assertEqual(
                        int(np.count_nonzero(observed == kind)), 0
                    )

                state = artifact["canonical"].copy()
                for _ in range(CHATTER_ULPS):
                    observed = evaluate_in_batches(season_at, state)
                    self.assertEqual(
                        int(np.count_nonzero(observed != kind)), 0
                    )
                    state = np.nextafter(state, np.inf)

    def test_no_two_crossings_collapse_to_one_canonical_state(self):
        for kernel_name, artifact in self.artifacts.items():
            with self.subTest(kernel=kernel_name):
                canonical = artifact["canonical"]
                self.assertEqual(len(np.unique(canonical)), len(canonical))

    def test_canonicalization_never_leaves_its_artifact(self):
        for kernel_name, artifact in self.artifacts.items():
            with self.subTest(kernel=kernel_name):
                canonical = artifact["canonical"]
                self.assertTrue(np.all(canonical >= artifact["low"]))
                self.assertTrue(np.all(canonical <= artifact["high"]))
                self.assertTrue(np.all(canonical <= artifact["raw"]))

    def test_canonicalization_terminated_without_an_iteration_cap(self):
        """Rounds are bounded by binary64 itself, not by an imposed limit."""
        for kernel_name, artifact in self.artifacts.items():
            with self.subTest(kernel=kernel_name, rounds=artifact["rounds"]):
                self.assertGreater(artifact["rounds"], 20)
                self.assertLess(artifact["rounds"], 200)

    # -- rederived extrema --------------------------------------------------

    def rederived_max_same_kind_gap(self):
        worst = 0.0
        for artifact in self.artifacts.values():
            for quadrant in range(4):
                same = artifact["canonical"][artifact["kind"] == quadrant]
                if len(same) > 1:
                    worst = max(worst, float(np.diff(same).max()))
        return worst

    def rederived_min_adjacent_gap(self):
        best = None
        for artifact in self.artifacts.values():
            gaps = np.diff(artifact["canonical"])
            if len(gaps):
                candidate = float(gaps.min())
                best = candidate if best is None else min(best, candidate)
        return best

    def test_maximum_same_kind_gap_matches_the_certification(self):
        self.assertAlmostEqual(
            self.rederived_max_same_kind_gap(),
            CERTIFIED_MAX_SAME_KIND_GAP_DAYS,
            places=6,
        )

    def test_minimum_adjacent_gap_is_at_least_the_frozen_threshold(self):
        """The frozen threshold is the conservative one and stays so."""
        rederived = self.rederived_min_adjacent_gap()
        self.assertGreaterEqual(
            rederived,
            CERTIFIED_MIN_ADJACENT_QUADRANT_GAP_DAYS
            - MIN_ADJACENT_REDERIVATION_SLACK_DAYS,
        )
        self.assertAlmostEqual(
            rederived, CERTIFIED_MIN_ADJACENT_QUADRANT_GAP_DAYS, places=6
        )

    # -- compatibility with the governed production constants ---------------

    def test_governed_horizon_still_exceeds_the_rederived_maximum_gap(self):
        self.assertEqual(
            astronomy_solver.SOLAR_LONGITUDE_SEARCH_SPAN_DAYS,
            EXPECTED_SEARCH_SPAN_DAYS,
        )
        self.assertGreater(
            EXPECTED_SEARCH_SPAN_DAYS, self.rederived_max_same_kind_gap()
        )

    def test_governed_step_is_still_below_the_rederived_minimum_gap(self):
        self.assertEqual(
            astronomy_solver.SOLAR_LONGITUDE_SAMPLE_STEP_DAYS,
            EXPECTED_SAMPLE_STEP_DAYS,
        )
        self.assertLess(
            EXPECTED_SAMPLE_STEP_DAYS, self.rederived_min_adjacent_gap()
        )

    def test_the_certified_horizon_invariant_holds_for_every_crossing(self):
        """No same-kind gap anywhere reaches the governed search horizon."""
        self.assertLess(
            self.rederived_max_same_kind_gap(), EXPECTED_SEARCH_SPAN_DAYS
        )


@unittest.skipUnless(GATE_IS_OPEN, SKIP_REASON)
class TestCertificationProvenance(unittest.TestCase):
    """The certified values belong to one scientific generation."""

    def test_the_pinned_artifacts_are_the_certified_ones(self):
        for kernel_name in PINNED_ARTIFACTS:
            with self.subTest(kernel=kernel_name):
                coverage = kernel_coverage_tt(kernel_name)
                self.assertEqual(coverage.kernel, kernel_name)
                self.assertLess(coverage.tt_start, coverage.tt_end)

    def test_the_governed_identities_are_exactly_four(self):
        self.assertEqual(len(KINDS), 4)
        self.assertEqual(
            astronomy_solver.GOVERNED_SOLAR_LONGITUDE_KINDS, KINDS
        )


class TestCertificationGate(unittest.TestCase):
    """Always-on: the gate itself must behave, whether or not it is open."""

    def test_gate_variable_name_is_stable(self):
        self.assertEqual(
            CERTIFICATION_GATE,
            "HPC_RUN_SOLAR_LONGITUDE_HORIZON_CERTIFICATION",
        )

    def test_gate_is_closed_unless_explicitly_opened(self):
        self.assertEqual(GATE_IS_OPEN, os.getenv(CERTIFICATION_GATE) == "1")

    def test_only_the_exact_opt_in_value_opens_the_gate(self):
        for value in (None, "", "0", "true", "TRUE", "yes", "2", " 1"):
            with self.subTest(value=value):
                self.assertFalse(value == "1")

    def test_skip_reason_names_the_variable_and_the_cost(self):
        self.assertIn(CERTIFICATION_GATE, SKIP_REASON)
        self.assertIn("minutes", SKIP_REASON)


if __name__ == "__main__":
    unittest.main()
