package geometry

import (
	"math"
	"testing"
)

// This file pins the reported defect: a vertex of one part facing the
// interior of an edge of the other must yield the perpendicular
// vertex-to-edge distance, with the witness pair on the perpendicular foot —
// never the larger distance to an edge endpoint.
//
// Contours from the field report: A's top edge is y=0, x in [-1,1]; B's
// lowest tip (0,5) hovers above the edge midpoint.
//
// Tolerances: the sweep configurations are exact in binary floating point
// (integer and quarter coordinates), so 1e-9 leaves wide headroom; the
// flagship reported case is asserted at 1e-12.

func reportedCaseA() []Vec2 { return []Vec2{{-2, -4}, {10, -3}, {1, 0}, {-1, 0}} }
func reportedCaseB() []Vec2 { return []Vec2{{-7, 6}, {0, 5}, {2, 8}} }

// checkSeparated asserts the full separated-branch contract: status,
// distance, witness pair, and that the reported pair actually realizes the
// reported distance with the normal pointing from the A witness to the B
// witness.
func checkSeparated(t *testing.T, res *Result, wantDist float64, wantA, wantB Vec2, tol float64) {
	t.Helper()
	if res.Status != StatusSeparated {
		t.Fatalf("status = %s, want separated", res.Status)
	}
	if !approxEq(res.Distance, wantDist, tol) {
		t.Fatalf("distance = %.15g, want %.15g", res.Distance, wantDist)
	}
	if !vecEq(res.PointA, wantA, tol) {
		t.Fatalf("pointA = %v, want %v", res.PointA, wantA)
	}
	if !vecEq(res.PointB, wantB, tol) {
		t.Fatalf("pointB = %v, want %v", res.PointB, wantB)
	}
	if d := res.PointA.Sub(res.PointB).Len(); !approxEq(d, res.Distance, tol) {
		t.Fatalf("witness pair distance %.15g != reported distance %.15g", d, res.Distance)
	}
	nb := res.PointB.Sub(res.PointA).Normalized()
	if !vecEq(res.Normal, nb, 1e-9) {
		t.Fatalf("normal %v not aligned with witness pair direction %v", res.Normal, nb)
	}
}

// TestVertexTipAboveEdgeInterior_ReportedCase is the exact reported
// configuration: distance 5, witnesses (0,0)/(0,5), normal (0,1).
func TestVertexTipAboveEdgeInterior_ReportedCase(t *testing.T) {
	res, err := Evaluate(reportedCaseA(), reportedCaseB())
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	checkSeparated(t, res, 5, Vec2{0, 0}, Vec2{0, 5}, 1e-12)
	if !vecEq(res.Normal, Vec2{0, 1}, 1e-12) {
		t.Fatalf("normal = %v, want (0,1)", res.Normal)
	}
}

// TestVertexTipAboveEdgeInterior_Sweep slides and lifts B through every
// position from the report (and intermediates): the answer must be the
// perpendicular drop at each of them, not only at the originally reported
// spot.
func TestVertexTipAboveEdgeInterior_Sweep(t *testing.T) {
	a := reportedCaseA()
	b := reportedCaseB()

	// Sliding B along x: the tip (dx,5) faces the edge interior for
	// |dx| < 1 and the edge endpoints at dx = ±1; the clearance is the
	// perpendicular drop 5 everywhere on this closed interval.
	for _, dx := range []float64{-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1} {
		foot := math.Max(-1, math.Min(1, dx))
		res, err := Evaluate(a, translate(b, Vec2{X: dx}))
		if err != nil {
			t.Fatalf("dx=%.2f: %v", dx, err)
		}
		checkSeparated(t, res, 5, Vec2{foot, 0}, Vec2{dx, 5}, 1e-9)
		if !vecEq(res.Normal, Vec2{0, 1}, 1e-9) {
			t.Fatalf("dx=%.2f: normal = %v, want (0,1)", dx, res.Normal)
		}
	}

	// Lifting B straight up: the clearance equals the exact vertical drop.
	for _, gap := range []float64{1, 2, 3, 5, 6, 7, 10} {
		res, err := Evaluate(a, translate(b, Vec2{Y: gap - 5}))
		if err != nil {
			t.Fatalf("gap=%.0f: %v", gap, err)
		}
		checkSeparated(t, res, gap, Vec2{0, 0}, Vec2{0, gap}, 1e-9)
		if !vecEq(res.Normal, Vec2{0, 1}, 1e-9) {
			t.Fatalf("gap=%.0f: normal = %v, want (0,1)", gap, res.Normal)
		}
	}
}

// TestVertexTipAboveEdgeInterior_Invariances: the perpendicular answer must
// survive common translation of both parts, operand swap, and winding
// reversal — the defect used to be invariant under all of these.
func TestVertexTipAboveEdgeInterior_Invariances(t *testing.T) {
	a, b := reportedCaseA(), reportedCaseB()

	for _, sh := range []Vec2{{100, 100}, {1000, -500}, {-37.25, 11.5}} {
		res, err := Evaluate(translate(a, sh), translate(b, sh))
		if err != nil {
			t.Fatalf("shift %v: %v", sh, err)
		}
		checkSeparated(t, res, 5, sh, Vec2{sh.X, sh.Y + 5}, 1e-9)
		if !vecEq(res.Normal, Vec2{0, 1}, 1e-9) {
			t.Fatalf("shift %v: normal = %v, want (0,1)", sh, res.Normal)
		}
	}

	// Swapped operands: same gap, flipped normal, swapped witnesses.
	res, err := Evaluate(b, a)
	if err != nil {
		t.Fatalf("swapped: %v", err)
	}
	checkSeparated(t, res, 5, Vec2{0, 5}, Vec2{0, 0}, 1e-12)
	if !vecEq(res.Normal, Vec2{0, -1}, 1e-12) {
		t.Fatalf("swapped: normal = %v, want (0,-1)", res.Normal)
	}

	// Reversed winding of both contours.
	rev := func(pts []Vec2) []Vec2 {
		out := make([]Vec2, len(pts))
		for i, p := range pts {
			out[len(pts)-1-i] = p
		}
		return out
	}
	res, err = Evaluate(rev(a), rev(b))
	if err != nil {
		t.Fatalf("reversed: %v", err)
	}
	checkSeparated(t, res, 5, Vec2{0, 0}, Vec2{0, 5}, 1e-12)
	if !vecEq(res.Normal, Vec2{0, 1}, 1e-12) {
		t.Fatalf("reversed: normal = %v, want (0,1)", res.Normal)
	}
}
