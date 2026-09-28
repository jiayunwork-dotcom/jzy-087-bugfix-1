package geometry

import (
	"math"
	"math/rand"
	"sort"
	"testing"
)

// Regression for the vertex-over-edge gap bug: a sharp vertex of B hangs
// over the INTERIOR of a straight edge of A. The clearance must be the
// perpendicular vertex-to-edge distance, with the A-witness at the foot of
// the perpendicular - never snapped to the edge endpoint.
//
//	A = (-2,-4),(10,-3),(1,0),(-1,0); top edge is y=0 for x in [-1,1].
//	B = (-7,6),(0,5),(2,8); lowest vertex (0,5) over the edge midpoint.
//
// The correct answer is distance 5, pointA=(0,0), pointB=(0,5),
// normal=(0,1). The broken kernel returned sqrt(26) ~= 5.0990 with
// pointA=(1,0) and normal ~=(-0.196,0.981): it kept a CSO edge whose true
// closest feature was one of its endpoints, then terminated on a repeated
// support point.
func TestVertexOverEdge_PerpendicularGap(t *testing.T) {
	a := []Vec2{{-2, -4}, {10, -3}, {1, 0}, {-1, 0}}
	b := []Vec2{{-7, 6}, {0, 5}, {2, 8}}
	res, err := Evaluate(a, b)
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	if res.Status != StatusSeparated {
		t.Fatalf("status = %s, want separated", res.Status)
	}
	if !approxEq(res.Distance, 5, exactGapTol) {
		t.Fatalf("distance = %.15g, want exactly 5 (got endpoint-snapped gap)", res.Distance)
	}
	if !vecEq(res.PointA, Vec2{X: 0, Y: 0}, exactGapTol) {
		t.Fatalf("pointA = %v, want perpendicular foot (0,0)", res.PointA)
	}
	if !vecEq(res.PointB, Vec2{X: 0, Y: 5}, exactGapTol) {
		t.Fatalf("pointB = %v, want (0,5)", res.PointB)
	}
	if !vecEq(res.Normal, Vec2{X: 0, Y: 1}, exactGapTol) {
		t.Fatalf("normal = %v, want (0,1)", res.Normal)
	}
	if d := res.PointA.Sub(res.PointB).Len(); !approxEq(d, res.Distance, exactGapTol) {
		t.Fatalf("witness distance %.15g != reported %.15g", d, res.Distance)
	}
}

// TestVertexOverEdge_SlideAlongEdge moves the B vertex across the whole
// interior span of the A edge and to both sides of its endpoints. At every
// position the gap is the constant vertical drop 5; the A-witness must be
// the foot (clamped to the edge endpoints only outside the span), and the
// witness pair must realize the reported distance.
func TestVertexOverEdge_SlideAlongEdge(t *testing.T) {
	a := []Vec2{{-2, -4}, {10, -3}, {1, 0}, {-1, 0}}
	for _, dx := range []float64{-1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0} {
		b := []Vec2{{-7 + dx, 6}, {dx, 5}, {2 + dx, 8}}
		res, err := Evaluate(a, b)
		if err != nil {
			t.Fatalf("dx=%g: %v", dx, err)
		}
		if !approxEq(res.Distance, 5, exactGapTol) {
			t.Fatalf("dx=%g: distance = %.15g, want 5", dx, res.Distance)
		}
		// Foot x is clamped to [-1,1] only once the vertex leaves the span.
		wantFootX := math.Max(-1, math.Min(1, dx))
		if !vecEq(res.PointA, Vec2{X: wantFootX, Y: 0}, exactGapTol) {
			t.Fatalf("dx=%g: pointA = %v, want foot (%g,0)", dx, res.PointA, wantFootX)
		}
		if !vecEq(res.PointB, Vec2{X: dx, Y: 5}, exactGapTol) {
			t.Fatalf("dx=%g: pointB = %v, want (%g,5)", dx, res.PointB, dx)
		}
		assertWitnessPair(t, res)
		assertNormalAlongWitnesses(t, res, exactGapTol)
	}
}

// TestVertexOverEdge_VaryingHeight checks several perpendicular drops, in
// particular the heights 6 and 7 for which the broken kernel reported
// 6.0828 and 7.0711 (sqrt(37), sqrt(50)).
func TestVertexOverEdge_VaryingHeight(t *testing.T) {
	a := []Vec2{{-2, -4}, {10, -3}, {1, 0}, {-1, 0}}
	for _, h := range []float64{0.5, 1, 3, 5, 6, 7, 10, 25.5} {
		// Triangle with lowest vertex (0,h); the other vertices stay above.
		b := []Vec2{{-7, h + 1}, {0, h}, {2, h + 3}}
		res, err := Evaluate(a, b)
		if err != nil {
			t.Fatalf("h=%g: %v", h, err)
		}
		if !approxEq(res.Distance, h, h*1e-12+1e-12) {
			t.Fatalf("h=%g: distance = %.15g, want %g", h, res.Distance, h)
		}
		if !vecEq(res.PointA, Vec2{X: 0, Y: 0}, 1e-10) {
			t.Fatalf("h=%g: pointA = %v, want (0,0)", h, res.PointA)
		}
		if !vecEq(res.PointB, Vec2{X: 0, Y: h}, 1e-10) {
			t.Fatalf("h=%g: pointB = %v, want (0,%g)", h, res.PointB, h)
		}
		assertWitnessPair(t, res)
		assertNormalAlongWitnesses(t, res, 1e-10)
	}
}

// TestVertexOverEdge_Invariants confirms the fix is independent of input
// ordering, polygon winding and the absolute coordinate frame.
func TestVertexOverEdge_Invariants(t *testing.T) {
	a := []Vec2{{-2, -4}, {10, -3}, {1, 0}, {-1, 0}}
	b := []Vec2{{-7, 6}, {0, 5}, {2, 8}}
	want := Vec2{X: 0, Y: 1}

	// Swap A and B: normal flips to point from the new A toward the new B.
	swap, err := Evaluate(b, a)
	if err != nil {
		t.Fatalf("swapped: %v", err)
	}
	if !approxEq(swap.Distance, 5, exactGapTol) {
		t.Fatalf("swapped distance = %.15g", swap.Distance)
	}
	if !vecEq(swap.Normal, want.Scale(-1), exactGapTol) {
		t.Fatalf("swapped normal = %v, want (0,-1)", swap.Normal)
	}
	assertWitnessPair(t, swap)

	// Clockwise winding of B.
	cwB := []Vec2{{2, 8}, {0, 5}, {-7, 6}}
	resCW, err := Evaluate(a, cwB)
	if err != nil {
		t.Fatalf("cw: %v", err)
	}
	if !approxEq(resCW.Distance, 5, exactGapTol) || !vecEq(resCW.Normal, want, exactGapTol) {
		t.Fatalf("cw: d=%.15g n=%v", resCW.Distance, resCW.Normal)
	}

	// Common translations, including large frames where relative tolerances
	// are what can be met with absolute coordinates ~1000.
	for _, sh := range []Vec2{{100, 100}, {1000, -500}, {-250.5, 10000}} {
		as := translate(a, sh)
		bs := translate(b, sh)
		res, err := Evaluate(as, bs)
		if err != nil {
			t.Fatalf("shift %v: %v", sh, err)
		}
		if !approxEqRel(res.Distance, 5, relTol, absTol) {
			t.Fatalf("shift %v: distance = %.15g", sh, res.Distance)
		}
		if !vecEq(res.PointA, Vec2{X: 0, Y: 0}.Add(sh), 1e-9*math.Max(1, math.Abs(sh.X))) {
			t.Fatalf("shift %v: pointA = %v", sh, res.PointA)
		}
		assertWitnessPair(t, res)
	}
}

// TestVertexOverEdge_Deterministic repeats the same request and requires
// bit-identical answers (the defect report noted the answer was stable but
// wrong; the correct answer must be stable as well).
func TestVertexOverEdge_Deterministic(t *testing.T) {
	a := []Vec2{{-2, -4}, {10, -3}, {1, 0}, {-1, 0}}
	b := []Vec2{{-7, 6}, {0, 5}, {2, 8}}
	first, err := Evaluate(a, b)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 10; i++ {
		r, err := Evaluate(a, b)
		if err != nil {
			t.Fatal(err)
		}
		if r.Distance != first.Distance || r.PointA != first.PointA ||
			r.PointB != first.PointB || r.Normal != first.Normal {
			t.Fatalf("repeat %d differs: %+v vs %+v", i, r, first)
		}
	}
}

// assertWitnessPair requires the reported closest points to be exactly the
// reported distance apart (within the stated kernel tolerance).
func assertWitnessPair(t *testing.T, res *Result) {
	t.Helper()
	d := res.PointA.Sub(res.PointB).Len()
	tol := 1e-10 * math.Max(1, math.Max(math.Abs(res.PointA.Len()), math.Abs(res.PointB.Len())))
	if !approxEq(d, res.Distance, tol) {
		t.Fatalf("witness pair distance %.15g != reported distance %.15g", d, res.Distance)
	}
}

// assertNormalAlongWitnesses requires the gap normal to be the unit vector
// from pointA toward pointB.
func assertNormalAlongWitnesses(t *testing.T, res *Result, tol float64) {
	t.Helper()
	nb := res.PointB.Sub(res.PointA).Normalized()
	if !vecEq(res.Normal, nb, tol) {
		t.Fatalf("normal %v does not point from pointA %v to pointB %v (want %v)",
			res.Normal, res.PointA, res.PointB, nb)
	}
	if l := res.Normal.Len(); math.Abs(l-1) > 1e-12 {
		t.Fatalf("normal not unit: len=%.15g", l)
	}
}

// brutePointToSegments is the independent oracle for the separated branch:
// minimum of every A-vertex-to-B-segment and B-vertex-to-A-segment distance,
// which for convex polygons is exactly the minimum inter-polygon distance
// (and covers vertex-edge, vertex-vertex and edge-edge witness types).
func brutePointToSegments(a, b []Vec2) float64 {
	best := math.Inf(1)
	segDist := func(p, x, y Vec2) float64 {
		e := y.Sub(x)
		if e.Len2() == 0 {
			return p.Sub(x).Len()
		}
		tt := clampUnit(p.Sub(x).Dot(e) / e.Len2())
		return p.Sub(x.Add(e.Scale(tt))).Len()
	}
	for i := range a {
		a0, a1 := a[i], a[(i+1)%len(a)]
		for j := range b {
			b0, b1 := b[j], b[(j+1)%len(b)]
			best = math.Min(best, segDist(a0, b0, b1))
			best = math.Min(best, segDist(a1, b0, b1))
			best = math.Min(best, segDist(b0, a0, a1))
			best = math.Min(best, segDist(b1, a0, a1))
		}
	}
	return best
}

// randomConvexIndependent builds a random strictly-convex CCW polygon by
// hulling uniform-in-disk points, independently of the kernel's own hull
// code (no shared implementation with what is under test). It rejects
// collinear hull points and nearly-zero edges so witnesses are unambiguous.
func randomConvexIndependent(rng *rand.Rand, n int, radius float64) []Vec2 {
	for {
		pts := make([]Vec2, 4*n)
		for i := range pts {
			r := radius * math.Sqrt(rng.Float64())
			th := rng.Float64() * 2 * math.Pi
			pts[i] = Vec2{X: r * math.Cos(th), Y: r * math.Sin(th)}
		}
		sorted := append([]Vec2(nil), pts...)
		sort.Slice(sorted, func(i, j int) bool {
			if sorted[i].X != sorted[j].X {
				return sorted[i].X < sorted[j].X
			}
			return sorted[i].Y < sorted[j].Y
		})
		cross := func(o, p, q Vec2) float64 { return p.Sub(o).Cross(q.Sub(o)) }
		lower := []Vec2{}
		for _, p := range sorted {
			for len(lower) >= 2 && cross(lower[len(lower)-2], lower[len(lower)-1], p) <= 0 {
				lower = lower[:len(lower)-1]
			}
			lower = append(lower, p)
		}
		upper := []Vec2{}
		for i := len(sorted) - 1; i >= 0; i-- {
			p := sorted[i]
			for len(upper) >= 2 && cross(upper[len(upper)-2], upper[len(upper)-1], p) <= 0 {
				upper = upper[:len(upper)-1]
			}
			upper = append(upper, p)
		}
		hull := append(lower[:len(lower)-1], upper[:len(upper)-1]...)
		if len(hull) < n {
			continue
		}
		ok := len(hull) >= 3
		for i := range hull {
			if hull[(i+1)%len(hull)].Sub(hull[i]).Len2() < 1e-6 {
				ok = false
			}
		}
		if ok {
			return hull
		}
	}
}

// satPenetration is the independent oracle for the penetrated branch: by the
// separating-axis theorem, two convex polygons are disjoint iff there is an
// edge normal on which their projected intervals do not overlap; when all
// axes overlap, the minimum overlap is the minimum-translation depth. It
// returns (depth, axis oriented from A toward B, true); the boolean is false
// when a separating axis exists. This implementation is fully independent
// of the GJK/EPA kernel.
func satPenetration(a, b []Vec2) (float64, Vec2, bool) {
	bestDepth := math.Inf(1)
	var bestAxis Vec2
	separated := false
	check := func(poly, other []Vec2) {
		for i := range poly {
			e := poly[(i+1)%len(poly)].Sub(poly[i])
			if e.Len2() == 0 {
				continue
			}
			n := Vec2{X: e.Y, Y: -e.X}.Normalized() // CCW: outward (right)
			loP, hiP := math.Inf(1), math.Inf(-1)
			loO, hiO := math.Inf(1), math.Inf(-1)
			for _, p := range poly {
				v := p.Dot(n)
				loP, hiP = math.Min(loP, v), math.Max(hiP, v)
			}
			for _, p := range other {
				v := p.Dot(n)
				loO, hiO = math.Min(loO, v), math.Max(hiO, v)
			}
			if hiO <= loP || hiP <= loO {
				separated = true // a separating axis exists
				continue
			}
			overlap := math.Min(hiP-loO, hiO-loP)
			if overlap < bestDepth {
				bestDepth = overlap
				// Orient n A->B: B moves along +n when its center lies on
				// the +n side of A's center; flip otherwise.
				if center(other).Sub(center(poly)).Dot(n) < 0 {
					n = n.Scale(-1)
				}
				bestAxis = n
			}
		}
	}
	check(a, b)
	check(b, a)
	if separated {
		return 0, Vec2{}, false
	}
	return bestDepth, bestAxis, true
}

func center(pts []Vec2) Vec2 {
	var c Vec2
	for _, p := range pts {
		c = c.Add(p)
	}
	return c.Scale(1 / float64(len(pts)))
}

// TestRandomConvex_VersusBruteForce_Distance is the broad differential test
// against the independent brute-force oracles. Random convex polygons are
// placed at a spread of relative offsets (including deliberately close
// ones, which specifically exercise the vertex-over-edge interior case),
// and BOTH conclusions are checked:
//
//   - separated: status, distance, witness-pair distance and normal
//     direction against the vertex-to-segment brute force;
//   - penetrated: penetration depth and normal against an independent SAT
//     minimum-translation oracle, plus the operational MTV check.
//
// Tolerances (coordinates O(10)):
//
//   - distance/depth: |d - oracle| <= 1e-8 + 1e-9*oracle
//   - witness pair realization: same
//   - normal vs witness/SAT direction: component error <= 1e-8
//   - after-MTV touch residual: <= 1e-7
func TestRandomConvex_VersusBruteForce_Distance(t *testing.T) {
	rng := rand.New(rand.NewSource(20240928))
	const N = 2000
	var separated, penetrated int
	for iter := 0; iter < N; iter++ {
		a := randomConvexIndependent(rng, 3+rng.Intn(6), 3.0)
		b0 := randomConvexIndependent(rng, 3+rng.Intn(6), 2.0)
		// Mostly clearly separated, but ~20% close placements so the
		// vertex-over-edge-interior and near-contact regimes are hit often.
		var sh Vec2
		if rng.Intn(5) == 0 {
			// Close: radii up to 3 and 2, centers O(1) apart -> a mix of
			// deep overlap, touch and hair-gap cases.
			sh = Vec2{X: rng.NormFloat64(), Y: rng.NormFloat64()}
		} else {
			theta := rng.Float64() * 2 * math.Pi
			r := 3.5 + rng.Float64()*6
			sh = Vec2{X: math.Cos(theta) * r, Y: math.Sin(theta) * r}
		}
		b := translate(b0, sh)

		res, err := Evaluate(a, b)
		if err != nil {
			t.Fatalf("iter %d: evaluate: %v", iter, err)
		}
		gap := brutePointToSegments(a, b)
		depth, satAxis, overlap := satPenetration(a, b)

		// First, the kernel's branch must agree with the independent
		// branch oracle (only enforce away from the touching boundary,
		// where SAT/brute-force flip deterministically and the kernel is
		// allowed to route boundary contact through EPA).
		const branchGap = 1e-7
		if gap > branchGap && !overlap && res.Status != StatusSeparated {
			t.Fatalf("iter %d: oracle says separated gap %.12g, kernel says %s",
				iter, gap, res.Status)
		}
		if overlap && depth > branchGap && res.Status != StatusPenetrated {
			t.Fatalf("iter %d: SAT says overlap depth %.12g, kernel says %s",
				iter, depth, res.Status)
		}

		switch res.Status {
		case StatusSeparated:
			separated++
			distTol := 1e-8 + 1e-9*gap
			if !approxEq(res.Distance, gap, distTol) {
				t.Fatalf("iter %d: distance %.12g != brute-force %.12g",
					iter, res.Distance, gap)
			}
			wd := res.PointA.Sub(res.PointB).Len()
			if !approxEq(wd, gap, distTol) {
				t.Fatalf("iter %d: witness distance %.12g != oracle %.12g",
					iter, wd, gap)
			}
			// Normal must be the unit witness direction.
			if !vecEq(res.Normal, res.PointB.Sub(res.PointA).Normalized(), 1e-8) {
				t.Fatalf("iter %d: normal %v != witness direction", iter, res.Normal)
			}
			if !pointInOrOn(t, a, res.PointA, 1e-8) {
				t.Fatalf("iter %d: pointA %v outside A", iter, res.PointA)
			}
			if !pointInOrOn(t, b, res.PointB, 1e-8) {
				t.Fatalf("iter %d: pointB %v outside B", iter, res.PointB)
			}
		case StatusPenetrated:
			penetrated++
			if math.Abs(res.Normal.Len()-1) > 1e-12 {
				t.Fatalf("iter %d: penetrated normal not unit: %v", iter, res.Normal)
			}
			// Direct comparison against the independent SAT depth/normal.
			// (Skipped in the near-boundary band where the SAT tie-break is
			// not unique and the depth is within the tolerance anyway.)
			if overlap && depth > 1e-5 {
				depthTol := 1e-8 + 1e-9*depth
				if !approxEq(res.PenetrationDepth, depth, depthTol) {
					t.Fatalf("iter %d: EPA depth %.12g != SAT depth %.12g",
						iter, res.PenetrationDepth, depth)
				}
				if !vecEq(res.Normal, satAxis, 1e-7) &&
					!vecEq(res.Normal, satAxis.Scale(-1), 1e-7) {
					t.Fatalf("iter %d: EPA normal %v != SAT axis %v",
						iter, res.Normal, satAxis)
				}
			}
			// The MTV translation must produce a zero-gap touch (neither a
			// remaining overlap nor a large spurious gap).
			freed := translate(b, res.Normal.Scale(res.PenetrationDepth))
			r2, err := Evaluate(a, freed)
			if err != nil {
				t.Fatalf("iter %d: post-MTV evaluate: %v", iter, err)
			}
			if r2.Status == StatusPenetrated && r2.PenetrationDepth > 1e-7 {
				t.Fatalf("iter %d: MTV depth %.12g insufficient, remains %.12g",
					iter, res.PenetrationDepth, r2.PenetrationDepth)
			}
			if r2.Status == StatusSeparated && r2.Distance > 1e-7 {
				t.Fatalf("iter %d: MTV depth %.12g excessive, leaves gap %.12g",
					iter, res.PenetrationDepth, r2.Distance)
			}
			if !pointInOrOn(t, a, res.PointA, 1e-8) {
				t.Fatalf("iter %d: contact pointA %v outside A", iter, res.PointA)
			}
			if !pointInOrOn(t, b, res.PointB, 1e-8) {
				t.Fatalf("iter %d: contact pointB %v outside B", iter, res.PointB)
			}
		}
	}
	if separated == 0 || penetrated == 0 {
		t.Fatalf("placement mix failed to cover both branches: sep=%d pen=%d",
			separated, penetrated)
	}
	t.Logf("compared %d cases: %d separated, %d penetrated vs brute force",
		N, separated, penetrated)
}

// reverse returns the vertices in the opposite boundary order (CW<->CCW).
func reverse(pts []Vec2) []Vec2 {
	out := make([]Vec2, len(pts))
	for i := range pts {
		out[i] = pts[len(pts)-1-i]
	}
	return out
}

// TestRandomConvex_OrderAndFrameInvariance runs the same differential
// comparison with (a) swapped arguments, (b) clockwise vertex order and
// (c) a large common translation, on every generated pair. The distance and
// depth must be invariant; the normal must flip consistently when A and B
// swap.
func TestRandomConvex_OrderAndFrameInvariance(t *testing.T) {
	rng := rand.New(rand.NewSource(777))
	const N = 800
	for iter := 0; iter < N; iter++ {
		a0 := randomConvexIndependent(rng, 3+rng.Intn(5), 2.0)
		b0 := randomConvexIndependent(rng, 3+rng.Intn(5), 1.5)
		theta := rng.Float64() * 2 * math.Pi
		r := 2.5 + rng.Float64()*5
		b0 = translate(b0, Vec2{X: math.Cos(theta) * r, Y: math.Sin(theta) * r})

		base, err := Evaluate(a0, b0)
		if err != nil {
			t.Fatalf("iter %d base: %v", iter, err)
		}

		// Swapped: distance invariant, normal flipped.
		sw, err := Evaluate(b0, a0)
		if err != nil {
			t.Fatalf("iter %d swapped: %v", iter, err)
		}
		if sw.Status != base.Status {
			t.Fatalf("iter %d swapped status %s != %s", iter, sw.Status, base.Status)
		}
		baseMag := base.Distance + base.PenetrationDepth
		if !approxEqRel(sw.Distance+sw.PenetrationDepth, baseMag, 1e-9, 1e-9) {
			t.Fatalf("iter %d swapped magnitude changed", iter)
		}
		if !vecEq(sw.Normal, base.Normal.Scale(-1), 1e-8) {
			t.Fatalf("iter %d swapped normal %v != -%v", iter, sw.Normal, base.Normal)
		}

		// Both contours clockwise: identical magnitude and normal.
		cw, err := Evaluate(reverse(a0), reverse(b0))
		if err != nil {
			t.Fatalf("iter %d cw: %v", iter, err)
		}
		if cw.Status != base.Status ||
			!approxEqRel(cw.Distance+cw.PenetrationDepth, baseMag, 1e-9, 1e-9) ||
			!vecEq(cw.Normal, base.Normal, 1e-8) {
			t.Fatalf("iter %d clockwise changed result: %+v vs %+v", iter, cw, base)
		}

		// Large common translation: invariant magnitude, normal, and the
		// translated witnesses.
		sh := Vec2{X: 1000, Y: -500}
		moved, err := Evaluate(translate(a0, sh), translate(b0, sh))
		if err != nil {
			t.Fatalf("iter %d moved: %v", iter, err)
		}
		frame := 1e-9 * 1000
		if moved.Status != base.Status ||
			!approxEqRel(moved.Distance+moved.PenetrationDepth, baseMag, 1e-9, frame) ||
			!vecEq(moved.Normal, base.Normal, frame) ||
			!vecEq(moved.PointA, base.PointA.Add(sh), frame) ||
			!vecEq(moved.PointB, base.PointB.Add(sh), frame) {
			t.Fatalf("iter %d common translation changed result", iter)
		}
	}
}

// TestRandomConvex_NearEdgeInterior specifically forces a vertex of B into
// the interior of an A edge by construction: take any random A, a random
// edge of it, and place a random B triangle whose vertex sits at a small
// known distance over a random interior point of that edge, with the rest
// of B on the outward side. The answer must be exactly the forced gap at
// every trial - this is the deterministic version of the original defect.
func TestRandomConvex_NearEdgeInterior(t *testing.T) {
	rng := rand.New(rand.NewSource(99))
	for iter := 0; iter < 500; iter++ {
		a := randomConvexIndependent(rng, 3+rng.Intn(5), 5.0)
		i := rng.Intn(len(a))
		p0, p1 := a[i], a[(i+1)%len(a)]
		e := p1.Sub(p0)
		// Outward unit normal: A is CCW, so right normal of p0->p1.
		nOut := Vec2{X: e.Y, Y: -e.X}.Normalized()
		// Random interior point strictly inside the open edge.
		u := 0.05 + 0.9*rng.Float64()
		foot := p0.Add(e.Scale(u))
		gap := 0.05 + rng.Float64()*4
		apex := foot.Add(nOut.Scale(gap))
		// Two more B vertices further outward and spread tangentially.
		tan := e.Normalized()
		s := 0.3 + rng.Float64()*1.5
		b := []Vec2{
			apex,
			apex.Add(nOut.Scale(1 + rng.Float64()*2)).Add(tan.Scale(s)),
			apex.Add(nOut.Scale(1 + rng.Float64()*2)).Add(tan.Scale(-s)),
		}
		// Validate convexity (the construction can produce a sliver for
		// tiny edges); skip invalid ones rather than biasing the oracle.
		if _, err := NewPolygon(b); err != nil {
			continue
		}
		res, err := Evaluate(a, b)
		if err != nil {
			t.Fatalf("iter %d: %v", iter, err)
		}
		if res.Status != StatusSeparated {
			// Other B vertices could bend toward A for tiny edges; such
			// constructions are skipped by the gap check below.
			continue
		}
		tol := 1e-8 + 1e-9*gap
		if !approxEq(res.Distance, gap, tol) {
			t.Fatalf("iter %d: gap %.12g, forced %.12g (pointA %v foot %v)",
				iter, res.Distance, gap, res.PointA, foot)
		}
		if !vecEq(res.PointA, foot, 1e-7) {
			t.Fatalf("iter %d: pointA %v not the perpendicular foot %v",
				iter, res.PointA, foot)
		}
		if !vecEq(res.PointB, apex, 1e-7) {
			t.Fatalf("iter %d: pointB %v, want apex %v", iter, res.PointB, apex)
		}
		assertWitnessPair(t, res)
	}
}
