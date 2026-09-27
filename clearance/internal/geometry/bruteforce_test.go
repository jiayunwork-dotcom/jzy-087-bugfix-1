package geometry

import (
	"math"
	"math/rand"
	"testing"
)

// TestRandomConvexPairsMatchBruteForce cross-checks the kernel against two
// independent oracles on random convex pairs, covering BOTH outcomes:
//
//   - separated: the distance must equal the brute-force minimum over all
//     edge pairs (point-to-segment distances), and the witness pair must
//     realize it with the normal aligned with the pair;
//   - penetrated: the depth must equal the SAT minimum translation
//     distance (minimum per-axis push over all edge-normal axes — exact
//     for convex polygons), and translating B by normal*depth must bring
//     the parts exactly to a touch.
//
// Tolerances (coordinates are O(10)):
//   - distance/depth vs oracle: 1e-7 relative, 1e-9 absolute;
//   - witness points on their polygon: 1e-7 absolute;
//   - witness-pair offset vs MTV, and residual after the MTV translation:
//     1e-6 absolute;
//   - configurations within 1e-3 of the touching boundary are skipped:
//     there the separated/penetrated decision itself is not robustly
//     decidable, and this test is not about the contact branch.
func TestRandomConvexPairsMatchBruteForce(t *testing.T) {
	const (
		oracleRel  = 1e-7
		oracleAbs  = 1e-9
		onPolyTol  = 1e-7
		contactTol = 1e-6
		skipBand   = 1e-3
	)
	rng := rand.New(rand.NewSource(20260926))
	var nSep, nPen int
	for iter := 0; iter < 400; iter++ {
		a := randomConvex(rng, 3+rng.Intn(5), 3.0)
		b0 := randomConvex(rng, 3+rng.Intn(5), 2.0)
		theta := rng.Float64() * 2 * math.Pi
		mag := 0.5 + rng.Float64()*6.5
		b := translate(b0, Vec2{X: math.Cos(theta) * mag, Y: math.Sin(theta) * mag})

		overlap, separated := satOracle(a, b)
		if math.Abs(overlap) < skipBand {
			continue
		}
		res, err := Evaluate(a, b)
		if err != nil {
			t.Fatalf("iter %d: %v", iter, err)
		}

		if separated {
			nSep++
			if res.Status != StatusSeparated {
				t.Fatalf("iter %d: oracle separated (axis gap %.3g), kernel says %s",
					iter, -overlap, res.Status)
			}
			want := bruteForceDistance(a, b)
			if !approxEqRel(res.Distance, want, oracleRel, oracleAbs) {
				t.Fatalf("iter %d: distance %.10g != brute-force %.10g", iter, res.Distance, want)
			}
			if d := res.PointA.Sub(res.PointB).Len(); !approxEqRel(d, res.Distance, oracleRel, oracleAbs) {
				t.Fatalf("iter %d: witness pair distance %.10g != reported %.10g", iter, d, res.Distance)
			}
			if !pointInOrOn(t, a, res.PointA, onPolyTol) {
				t.Fatalf("iter %d: pointA %v not on A", iter, res.PointA)
			}
			if !pointInOrOn(t, b, res.PointB, onPolyTol) {
				t.Fatalf("iter %d: pointB %v not on B", iter, res.PointB)
			}
			nb := res.PointB.Sub(res.PointA).Normalized()
			if !vecEq(res.Normal, nb, onPolyTol) {
				t.Fatalf("iter %d: normal %v not aligned with witness pair %v", iter, res.Normal, nb)
			}
			continue
		}

		nPen++
		if res.Status != StatusPenetrated {
			t.Fatalf("iter %d: oracle penetrated (overlap %.3g), kernel says %s",
				iter, overlap, res.Status)
		}
		if !approxEqRel(res.PenetrationDepth, overlap, oracleRel, oracleAbs) {
			t.Fatalf("iter %d: depth %.10g != SAT MTVD %.10g", iter, res.PenetrationDepth, overlap)
		}
		if !pointInOrOn(t, a, res.PointA, onPolyTol) {
			t.Fatalf("iter %d: contact pointA %v not on A", iter, res.PointA)
		}
		if !pointInOrOn(t, b, res.PointB, onPolyTol) {
			t.Fatalf("iter %d: contact pointB %v not on B", iter, res.PointB)
		}
		// The contact pair must be offset by exactly the MTV (A-side minus
		// B-side equals normal*depth).
		mtv := res.Normal.Scale(res.PenetrationDepth)
		if d := res.PointA.Sub(res.PointB).Sub(mtv).Len(); d > contactTol {
			t.Fatalf("iter %d: contact pair offset != MTV (residual %.3g)", iter, d)
		}
		// Operational meaning of the MTV: translating B by it brings the
		// parts exactly to a touch (zero-depth contact, either branch may
		// report it within tolerance).
		r2, err := Evaluate(a, translate(b, mtv))
		if err != nil {
			t.Fatalf("iter %d: re-evaluate after MTV: %v", iter, err)
		}
		switch r2.Status {
		case StatusPenetrated:
			if r2.PenetrationDepth > contactTol {
				t.Fatalf("iter %d: after MTV translation depth %.3g remains", iter, r2.PenetrationDepth)
			}
		case StatusSeparated:
			if r2.Distance > contactTol {
				t.Fatalf("iter %d: after MTV translation gap %.3g remains", iter, r2.Distance)
			}
		}
	}
	if nSep < 25 || nPen < 25 {
		t.Fatalf("random mix too skewed to cover both branches: %d separated, %d penetrated", nSep, nPen)
	}
	t.Logf("covered %d separated and %d penetrated random pairs", nSep, nPen)
}

// satOracle is an independent separating-axis reference. Along one
// (unoriented) axis the push distance is the cheaper of shoving B along +u
// (maxA-minB) or along -u (maxB-minA); it is negative exactly when the
// projections are disjoint (a separating axis). For convex polygons the
// minimum of this push over all edge-normal axes equals the exact minimum
// translation distance. (The naive "interval overlap"
// min(maxA,maxB)-max(minA,minB) is only the intersection length and
// underestimates the push whenever one projection nests inside the other.)
// Axis orientation is irrelevant, so no winding handling is needed.
func satOracle(a, b []Vec2) (overlap float64, separated bool) {
	overlap = math.Inf(1)
	for _, poly := range [][]Vec2{a, b} {
		n := len(poly)
		for i := 0; i < n; i++ {
			e := poly[(i+1)%n].Sub(poly[i])
			if e.Len2() == 0 {
				continue
			}
			axis := e.PerpLeft().Normalized()
			minA, maxA := projectionInterval(a, axis)
			minB, maxB := projectionInterval(b, axis)
			if push := math.Min(maxA-minB, maxB-minA); push < overlap {
				overlap = push
			}
		}
	}
	return overlap, overlap < 0
}

func projectionInterval(pts []Vec2, axis Vec2) (float64, float64) {
	lo := pts[0].Dot(axis)
	hi := lo
	for _, p := range pts[1:] {
		if v := p.Dot(axis); v < lo {
			lo = v
		} else if v > hi {
			hi = v
		}
	}
	return lo, hi
}
