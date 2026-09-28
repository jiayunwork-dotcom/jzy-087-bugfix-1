package geometry

import "math"

// reduceSegment reduces a two-vertex simplex to its feature (one vertex or
// the edge interior) closest to the origin. The slice is stored with the
// newest point first, s[0]=w2, s[1]=w1, but the barycentric decision is
// order-independent; kept points are returned newest-first.
//
// It returns the reduced simplex (one or two support points), the next
// search direction (from the closest point toward the origin) and a flag
// that is true when the origin lies on the feature (boundary contact).
func reduceSegment(s []SupportPoint, tol float64) (contained bool, out []SupportPoint, dir Vec2) {
	w1, w2 := s[1].V, s[0].V
	e := w2.Sub(w1)
	d2 := w1.Dot(e) * -1
	if d2 <= 0 {
		// w1 vertex region.
		if w1.Len() <= tol {
			return true, []SupportPoint{s[1]}, Vec2{}
		}
		return false, []SupportPoint{s[1]}, w1.Scale(-1)
	}
	d1 := w2.Dot(e)
	if d1 <= 0 {
		// w2 vertex region.
		if w2.Len() <= tol {
			return true, []SupportPoint{s[0]}, Vec2{}
		}
		return false, []SupportPoint{s[0]}, w2.Scale(-1)
	}
	// Edge interior region: closest point by barycentric weights.
	inv := 1 / (d1 + d2)
	q := w1.Scale(d1 * inv).Add(w2.Scale(d2 * inv))
	if q.Len() <= tol {
		// Boundary contact: keep the edge so EPA can grow from it.
		return true, []SupportPoint{s[0], s[1]}, Vec2{}
	}
	return false, []SupportPoint{s[0], s[1]}, q.Scale(-1)
}

// reduceTriangle solves the Voronoi region of the origin against a
// three-vertex simplex and returns the reduced closest feature.
//
// It is a direct port of the barycentric simplex solver used by Box2D
// (b2Simplex::Solve3): the plane is partitioned into the three vertex
// cones, the three edge slabs and the triangle interior. Crucially, an
// edge is only kept when the origin is both inside its endpoint slabs
// AND beyond its supporting line on the exterior side (the n123
// sign tests). An earlier implementation dropped the exterior-side test
// and, when two edge lines were violated, simply kept the most-violated
// edge. That misclassification could retain an edge whose true closest
// point was one of its endpoints: the next support point then repeated an
// existing simplex vertex and the iteration terminated with the edge's
// (larger) distance - the reported gap pointed at the edge endpoint
// instead of the perpendicular foot.
//
// The slice stores the newest support point first: s[0]=w3, s[1]=w2,
// s[2]=w1; the Box2D naming below uses oldest-first w1,w2,w3.
func reduceTriangle(s []SupportPoint, tol float64) (contained bool, out []SupportPoint, dir Vec2) {
	// Map newest-first to the Box2D oldest-first convention.
	v1, v2, v3 := s[2], s[1], s[0]
	w1, w2, w3 := v1.V, v2.V, v3.V

	e12 := w2.Sub(w1)
	e13 := w3.Sub(w1)
	e23 := w3.Sub(w2)
	d12_1 := w2.Dot(e12)
	d12_2 := w1.Dot(e12) * -1
	d13_1 := w3.Dot(e13)
	d13_2 := w1.Dot(e13) * -1
	d23_1 := w3.Dot(e23)
	d23_2 := w2.Dot(e23) * -1

	// Signed doubled triangle area and the three vertex barycentric
	// numerators (Box2D: n123 * cross of the other two vertices).
	n123 := e12.Cross(e13)
	d123_1 := n123 * w2.Cross(w3)
	d123_2 := n123 * w3.Cross(w1)
	d123_3 := n123 * w1.Cross(w2)

	// Vertex regions.
	if d12_2 <= 0 && d13_2 <= 0 {
		return vertexContained(v1, w1, tol)
	}
	// Edge w1-w2 (exterior side tested via d123_3 <= 0).
	if d12_1 > 0 && d12_2 > 0 && d123_3 <= 0 {
		inv := 1 / (d12_1 + d12_2)
		q := w1.Scale(d12_1 * inv).Add(w2.Scale(d12_2 * inv))
		return edgeResult([]SupportPoint{v2, v1}, q, tol)
	}
	// Edge w1-w3 (exterior side via d123_2 <= 0).
	if d13_1 > 0 && d13_2 > 0 && d123_2 <= 0 {
		inv := 1 / (d13_1 + d13_2)
		q := w1.Scale(d13_1 * inv).Add(w3.Scale(d13_2 * inv))
		return edgeResult([]SupportPoint{v3, v1}, q, tol)
	}
	if d12_1 <= 0 && d23_2 <= 0 {
		return vertexContained(v2, w2, tol)
	}
	if d13_1 <= 0 && d23_1 <= 0 {
		return vertexContained(v3, w3, tol)
	}
	// Edge w2-w3 (exterior side via d123_1 <= 0).
	if d23_1 > 0 && d23_2 > 0 && d123_1 <= 0 {
		inv := 1 / (d23_1 + d23_2)
		q := w2.Scale(d23_1 * inv).Add(w3.Scale(d23_2 * inv))
		return edgeResult([]SupportPoint{v3, v2}, q, tol)
	}

	// Origin is on or inside the triangle.
	return true, []SupportPoint{v3, v2, v1}, Vec2{}
}

// vertexContained builds the reduction result for a single-vertex feature.
func vertexContained(v SupportPoint, w Vec2, tol float64) (bool, []SupportPoint, Vec2) {
	if w.Len() <= tol {
		return true, []SupportPoint{v}, Vec2{}
	}
	return false, []SupportPoint{v}, w.Scale(-1)
}

// edgeResult builds the reduction result for an interior edge feature,
// keeping the edge (newest first) and pointing toward the origin from q.
func edgeResult(kept []SupportPoint, q Vec2, tol float64) (bool, []SupportPoint, Vec2) {
	if q.Len() <= tol {
		return true, kept, Vec2{}
	}
	return false, kept, q.Scale(-1)
}

// closestLineFeature reduces a degenerate (collinear) triangle to the segment
// or vertex closest to the origin and gives the direction toward the origin.
func closestLineFeature(s []SupportPoint, tol float64) (out []SupportPoint, dir Vec2) {
	bestD := math.Inf(1)
	var bestQ Vec2
	var bestP [2]SupportPoint
	var bestT float64
	pairs := [][2]SupportPoint{{s[0], s[1]}, {s[1], s[2]}, {s[2], s[0]}}
	for _, pr := range pairs {
		ab := pr[1].V.Sub(pr[0].V)
		if ab.Len2() <= tol*tol {
			if d := pr[0].V.Len(); d < bestD {
				bestD = d
				bestQ = pr[0].V
				bestP = pr
				bestT = 0
			}
			continue
		}
		t := clampUnit(pr[0].V.Scale(-1).Dot(ab) / ab.Len2())
		q := pr[0].V.Add(ab.Scale(t))
		if d := q.Len(); d < bestD {
			bestD = d
			bestQ = q
			bestP = pr
			bestT = t
		}
	}
	if bestT <= paramTol || bestT >= 1-paramTol {
		sp := bestP[0]
		if bestT >= 1-paramTol {
			sp = bestP[1]
		}
		return []SupportPoint{sp}, sp.V.Scale(-1)
	}
	return []SupportPoint{bestP[1], bestP[0]}, bestQ.Scale(-1)
}

func clampUnit(t float64) float64 {
	switch {
	case t < 0:
		return 0
	case t > 1:
		return 1
	default:
		return t
	}
}
