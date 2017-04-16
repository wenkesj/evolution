// Package opt implements Adam optimization
package opt

import (
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec64"
	"math"
)

// Adam is a data structure for Adaptive Moment Estimation
type Adam struct {
	stepSize, beta1, beta2, epsilon float64
	m, v                            []anyvec.Vector
	t                               float64
}

// NewAdam creates a new Adam instance to optimize with.
func NewAdam(
	params []anyvec.Vector, stepSize, beta1, beta2, epsilon float64) *Adam {
	adam := new(Adam)
	adam.v = make([]anyvec.Vector, len(params))
	adam.m = make([]anyvec.Vector, len(params))
	for i := range adam.v {
		adam.v[i] = anyvec64.MakeVector(params[i].Len())
		adam.m[i] = anyvec64.MakeVector(params[i].Len())
	}
	adam.stepSize = stepSize
	adam.t = 0
	adam.beta1 = beta1
	adam.beta2 = beta2
	adam.epsilon = epsilon
	return adam
}

func (adam *Adam) Update(deltas, params []anyvec.Vector) []anyvec.Vector {
	adam.t += 1
	a := adam.stepSize *
		math.Sqrt(1.0 - math.Pow(adam.beta2, adam.t)) /
		(1.0 - math.Pow(adam.beta1, adam.t))

	for i := range adam.v {
		delta2 := deltas[i].Copy()
		adam.m[i].Scale(adam.beta2)
		deltas[i].Scale(1.0 - adam.beta1)
		adam.m[i].Add(deltas[i])

		adam.v[i].Scale(adam.beta1)
		anyvec.Pow(delta2, anyvec64.MakeNumeric(2))
		delta2.Scale(1.0 - adam.beta2)
		adam.v[i].Add(delta2)

		step := adam.m[i].Copy()
		tmpV := adam.v[i].Copy()

		step.Scale(-a)
		anyvec.Pow(tmpV, anyvec64.MakeNumeric(0.5))
		tmpV.AddScaler(adam.epsilon)
		step.Div(tmpV)
		params[i].Add(step)
	}
	return params
}
