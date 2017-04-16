// Package policy implements a generic policy for evaluation of an environment.
package policy

import (
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec64"
	"github.com/wenkesj/evolution/agent"
)

// Policy is an arbirtrary runner/evaluater for an environment.
type Policy struct {
	StepLimit int
}

// New returns a new policy from the stepLimit
func New(stepLimit int) *Policy {
	p := new(Policy)
	p.StepLimit = stepLimit
	return p
}

// Rollout runs an entire episode of the agents environment and returns the
// rewards and epochs ran.
func (p *Policy) Rollout(a *agent.Agent, render bool) (anyvec.Vector, int) {
	var epoch int
	var reward float64
	var done bool
	var err error
	if err != nil {
		panic(err)
	}

	rewards := []float64{}
	observation, err := a.Client.Reset(a.Id)
	if err != nil {
		panic(err)
	}

	for epoch < p.StepLimit {
		epoch++
		action := a.Action(observation).(int)

		observation, reward, done, _, err = a.Client.Step(a.Id, action, render)
		if err != nil {
			panic(err)
		}
		rewards = append(rewards, reward)
		if done {
			break
		}
	}

	return anyvec64.MakeVectorData(anyvec64.MakeNumericList(rewards)), epoch
}
