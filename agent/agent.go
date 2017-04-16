// Package agent implements a neural network worker.
package agent

import (
	gym "github.com/openai/gym-http-api/binding-go"
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anyvec/anyvec64"
	"math/rand"
)

// Agent is a container for an environment/instance and works independently.
type Agent struct {
	Client *gym.Client
	Id     gym.InstanceID
	R      *rand.Rand
	Net    anynet.Net
}

// New creates a new Agent
func New(
	client *gym.Client, id gym.InstanceID, net anynet.Net, r *rand.Rand) *Agent {
	agent := new(Agent)
	agent.Client = client
	agent.Id = id
	agent.R = r
	agent.Net = net
	return agent
}

// Action returns an action from the given observation
func (agent *Agent) Action(observation interface{}) interface{} {
	// Set the parameters of the network from the list of parameters
	// and propagate signals
	obs, _ := observation.([]float64)
	input := anydiff.NewConst(
		anyvec64.MakeVectorData(anyvec64.MakeNumericList(obs)))
	output := agent.Net.Apply(input, 1).Output()
	res, _ := output.Data().([]float64)
	var ret int
	if res[0] > 0 {
		ret = 1
	} else {
		ret = 0
	}
	return ret
}
