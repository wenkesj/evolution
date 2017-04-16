package main

import (
	"flag"
	"fmt"
	"math/rand"
	"path"
	"sync"

	"github.com/gonum/plot"
	"github.com/gonum/plot/plotter"
	"github.com/gonum/plot/plotutil"
	"github.com/gonum/plot/vg"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec64"
	"github.com/wenkesj/evolution/agent"
	"github.com/wenkesj/evolution/env"
	"github.com/wenkesj/evolution/noise"
	"github.com/wenkesj/evolution/opt"
	"github.com/wenkesj/evolution/policy"
	"github.com/wenkesj/evolution/util"
)

var (
	creator = anyvec64.CurrentCreator()
	net     = newNet()
)

// newNet creates a new CartPole-v0 net
func newNet() anynet.Net {
	return anynet.Net{
		anynet.NewFC(creator, 4, 4),
		anynet.Tanh,
		anynet.NewFC(creator, 4, 4),
		anynet.Tanh,
		anynet.NewFC(creator, 4, 1),
		anynet.Tanh,
	}
}

// setNetParams sets the params of the network
func setNetParams(net anynet.Net, params []anyvec.Vector) {
	parameters := net.Parameters()
	for i, param := range parameters {
		param.Vector.Set(params[i])
	}
}

func main() {
	var baseURL, environment, monPath string
	var renderAgent, renderFinal bool
	var numAgents, globalStepLimit, episodeLimit, testRuns int
	var globalSeed int64
	var noiseStdDeviation, l2Coefficient, stepSize,
	 	beta1, beta2, epsilon, cutoffEpoch float64

	flag.StringVar(&baseURL, "url", "http://localhost:5000", "openai/gym-http-api url")
	flag.StringVar(&environment, "env", "CartPole-v0", "openai/gym environment")
	flag.StringVar(&monPath, "outmonitor", "", "path to save openai/gym environment monitor")
	flag.BoolVar(&renderAgent, "renderagent", false, "render openai/gym environment for agents (not recommended)")
	flag.BoolVar(&renderFinal, "renderfinal", false, "render openai/gym environment final test (recommended)")
	flag.IntVar(&numAgents, "agents", 2, "number of agents")
	flag.Int64Var(&globalSeed, "seed", 0, "random seed")
	flag.IntVar(&globalStepLimit, "steplimit", 100000, "openai/gym environment step limit")
	flag.IntVar(&episodeLimit, "episodes", 100, "number of episodes to run")
	flag.IntVar(&testRuns, "finalepisodes", 5, "number of episodes to run after training")
	flag.Float64Var(&cutoffEpoch, "cutoff", 180.0, "average agent cutoff training")
	flag.Float64Var(&noiseStdDeviation, "std", 0.02, "noise standard deviation")
	flag.Float64Var(&l2Coefficient, "l2", 0.005, "l2 regularization coefficient")
	flag.Float64Var(&stepSize, "stepsize", 0.01, "optimizer stepsize")
	flag.Float64Var(&beta1, "beta1", 0.9, "optimizer beta1 (adam)")
	flag.Float64Var(&beta2, "beta2", 0.999, "optimizer beta2 (adam)")
	flag.Float64Var(&epsilon, "epsilon", 1e-8, "optimizer epsilon (adam)")
	flag.Parse()

	// Global seeder for initialization
	seeder := rand.New(rand.NewSource(globalSeed))

	// Create the global policy
	p := policy.New(globalStepLimit)

	// Get the initial parameters of the network
	var paramsDimensions int
	parameters := net.Parameters()
	params := make([]anyvec.Vector, len(parameters))
	for i, param := range parameters {
		params[i] = param.Vector
		paramsDimensions += params[i].Len()
	}

	// Create the optimizer
	optimizer := opt.NewAdam(params, stepSize, beta1, beta2, epsilon)

	// Create the noise table
	noiseTable := noise.New(seeder.Int63(), numAgents*paramsDimensions)

	// Create agents
	agents := make([]*agent.Agent, numAgents)
	for i := range agents {
		client, id, err := env.New(baseURL, environment)
		if err != nil {
			panic(err)
		}

		agents[i] = agent.New(
			client, id, newNet(), rand.New(rand.NewSource(seeder.Int63())))
	}

	// Rollout episodes
	wg := new(sync.WaitGroup)
	var averageEpochs []float64
	for episode := 0; episode < episodeLimit; episode++ {
		fmt.Printf("\rEPISODE %d out of %d\n", episode, episodeLimit)
		fmt.Printf("\rAGENTS %d\n", len(agents))

		// Accumulate results
		var allRewards [][2]float64
		var allEpochs [][2]int

		// Share parameters to all agents and compute independently on buffered
		// channels
		rewards := make(chan [2]float64, len(agents))
		epochs := make(chan [2]int, len(agents))
		noiseVectors := make(chan anyvec.Vector, len(agents))

		for _, worker := range agents {
			wg.Add(1)
			go func(
				wg *sync.WaitGroup, worker *agent.Agent, params []anyvec.Vector) {
				defer wg.Done()
				// Make random perturbations
				posParams := make([]anyvec.Vector, len(params))
				negParams := make([]anyvec.Vector, len(params))
				noiseIndex := noiseTable.SampleIndex(worker.R, paramsDimensions)
				noiseVector := noiseTable.Chunk(noiseIndex, paramsDimensions)
				noiseVector.Scale(anyvec64.MakeNumeric(noiseStdDeviation))

				for i := range params {
					posParams[i] = params[i].Copy()
					posParams[i].Add(noiseVector)
					negParams[i] = params[i].Copy()
					negParams[i].Sub(noiseVector)
				}

				// Rollout with the new params
				setNetParams(worker.Net, posParams)
				posRewards, posEpochs := p.Rollout(worker, renderAgent)
				setNetParams(worker.Net, negParams)
				negRewards, negEpochs := p.Rollout(worker, renderAgent)
				posSum, _ := anyvec.Sum(posRewards).(float64)
				negSum, _ := anyvec.Sum(negRewards).(float64)

				// Send index, rewards, and epochs
				noiseVectors <- noiseVector
				rewards <- [2]float64{posSum, negSum}
				epochs <- [2]int{posEpochs, negEpochs}
			}(wg, worker, params)
		} // end agents send
		wg.Wait()
		close(noiseVectors)
		close(rewards)
		close(epochs)

		// Calculate mild-statistics
		sumEpochs := 0
		for epoch := range epochs {
			for i := range epoch {
				sumEpochs += epoch[i]
			}
			allEpochs = append(allEpochs, epoch)
		}
		averageEpoch := float64(sumEpochs) / float64(len(allEpochs)*2)
		fmt.Printf("\rAVERAGE EPOCH %.2f\033[F\033[F", averageEpoch)
		averageEpochs = append(averageEpochs, averageEpoch)

		// Noise
		v := 0
		contiguousNoise := make([]float64, len(agents)*paramsDimensions)
		for noiseVector := range noiseVectors {
			for _, element := range noiseVector.Data().([]float64) {
				contiguousNoise[v] = element
				v++
			}
		}

		// Compute ranks
		for reward := range rewards {
			allRewards = append(allRewards, reward)
		}
		rankedRewards := util.ComputeCenteredRank(allRewards)

		// Get the reward differences
		rewardResults := make([]float64, len(rankedRewards))
		for i := range rankedRewards {
			rewardResults[i] = rankedRewards[i][0] - rankedRewards[i][1]
		}

		// Transpose the vector and the matrix
		// (N,P) X (1,N) = [(P,N) X (N,1)]^T = (P,1) => [(P,1)]^T => (1, P)
		noiseMatrix := anyvec64.MakeVectorData(anyvec64.MakeNumericList(contiguousNoise))
		rewardsMatrix := anyvec64.MakeVectorData(anyvec64.MakeNumericList(rewardResults))

		gradients := anyvec64.MakeVector(paramsDimensions)
		anyvec.Gemv(
			true, len(agents), paramsDimensions, anyvec64.MakeNumeric(1),
			noiseMatrix, paramsDimensions, rewardsMatrix, 1, anyvec64.MakeNumeric(0),
			gradients, 1)

		// Scaled gradients
		gradients.Scale(anyvec64.MakeNumeric(1.0 / float64(len(rankedRewards))))

		// Create the parameter deltas
		deltas := make([]anyvec.Vector, len(params))
		for i := range params {
			deltas[i] = params[i].Copy()
			deltas[i].Scale(anyvec64.MakeNumeric(l2Coefficient))
			deltas[i].Sub(gradients)
		}

		// Update the parameters from the deltas
		params = optimizer.Update(deltas, params)
		if averageEpoch >= cutoffEpoch {
			break
		}
	} // end episode

	for i := range agents {
		agents[i].Client.Close(agents[i].Id)
	}

	// Make final test
	client, id, err := env.New(baseURL, environment)
	if err != nil {
		panic(err)
	}
	defer client.Close(id)
	finalAgent := agent.New(
		client, id, newNet(), rand.New(rand.NewSource(seeder.Int63())))
	setNetParams(finalAgent.Net, params)

	if monPath != "" {
		err = finalAgent.Client.StartMonitor(
			finalAgent.Id, monPath, true, false, true)
		defer finalAgent.Client.CloseMonitor(finalAgent.Id)
		if err != nil {
			panic(err)
		}

		outPlotPoints := make(plotter.XYs, len(averageEpochs))
		outPlot, err := plot.New()
		outPlot.Title.Text = "Average Epochs"
		outPlot.X.Label.Text = "Episode"
		outPlot.Y.Label.Text = "Reward"

		for i, average := range averageEpochs {
			outPlotPoints[i].X, outPlotPoints[i].Y = float64(i), average
		}

		err = plotutil.AddLines(outPlot, "Average Epochs", outPlotPoints)
		if err != nil {
			panic(err)
		}

		if err = outPlot.Save(6*vg.Inch, 6*vg.Inch, path.Join(monPath, "averages.png")); err != nil {
			panic(err)
		}
	}

	fmt.Println()
	// Ensure env doesn't roll over
	p.StepLimit = 10000000000
	testE := 0
	maxE := -1
	for i := 0; i < testRuns; i++ {
		fmt.Printf("\rTEST EPISODE %d out of %d\n", i, testRuns)
		_, epochs := p.Rollout(finalAgent, renderFinal)
		testE += epochs
		if epochs > maxE {
			maxE = epochs
		}
	}
	averageE := float64(testE) / float64(testRuns+1)

	fmt.Println()
	fmt.Printf(
		"(test) MAX EPOCHS %d TOTAL EPOCHS %d AVERAGE %.2f", maxE, testE, averageE)
}
