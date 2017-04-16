# evolution
This is a _local_, not _distributed_, [go](https://golang.org/), not _python_, implementation of the [Evolution Strategies as a Scalable
Alternative to Reinforcement Learning](https://arxiv.org/abs/1703.03864)
(Salimans et. al). The original starter from the paper can be found
[openai/evolution-strategies-starter](https://github.com/openai/evolution-strategies-starter).
Under the covers it uses the [openai/gym-http-api](https://github.com/openai/gym-http-api),
more specifically [binding-go](https://github.com/openai/gym-http-api/tree/master/binding-go),
and uses [unixpickle/anynet](https://github.com/unixpickle/anynet) and
[unixpickle/anyvec](https://github.com/unixpickle/anyvec) for efficient
high-level vector computation. Enjoy!

## instructions
The goal is to solve [CartPole-v0](https://gym.openai.com/envs/CartPole-v0),
This requires 195 epochs/reward over 100 episodes. Install [openai/gym](https://github.com/openai/gym),
[openai/gym-http-api](https://github.com/openai/gym-http-api) is a dependency
required from the Go source.

Get the binary. Clone, download, or whatever you want, or just
```sh
$ go get github.com/wenkesj/evolution
```

In a seperate terminal, open the gym from wherever
`github.com/openai/gym-http-api` is located in your fs.
```
$ python gym_http_server.py
```

Run the trainer and evaluater with whatever concauction you choose.

```sh
$ # 200 episodes of "training" by 2 agents and 100
$ # finalepisodes of evaluation with a single agent
$ # Saving results to a directory "~/agents2eps200"
$ evolution --outmonitor ~/agents2eps200 \
  --finalepisodes 100 \
  --episodes 200
  --agents 2
```

## example results
<p align="center">
  <img src="/averages.png" alt="cartpole average training example"/>
</p>

So, after 42 episodes, the 2 agents evolve enough to simply destroy at the game on
their own. In this simple case, we apply a cutoff average reward of 195 or above
for both agents, signifying the parameters on average should be able to solve
the game with a single offspring. So we test that fact,

<p align="center">
  <img src="/averages.gif" alt="cartpole average evaluation example"/>
</p>

And it works! We get 198.5 average reward over 100 episodes!

## roadmap
- [ ] Parallelize where needed to avoid embarrassment :smirk:
- [ ] 32/64 bit support?
- [ ] Support multiple environments [gym-http-api#47](https://github.com/openai/gym-http-api/pull/47)
- [ ] Serialize/deserialize networks the [anynet](https://github.com/unixpickle/anynet) way
  - [ ] Goals `$ evolution -net net.proto -env Pong-v0 ...`
  - [ ] Input network for specific environment (i.e. Pong-v0)
  - [ ] Save/load
- [ ] Optimizations
  - [ ] Adam (ADAM!!!!! :wink:), SGD variable?
  - [ ] [unixpickle/cudavec](https://github.com/unixpickle/cudavec) for gigs
- [ ] Plotting, statistics, performance profiling, uploading

## disclaimer
This is a project for my [Complex Systems and Networks](http://www.ece.uc.edu/~aminai/EECE7065.pdf)
class. This isn't meant to be comparable to the original work; I'm not a master
coder/statistical god/andrej karpathy, I just thought this was a cool idea.
This is an implementation with results and intrepretation.
