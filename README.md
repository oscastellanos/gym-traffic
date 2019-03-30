This repository contains a PIP package which is an OpenAI environment for
simulating an enironment in which an agent is a traffic signal controller.


## Installation

Install the [OpenAI gym](https://gym.openai.com/docs/).

Then install this package via

```
pip install -e .
```

## Usage

```
import gym
import gym_traffic

env = gym.make('traffic-v1')
```
