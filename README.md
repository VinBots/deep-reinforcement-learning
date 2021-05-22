[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135602-b0335606-7d12-11e8-8689-dd1cf9fa11a9.gif "Trained Agents"


[image2]: https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png "Kernel"


<h1 align="center">
  <br>
  <a href="https://github.com/VinBots/"><img src="logo_drlnd.jpg" width="278" height="208" alt="DRL"></a>
</h1>

<h4 align="center">Initiation to Deep Reinforcement Learning </h4>
<p align="center">
  <a href="#about">About</a> •
  <a href="#learning">Learning</a> •
  <a href="#installation">Installation</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#references">References</a> •
  <a href="#credits">Credits</a> •
  <a href="#going-further">Going Further</a>
</p>

---
<h1 align="center">
  <br>
  <a href=""><img src="https://user-images.githubusercontent.com/10624937/42135602-b0335606-7d12-11e8-8689-dd1cf9fa11a9.gif" width="600" alt="DRL"></a>
</h1>
<h5 align="center">A suite of Deep RL tasks in Unity / Gym environments</h2>


## About

The tutorials lead you through implementing various algorithms in reinforcement learning.  All of the code is in PyTorch (v0.4) and Python 3.

* [Dynamic Programming](https://github.com/udacity/deep-reinforcement-learning/tree/master/dynamic-programming): Implement Dynamic Programming algorithms such as Policy Evaluation, Policy Improvement, Policy Iteration, and Value Iteration. 
* [Monte Carlo](https://github.com/udacity/deep-reinforcement-learning/tree/master/monte-carlo): Implement Monte Carlo methods for prediction and control. 
* [Temporal-Difference](https://github.com/udacity/deep-reinforcement-learning/tree/master/temporal-difference): Implement Temporal-Difference methods such as Sarsa, Q-Learning, and Expected Sarsa. 
* [Discretization](https://github.com/udacity/deep-reinforcement-learning/tree/master/discretization): Learn how to discretize continuous state spaces, and solve the Mountain Car environment.
* [Tile Coding](https://github.com/udacity/deep-reinforcement-learning/tree/master/tile-coding): Implement a method for discretizing continuous state spaces that enables better generalization.
* [Deep Q-Network](https://github.com/udacity/deep-reinforcement-learning/tree/master/dqn): Explore how to use a Deep Q-Network (DQN) to navigate a space vehicle without crashing.
* [Robotics](https://github.com/dusty-nv/jetson-reinforcement): Use a C++ API to train reinforcement learning agents from virtual robotic simulation in 3D. (_External link_)
* [Hill Climbing](https://github.com/udacity/deep-reinforcement-learning/tree/master/hill-climbing): Use hill climbing with adaptive noise scaling to balance a pole on a moving cart.
* [Cross-Entropy Method](https://github.com/udacity/deep-reinforcement-learning/tree/master/cross-entropy): Use the cross-entropy method to train a car to navigate a steep hill.
* [REINFORCE](https://github.com/udacity/deep-reinforcement-learning/tree/master/reinforce): Learn how to use Monte Carlo Policy Gradients to solve a classic control task.
* **Proximal Policy Optimization**: Explore how to use Proximal Policy Optimization (PPO) to solve a classic reinforcement learning task. (_Coming soon!_)
* **Deep Deterministic Policy Gradients**: Explore how to use Deep Deterministic Policy Gradients (DDPG) with OpenAI Gym environments.
  * [Pendulum](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum): Use OpenAI Gym's Pendulum environment.
  * [BipedalWalker](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-bipedal): Use OpenAI Gym's BipedalWalker environment.
* [Finance](https://github.com/udacity/deep-reinforcement-learning/tree/master/finance): Train an agent to discover optimal trading strategies.

## Learning
* Multi-Agent collaboration environment
* Soft Actor-Critic algorithm

## Installation

**1. Download the environment**

Download the environment from one of the links below. Then, unzip (or decompress) the file. You need only select the environment that matches your operating system:

    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)
 

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```
	
2. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
	- Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).
	
3. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

5. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 

![Kernel][image2]


## Configuration

Follow the instructions in `Tennis.ipynb`

## References

* Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor, Haarnoja et al, 2018
* Soft Actor-Critic Algorithms and Applications, Haarnoja et al, 2018
* Learning to Walk via Deep Reinforcement Learning, Haarnoja et al, 2018
* [Open AI](https://spinningup.openai.com/en/latest/algorithms/sac.html)
* [Soft Learning Repo](https://github.com/rail-berkeley/softlearning)


## Credits

* Udacity [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) program.  

## Going Further

* TBC [work in progress]


## OpenAI Gym Benchmarks

### Classic Control
- `Acrobot-v1` with [Tile Coding](https://github.com/udacity/deep-reinforcement-learning/blob/master/tile-coding/Tile_Coding_Solution.ipynb) and Q-Learning  
- `Cartpole-v0` with [Hill Climbing](https://github.com/udacity/deep-reinforcement-learning/blob/master/hill-climbing/Hill_Climbing.ipynb) | solved in 13 episodes
- `Cartpole-v0` with [REINFORCE](https://github.com/udacity/deep-reinforcement-learning/blob/master/reinforce/REINFORCE.ipynb) | solved in 691 episodes 
- `MountainCarContinuous-v0` with [Cross-Entropy Method](https://github.com/udacity/deep-reinforcement-learning/blob/master/cross-entropy/CEM.ipynb) | solved in 47 iterations
- `MountainCar-v0` with [Uniform-Grid Discretization](https://github.com/udacity/deep-reinforcement-learning/blob/master/discretization/Discretization_Solution.ipynb) and Q-Learning | solved in <50000 episodes
- `Pendulum-v0` with [Deep Deterministic Policy Gradients (DDPG)](https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/DDPG.ipynb)

### Box2d
- `BipedalWalker-v2` with [Deep Deterministic Policy Gradients (DDPG)](https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-bipedal/DDPG.ipynb)
- `CarRacing-v0` with **Deep Q-Networks (DQN)** | _Coming soon!_
- `LunarLander-v2` with [Deep Q-Networks (DQN)](https://github.com/udacity/deep-reinforcement-learning/blob/master/dqn/solution/Deep_Q_Network_Solution.ipynb) | solved in 1504 episodes

### Toy Text
- `FrozenLake-v0` with [Dynamic Programming](https://github.com/udacity/deep-reinforcement-learning/blob/master/dynamic-programming/Dynamic_Programming_Solution.ipynb)
- `Blackjack-v0` with [Monte Carlo Methods](https://github.com/udacity/deep-reinforcement-learning/blob/master/monte-carlo/Monte_Carlo_Solution.ipynb)
- `CliffWalking-v0` with [Temporal-Difference Methods](https://github.com/udacity/deep-reinforcement-learning/blob/master/temporal-difference/Temporal_Difference_Solution.ipynb)