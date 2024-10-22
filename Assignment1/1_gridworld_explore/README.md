
## Tasks

Tasks are :
 - to implement the computation of the return of an episode in gridworld.py, see runEpisode function, and the computation of the estimated value of the start state.
 - to provide mean value and the standard deviation over the episodes
 - to find the episode count to be sure mean estimate is within a specific confidence interval of the true mean


### First Subtask: Implementing return computation and value estimation

**Function Overview:** ``runEpisodefunction`` simulates a single episode in the environment, starting from an initial state and progressing step by step until it reaches a terminal state. During each step, the agent selects an action based on the current state, receives a reward, and updates its learning process. The function computes the total return (sum of discounted rewards) accumulated during the episode.

**Initialize Variables:** 

`returns = 0`: This variable keeps track of the total return for the episode. It starts at 0 and will accumulate the discounted rewards as the episode progresses. 

`gamma = 1`: This is the initial discount factor multiplier. It will be updated after each step to discount future rewards based on the discount factor 𝛾 of the value function.

**Start of the Episode**: The function enters a while loop that runs until a terminal state is reached.

**Display and Pause (Optional for Visualization)**:
The current state is displayed, and the program pauses to allow visualization of the agent’s progress.

**Check for Terminal State**: `environment.getPossibleActions(state)` retrieves the possible actions in the current state. If no actions are available, the episode ends, and the total returns are returned.

**Agent Chooses Action**: The agent selects an action based on the current state through action = decision(state). This is typically done using the agent's decision policy.
If the agent fails to return an action, an error is raised.

**Execute Action and Transition**: `environment.doAction(action)` performs the selected action, transitioning to the next state and receiving the reward for that action. The resulting next state and reward are printed for logging purposes.

**Update the Agent**: The agent's learning process is updated using agent.update(state, action, nextState, reward). This helps the agent improve its policy or value estimation based on the current experience.

**Accumulate Discounted Return**: The reward from the current step is multiplied by gamma(discount) factor and added to returns. `gamma` is then updated by multiplying it by the discount rate `discount`, which means future rewards will be increasingly discounted in subsequent steps.

```
returns += gamma * reward
gamma *= discount

```

**Loop Until Episode Ends**: The loop continues until the agent reaches a terminal state, at which point the total return for the episode is returned.

### Second Subtask : Mean and Standard Deviation in Gridworld

**Overview**: In the main loop, we simulate episodes in a Gridworld environment using different types of agents. 
After running the specified number of episodes, we calculate the mean and standard deviation of the returns from these episodes. Additionally, we apply the Central Limit Theorem to determine the required number of episodes for a specified confidence interval.

#### Mathematical Concepts
**Mean**
The mean (or average) of a set of values is calculated by summing all the values and dividing by the total number of values. In the context of our Gridworld episodes, we calculate the mean return from the `returns_sum` list as follows:

```math
Mean = (1/n) * Σ (x_i) for i = 1 to n
```
where 𝑛 is the number of episodes and xi is the return from episode i.

**Calculation in the code**:

After appending each result in the `returns_sum` list, using the numpy library ( `import numpy as np` )

```mean = np.mean(returns_sum)```

**Standard Deviation**

The standard deviation measures the amount of variation or dispersion in a set of values. It is calculated using the following formula:

```math
σ = sqrt((1/(n-1)) * Σ(x_i - μ)²)
```
where: 
σ is the standard deviation, 
μ is the mean of the returns, and 
n is the number of episodes. 

The standard deviation provides insight into how much the episode returns deviate from the mean return.

**Calculation in the code**:


After appending each result in the `returns_sum` list, using the numpy library ( `import numpy as np` )

```std = np.std(returns_sum)```

### Third Subtask: Central Limit Theorem in Gridworld

The Central Limit Theorem states that the sampling distribution of the sample mean will approach a normal distribution as the sample size n increases, regardless of the original distribution of the data, provided the samples are independent and identically distributed (i.i.d.).

This means that for a sufficiently large number of episodes, the distribution of the sample mean will tend to be normally distributed. This allows us to use the mean and standard deviation of the sample to estimate confidence intervals for the true mean of the population.

**Confidence Interval**

A confidence interval gives an estimated range of values which is likely to include the population parameter, with a specified level of confidence. In tis code, we calculate the required number of episodes to achieve a confidence interval of ±1.96 times the standard error of the mean, corresponding to a 95% confidence level.

The required number of episodes can be calculated using the following formula derived from the Central Limit Theorem:

```math
n = (z * σ / E)²
```
where:
n = required number of episodes,
z = z-value corresponding to the desired confidence level (e.g., 1.96 for 95% confidence),
σ = standard deviation of the episode returns,
E = margin of error (in this case, we want the margin of error to be ±0.0004).

**Calculation in the code**:

Required Episodes for Confidence Interval:

Finally, using the calculated standard deviation, we determine the required number of episodes to achieve a 1.96 confidence interval:

``` required_episodes = (z_value * std / confidence_interval) ** 2 ```

