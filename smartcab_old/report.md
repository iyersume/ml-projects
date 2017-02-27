## Report for Smartcab system

### Implement a Basic Driving Agent

***QUESTION:*** _Observe what you see with the agent's behavior as it takes
random actions. Does the **smartcab** eventually make it to the destination?
Are there any other interesting observations to note?_

As expected, agent takes random actions at each intersections. There are
multiple instances when the action taken is against the signal at the
intersection.

The agent eventually does make it to the destination in most cases for multiple
runs.

### Inform the Driving Agent

***QUESTION:*** _What states have you identified that are appropriate for modeling the **smartcab** and environment? Why do you believe each of these states to be appropriate for this problem?_

The state space needs to consider following dimensions and their possible values:

Next waypoint: None (already at goal), Forward, Left, Right
Light: Green or Red
Oncoming: None (no vehicle), Forward, Left, Right
Left: None (no vehicle), Forward, Left, Right
Right: None (no vehicle), Forward, Left, Right

The above states give the complete picture of the environment at each step. The only information missing there is the deadline. Even though that is a discrete quantity, it that takes a lot of values and does provide enough information to help the agent.

***OPTIONAL:*** _How many states in total exist for the **smartcab** in this environment? Does this number seem reasonable given that the goal of Q-Learning is to learn and make informed decisions about each state? Why or why not?_

A total of 2*4*4*4*4 = 512 states exist in this environment, with 4 actions
possible in each state. Thus total state-action pairs is 2048, which is
reasonable for Q-learning, since total memory required will be around 16 KB and
it won't take too long to learn that within that space.


### Implement a Q-Learning Driving Agent

***QUESTION:*** _What changes do you notice in the agent's behavior when compared to the basic driving agent when random actions were always taken? Why is this behavior occurring?_

The Q-learning agent slowly learns from it's environment, with the total time to reach the destination decreasing with trials. With the random-action agent,
it would exceed the limit equally often in the begining of the trials compared
to th end. With the Q-learning agent, however, the number of time exceeds reduces with trials, with very few time exceeded by the end of the 100 trials.


### Improve the Q-Learning Driving Agent
***QUESTION:*** _Report the different values for the parameters tuned in your basic implementation of Q-Learning. For which set of parameters does the agent perform best? How well does the final driving agent perform?_




***QUESTION:*** _Does your agent get close to finding an optimal policy, i.e. reach the destination in the minimum possible time, and not incur any penalties? How would you describe an optimal policy for this problem?_



