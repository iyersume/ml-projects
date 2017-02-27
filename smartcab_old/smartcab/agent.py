import random
import numpy
from environment import Agent, Environment, TrafficLight
from planner import RoutePlanner
from simulator import Simulator


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'white'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.valid_lights = ['green', 'red']
        self.n_actions = len(self.env.valid_actions)
        self.Q_table = self.reset_q_table()
        self.iteration = 1
        self.prev_state = (0, 0, 0, 0, 0)
        self.prev_action = 0

    def reset_q_table(self):
        """ Initialize the Q table using optimistic values to ensure all actions
        are explored
        """
        INIT_Q = 10.0
        return numpy.ones(
            (
                # state
                self.n_actions,   # next waypoint action
                len(self.valid_lights),   # light (green or red)
                self.n_actions,   # oncoming traffic action
                self.n_actions,   # left traffic action
                self.n_actions,   # right traffic action

                # action
                self.n_actions    # agent action
            ), dtype=float
        ) * INIT_Q

    def get_q(self, state, action):
        # state is a tuple containing:
        #   (next waypoint, light state, oncoming, left, right)
        index = (state[0], state[1], state[2], state[3], state[4], action)
        return self.Q_table[index]

    def get_max_q_val_action(self, state):
        index = (state[0], state[1], state[2], state[3], state[4])
        state_vals = self.Q_table[index]
        max_action = numpy.argmax(state_vals)
        return state_vals[max_action], max_action

    def update_q(self, new_state, reward):
        state, action = self.prev_state, self.prev_action
        if state and action:
            index = (state[0], state[1], state[2], state[3], state[4], action)
            alpha = 1.0 / self.iteration
            future_utility = self.get_max_q_val_action(new_state)[0]
            self.Q_table[index] = ((1 - alpha) * self.Q_table[index] +
                                   alpha * (reward + future_utility))

    def update_params(self, state, action):
        self.prev_state = state
        self.prev_action = action

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        # self.Q_table = self.reset_q_table()
        self.iteration += 1

    def get_next_action(self, state):
        # epsilon-greedy action selection, with decay of epsilon value
        INIT_EPSILON = 0.9
        if random.random() < INIT_EPSILON / self.iteration:
            return random.choice(self.env.valid_actions)
        else:
            return self.env.valid_actions[self.get_max_q_val_action(state)[1]]

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()
        next_waypoint_index = self.env.valid_actions.index(self.next_waypoint)
        # from route planner, also displayed by simulator

        # inputs gives {'light': light, 'oncoming': oncoming, 'left': left, 'right': right}
        inputs = self.env.sense(self)
        inputs_indices = ([self.valid_lights.index(inputs['light'])] +
                          [self.env.valid_actions.index(inputs[i])
                           for i in ('oncoming', 'left', 'right')])

        deadline = self.env.get_deadline(self)

        # TODO: Update state
        state = [next_waypoint_index] + inputs_indices

        # TODO: Select action according to your policy
        # action = random.choice(self.env.valid_actions)
        action = self.get_next_action(state)
        action_index = self.env.valid_actions.index(action)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        self.update_q(state, reward)
        self.update_params(state, action_index)

        # print("LearningAgent.update(): deadline = {}, "
        #       "inputs = {}, action = {}, reward = {}, "
        #       " Q value = {}".
        #       format(deadline, inputs, action, reward,
        #              self.get_q(self.prev_state, self.prev_action)))  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment(num_dummies=2)  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.15, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
