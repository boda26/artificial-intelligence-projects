import numpy as np
import utils

class Agent:    
    def __init__(self, actions, Ne=40, C=40, gamma=0.7):
        # HINT: You should be utilizing all of these
        self.actions = actions
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma
        self.reset()
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        self.t = 0
        
    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path,self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        # HINT: These variables should be used for bookkeeping to store information across time-steps
        # For example, how do we know when a food pellet has been eaten if all we get from the environment
        # is the current number of points? In addition, Q-updates requires knowledge of the previously taken
        # state and action, in addition to the current state from the environment. Use these variables
        # to store this kind of information.
        self.points = 0
        self.s = None
        self.a = None
        self.t = 0
    
    def act(self, environment, points, dead):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: chosen action between utils.UP, utils.DOWN, utils.LEFT, utils.RIGHT

        Tip: you need to discretize the environment to the state space defined on the webpage first
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the playable board)
        '''
        s_prime = self.generate_state(environment)
        # self.t += 1

        if self._train:
            # decide the reward
            if dead:
                reward = -1
            elif points > self.points:
                reward = 1
                self.points = points
            else:
                reward = -0.1
            # check whether on time t=0
            # update N and Q
            if self.s is not None and self.a is not None:
                self.N[self.s][self.a] += 1
                lr = self.C / (self.C + self.N[self.s][self.a])
                # max_new_val = float('-inf')
                # for val in self.Q[s_prime]:
                #     if val > max_new_val:
                #         max_new_val = val
                self.Q[self.s][self.a] = self.Q[self.s][self.a] + lr * (reward + self.gamma * max(self.Q[s_prime]) - self.Q[self.s][self.a])
            # reset the agent if dead
        if dead:
            self.reset()
            return self.a
        
        # choosing the optimal action
        a_prime = None
        if self._train:
            if self.N[s_prime][utils.RIGHT] < self.Ne:
                a_prime = utils.RIGHT
            elif self.N[s_prime][utils.LEFT] < self.Ne:
                a_prime = utils.LEFT
            elif self.N[s_prime][utils.DOWN] < self.Ne:
                a_prime = utils.DOWN
            elif self.N[s_prime][utils.UP] < self.Ne:
                a_prime = utils.UP
        if a_prime is None:
            max_Q = float('-inf')
            for val in self.Q[s_prime]:
                if val > max_Q:
                    max_Q = val
            if self.Q[s_prime][utils.RIGHT] == max_Q:
                a_prime = utils.RIGHT
            elif self.Q[s_prime][utils.LEFT] == max_Q:
                a_prime = utils.LEFT
            elif self.Q[s_prime][utils.DOWN] == max_Q:
                a_prime = utils.DOWN
            else:
                a_prime = utils.UP
        self.s = s_prime
        self.a = a_prime
        return a_prime


    def generate_state(self, environment):
        # TODO: Implement this helper function that generates a state given an environment
        snake_head_x, snake_head_y, snake_body, food_x, food_y = environment
        # food_dir_x
        if food_x < snake_head_x:
            food_dir_x = 1
        elif food_x > snake_head_x:
            food_dir_x = 2
        else:
            food_dir_x = 0
        
        # food_dir_y
        if food_y < snake_head_y:
            food_dir_y = 1
        elif food_y > snake_head_y:
            food_dir_y = 2
        else:
            food_dir_y = 0
        
        # adjoining_wall_x
        if snake_head_x == 1:
            adjoining_wall_x = 1
        elif snake_head_x == utils.DISPLAY_WIDTH - 2:
            adjoining_wall_x = 2
        else:
            adjoining_wall_x = 0
        
        # adjoining_wall_y
        if snake_head_y == 1:
            adjoining_wall_y = 1
        elif snake_head_y == utils.DISPLAY_HEIGHT - 2:
            adjoining_wall_y = 2
        else:
            adjoining_wall_y = 0

        # adjoining_body
        adjoining_body_top = 0
        adjoining_body_bottom = 0
        adjoining_body_left = 0
        adjoining_body_right = 0
        for pos in snake_body:
            body_x, body_y = pos
            if body_x == snake_head_x and body_y == snake_head_y - 1:
                adjoining_body_top = 1
            if body_x == snake_head_x and body_y == snake_head_y + 1:
                adjoining_body_bottom = 1
            if body_y == snake_head_y and body_x == snake_head_x - 1:
                adjoining_body_left = 1
            if body_y == snake_head_y and body_x == snake_head_x + 1:
                adjoining_body_right = 1
        return (food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)
