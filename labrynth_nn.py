from laberint_game import LaberintGame
from random import randint
import numpy as np
import tflearn
import time
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean
from collections import Counter

class LabrynthNN:
    def __init__(self, initial_games = 200, test_games = 150, goal_steps = 3000, lr = 1e-2, filename = 'labrynth_nn.tflearn'):
        self.initial_games = initial_games
        self.test_games = test_games
        self.goal_steps = goal_steps
        self.lr = lr
        self.filename = filename

    def initial_population(self):
        training_data = []
        for _ in range(self.initial_games):
            game = LaberintGame()
            _, _, board = game.start()
            prev_observation = self.generate_observation(board)
            for _ in range(self.goal_steps):
                action, game_action = self.generate_action(prev_observation)
                done, score, board  = game.step(game_action)
                if done and score < -99:
                    training_data.append([self.add_action_to_observation(prev_observation, action), -1])
                    break
                elif done and score > 99:
                    training_data.append([self.add_action_to_observation(prev_observation, action), 1])
                    break
                elif self.same_observation(prev_observation, self.generate_observation(board)): 
                    training_data.append([self.add_action_to_observation(prev_observation, action), -0.5])
                    prev_observation = self.generate_observation(board)
                else:
                    training_data.append([self.add_action_to_observation(prev_observation, action), 0.85])
                    prev_observation = self.generate_observation(board)
        print(len(training_data))
        return training_data

    def generate_action(self, obs):
        game_action = randint(0,3)
        action = self.get_action_num(game_action)
        return action, game_action

    def get_game_action(self, action):
        game_action = (action + 1) * 2
        return game_action

    def get_action_num(self, game_action):
        action = game_action/2 - 1
        return action

    def generate_observation(self, board):
        player_pos_x, player_pos_y = self.get_player_position(board)
        surr_up, surr_right, surr_down, surr_left = self.get_player_surrounding(player_pos_x, player_pos_y, board)
        return np.array([surr_up, surr_right, surr_down, surr_left])
        
    def add_action_to_observation(self, observation, action):
        return np.append([action], observation)
    
    def get_player_position(self, board):
        for i in range(len(board)):
            for j in range(len(board[i])):
                if(board[i][j] == 'A'):
                    return i, j
        
    def same_observation(self, obs1, obs2):
        obs_comp = obs1 == obs2
        return obs_comp.all()
    
    def get_player_surrounding(self, player_pos_x, player_pos_y, board):
        right = self.normalize_surrounding(board[player_pos_x][player_pos_y+1])
        up = self.normalize_surrounding(board[player_pos_x-1][player_pos_y])
        left = self.normalize_surrounding(board[player_pos_x][player_pos_y-1])
        down = self.normalize_surrounding(board[player_pos_x+1][player_pos_y])
        return up, right, left, down
        
    def normalize_surrounding(self, key):
        if key == '.':
            return 0.7
        elif key == 'w':
            return -0.85
        elif key == 't':
            return -1
        elif key == 'x':
            return 1       
        
    def model(self):
        network = input_data(shape=[None, 5, 1], name='input')
        network = fully_connected(network, 25, activation='relu')
        network = fully_connected(network, 25, activation='relu')
        network = fully_connected(network, 1, activation='linear')
        network = regression(network, optimizer='adam', learning_rate=self.lr, loss='mean_square', name='target')
        model = tflearn.DNN(network, tensorboard_dir='log')
        return model
        
    def train_model(self, training_data, model):
        X = np.array([i[0] for i in training_data]).reshape(-1, 5, 1)
        y = np.array([i[1] for i in training_data]).reshape(-1, 1)
        model.fit(X,y, n_epoch = 1, shuffle = True, run_id = self.filename)
        model.save(self.filename)
        return model
        
    def highest_action(self, model, obs):
        predictions = []
        for action in range(0, 4):
           action_num = self.get_action_num(action)
           predictions.append(model.predict(self.add_action_to_observation(obs, action_num).reshape(-1, 5, 1)))
        action = np.argmax(np.array(predictions))
        return action
    
    def next_action(self, model, obs, prev_action):
        left_action = self.turn_left(prev_action)
        right_action = self.turn_right(prev_action)
        if self.action_is_valid(model, obs, left_action):
            action = left_action
        elif self.action_is_valid(model, obs, prev_action):
            action = prev_action
        elif self.action_is_valid(model, obs, right_action):
            action = right_action
        else:                
            action = self.turn_left(left_action)            
        return action
    
    def action_is_valid(self, model, obs, action):
        action_num = self.get_action_num(action)
        prediction = model.predict(self.add_action_to_observation(obs, action_num).reshape(-1, 5, 1))
        return (prediction > -0.35).all()
            
    def turn_left(self, action):
        new_action = action + 1
        if new_action > 3:
            new_action = 0
        return new_action
            
    def turn_right(self, action):
        new_action = action - 1
        if new_action < 0:
            new_action = 3
        return new_action

    def test_model(self, model):
        steps_arr = []
        for _ in range(self.test_games):
            steps = 0
            game_memory = []
            game = LaberintGame()
            _, _, board = game.start()
            prev_observation = self.generate_observation(board)
            prev_action = self.highest_action(model, prev_observation)
            for _ in range(self.goal_steps):
                action = self.next_action(model, prev_observation, prev_action)
                done, _, board = game.step(action)
                game_memory.append([prev_observation, action])
                if done:
                    break
                else:
                    prev_action = action
                    prev_observation = self.generate_observation(board)
                    steps += 1
            steps_arr.append(steps)
        print('Average steps:',mean(steps_arr))
        print(Counter(steps_arr))
        
    def visualize_game(self, model):
        game = LaberintGame(gui = True)
        _, _, board = game.start()
        prev_observation = self.generate_observation(board)
        prev_action = self.highest_action(model, prev_observation)
        for _ in range(self.goal_steps):
            action = self.next_action(model, prev_observation, prev_action)
            done, _, board  = game.step(action)
            if done:
                break
            else:
                prev_action = action
                prev_observation = self.generate_observation(board)

    def train(self):
        training_data = self.initial_population()
        nn_model = self.model()
        nn_model = self.train_model(training_data, nn_model)
        self.test_model(nn_model)

    def visualize(self):
        nn_model = self.model()
        nn_model.load(self.filename)
        self.visualize_game(nn_model)

    def test(self):
        nn_model = self.model()
        nn_model.load(self.filename)
        self.test_model(nn_model)

if __name__ == "__main__":
    print(time.strftime('%X %x'))
    LabrynthNN().train()
    print(time.strftime('%X %x'))
        
