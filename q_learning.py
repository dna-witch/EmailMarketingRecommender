import os
import json
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

# import custom modules
from campaign_env import CampaignEnv
from metrics import Metrics

# ignore warnings
import warnings
warnings.filterwarnings('ignore')


class QLearning:
    """This class defines the Q-learning agent for the email marketing campaign data. 
    The class provides methods to train the agent and predict actions based on the learned Q-values."""
    def __init__(self):
        self.data = pd.read_csv('data/transformed_data.csv', low_memory=False)
        self.env = CampaignEnv()
        self.metrics = Metrics()
        self.q_values = np.zeros((self.env.get_number_of_states(), 4))  # 4 possible actions
        self.env.generate_state_index_map()
        self.alpha = 0.1  # learning rate
        self.gamma = 0.9  # discount factor
        # self.epsilon = 0.1  # exploration-exploitation trade-off ratio

        self.num_episodes = 1000
        # self.rewards = []
    
    def select_action(self, state, epsilon):
        """Select an action based on epsilon-greedy policy."""
        if np.random.rand() <= epsilon:
            # Explore - choose a random action
            return random.randint(1, 4)  # 4 possible actions, numbered 1 to 4
        else:
            # Exploit - choose the action with the highest Q-value for the current state
            return np.argmax(self.q_values[state])
    
    def update_q_values(self, state, action, reward, next_state):
        """Update the Q-values based on the Q-learning update rule."""
        action_index = action - 1  # Convert action number to index (0 to 3)
        best_next_action = np.argmax(self.q_values[next_state])
        # Q-learning update rule
        self.q_values[state][action_index] += round(self.alpha * (reward + self.gamma * self.q_values[next_state][best_next_action] - self.q_values[state][action_index]), 5)
        # self.q_values[state][action] += self.alpha * (reward + self.gamma * np.max(self.q_values[next_state]) - self.q_values[state][action])
    
    def train_agent(self):
        """Train the Q-learning agent."""
        print(f"Training the agent for {self.num_episodes} episodes...")
        steps = 0

        # Start training
        for episode in range(self.num_episodes):
            state = self.env.get_state(None)
            state_index = self.env.map_state_to_index(state)
            done = False

            while not done:
                # Select action using epsilon-greedy policy
                action = self.select_action(state_index, epsilon=1)

                # Observe reward and next state after taking action
                reward, done = self.env.take_action(state, action, self.data)

                # Update Q-values based on reward and next state
                next_state = self.env.get_state(state)
                next_state_index = self.env.map_state_to_index(next_state)
                self.update_q_values(state_index, action, reward, next_state_index)

                # Update state and step count
                state_index = next_state_index
                steps += 1
        
        # Save the Q-values to a CSV file
        np.savetxt('results/q_table.csv', self.q_values, delimiter=',')
        print("Training complete. Q-values saved to 'results/q_table.csv'.")
        print("Q_table:")
        print(self.q_values)
    
    def predict_action(self, state):
        """Predict the action to take based on the current state."""
        state_index = self.env.map_state_to_index(state)

        # Load saved Q-table
        self.q_values = np.loadtxt('results/q_table.csv', delimiter=',')

        # Choose action with the highest Q-value for the current state
        action = np.argmax(self.q_values[state_index])

        return action
    
    def test_agent(self):
        data = pd.read_csv('data/transformed_data.csv', low_memory=False)

        # Split data into training and test sets
        # Data is stratified according to the 'Action' column
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['Action'])

        # Get the actual actions from the test data
        actual_actions = test_data['Action']
        # Initialize an empty list to store predicted actions
        predicted_actions = []

        # Predict actions for each row in the test data
        for _, row in test_data.iterrows():
            age_group = row['Age_binned']
            tenure_group = row['Tenure_binned']
            gender = row['Gender']
            user_type = row['Type']
            subject_line = row['SubjectLine_ID']

            # Construct state from state variables
            state = (age_group, tenure_group, gender, user_type, subject_line)

            # Predict the action to take based on the current state using Q-table
            action = self.predict_action(state)

            # Append the predicted action to the list
            predicted_actions.append(action)

            if action == 0:
                action_label = 'Do not send email.'
            elif action == 1:
                action_label = 'Send email with Subject Line 1.'
            elif action == 2:
                action_label = 'Send email with Subject Line 2.'
            elif action == 3:
                action_label = 'Send email with Subject Line 3.'
            else:
                action_label = 'Invalid action.'
            
        # Calculate and print the accuracy of the agent
        accuracy = self.metrics.calculate_accuracy(predicted_actions, actual_actions)
        print(f"Accuracy of the Q-learning agent: {accuracy:.2f}%")

if __name__ == '__main__':
    agent = QLearning()
    agent.train_agent()
    agent.test_agent()