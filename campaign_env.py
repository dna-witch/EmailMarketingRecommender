import numpy as np
import pandas as pd
import random

# ignore warnings
import warnings
warnings.filterwarnings('ignore')


class CampaignEnv:
    """This class defines the environment for the email marketing campaign data. 
    The class provides methods to initialize the environment, take actions, and get rewards."""
    def __init__(self):
        self.num_age_groups = 6
        self.num_tenure_groups = 4
        self.num_genders = 2
        self.num_types = 2
        self.num_subject_lines = 3

        self.state_index_map = {}
        self.index_state_map = {}
        self.index_counter = 0
    
    def get_number_of_states(self):
        return self.num_age_groups * self.num_tenure_groups * self.num_genders * self.num_types * self.num_subject_lines

    def take_action(self, state, action, data):
        """Take an action in the environment and return the next state and reward."""
        # Get the current state
        age_group, tenure_group, gender, user_type, subject_line = state
        DEFAULT_CONVERSION_RATE = 0.0  # Default conversion rate if no matching data is available

        # Get the conversion rate for the current state
        try:
            conversion_rate = data[
                (data['Age_binned'] == age_group) &
                (data['Tenure_binned'] == tenure_group) &
                (data['Gender'] == gender) &
                (data['Type'] == user_type) &
                (data['SubjectLine_ID'] == subject_line)
                ]['Conversion_Rate'].values[0]
        except IndexError:
            conversion_rate = DEFAULT_CONVERSION_RATE
        except Exception as e:
            print(f"An error occurred: {e}")
            conversion_rate = DEFAULT_CONVERSION_RATE
        
        # Determine the reward based on the conversion rate
        conversion_rate_threshold = 25.0  # Conversion rate threshold for a successful campaign

        if action == 1 or action == 2:  # Send an email with Subject Line 1 or 2
            if conversion_rate >= conversion_rate_threshold:
                reward = 10
            else:
                reward = 5
            done = True
        elif action == 3:  # Send an email with Subject Line 3
            if conversion_rate >= conversion_rate_threshold:
                reward = 5
            else:
                reward = 2.5
            done = True
        elif action == 4:  # Do not send an email
            reward = 0
            done = False
        else:
            raise ValueError("Invalid action. Please choose an action from [1, 2, 3, 4].")
        
        return reward, done
    
    def get_state(self, current_state):
        # Markov Property - The future is independent of the past given the present
        if current_state is None:
            # Initialize the state
            return (
                np.random.randint(self.num_age_groups),
                np.random.randint(self.num_tenure_groups),
                np.random.randint(self.num_genders),
                np.random.randint(self.num_types),
                np.random.randint(0, self.num_subject_lines + 1)
            )
        else:
            # Get the next state by incrementing each state variable by 1
            next_state = list(current_state)
            next_state[4] += 1  # Increment the subject line ID
            if next_state[4] >= self.num_subject_lines:
                next_state[4] = 1  # Wrap around if subject line ID exceeds the number of subject lines
                next_state[3] += 1  # Increment the type
                if next_state[3] >= self.num_types:
                    next_state[3] = 0  # Wrap around if type exceeds the number of types
                    next_state[2] += 1  # Increment the gender
                    if next_state[2] >= self.num_genders:
                        next_state[2] = 0  # Wrap around if gender exceeds the number of genders
                        next_state[1] += 1  # Increment the tenure group
                        if next_state[1] >= self.num_tenure_groups:
                            next_state[1] = 0  # Wrap around if tenure group exceeds the number of tenure groups
                            next_state[0] += 1  # Increment the age group
                            if next_state[0] >= self.num_age_groups:
                                next_state[0] = 0  # Wrap around if age group exceeds the number of age groups
            return tuple(next_state)
    
    def generate_state_index_map(self):
        index_counter = 0

        for age_group in range(self.num_age_groups):
            for tenure_group in range(self.num_tenure_groups):
                for gender in range(self.num_genders):
                    for user_type in range(self.num_types):
                        for subject_line in range(0, self.num_subject_lines+1):
                            state = (age_group, tenure_group, gender, user_type, subject_line)
                            self.state_index_map[state] = index_counter
                            self.index_state_map[index_counter] = state
                            index_counter += 1
        
        # save the state map to a text file
        with open('data/state_index_map.txt', 'w') as file:
            for state, index in self.state_index_map.items():
                file.write(f"State: {state}, Index: {index}\n")
        print(f"State index map saved to 'data/state_index_map.txt'.")

    def map_state_to_index(self, state):
        if state not in self.state_index_map:
            self.state_index_map[state] = self.index_counter
            self.index_state_map[self.index_counter] = state
            self.index_counter += 1
        return self.state_index_map[state]

    def map_index_to_state(self, index):
        return self.index_state_map[index]