# import custom modules
from q_learning import QLearning

from flask import Flask, request

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    """Predict action to take based on the current state.
    The state information is received as a POST request."""
    data = request.get_json()
    input_state = data['input_state']
    if not input_state:
        raise ValueError("No input state provided.")
    
    # Encode the input state variables
    age_group_mapping = {0: [0, 20], 1: [21, 30], 3: [31, 40], 4: [41, 50], 5: [51, 60], 6: [61, 70]}
    tenure_group_mapping = {0: [0, 11], 1: [12, 14], 2: [15, 22], 3: [23, 38]}
    gender_mapping = {'M': 0, 'F': 1}
    type_mapping = {'B': 0, 'C': 1}

    age_group = next(key for key, value in age_group_mapping.items() if value[0] <= input_state['Age'] <= value[1])
    tenure_group = next(key for key, value in tenure_group_mapping.items() if value[0] <= input_state['Tenure'] <= value[1])
    gender = gender_mapping[input_state['Gender']]
    user_type = type_mapping[input_state['Type']]
    subject_line = input_state['SubjectLine_ID']

    state = (age_group, tenure_group, gender, user_type, subject_line)

    # Get the action to take
    action = agent.predict_action(state)

if __name__ == '__main__':
    flaskPort = 8786
    agent = QLearning()
    app.run(host='0.0.0.0', port=flaskPort)