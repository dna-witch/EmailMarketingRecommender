
class Metrics:
    @staticmethod
    def calculate_accuracy(predicted_actions, actual_actions):
        """Calculate the accuracy of the Q-learning by comparing the percentage of predicted actions that match the actual actions."""
        
        # Double check to make sure that there are enough predictions to make a 1:1 comparison with the actual actions
        if len(predicted_actions) != len(actual_actions):
            raise ValueError("The number of predicted actions does not match the number of actual actions.")
        
        # Calculate the number of correct predictions
        correct = sum([1 for pred, actual in zip(predicted_actions, actual_actions) if pred == actual])
        total = len(predicted_actions)

        # Calculate the accuracy
        accuracy = (correct / total) * 100

        return accuracy