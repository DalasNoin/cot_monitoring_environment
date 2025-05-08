from tool_box.base_tool import BaseTool
import json
import os


class PredictActivationsTool(BaseTool):
    name = "guess_activation"
    def __init__(self):
        super().__init__()
        self.description_text = """
        Use this tool to provide your guess for the activation of the current sentence based on the explanation of the neuron. Make sure to call this tool for each sentence separately.
        Maximum activation is 30.0 and minimum activation is 0.0.
        An activation between 0.0 and 1.0 means that the neuron is not activated as the explanation is not relevant to the sentence.
        An activation between 1.0 and 10.0 means that the neuron is activated slightly as there is some overlap between the explanation and the sentence.
        An activation between 10.0 and 30.0 means that the neuron is highly activated as the explanation is highly relevant to the sentence.
        """
        self.parameter_definitions = {
            "activation": {
                "description": "Floating point value representing your guess for the activation of the current sentence. Make sure to provide a value between 0.0 and 30.0.",
                "type": 'number',
                "required": True
            },
            "sentence_number": {
                "description": "The sentence number for which you're making a guess for.",
                "type": 'int',
                "required": True
            },

        }
        self._parameter_example_usage = {
            "activation": 15.7,
            "sentence_number": 1
        }

        self.data = {}


    def description(self):
        return {
            "name": self.name,
            "description": self.description_text,
            "parameter_definitions": self.parameter_definitions
        }

    def run(self, activation: float, sentence_number: int) -> str:
        # The tool doesn't actually generate or execute any plans
        # It just returns the plan provided by the agent
        self.data[sentence_number] = activation
        return f"Your guess for sentence number {sentence_number} is an activation of {activation}."

    def finalisation(self):
        pass
        # with open("sae_labelling/activations_finalisation.json", 'w') as file:
        #     json.dump(self.data, file)

    def __call__(self, activation: float, sentence_number: int) -> str:
        return self.run(activation, sentence_number)


class GuessActivationsTool(BaseTool):
    name = "guess_activations"

    def __init__(self, sentence_count: int = None):
        super().__init__()
        self.sentence_count = sentence_count
        self.description_text = """
        Use this tool to provide your guesses for the activations of multiple sentences at once.
        Provide a list of floating-point values, where the first value in the list corresponds to the
        first sentence, the second value to the second sentence, and so on.
        For example, if 8 sentences are shown, the list must contain 8 activation values.
        Maximum activation is 30.0 and minimum activation is 0.0.
        An activation between 0.0 and 1.0 means that the neuron is not activated as the explanation is not relevant to the sentence.
        An activation between 1.0 and 10.0 means that the neuron is activated slightly as there is some overlap between the explanation and the sentence.
        An activation between 10.0 and 30.0 means that the neuron is highly activated as the explanation is highly relevant to the sentence.
        If a sentence_count was specified when this tool was initialized, the provided list of activations
        will be checked against this count. If the lengths do not match, an informative message will be
        returned instead of an error, explaining the mismatch.
        """
        self.parameter_definitions = {
            "activation_values": {
                "description": "A list of floating point values representing your guesses for the activations of the sentences. The order of values in the list MUST correspond to the order of the sentences (e.g., the first value for the first sentence, second for the second, etc.). Each value must be between 0.0 and 30.0. If a sentence_count was specified when this tool was initialized, the length of this list must match that count.",
                "type": 'array',  # Describes a list of items, items type detailed in description.
                "required": True
            }
        }
        self._parameter_example_usage = {
            "activation_values": [15.7, 0.5, 22.1, 8.0]
        }
        self.data = {}  # To store the activations

    def set_sentence_count(self, sentence_count: int):
        self.sentence_count = sentence_count

    def description(self):
        return {
            "name": self.name,
            "description": self.description_text,
            "parameter_definitions": self.parameter_definitions
        }

    def run(self, activation_values: list[float]) -> str:
        if not isinstance(activation_values, list):
             return "Error: 'activation_values' must be a list."

        if not activation_values: # Check for empty list
            return "Error: 'activation_values' list cannot be empty if sentences are present."

        for i, val in enumerate(activation_values):
            if not isinstance(val, (float, int)):
                return f"Error: Item at index {i} ('{val}') in 'activation_values' is not a valid number. All items must be numbers."
            try:
                num_val = float(val)
            except ValueError:
                return f"Error: Item at index {i} ('{val}') in 'activation_values' cannot be converted to a float."
            if not (0.0 <= num_val <= 30.0):
                return (f"Error: Activation value {num_val} at index {i} is outside the allowed range (0.0 to 30.0). "
                        f"Please ensure all values are within this range.")

        if self.sentence_count is not None:
            if len(activation_values) != self.sentence_count:
                return (f"Input Mismatch: Expected {self.sentence_count} activation values based on tool initialization, "
                        f"but received {len(activation_values)}. You provided {len(activation_values)} values: {activation_values}. "
                        f"Please ensure the list of activations matches the expected number of sentences ({self.sentence_count}).")
        
        # Store data, mapping sentence number to activation value
        self.data = {idx: float(val) for idx, val in enumerate(activation_values)}
        
        return f"Your guesses for {len(activation_values)} sentences have been recorded: {list(self.data.values())}."

    def finalisation(self):
        pass
        # Example for saving data:
        # with open("sae_labelling/multiple_activations_finalisation.json", 'w') as file:
        #     json.dump(self.data, file)

    def __call__(self, activation_values: list[float]) -> str:
        return self.run(activation_values=activation_values)