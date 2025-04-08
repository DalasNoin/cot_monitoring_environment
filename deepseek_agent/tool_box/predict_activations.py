from tool_box.base_tool import BaseTool
import json
import os


class PredictActivationsTool(BaseTool):
    name = "guess_activation"
    def __init__(self):
        super().__init__()
        self.description_text = """
        Use this tool to provide your guess for the activation of the current sentence based on the explanation of the neuron. 
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
        with open("sae_labelling/activations_finalisation.json", 'w') as file:
            json.dump(self.data, file)

    def __call__(self, activation: float, sentence_number: int) -> str:
        return self.run(activation, sentence_number)