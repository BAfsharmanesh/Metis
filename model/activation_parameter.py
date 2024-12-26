# Copyright 2024 Samsung Electronics Co., Ltd. All Rights Reserved
from utils import ModelConfig


class GPTActivationAndParam:
    def __init__(self, model_config: ModelConfig, model_params):
        """
        Initialize the GPTActivationAndParam

        Parameters:
             hidden_size (int): The hidden size of the model.
             sequence_length (int): The sequence length of the input.
             num_layers (int): The total number of layers in the GPT model.
             vocab_size (int): The size of the vocabulary.
             attention_head_size (int): The size of each attention head.
        """
        self.hidden_size = model_config.hidden_size
        self.sequence_length = model_config.sequence_length
        self.num_layers = model_config.num_layers
        self.vocab_size = model_config.vocab_size
        self.attention_head_size = model_config.attention_head_size
        self.input_params = float(model_params[0])
        self.output_params = float(model_params[-1])
        self.transformer_params = float(model_params[1])

    def get_num_layers(self):
        return self.num_layers

    def get_activation_size(self, layer_id, batch_size, tp_deg):
        if layer_id == (self.num_layers - 1):
            return batch_size * self.sequence_length * self.vocab_size / tp_deg
        return batch_size * self.sequence_length * self.hidden_size

    def get_parameter_size(self, tp_deg):
        parameters = [self.input_params/tp_deg]
        parameters += [self.transformer_params/tp_deg for i in range(self.num_layers-2)]
        parameters.append(self.output_params/tp_deg)
        return parameters

    def get_parameter_size_by_stage(self, tp_deg, start_layer_id, end_layer_id):
        num_transformer_layer = end_layer_id - start_layer_id
        parameters = 0
        if start_layer_id == 0:
            parameters += (self.input_params / tp_deg)
            num_transformer_layer -= 1
        if end_layer_id == self.num_layers:
            parameters += (self.output_params / tp_deg)
            num_transformer_layer -= 1

        parameters += (self.transformer_params / tp_deg * num_transformer_layer)
        return parameters

class GeneralModelActivationAndParam:
    def __init__(self, num_layers, model_params, model_activations):
        """
        Initialize the GeneralModelActivationAndParam

        Parameters:
            model_params (list): A list of parameter sizes for the model, where:
                - model_params[0]: Parameters for the input layer.
                - model_params[-1]: Parameters for the output layer.
                - model_params[1:-1]: Parameters for each transformer layer.
            model_activations (list): A list of activation sizes for each layer.
        """
        self.num_layers = num_layers
        self.model_params = model_params
        self.model_activations = model_activations

    def get_num_layers(self):
        """Returns the total number of layers in the model."""
        return self.num_layers

    def get_activation_size(self, layer_id, batch_size, tp_deg):
        """
        Returns the activation size for a specific layer.

        Parameters:
            layer_id (int): ID of the layer (0-indexed).
            batch_size (int): Number of sequences in the batch.
            tp_deg (int): Tensor parallelism degree.

        Returns:
            float: Activation size for the given layer.
        """
        if layer_id < 0 or layer_id >= self.num_layers:
            raise ValueError("Invalid layer_id: must be in the range [0, num_layers - 1]")
        activation_size = self.model_activations[layer_id]
        return (batch_size * activation_size) / tp_deg

    def get_parameter_size(self, tp_deg):
        """
        Returns the parameter sizes for all layers.

        Parameters:
            tp_deg (int): Tensor parallelism degree.

        Returns:
            list: A list of parameter sizes for all layers.
        """
        return [param / tp_deg for param in self.model_params]

    def get_parameter_size_by_stage(self, tp_deg, start_layer_id, end_layer_id):
        """
        Returns the parameter size for a range of layers.

        Parameters:
            tp_deg (int): Tensor parallelism degree.
            start_layer_id (int): Starting layer ID (inclusive).
            end_layer_id (int): Ending layer ID (exclusive).

        Returns:
            float: Total parameter size for the specified range of layers.
        """
        if start_layer_id < 0 or end_layer_id > self.num_layers or start_layer_id > end_layer_id:
            raise ValueError("Invalid layer range: ensure 0 <= start_layer_id < end_layer_id <= num_layers")
        
        parameters = sum(self.model_params[start_layer_id:end_layer_id]) / tp_deg
        return parameters
