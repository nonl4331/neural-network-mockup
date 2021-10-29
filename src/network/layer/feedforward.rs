use crate::network::change::FeedForwardChange;
use crate::network::change::LayerChange;
use crate::network::layer::{ActivationFunction, InitType, LayerTrait, Neuron};

use crate::network::utility::{
    d_sigmoid, get_weights_vec, hadamard_product, normalised_xavier_init, sigmoid, transpose,
    update_change_neurons, xavier_init, Float,
};

pub struct FeedForward {
    activation_function: ActivationFunction,
    neurons: Vec<Neuron>,
    z_values: Vec<Float>,
    output: Vec<Float>,
}

impl LayerTrait for FeedForward {
    fn forward(&mut self, input: Vec<Float>) {
        assert_eq!(self.neurons[0].weights.len(), input.len());
        let mut z_values = Vec::new();

        for neuron in &self.neurons {
            let z = neuron
                .weights
                .iter()
                .zip(input.iter())
                .fold(0.0, |acc, (input, weight)| acc + input * weight)
                + neuron.bias;

            z_values.push(z);
        }
        let output = z_values.iter().map(|&z| sigmoid(z)).collect();

        self.z_values = z_values;
        self.output = output;
    }
    fn backward(
        &mut self,
        a: &Vec<Float>,
        layer_change: &mut LayerChange,
        error_input: &Vec<Float>,
        weights: Vec<Vec<Float>>,
        eta: Float,
    ) -> (Vec<f32>, Vec<Vec<f32>>) {
        let weights = transpose(weights);

        let c_da: Vec<Float> = weights
            .iter()
            .map(|weights| hadamard_product(weights, &error_input).iter().sum())
            .collect();

        let a_dz: Vec<Float> = match self.activation_function {
            ActivationFunction::Sigmoid => self.z_values.iter().map(|&z| d_sigmoid(z)).collect(),
            ActivationFunction::Softmax => {
                panic!("a_dz not implemented for softmax yet!");
            }
        };

        let errors = hadamard_product(&c_da, &a_dz);

        update_change_neurons(layer_change.get_neurons_mut(), &errors, &a, eta);

        (errors, get_weights_vec(&self.neurons))
    }
    fn get_len(&self) -> usize {
        self.neurons.len()
    }
    fn get_output(&self) -> Vec<Float> {
        self.output.clone()
    }
    fn get_z_values(&self) -> Vec<Float> {
        self.z_values.clone()
    }
    fn get_layer_change(&self) -> LayerChange {
        LayerChange::FeedForwardChange(FeedForwardChange::new(
            self.neurons.len(),
            self.neurons[0].weights.len(),
        ))
    }
    fn update(&mut self, changes: &LayerChange, mini_batch_size: usize) {
        for (neuron, neuron_change) in self.neurons.iter_mut().zip(changes.get_neurons()) {
            neuron.update(neuron_change, mini_batch_size);
        }
    }
}

impl FeedForward {
    pub fn new(
        init_type: InitType,
        activation_function: ActivationFunction,
        input_size: usize,
        length: usize,
    ) -> Self {
        let mut neurons = Vec::new();
        for _ in 0..length {
            let mut weights = Vec::new();
            for _ in 0..input_size {
                match init_type {
                    InitType::Xavier => {
                        weights.push(xavier_init(input_size));
                    }
                    InitType::NormalisedXavier => {
                        weights.push(normalised_xavier_init(input_size, length));
                    }
                }
            }
            neurons.push(Neuron::new(weights, 0.0));
        }
        FeedForward {
            neurons,
            activation_function,
            z_values: Vec::new(),
            output: Vec::new(),
        }
    }
}
