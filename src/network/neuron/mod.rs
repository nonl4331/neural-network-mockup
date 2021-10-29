use crate::network::change::NeuronChange;
use crate::network::Float;

pub mod activation_function;
pub mod cost_function;

pub struct Neuron {
    pub weights: Vec<Float>,
    pub bias: Float,
}

impl Neuron {
    pub fn new(weights: Vec<Float>, bias: Float) -> Self {
        Neuron { weights, bias }
    }

    pub fn update(&mut self, neuron_change: &NeuronChange, mini_batch_size: usize) {
        self.bias -= neuron_change.bias / mini_batch_size as Float;
        for (weight, change) in self.weights.iter_mut().zip(neuron_change.weights.iter()) {
            *weight -= change / mini_batch_size as Float;
        }
    }
}
