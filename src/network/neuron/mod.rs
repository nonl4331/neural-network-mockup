use crate::network::{change::NeuronChange, Float};

pub mod activation_function;
pub mod cost_function;
pub mod initialisation;

pub struct Neuron {
    pub bias: Float,
    pub weights: Vec<Float>,
}

impl Neuron {
    pub fn new(weights: Vec<Float>, bias: Float) -> Self {
        Neuron { bias, weights }
    }

    pub fn update(&mut self, neuron_change: &NeuronChange, mini_batch_size: usize) {
        self.bias -= neuron_change.bias / mini_batch_size as Float;
        for (weight, change) in self.weights.iter_mut().zip(neuron_change.weights.iter()) {
            *weight -= change / mini_batch_size as Float;
        }
    }
}
