use crate::network::{change::NeuronChange, Float, Regularisation};

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

    pub fn update(
        &mut self,
        learning_rate: Float,
        neuron_change: &NeuronChange,
        mini_batch_size: usize,
        regularisation: &Regularisation,
    ) {
        self.bias -= neuron_change.bias / mini_batch_size as Float;
        match regularisation {
            Regularisation::L1(lambda) => {
                for (weight, change) in self.weights.iter_mut().zip(neuron_change.weights.iter()) {
                    *weight *= 1.0 - learning_rate * lambda / mini_batch_size as Float;
                    *weight -= change / mini_batch_size as Float;
                }
            }
            Regularisation::L2(lambda) => {
                for (weight, change) in self.weights.iter_mut().zip(neuron_change.weights.iter()) {
                    *weight -= change * learning_rate / mini_batch_size as Float
                        + learning_rate * lambda * weight.signum() / mini_batch_size as Float;
                }
            }
            Regularisation::None => {
                for (weight, change) in self.weights.iter_mut().zip(neuron_change.weights.iter()) {
                    *weight -= change * learning_rate / mini_batch_size as Float;
                }
            }
        }
    }
}
