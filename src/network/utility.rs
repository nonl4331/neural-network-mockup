use crate::network::change::NeuronChange;
use crate::network::neuron::Neuron;

use rand::{thread_rng, Rng};
use rand_distr::{Distribution, Normal};

pub type Float = f32;

pub fn xavier_init(in_num: usize) -> Float {
    let mut rng = thread_rng();
    let val = (1.0 / in_num as Float).sqrt();
    rng.gen_range((-val)..val)
}

pub fn normalised_xavier_init(in_num: usize, self_num: usize) -> Float {
    let mut rng = thread_rng();
    let val = (6.0 / (in_num + self_num) as Float).sqrt();
    rng.gen_range((-val)..val)
}

pub fn he_init(in_num: usize) -> Float {
    let mut rng = thread_rng();
    let std = (2.0 / in_num as Float).sqrt();
    let normal = Normal::new(0.0, std).unwrap();
    normal.sample(&mut rng)
}

pub fn update_change_neurons(
    neurons: &mut Vec<NeuronChange>,
    errors: &Vec<Float>,
    a: &Vec<Float>,
    eta: Float,
) {
    for (neuron, error) in neurons.iter_mut().zip(errors) {
        neuron.update(a, *error, -eta);
    }
}

pub fn hadamard_product(a: &Vec<Float>, b: &Vec<Float>) -> Vec<Float> {
    a.iter().zip(b).map(|(&a, b)| a * b).collect()
}

pub fn update_change_bias(bias: &mut Float, error: Float, eta: Float) {
    *bias -= eta * error;
}

pub fn update_change_weights(weights: &mut Vec<Float>, error: Float, a: &Vec<Float>, eta: Float) {
    for (i, weight) in weights.iter_mut().enumerate() {
        *weight -= eta * error * a[i];
    }
}

pub fn get_weights_vec(neurons: &Vec<Neuron>) -> Vec<Vec<Float>> {
    let mut weights = Vec::new();
    for neuron in neurons {
        weights.push(neuron.weights.clone());
    }
    weights
}

// change to flat memory layout
pub fn transpose(v: Vec<Vec<Float>>) -> Vec<Vec<Float>> {
    assert!(!v.is_empty());
    (0..v[0].len())
        .map(|i| {
            v.iter()
                .map(|inner| inner[i].clone())
                .collect::<Vec<Float>>()
        })
        .collect()
}
