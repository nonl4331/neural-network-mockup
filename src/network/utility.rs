use crate::network::Neuron;

pub type Float = f32;

pub fn get_weights_vec(neurons: &Vec<Neuron>) -> Vec<Vec<Float>> {
    let mut weights = Vec::new();
    for neuron in neurons {
        weights.push(neuron.weights.clone());
    }
    weights
}

pub fn hadamard_product(a: &Vec<Float>, b: &Vec<Float>) -> Vec<Float> {
    a.iter().zip(b).map(|(&a, b)| a * b).collect()
}

/// max() over a slice of floats, gets the index of a largest value
pub fn max_index(nets: &[Float]) -> usize {
    let mut max = Float::NEG_INFINITY;
    let mut index = 0;
    for (i, n) in nets.iter().enumerate() {
        if *n > max {
            index = i;
            max = *n;
        }
    }
    index
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
