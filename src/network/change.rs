use crate::network::Float;

pub enum LayerChange {
    FeedForwardChange(FeedForwardChange),
    OutputLayerChange(OutputLayerChange),
    None,
}

impl LayerChange {
    pub fn get_neurons(&self) -> &Vec<NeuronChange> {
        match self {
            LayerChange::FeedForwardChange(layer) => &layer.neurons,
            LayerChange::OutputLayerChange(layer) => &layer.neurons,
            LayerChange::None => unreachable!(),
        }
    }

    pub fn update(&mut self, errors: &Vec<Float>, a: &Vec<Float>, eta: Float) {
        match self {
            LayerChange::FeedForwardChange(layer) => {
                for (neuron, error) in layer.neurons.iter_mut().zip(errors.iter()) {
                    neuron.update(a, *error, eta);
                }
            }
            LayerChange::OutputLayerChange(layer) => {
                for (neuron, error) in layer.neurons.iter_mut().zip(errors.iter()) {
                    neuron.update(a, *error, eta);
                }
            }
            LayerChange::None => unreachable!(),
        }
    }
}

pub struct FeedForwardChange {
    neurons: Vec<NeuronChange>,
}

impl FeedForwardChange {
    pub fn new(length: usize, weight_connections: usize) -> Self {
        let neurons = vec![
            NeuronChange {
                weights: vec![0.0; weight_connections],
                bias: 0.0
            };
            length
        ];
        FeedForwardChange { neurons }
    }
}

#[derive(Clone)]
pub struct NeuronChange {
    pub weights: Vec<Float>,
    pub bias: Float,
}

impl NeuronChange {
    pub fn update(&mut self, a: &Vec<Float>, error: Float, eta: Float) {
        self.bias += eta * error;
        for (weight, a_i) in self.weights.iter_mut().zip(a) {
            *weight += eta * error * a_i;
        }
    }
}

pub struct OutputLayerChange {
    neurons: Vec<NeuronChange>,
}

impl OutputLayerChange {
    pub fn new(length: usize, weight_connections: usize) -> Self {
        let neurons = vec![
            NeuronChange {
                weights: vec![0.0; weight_connections],
                bias: 0.0
            };
            length
        ];
        OutputLayerChange { neurons }
    }
}
