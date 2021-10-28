use crate::network::utility::update_change_bias;
use crate::network::utility::update_change_weights;
use crate::network::utility::Float;

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
            LayerChange::None => panic!("get_neurons called on none!"),
        }
    }
    pub fn get_neurons_mut(&mut self) -> &mut Vec<NeuronChange> {
        match self {
            LayerChange::FeedForwardChange(layer) => &mut layer.neurons,
            LayerChange::OutputLayerChange(layer) => &mut layer.neurons,
            LayerChange::None => panic!("get_neurons called on none!"),
        }
    }
}

#[derive(Clone)]
pub struct NeuronChange {
    pub weights: Vec<Float>,
    pub bias: Float,
}

impl NeuronChange {
    pub fn update(&mut self, a: &Vec<Float>, error: Float, eta: Float) {
        update_change_bias(&mut self.bias, error, eta);
        update_change_weights(&mut self.weights, error, a, eta);
    }
}
