pub mod layer;

mod utility;

mod change;

use crate::network::change::LayerChange;
use crate::network::layer::LayerTrait;
use crate::network::utility::Float;
use layer::Layer;

use rand::prelude::SliceRandom;
use rand::thread_rng;

type Data = Vec<(Vec<Float>, Vec<Float>)>;

pub struct Network {
    layers: Vec<Layer>,
}

impl Network {
    pub fn from_layers(layers: Vec<Layer>) -> Self {
        Network { layers }
    }

    pub fn get_layer_changes(&self) -> Vec<LayerChange> {
        let mut changes = Vec::new();
        for layer in &self.layers {
            changes.push(layer.get_layer_change());
        }
        changes
    }

    pub fn apply_layer_changes(&mut self, changes: &Vec<LayerChange>, mini_batch_size: usize) {
        assert_eq!(self.layers.len(), changes.len());

        for (layer, layer_change) in self.layers.iter_mut().zip(changes) {
            layer.update(layer_change, mini_batch_size);
        }
    }

    pub fn forward(&mut self, input: Vec<Float>) -> Vec<Float> {
        let mut next_input = input;
        for layer in &mut self.layers {
            layer.forward(next_input);
            next_input = layer.get_output();
        }
        next_input
    }

    fn backpropagation(
        &mut self,
        input: &Vec<Float>,
        changes: &mut Vec<LayerChange>,
        expected_output: &Vec<Float>,
        eta: Float,
    ) -> Vec<Float> {
        let len = self.layers.len();
        assert_eq!(expected_output.len(), self.layers[len - 1].get_len());
        let mut z_values = Vec::new();
        let mut outputs = Vec::new();
        let mut output = input.clone();
        for layer in &mut self.layers {
            layer.forward(output);
            output = layer.get_output();
            outputs.push(output.clone());
            z_values.push(layer.get_z_values());
        }

        let expected_output = vec![expected_output.clone(); 1];

        let (mut errors, mut weights) = self.layers[len - 1].backward(
            &outputs[outputs.len() - 2],
            &mut changes[len - 1],
            &output,
            expected_output,
            eta,
        );

        for i in 0..(len - 1) {
            let layer_index = len - (i + 2);

            let outputs_index = if i == len - 2 { 0 } else { len - (i + 3) };

            let (terrors, tweights) = self.layers[layer_index].backward(
                &outputs[outputs_index],
                &mut changes[layer_index],
                &errors,
                weights,
                eta,
            );

            errors = terrors;
            weights = tweights;
        }

        output
    }
    pub fn sgd(
        &mut self,
        mut training_data: Data,
        test_data: Option<Data>,
        epochs: usize,
        mini_batch_size: usize,
        eta: Float,
    ) {
        for i in 0..epochs {
            training_data.shuffle(&mut thread_rng());

            let mini_batches = training_data.chunks(mini_batch_size);

            for mini_batch in mini_batches {
                let mut changes = self.get_layer_changes();

                for data in mini_batch {
                    self.backpropagation(&data.0, &mut changes, &data.1, eta);
                }
                self.apply_layer_changes(&changes, mini_batch_size);
            }

            if test_data.is_some() {
                let mut correct = 0;
                let num = test_data.as_ref().unwrap().len();

                for data in test_data.as_ref().unwrap() {
                    let output = self.forward(data.0.clone());
                    if get_index_max(&output) == get_index_max(&data.1) {
                        correct += 1;
                    }
                }

                println!(
                    "Epoch {}: {} / {} ({}%)",
                    i,
                    correct,
                    num,
                    (correct * 100) as Float / num as Float
                );
            } else {
                println!("Epoch {} complete.", i);
            }
        }
    }
}

use std::cmp::Ordering;

fn get_index_max(nets: &Vec<Float>) -> usize {
    let index_of_max: Option<usize> = nets
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        .map(|(index, _)| index);
    index_of_max.unwrap()
}
