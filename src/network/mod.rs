pub mod layer;

pub mod neuron;

mod change;

mod utility;

use crate::front_end::graph_results;

pub use neuron::{
    activation_function::ActivationFunction,
    cost_function::{CostFunction, Regularisation},
    initialisation::InitType,
    Neuron,
};

use {
    change::LayerChange,
    layer::{Layer, LayerTrait},
    utility::{max_index, Float},
};

use rand::prelude::SliceRandom;
use rand::thread_rng;

type Data = Vec<(Vec<Float>, Vec<Float>)>;

pub struct Network {
    layers: Vec<Layer>,
}

impl Network {
    fn apply_layer_changes(
        &mut self,
        changes: &Vec<LayerChange>,
        learning_rate: Float,
        mini_batch_size: usize,
    ) {
        assert_eq!(self.layers.len(), changes.len());

        for (layer, layer_change) in self.layers.iter_mut().zip(changes) {
            layer.update(
                layer_change,
                learning_rate,
                mini_batch_size,
                &crate::network::Regularisation::None,
            );
        }
    }

    fn backpropagation(
        &mut self,
        input: &Vec<Float>,
        changes: &mut Vec<LayerChange>,
        expected_output: &Vec<Float>,
    ) -> Vec<Float> {
        let len = self.layers.len();
        assert_eq!(expected_output.len(), self.layers[len - 1].neuron_count());
        let mut z_values = Vec::new();
        let mut outputs = Vec::new();
        let mut output = input.clone();
        for layer in &mut self.layers {
            layer.forward(output);
            output = layer.last_output();
            outputs.push(output.clone());
            z_values.push(layer.last_z_values());
        }

        let expected_output = vec![expected_output.clone(); 1];

        let (mut errors, mut weights) = self.layers[len - 1].backward(
            &outputs[outputs.len() - 2],
            &mut changes[len - 1],
            &output,
            expected_output,
        );

        for i in 0..(len - 1) {
            let layer_index = len - (i + 2);

            let outputs_index = if i == len - 2 { 0 } else { len - (i + 3) };

            let (terrors, tweights) = self.layers[layer_index].backward(
                &outputs[outputs_index],
                &mut changes[layer_index],
                &errors,
                weights,
            );

            errors = terrors;
            weights = tweights;
        }

        output
    }

    fn empty_layer_changes(&self) -> Vec<LayerChange> {
        let mut changes = Vec::new();
        for layer in &self.layers {
            changes.push(layer.empty_layer_change());
        }
        changes
    }

    pub fn forward(&mut self, input: Vec<Float>) -> Vec<Float> {
        let mut next_input = input;
        for layer in &mut self.layers {
            layer.forward(next_input);
            next_input = layer.last_output();
        }
        next_input
    }

    pub fn from_layers(layers: Vec<Layer>) -> Self {
        Network { layers }
    }

    pub fn sgd(
        &mut self,
        mut training_data: Data,
        test_data: Option<Data>,
        epochs: usize,
        mini_batch_size: usize,
        learning_rate: Float,
    ) {
        let mut results: Vec<(f32, f64)> = Vec::new();
        if test_data.is_some() {
            let mut correct = 0;
            let num = test_data.as_ref().unwrap().len();

            for data in test_data.as_ref().unwrap() {
                let output = self.forward(data.0.clone());
                if max_index(&output) == max_index(&data.1) {
                    correct += 1;
                }
            }

            let percent_correct = (correct * 100) as Float / num as Float;
            println!("Epoch 0: {} / {} ({}%)", correct, num, percent_correct);
        }

        let mut max_correct = 0;
        let mut min_correct = test_data.as_ref().unwrap().len();
        for i in 0..epochs {
            training_data.shuffle(&mut thread_rng());

            let mini_batches = training_data.chunks(mini_batch_size);

            for mini_batch in mini_batches {
                let mut changes = self.empty_layer_changes();

                for data in mini_batch {
                    self.backpropagation(&data.0, &mut changes, &data.1);
                }
                self.apply_layer_changes(&changes, learning_rate, mini_batch_size);
            }

            if test_data.is_some() {
                let mut correct = 0;
                let num = test_data.as_ref().unwrap().len();

                for data in test_data.as_ref().unwrap() {
                    let output = self.forward(data.0.clone());
                    if max_index(&output) == max_index(&data.1) {
                        correct += 1;
                    }
                }
                if correct > max_correct {
                    max_correct = correct;
                }

                if correct < min_correct {
                    min_correct = correct;
                }

                let percent_correct = (correct * 100) as Float / num as Float;
                println!(
                    "Epoch {}: {} / {} ({}%)",
                    i + 1,
                    correct,
                    num,
                    percent_correct
                );
                results.push(((i + 1) as Float, percent_correct as f64));
            } else {
                println!("Epoch {} complete.", i + 1);
            }
        }
        if test_data.is_some() {
            let num = test_data.as_ref().unwrap().len();
            println!(
                "Highest accuracy: {} / {} ({}%)",
                max_correct,
                num,
                (max_correct * 100) as Float / num as Float
            );
            graph_results(
                epochs,
                results,
                (min_correct * 100) as f64 / num as f64,
                (max_correct * 100) as f64 / num as f64,
            );
        }
    }
}
