use std::ops;

use crate::{matrix::Matrix, output_activation_type::OutputActivationType};

#[derive(Clone)]
pub struct Gradients {
    pub weights: Vec<Matrix>,
    pub biases: Vec<Matrix>,
}

impl Gradients {
    pub fn new(network_shape: &impl NeuralNetwork) -> Self {
        let layer_value = network_shape.get_layer_value();

        let mut weights = Vec::<Matrix>::new();
        let mut biases = Vec::<Matrix>::new();

        for i in 0..layer_value {
            let m = network_shape.get_weight_matrix(i);
            weights.push(Matrix::new(m.rows, m.cols));
            biases.push(Matrix::new(1, m.cols));
        }

        Self {
            weights: weights,
            biases: biases,
        }
    }
}

impl ops::AddAssign<Gradients> for Gradients {
    fn add_assign(&mut self, rhs: Gradients) {
        for index in 0..self.weights.len() {
            self.weights[index] += &rhs.weights[index];
            self.biases[index] += &rhs.biases[index];
        }
    }
}

/// ニューラルネットワークの学習に必要な中間データなどを格納する
pub struct NetworkWorkspace {
    pub layer_inputs: Vec<Matrix>,
    /// layer_outputs は入力層のデータも保持しているため、Vec の長さが 1 長い
    pub layer_outputs: Vec<Matrix>,
    pub layer_deltas: Vec<Matrix>,
    pub error: f64,
    pub next_weights: Vec<Matrix>,
    pub next_biases: Vec<Matrix>,
}

impl NetworkWorkspace {
    pub fn new_for_network(network_shape: &impl NeuralNetwork, sample_value: usize) -> Self {
        let layer_value = network_shape.get_layer_value();

        let mut layer_inputs = Vec::<Matrix>::new();
        let mut layer_outputs = Vec::<Matrix>::new();
        let mut layer_deltas = Vec::<Matrix>::new();
        let mut next_weights = Vec::<Matrix>::new();
        let mut next_biases = Vec::<Matrix>::new();

        layer_outputs.push(Matrix::new(
            sample_value,
            network_shape.get_weight_matrix(0).rows,
        ));

        for i in 0..layer_value {
            let current_matrix = network_shape.get_weight_matrix(i);
            let current_node_value = current_matrix.cols;
            layer_inputs.push(Matrix::new(sample_value, current_node_value));
            layer_outputs.push(Matrix::new(sample_value, current_node_value));
            layer_deltas.push(Matrix::new(sample_value, current_node_value));
            next_weights.push(Matrix::new(current_matrix.rows, current_matrix.cols));
            next_biases.push(Matrix::new(1, current_node_value));
        }

        let error = 0.0;

        NetworkWorkspace {
            layer_inputs: layer_inputs,
            layer_outputs: layer_outputs,
            layer_deltas: layer_deltas,
            error: error,
            next_weights: next_weights,
            next_biases: next_biases,
        }
    }
}

pub trait NeuralNetwork {
    fn new(nodes_values: Vec<usize>, sample_value: usize) -> Self;
    fn forward_and_backward(
        &self,
        inputs: &Matrix,
        expects: &Matrix,
        workspace: &mut NetworkWorkspace,
        eta: f64,
    );
    fn update_weights(&mut self, next_weights: &mut Vec<Matrix>, next_biases: &mut Vec<Matrix>);
    fn set_activations(&mut self, activations: &mut Vec<fn(&f64) -> f64>);
    fn set_differential_activation(&mut self, differentials: &mut Vec<fn(&f64) -> f64>);
    fn set_output_activation_type(&mut self, activation_type: OutputActivationType);
    fn get_layer_value(&self) -> usize;
    fn get_input_node_value(&self) -> usize;
    fn get_output_node_value(&self) -> usize;
    fn get_weight_matrix(&self, index: usize) -> &Matrix;
    fn get_zero_weights(&self) -> Vec<Matrix> {
        let layer_value = self.get_layer_value();
        let mut return_weights = Vec::<Matrix>::new();

        for i in 0..layer_value {
            let current_matrix = self.get_weight_matrix(i);
            return_weights.push(Matrix::new(current_matrix.rows, current_matrix.cols));
        }

        return_weights
    }
    fn get_zero_biases(&self) -> Vec<Matrix> {
        let layer_value = self.get_layer_value();
        let mut return_weights = Vec::<Matrix>::new();

        for i in 0..layer_value {
            let current_matrix = self.get_weight_matrix(i);
            return_weights.push(Matrix::new(1, current_matrix.cols));
        }

        return_weights
    }
    fn export_ron(&self) -> Result<(), Box<dyn std::error::Error>>;
}
