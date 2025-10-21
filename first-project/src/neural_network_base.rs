use core::net;
use std::ops;

use crate::{matrix::Matrix};

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
    pub local_gradients: Gradients,
}

impl NetworkWorkspace {
    pub fn new_for_network(network_shape: &impl NeuralNetwork) -> Self {
        let layer_value = network_shape.get_layer_value();

        let mut layer_inputs = Vec::<Matrix>::new();
        let mut layer_outputs = Vec::<Matrix>::new();
        let mut layer_deltas = Vec::<Matrix>::new();

        layer_outputs.push(Matrix::new(1, network_shape.get_weight_matrix(0).rows));

        for i in 0..layer_value {
            let mut current_node_value = network_shape.get_weight_matrix(0).cols;
            layer_inputs.push(Matrix::new(1, current_node_value));
            layer_outputs.push(Matrix::new(1, current_node_value));
            layer_deltas.push(Matrix::new(1, current_node_value));
        }

        let error = 0.0;
        let local_gradients = Gradients::new(network_shape);

        NetworkWorkspace {
            layer_inputs: layer_inputs,
            layer_outputs: layer_outputs,
            layer_deltas: layer_deltas,
            error: error,
            local_gradients: local_gradients,
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
    ) -> Result<Gradients, String>;
    fn update_weights(&mut self, workspace: &mut NetworkWorkspace);
    fn get_layer_value(&self) -> usize;
    fn get_weight_matrix(&self, index: usize) -> &Matrix;

    fn relu(x: &f64) -> f64 {
        x.max(0.0)
    }

    fn differential_relu(x: &f64) -> f64 {
        if *x > 0.0 { 1.0 } else { 0.0 }
    }

    fn leaky_relu(x: &f64) -> f64 {
        if *x > 0.0 { *x } else { 0.01 * *x }
    }

    fn differential_leaky_relu(x: &f64) -> f64 {
        if *x > 0.0 { 1.0 } else { 0.01 }
    }

    fn softmax(z: &mut Matrix) {
        let nodes_exp_sum: f64 = z.data.iter().map(|x| x.exp()).sum();
        for i in 0..z.rows {
            for j in 0..z.cols {
                let before_normalize = z.get(i, j).unwrap();
                z.set(i, j, before_normalize.exp() / nodes_exp_sum).unwrap();
            }
        }
    }
}
