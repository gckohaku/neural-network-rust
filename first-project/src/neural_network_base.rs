use std::ops;

use crate::matrix::Matrix;

pub struct Gradients {
	weights: Vec<Matrix>,
	biases: Vec<Matrix>,
}

impl ops::AddAssign<Gradients> for Gradients {
	fn add_assign(&mut self, rhs: Gradients) {
		for index in 0..self.weights.len() {
			self.weights[index] += &rhs.weights[index];
			self.biases[index] += &rhs.biases[index];
		}
	}
}

pub trait NeuralNetwork {
    fn new(nodes_values: Vec<usize>, sample_value: usize) -> Self;
	fn forward_and_backward(&self, inputs: &Matrix, expects: &Matrix, workspace: &mut Self::Workspace) -> Result<Gradients, String>;
    fn update(&mut self, workspace: &mut Self::Workspace);

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

    // 関連型
    type Workspace;
}
