use crate::matrix::Matrix;

pub trait NeuralNetwork {
	fn new(nodes_values: Vec<usize>, sample_value: usize) -> Self;
	fn forward(&self, input: &Matrix, workspace: &mut Self::Workspace) -> Result<Matrix, String>;
	fn backward(&self, expects: &Matrix, workspace: &mut Self::Workspace);
	fn update(&mut self, workspace: &mut Self::Workspace);

	// 関連型
	type Workspace;
}

pub fn relu(x: &f64) -> f64 {
    x.max(0.0)
}

pub fn differential_relu(x: &f64) -> f64 {
    if *x > 0.0 { 1.0 } else { 0.0 }
}

pub fn leaky_relu(x: &f64) -> f64 {
    if *x > 0.0 { *x } else { 0.01 * *x }
}

pub fn differential_leaky_relu(x: &f64) -> f64 {
    if *x > 0.0 { 1.0 } else { 0.01 }
}