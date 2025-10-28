use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct RonNNData {
	pub layer_value: usize,
	pub node_values: Vec<usize>,
	pub layers: Vec<LayerInfo>
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LayerInfo {
	pub weights: Vec<f64>,
	pub biases: Vec<f64>
}