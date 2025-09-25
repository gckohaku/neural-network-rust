use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct RonNNData {
	layer_value: usize,
	node_values: Vec<usize>,
	layers: Vec<LayerInfo>
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LayerInfo {
	weights: Vec<f64>,
	biases: Vec<f64>
}