mod Matrix;

pub struct NeuralNetwork {
	layers: Vec<Matrix>,
	nodes: Vec<Matrix>,
	weights: Vec<Matrix>,
	biases: Vec<Matrix>,
	deltas: Vec<Matrix>,
}