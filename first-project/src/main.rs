use crate::neural_network::NeuralNetwork;

mod matrix;
mod neural_network;
mod auto_apply_ops;

fn main() {
	let nn = NeuralNetwork::new(vec![3, 2, 1]);
	
	println!("{:#?}", nn);
}