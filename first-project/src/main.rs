use std::ops;

use crate::{macros::apply_ops::apply_arithmetics, matrix::Matrix, neural_network::NeuralNetwork};

mod matrix;
mod neural_network;
mod auto_apply_ops;
mod macros;

fn main() {
	let nn = NeuralNetwork::new(vec![3, 2, 1]);
	
	apply_arithmetics_base!(Matrix, Matrix, ops::Add, add, +);
}