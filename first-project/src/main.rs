use std::ops;

use crate::{matrix::Matrix, neural_network::NeuralNetwork};

mod matrix;
mod neural_network;

fn main() {
    let nn = NeuralNetwork::new(vec![3, 2, 1]);
}
