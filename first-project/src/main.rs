use std::{cmp::Ordering, collections::HashMap};

use crate::{matrix::Matrix, neural_network::NeuralNetwork, rand::Rand};

mod matrix;
mod neural_network;
mod rand;

fn main() {
    let mut nn = NeuralNetwork::new(vec![3, 4, 3, 2]);

    nn.forward(&Matrix::new_from_vec(3, 1, vec![1.0, 2.0, 1.5]).unwrap())
        .expect("Failed to forward pass");

    println!("{:6.2}", nn);
}
