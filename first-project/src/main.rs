use std::{cmp::Ordering, collections::HashMap};

use crate::{
    matrix::Matrix,
    neural_network::{NeuralNetwork, differential_relu, relu},
};

mod matrix;
mod neural_network;
mod rand;

fn main() {
    let mut nn = NeuralNetwork::new(vec![3, 4, 3, 2]);
    nn.set_activations(&mut vec![relu, relu]);
    nn.set_differential_activation(&mut vec![differential_relu, differential_relu]);
    nn.set_use_softmax_flags(&mut vec![false, false, true]);

    nn.forward(&Matrix::new_from_vec(3, 1, vec![1.0, 2.0, 1.5]).unwrap())
        .expect("Failed to forward pass");

    println!("{:6.2}", nn);
}
