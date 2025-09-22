use std::{cmp::Ordering, collections::HashMap};

use crate::{
    output_activation_type::OutputActivationType, matrix::Matrix, neural_network::{differential_relu, relu, NeuralNetwork}
};

mod matrix;
mod neural_network;
mod rand;
mod output_activation_type;

fn main() {
    // iris dataset 用ニューラルネットワーク
    let mut nn = NeuralNetwork::new(vec![4, 8, 4, 3], 10);
    nn.set_activations(&mut vec![relu, relu, relu]);
    nn.set_differential_activation(&mut vec![differential_relu, differential_relu, differential_relu]);
    nn.set_output_activation_type(OutputActivationType::SoftmaxAndCrossEntropy);

    
}

fn generate_shuffle_array(value: usize) -> Vec<usize> {
    let v = (0..value).collect::<Vec<usize>>();
    let r = rand::Rand::new();

    for i in (1..value).rev() {
        
    }

    v
}