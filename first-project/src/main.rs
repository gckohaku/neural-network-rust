use std::{cmp::Ordering, collections::HashMap};

use crate::{
    matrix::Matrix,
    neural_network::{NeuralNetwork, differential_relu, relu},
    output_activation_type::OutputActivationType,
};

mod matrix;
mod neural_network;
mod output_activation_type;
mod rand;

fn main() {
    // ミニバッチ内のサンプルサイズを指定
    let mini_batch_sample_size = 10;

    // iris dataset 用ニューラルネットワーク
    let mut nn = NeuralNetwork::new(vec![4, 8, 4, 3], mini_batch_sample_size);
    nn.set_activations(&mut vec![relu, relu, relu]);
    nn.set_differential_activation(&mut vec![
        differential_relu,
        differential_relu,
        differential_relu,
    ]);
    nn.set_output_activation_type(OutputActivationType::SoftmaxAndCrossEntropy);
}

fn generate_shuffle_array(value: usize) -> Vec<usize> {
    let mut v = (0..value).collect::<Vec<usize>>();
    let r = rand::Rand::new();

    for i in 0..=(value - 2) {
        let j = r.rand_usize_range(i, value);
        v.swap(i, j);
    }

    v
}
