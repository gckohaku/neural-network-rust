use std::{cmp::Ordering, collections::HashMap};

use crate::{
    iris_nn::{iris_analyze, iris_nn_process}, matrix::Matrix, neural_network::{differential_relu, relu, NeuralNetwork}, output_activation_type::OutputActivationType, rand::Rand
};

mod matrix;
mod neural_network;
mod output_activation_type;
mod rand;
mod ron_data;
mod iris_nn;

fn main() {
    // iris_nn_process();

    iris_analyze();
}