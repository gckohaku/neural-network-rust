use crate::{matrix::Matrix, neural_network::NeuralNetwork};

mod matrix;
mod neural_network;
mod rand;

fn main() {
    let mut nn = NeuralNetwork::new(vec![3, 2, 1]);

    nn.forward(&Matrix::new_from_vec(3, 1, vec![1.0, 2.0, 1.5]).unwrap())
        .expect("Failed to forward pass");

    println!("{:5.1}", nn);
}
