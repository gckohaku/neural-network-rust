use crate::{matrix::Matrix, neural_network::NeuralNetwork, rand::Rand};

mod matrix;
mod neural_network;
mod rand;

fn main() {
    // let mut nn = NeuralNetwork::new(vec![3, 2, 1]);

    // nn.forward(&Matrix::new_from_vec(3, 1, vec![1.0, 2.0, 1.5]).unwrap())
    //     .expect("Failed to forward pass");

    // println!("{:5.1}", nn);

    let mut r = Rand::new();

    for _ in 0 .. 50 {
        println!("{:.20}", r.rand_f64());
    }

    // TODO: 正規分布に従った乱数の生成がしっかりと実装できているかをテスト
}
