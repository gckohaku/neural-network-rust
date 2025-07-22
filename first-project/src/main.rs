use std::collections::HashMap;

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

    // 複数回乱数を回してテスト
    // for _ in 0 .. 50 {
    //     println!("{:.20}", r.rand_f64());
    // }

    // TODO: 正規分布に従った乱数の生成がしっかりと実装できているかをテスト
    let division_values = 20;
    let min_value = 0.0;
    let max_value = 1.0;
    let value_range = max_value - min_value;
    let values_length = 100000;
    let mut counts = HashMap::new();

    for i in 0 .. division_values {
        let key = format!("{:.2} <= v < {:.2}", min_value + (value_range / division_values as f64) * i as f64, min_value + (value_range / division_values as f64) * (i + 1) as f64);
        println!("{}", key);

        counts.insert(key, 0);
    }

    for _i in 0 .. values_length {
        let value = r.rand_f64();
        println!("{:.20}", value);
        println!("{:.20}", (value_range / division_values as f64));

        let category = (((value - min_value) * division_values as f64) / division_values as f64).ceil();

        let key = format!("{:.2} <= v < {:.2}", min_value + (value_range / division_values as f64) * category, min_value + (value_range / division_values as f64) * (category + 1.0));
        println!("{}", key);

        let new_value = counts.get(&key).unwrap() + 1;

        counts.insert(key, new_value);
    }

    for data in counts {
        println!("{:?}", data)
    }
}
