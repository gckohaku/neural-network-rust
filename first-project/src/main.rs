use std::{cmp::Ordering, collections::HashMap};

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
    let division_values = 64;
    let min_value = -4.0;
    let max_value = 4.0;
    let value_range = max_value - min_value;
    let values_length = 100000;
    let mut counts = HashMap::new();
    let mut loss_count = 0;

    for i in 0..division_values {
        let key = format!(
            "{:.5} <= v < {:.5}",
            min_value + (value_range / division_values as f64) * i as f64,
            min_value + (value_range / division_values as f64) * (i + 1) as f64
        );
        // println!("{}", key);

        counts.insert(key, 0);
    }

    for _i in 0..values_length {
        let value = r.normal(0.0, 1.0);
        // println!("{:.20}", value);

        // ここの式が不適切
        let category = ((value - min_value) * division_values as f64).floor();
        println!("{}", category);

        let key = format!(
            "{:.5} <= v < {:.5}",
            min_value + (value_range / division_values as f64) * category,
            min_value + (value_range / division_values as f64) * (category + 1.0)
        );
        // println!("{}", key);

        if counts.contains_key(&key) {
            let new_value = counts.get(&key).unwrap() + 1;
            counts.insert(key, new_value);
        }
        else {
            loss_count += 1;
        }
    }

    let mut keys = counts.keys().cloned().collect::<Vec<String>>();
    keys.sort_by(|a, b| {
        if a.chars().nth(0).unwrap() == '-' && b.chars().nth(0).unwrap() != '-' {
            return Ordering::Less;
        } else if a.chars().nth(0).unwrap() != '-' && b.chars().nth(0).unwrap() == '-' {
            return Ordering::Greater;
        } else if a.chars().nth(0).unwrap() == '-' && b.chars().nth(0).unwrap() == '-' {
            return b.cmp(a);
        }
        a.cmp(b)
    });

    for key in keys {
        let freq = counts.get(&key).unwrap();

        println!("{}: {}", key, freq)
    }
    println!("loss count: {}", loss_count)
}
