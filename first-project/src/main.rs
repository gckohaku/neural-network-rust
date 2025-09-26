use std::{cmp::Ordering, collections::HashMap};

use crate::{
    matrix::Matrix,
    neural_network::{differential_relu, relu, NeuralNetwork},
    output_activation_type::OutputActivationType, rand::Rand,
};

mod matrix;
mod neural_network;
mod output_activation_type;
mod rand;
mod ron_data;

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

    let epoch_value = 100000;
    let r = Rand::new();

    for epoch in 0..epoch_value {
        let shuffle_index = generate_shuffle_array(irisdata::IRIS_DATA.len(), r);
        let mut epoch_error = 0.0;

        for indexes in shuffle_index.chunks(mini_batch_sample_size) {
            let mut batch = Vec::new();

            for index in indexes {
                batch.push(&irisdata::IRIS_DATA[*index]);
            }

            let mut batch_data: Vec<f64> = Vec::new();
            let mut expect_data: Vec<f64> = Vec::new();

            for data in batch {
                batch_data.push(data.sepal_length as f64);
                batch_data.push(data.sepal_width as f64);
                batch_data.push(data.petal_length as f64);
                batch_data.push(data.petal_width as f64);

                expect_data.push(if data.species == irisdata::Species::IrisSetosa {
                    1.0
                } else {
                    0.0
                });
                expect_data.push(if data.species == irisdata::Species::IrisVersicolor {
                    1.0
                } else {
                    0.0
                });
                expect_data.push(if data.species == irisdata::Species::IrisVirginica {
                    1.0
                } else {
                    0.0
                });
            }

            let input_node_value = nn.get_input_node_value();
            let output_node_value = nn.get_output_node_value();

            let inputs =
                Matrix::new_from_vec(mini_batch_sample_size, input_node_value, batch_data).unwrap();
            let expects =
                Matrix::new_from_vec(mini_batch_sample_size, output_node_value, expect_data)
                    .unwrap();

            // println!("{:7.2}", &expects);

            nn.forward(&inputs, &expects).unwrap();
            epoch_error += nn.get_error();
            nn.backward(&expects, 0.1).unwrap();
        }

        println!("epoch {:6} error: {:13.10}", epoch, epoch_error);
    }

    nn.export_ron();
}

fn generate_shuffle_array(value: usize, r: Rand) -> Vec<usize> {
    let mut v = (0..value).collect::<Vec<usize>>();

    for i in 0..=(value - 2) {
        let j = r.rand_usize_range(i, value);
        v.swap(i, j);
    }

    v
}
