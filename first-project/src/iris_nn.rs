use std::collections::HashMap;

use crate::{
    matrix::Matrix, neural_network::*, output_activation_type::OutputActivationType, rand::Rand,
};

pub fn iris_nn_process() {
    // ミニバッチ内のサンプルサイズを指定
    let mini_batch_sample_size = 10;

    // iris dataset 用ニューラルネットワーク
    let mut nn = NeuralNetwork::new(vec![4, 6, 8, 6, 4, 3], mini_batch_sample_size);
    nn.set_activations(&mut vec![relu, relu, relu, relu, relu]);
    nn.set_differential_activation(&mut vec![
        differential_relu,
        differential_relu,
        differential_relu,
        differential_relu,
        differential_relu,
    ]);
    nn.set_output_activation_type(OutputActivationType::SoftmaxAndCrossEntropy);

    let epoch_value = 1000;
    let mut r = Rand::new();

    for epoch in 0..epoch_value {
        let shuffle_index = generate_shuffle_array(irisdata::IRIS_DATA.len(), &mut r);
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
            nn.backward(&expects, 0.00001).unwrap();
        }

        println!("epoch {:6} error: {:13.10}", epoch + 1, epoch_error);
    }

    nn.export_ron();
}

fn generate_shuffle_array(value: usize, r: &mut Rand) -> Vec<usize> {
    let mut v = (0..value).collect::<Vec<usize>>();

    for i in 0..=(value - 2) {
        let j = r.rand_usize_range(i, value);
        v.swap(i, j);
    }

    v
}

struct IrisValues {
    sepal_length: f32,
    sepal_width: f32,
    petal_length: f32,
    petal_width: f32,
}

impl IrisValues {
    fn new() -> IrisValues {
        IrisValues {
            sepal_length: 0.0,
            sepal_width: 0.0,
            petal_length: 0.0,
            petal_width: 0.0,
        }
    }
}

pub fn iris_analyze() {
    // 最小値、最大値、中央値、平均値、最頻値
    let mut maximums = IrisValues::new();
    let mut minimums = IrisValues::new();
    let mut medians = IrisValues::new();
    let mut averages = IrisValues::new();
    let mut modes = IrisValues::new();

    let sepal_lengths: Vec<f32> = irisdata::IRIS_DATA.iter().map(|d| d.sepal_length).collect();
    let sepal_width: Vec<f32> = irisdata::IRIS_DATA.iter().map(|d| d.sepal_width).collect();
    let petal_length: Vec<f32> = irisdata::IRIS_DATA.iter().map(|d| d.petal_length).collect();
    let petal_width: Vec<f32> = irisdata::IRIS_DATA.iter().map(|d| d.petal_width).collect();

    maximums.sepal_length = sepal_lengths.iter().fold(0.0 / 0.0, |m, v| v.max(m));
    maximums.sepal_width = sepal_width.iter().fold(0.0 / 0.0, |m, v| v.max(m));
    maximums.petal_length = petal_length.iter().fold(0.0 / 0.0, |m, v| v.max(m));
    maximums.petal_width = petal_width.iter().fold(0.0 / 0.0, |m, v| v.max(m));

    minimums.sepal_length = sepal_lengths.iter().fold(0.0 / 0.0, |m, v| v.min(m));
    minimums.sepal_width = sepal_width.iter().fold(0.0 / 0.0, |m, v| v.min(m));
    minimums.petal_length = petal_length.iter().fold(0.0 / 0.0, |m, v| v.min(m));
    minimums.petal_width = petal_width.iter().fold(0.0 / 0.0, |m, v| v.min(m));

    averages.sepal_length = sepal_lengths.iter().sum::<f32>() / sepal_lengths.len() as f32;
    averages.sepal_width = sepal_width.iter().sum::<f32>() / sepal_width.len() as f32;
    averages.petal_length = petal_length.iter().sum::<f32>() / petal_length.len() as f32;
    averages.petal_width = petal_width.iter().sum::<f32>() / petal_width.len() as f32;

    let half_index = irisdata::IRIS_DATA.len() / 2;
    println!("half index: {}", half_index);

    let mut sorted_sepal_lengths = sepal_lengths.clone();
    sorted_sepal_lengths.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut sorted_sepal_width = sepal_width.clone();
    sorted_sepal_width.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut sorted_petal_lengths = petal_length.clone();
    sorted_petal_lengths.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut sorted_petal_width = petal_width.clone();
    sorted_petal_width.sort_by(|a, b| a.partial_cmp(b).unwrap());

    if irisdata::IRIS_DATA.len() % 2 == 0 {
        medians.sepal_length =
            (sorted_sepal_lengths[half_index] + sorted_sepal_lengths[half_index - 1]) / 2.0;
        medians.sepal_width =
            (sorted_sepal_width[half_index] + sorted_sepal_width[half_index - 1]) / 2.0;
        medians.petal_length =
            (sorted_petal_lengths[half_index] + sorted_petal_lengths[half_index - 1]) / 2.0;
        medians.petal_width =
            (sorted_petal_width[half_index] + sorted_petal_width[half_index - 1]) / 2.0;
    } else {
        medians.sepal_length = sorted_sepal_lengths[half_index];
        medians.sepal_width = sorted_sepal_width[half_index];
        medians.petal_length = sorted_petal_lengths[half_index];
        medians.petal_width = sorted_petal_width[half_index];
    }

    // let mut sepal_length_map = HashMap::<f32, usize>::new();
    // let mut sepal_width_map = HashMap::<f32, usize>::new();
    // let mut petal_length_map = HashMap::<f32, usize>::new();
    // let mut petal_width_map = HashMap::<f32, usize>::new();

    for datum in irisdata::IRIS_DATA {}

    // 範囲、分散 (標準偏差)、歪度、尖度


    println!("maximums:");
    println!("  sepal length: {}", maximums.sepal_length);
    println!("  sepal width: {}", maximums.sepal_width);
    println!("  petal length: {}", maximums.petal_length);
    println!("  petal width: {}", maximums.petal_width);

    println!("minimums:");
    println!("  sepal length: {}", minimums.sepal_length);
    println!("  sepal width: {}", minimums.sepal_width);
    println!("  petal length: {}", minimums.petal_length);
    println!("  petal width: {}", minimums.petal_width);

    println!("averages:");
    println!("  sepal length: {}", averages.sepal_length);
    println!("  sepal width: {}", averages.sepal_width);
    println!("  petal length: {}", averages.petal_length);
    println!("  petal width: {}", averages.petal_width);

    println!("medians:");
    println!("  sepal length: {}", medians.sepal_length);
    println!("  sepal width: {}", medians.sepal_width);
    println!("  petal length: {}", medians.petal_length);
    println!("  petal width: {}", medians.petal_width);

}
