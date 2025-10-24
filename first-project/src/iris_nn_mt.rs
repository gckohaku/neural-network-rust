use std::{sync::Arc, time};

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    fully_connected_network::*,
    iris_normalization::iris_normalization,
    matrix::Matrix,
    neural_network_base::{NetworkWorkspace, NeuralNetwork},
    neural_network_functions::{differential_relu, relu},
    output_activation_type::OutputActivationType,
    rand::Rand,
};

#[derive(Clone)]
struct SampleData {
    inputs: Vec<f64>,
    expects: Vec<f64>,
}

impl SampleData {
    fn new() -> SampleData {
        SampleData {
            inputs: vec![],
            expects: vec![],
        }
    }
}

pub fn iris_nn_process() {
    let normalization_data = iris_normalization(&irisdata::IRIS_DATA);
    // println!("{:?}", &normalization_data);

    // ミニバッチ内のサンプルサイズを指定
    let mini_batch_sample_size = 10;

    // iris dataset 用ニューラルネットワーク
    let mut nn = FullyConnectedNetwork::new(vec![4, 12, 24, 36, 24, 12, 3], 1);
    nn.set_activations(&mut vec![relu, relu, relu, relu, relu, relu]);
    nn.set_differential_activation(&mut vec![
        differential_relu,
        differential_relu,
        differential_relu,
        differential_relu,
        differential_relu,
        differential_relu,
    ]);
    nn.set_output_activation_type(OutputActivationType::SoftmaxAndCrossEntropy);

    let epoch_value = 50000;
    let mut r = Rand::new();

    // 現在の時刻
    let epochs_now = time::Instant::now();

    for epoch in 0..epoch_value {
        let shuffle_index = generate_shuffle_array(normalization_data.len(), &mut r);
        let mut epoch_error: f64 = 0.0;

        for indexes in shuffle_index.chunks(mini_batch_sample_size) {
            let mut batch = Vec::new();

            for index in indexes {
                batch.push(&normalization_data[*index]);
            }

            let mut mini_batch_data = Vec::<SampleData>::new();

            for data in batch {
                let mut sample_data = SampleData::new();

                sample_data.inputs.push(data.sepal_length as f64);
                sample_data.inputs.push(data.sepal_width as f64);
                sample_data.inputs.push(data.petal_length as f64);
                sample_data.inputs.push(data.petal_width as f64);

                sample_data
                    .expects
                    .push(if data.species == irisdata::Species::IrisSetosa {
                        1.0
                    } else {
                        0.0
                    });
                sample_data
                    .expects
                    .push(if data.species == irisdata::Species::IrisVersicolor {
                        1.0
                    } else {
                        0.0
                    });
                sample_data
                    .expects
                    .push(if data.species == irisdata::Species::IrisVirginica {
                        1.0
                    } else {
                        0.0
                    });

                mini_batch_data.push(sample_data);
            }

            let input_node_value = nn.get_input_node_value();
            let output_node_value = nn.get_output_node_value();

            // 誤差と次の重みとバイアスの平均を受け取るタプルを返すようにする
            let (error, mut next_weights, mut next_biases) = mini_batch_data
                .par_iter()
                .map(|data| {
                    let mut workspace = NetworkWorkspace::new_for_network(&nn, 1);

                    nn.forward_and_backward(
                        &Matrix::new_from_vec(1, input_node_value, data.inputs.clone()).unwrap(),
                        &Matrix::new_from_vec(1, output_node_value, data.expects.clone()).unwrap(),
                        &mut workspace,
                        0.0007,
                    );

                    (
                        workspace.error,
                        workspace.next_weights,
                        workspace.next_biases,
                    )
                })
                .reduce(
                    || (0.0, Vec::new(), Vec::new()),
                    |mut current, next| {
                        if current.1.len() == 0 {
                            return next;
                        }

                        current.0 += next.0;
                        for i in 0..current.1.len() {
                            current.1[i] += &next.1[i];
                            current.2[i] += &next.2[i];
                        }

                        current
                    },
                );

            nn.update_weights(&mut next_weights, &mut next_biases);

            epoch_error += error;
        }

        if (epoch + 1) % 500 == 0 {
            println!(
                "epoch {:6} error: {:13.10}",
                epoch + 1,
                epoch_error / irisdata::IRIS_DATA.len() as f64
            );
            // print!("{:8.4}", nn.get_output_nodes());
        }

        // println!("{:?}", nn.get_weight_matrix(5)[(0, 0)]);
    }

    println!(
        "epochs process duration: {:?}sec.",
        epochs_now.elapsed().as_secs_f64()
    );

    // nn.export_ron();
}

fn generate_shuffle_array(value: usize, r: &mut Rand) -> Vec<usize> {
    let mut v = (0..value).collect::<Vec<usize>>();

    for i in 0..=(value - 2) {
        let j = r.rand_usize_range(i, value);
        v.swap(i, j);
    }

    v
}
