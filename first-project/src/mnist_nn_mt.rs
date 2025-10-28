use std::time;

use mnist::MnistBuilder;

use crate::{
    fully_connected_network::FullyConnectedNetwork,
    matrix::Matrix,
    neural_network_base::{NetworkWorkspace, NeuralNetwork},
    neural_network_functions::{differential_relu, relu},
    output_activation_type::OutputActivationType,
    rand::Rand,
    utilities::shuffle::generate_shuffle_array,
};

pub fn mnist_process() {
    let mini_batch_sample_size = 100;

    // ニューラルネットワーク初期化
    let mut nn = FullyConnectedNetwork::new(
        vec![784, 1568, 784, 392, 196, 49, 10],
        mini_batch_sample_size,
    );
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

    let input_node_value = nn.get_input_node_value();
    let output_node_value = nn.get_output_node_value();

    let mut r = Rand::new();
    let mut workspace = NetworkWorkspace::new_for_network(&nn, mini_batch_sample_size);

    let epoch_value = 10;

    let image_dot_value = 28 * 28;
    let training_value = 1_000;
    let validation_value = 5_000;
    let test_value = 5_000;

    let mnist = MnistBuilder::new()
        .label_format_one_hot()
        .training_set_length(training_value)
        .validation_set_length(validation_value)
        .test_set_length(test_value)
        .training_images_filename("train-images.idx3-ubyte")
        .training_labels_filename("train-labels.idx1-ubyte")
        .test_images_filename("t10k-images.idx3-ubyte")
        .test_labels_filename("t10k-labels.idx1-ubyte")
        .finalize()
        .normalize();

    println!("{}", mnist.trn_img.len());

    // 現在の時刻
    let epochs_now = time::Instant::now();

    for epoch in 0..epoch_value {
        let shuffle_index = generate_shuffle_array(training_value.try_into().unwrap(), &mut r);
        let mut epoch_error = 0.0;

        let mut mini_batch_count = 0;

        for indexes in shuffle_index.chunks(mini_batch_sample_size) {
            mini_batch_count += 1;
            let mut batch_data: Vec<f64> = Vec::new();
            let mut expect_data: Vec<f64> = Vec::new();

            for index in indexes {
                let trn_data: &Vec<f64> = &mnist.trn_img
                    [(*index) * image_dot_value..((*index) + 1) * image_dot_value]
                    .iter()
                    .map(|&x| x as f64)
                    .collect();
                batch_data.extend_from_slice(trn_data);

                let trn_label: &Vec<f64> = &mnist.trn_lbl[(*index) * 10..((*index) + 1) * 10]
                    .iter()
                    .map(|&x| x as f64)
                    .collect();
                expect_data.extend_from_slice(trn_label);
            }

            let inputs =
                Matrix::new_from_vec(mini_batch_sample_size, input_node_value, batch_data).unwrap();
            let expects =
                Matrix::new_from_vec(mini_batch_sample_size, output_node_value, expect_data)
                    .unwrap();

            nn.forward_and_backward(&inputs, &expects, &mut workspace, 0.001);
            epoch_error += workspace.error;
            if (workspace.error.is_nan()) {
                panic!("error is NaN");
            }
            nn.update_weights(&mut workspace.next_weights, &mut workspace.next_biases);

            println!("mini batch count: {}", mini_batch_count);
        }

        if (epoch + 1) % 1 == 0 {
            println!(
                "epoch {:6} error: {:13.10}",
                epoch + 1,
                epoch_error / training_value as f64
            );
        }
    }

    println!(
        "epochs process duration: {:?}sec.",
        epochs_now.elapsed().as_secs_f64()
    );
}
