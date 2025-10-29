use std::{
    cell::{RefCell, UnsafeCell},
    sync::Arc,
    time,
};

use mnist::MnistBuilder;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    fully_connected_network::FullyConnectedNetwork,
    matrix::Matrix,
    neural_network_base::{NetworkWorkspace, NeuralNetwork},
    neural_network_functions::{differential_relu, relu},
    output_activation_type::OutputActivationType,
    rand::Rand,
    utilities::shuffle::generate_shuffle_array,
};

struct MiniBatchParSample {
    input: Vec<f64>,
    expect: Vec<f64>,
}

pub fn mnist_process() {
    let mini_batch_sample_size = 100;

    // ニューラルネットワーク初期化
    let mut nn = FullyConnectedNetwork::new(vec![784, 1568, 784, 392, 196, 49, 10], 1);
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
    let mut workspace = NetworkWorkspace::new_for_network(&nn, 1);

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

    let arc_nn = Arc::new(nn.clone());

    // Arc<NeuralNetwork> を格納するスロットを作成
    thread_local!(
        static NN_ARC: UnsafeCell<Option<Arc<FullyConnectedNetwork>>> = UnsafeCell::new(None);
    );

    //  スレッドローカルなワークスペースを作成
    thread_local!(
        static WORKSPACE: UnsafeCell<Option<Arc<NetworkWorkspace>>> = UnsafeCell::new(None);
    );

    // 現在の時刻
    let epochs_now: time::Instant = time::Instant::now();

    for epoch in 0..epoch_value {
        let shuffle_index = generate_shuffle_array(training_value.try_into().unwrap(), &mut r);
        let mut epoch_error = 0.0;

        let mut mini_batch_count = 0;
        let workspace = NetworkWorkspace::new_for_network(&nn, 1);

        for indexes in shuffle_index.chunks(mini_batch_sample_size) {
            mini_batch_count += 1;
            let mut mini_batch_par_sample: Vec<MiniBatchParSample> = Vec::new();

            for index in indexes {
                let mut batch_data: Vec<f64> = Vec::new();
                let mut expect_data: Vec<f64> = Vec::new();

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

                mini_batch_par_sample.push(MiniBatchParSample {
                    input: batch_data,
                    expect: expect_data,
                });
            }

            // let inputs =
            //     Matrix::new_from_vec(mini_batch_sample_size, input_node_value, batch_data).unwrap();
            // let expects =
            //     Matrix::new_from_vec(mini_batch_sample_size, output_node_value, expect_data)
            //         .unwrap();

            let (error, next_weights, next_biases) = mini_batch_par_sample
                .par_iter()
                .map(|sample| {
                    // NN_ARC と WORKSPACE が初期化されていなければ初期化する
                    let local_nn_ref = NN_ARC.with(|nn_cell| {
                        let nn_slot = unsafe { &mut *nn_cell.get() };

                        if nn_slot.is_none() {
                            // Arc::clone() で一度だけ初期化
                            let cloned_arc = Arc::clone(&arc_nn);

                            // 同時に WORKSPACE も初期化
                            WORKSPACE.with(|ws_cell| {
                                let ws_slot = unsafe { &mut *ws_cell.get() };
                                *ws_slot = Some(
                                    NetworkWorkspace::new_for_network(cloned_arc.as_ref(), 1)
                                        .into(),
                                );
                            });

                            // NN_ARC スロットに格納
                            *nn_slot = Some(cloned_arc).into();
                        }
                        // NN_ARC スロットからネットワーク参照を取り出して利用
                        nn_slot.as_ref().unwrap()
                    });

                    // ワークスペースにアクセス (毎回)
                    let mut workspace = WORKSPACE.with(|ws_cell| {
                        let ws_slot = unsafe { &mut *ws_cell.get() };
                        ws_slot.as_mut().unwrap()
                    });

                    // let local_nn = Arc::clone(&arc_nn);

                    local_nn_ref.forward_and_backward(
                        &Matrix::new_from_vec(1, input_node_value, sample.input.clone()).unwrap(),
                        &Matrix::new_from_vec(1, output_node_value, sample.expect.clone()).unwrap(),
                        &mut workspace,
                        0.001,
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

            let mut average_weights = Vec::new();
            let mut average_biases = Vec::new();
            for i in 0..next_weights.len() {
                average_weights.push((&next_weights[i] / mini_batch_sample_size as f64).unwrap());
                average_biases.push((&next_biases[i] / mini_batch_sample_size as f64).unwrap());
            }

            nn.update_weights(&mut average_weights, &mut average_biases);

            epoch_error += error;

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
