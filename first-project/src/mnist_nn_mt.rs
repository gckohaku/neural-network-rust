use std::{
    cell::UnsafeCell,
    io::{self, Write},
    sync::{Arc, RwLock},
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

struct MiniBatchParChunk {
    input: Matrix,
    expect: Matrix,
}

pub fn mnist_process(
    epochs: i32,
    chunk_size: usize,
    mini_batch_iteration: usize,
    validation_iteration: usize,
    test_iteration: usize,
    training_max_value: u32,
    validation_max_value: u32,
    test_max_value: u32,
    validation_chunk_size: usize,
    test_chunk_size: usize,
) {
    let epoch_value = epochs;
    let mini_batch_sample_size: usize = chunk_size * mini_batch_iteration;
    let validation_sample_size: usize = validation_chunk_size * validation_iteration;
    let test_sample_size: usize = test_chunk_size * test_iteration;

    let image_dot_value = 28 * 28;
    let training_mini_batch_value = training_max_value / mini_batch_sample_size as u32;
    let validation_mini_batch_value = validation_max_value / validation_sample_size as u32;
    let test_mini_batch_value = test_max_value / test_sample_size as u32;

    let training_value = mini_batch_sample_size as u32 * training_mini_batch_value;
    let validation_value = validation_sample_size as u32 * validation_mini_batch_value;
    let test_value = test_sample_size as u32 * test_mini_batch_value;

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
    let mut workspace = NetworkWorkspace::new_for_network(&nn, chunk_size);
    let mut validation_workspace =
        NetworkWorkspace::new_for_network(&nn, validation_value as usize);

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

    let arc_nn = Arc::new(RwLock::new(nn.clone()));

    // Arc<NeuralNetwork> を格納するスロットを作成
    thread_local!(
        static NN_ARC: UnsafeCell<Option<Arc<RwLock<FullyConnectedNetwork>>>> =
            UnsafeCell::new(None);
    );

    //  スレッドローカルなワークスペースを作成
    thread_local!(
        static WORKSPACE: UnsafeCell<Option<NetworkWorkspace>> = UnsafeCell::new(None);
    );

    let workspace = NetworkWorkspace::new_for_network(&nn, 1);

    // 現在の時刻
    let epochs_now: time::Instant = time::Instant::now();

    for epoch in 0..epoch_value {
        let shuffle_index = generate_shuffle_array(training_value.try_into().unwrap(), &mut r);
        let mut epoch_error = 0.0;

        let mut mini_batch_count = 0;

        for indexes in shuffle_index.chunks(mini_batch_sample_size) {
            mini_batch_count += 1;
            let mut mini_batch_par_chunk: Vec<MiniBatchParChunk> = Vec::new();
            let mut batch_data: Vec<f64> = Vec::new();
            let mut expect_data: Vec<f64> = Vec::new();

            let mut current_value_in_chunk = 0;
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

                current_value_in_chunk += 1;
                if current_value_in_chunk == chunk_size {
                    mini_batch_par_chunk.push(MiniBatchParChunk {
                        input: Matrix::new_from_vec(
                            chunk_size,
                            image_dot_value,
                            batch_data.clone(),
                        )
                        .unwrap(),
                        expect: Matrix::new_from_vec(chunk_size, 10, expect_data.clone()).unwrap(),
                    });
                    batch_data.clear();
                    expect_data.clear();
                    current_value_in_chunk = 0;
                }
            }

            // let inputs =
            //     Matrix::new_from_vec(mini_batch_sample_size, input_node_value, batch_data).unwrap();
            // let expects =
            //     Matrix::new_from_vec(mini_batch_sample_size, output_node_value, expect_data)
            //         .unwrap();

            // let now = time::Instant::now();
            let (error, next_weights, next_biases) = mini_batch_par_chunk
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
                                    NetworkWorkspace::new_for_network(
                                        &(*cloned_arc.read().unwrap()),
                                        1,
                                    )
                                    .into(),
                                );
                            });

                            // NN_ARC スロットに格納
                            *nn_slot = Some(cloned_arc);
                        }
                        // NN_ARC スロットからネットワーク参照を取り出して利用
                        nn_slot.as_ref().unwrap()
                    });

                    // let now = time::Instant::now();

                    // ワークスペースにアクセス (毎回)
                    let mut workspace = WORKSPACE.with(|ws_cell| {
                        let ws_slot = unsafe { &mut *ws_cell.get() };
                        ws_slot.as_mut().unwrap()
                    });

                    let read_lock_nn = local_nn_ref.read().unwrap();

                    read_lock_nn.forward_and_backward(
                        &sample.input,
                        &sample.expect,
                        &mut workspace,
                        0.0001,
                    );

                    // if workspace.error.is_nan() {
                    //     panic!("error is nan\niteration: {}", mini_batch_count);
                    // }

                    (
                        workspace.error,
                        workspace.next_weights.clone(),
                        workspace.next_biases.clone(),
                    )
                })
                .reduce(
                    || (0.0, nn.get_zero_weights(), nn.get_zero_biases()),
                    |mut current, next| {
                        for i in 0..current.1.len() {
                            current.1[i] += &next.1[i];
                            current.2[i] += &next.2[i];
                        }

                        current.0 += next.0;

                        current
                    },
                );
            // println!("mini batch calc time: {:?}", now.elapsed());

            let mut average_weights = Vec::new();
            let mut average_biases = Vec::new();
            for i in 0..next_weights.len() {
                average_weights.push((&next_weights[i] / mini_batch_iteration as f64).unwrap());
                average_biases.push((&next_biases[i] / mini_batch_iteration as f64).unwrap());
            }

            let mut write_lock_nn = arc_nn.write().unwrap();
            write_lock_nn.update_weights(&mut average_weights, &mut average_biases);
            epoch_error += error;

            print!(
                "\rmini batch count: {} / {}",
                mini_batch_count, training_mini_batch_value
            );
            io::stdout().flush().unwrap();
        }

        if (epoch + 1) % 1 == 0 {
            println!(
                "\nepoch {:6} error: {:13.10}",
                epoch + 1,
                epoch_error / training_value as f64
            );
        }

        println!("validation test:");

        let samples = generate_validation_data(&mnist.val_img, &mnist.val_lbl, validation_value as usize, validation_chunk_size);

        let validation_result = parallel_forward_only(samples, &arc_nn, &NN_ARC, &WORKSPACE);

        println!("collect rate: {}", validation_result.1 as f64 / validation_value as f64);
        println!("              ({} / {})", validation_result.1, validation_value);
        println!("error: {}\n", validation_result.0 / validation_value as f64);
    }

    println!("after leaning test:");

    let test_samples = generate_validation_data(&mnist.tst_img, &mnist.tst_lbl, test_value as usize, test_chunk_size);

    let test_result = parallel_forward_only(test_samples, &arc_nn, &NN_ARC, &WORKSPACE);

    println!("collect rate: {}", test_result.1 as f64 / validation_value as f64);
    println!("              ({} / {})", test_result.1, test_value);
    println!("error: {}\n", test_result.0 / test_value as f64);

    println!(
        "epochs process duration: {:?}sec.",
        epochs_now.elapsed().as_secs_f64()
    );
}

fn generate_validation_data(
    img_data: &Vec<f32>,
    lbl_data: &Vec<u8>,
    data_value: usize,
    chunk_size: usize,
) -> Vec<MiniBatchParChunk> {
    let img_data_f64: Vec<f64> = img_data.iter().map(|&x| x as f64).collect();
    let img_lbl_f64: Vec<f64> = lbl_data.iter().map(|&x| x as f64).collect();
    let image_dot_value = 28 * 28;

    let mut mini_batch_par_chunk: Vec<MiniBatchParChunk> = Vec::new();
    let mut batch_data: Vec<f64> = Vec::new();
    let mut expect_data: Vec<f64> = Vec::new();

    let mut current_value_in_chunk = 0;

    for index in 0..data_value {
        let single_data = &img_data_f64[index * image_dot_value..(index + 1) * image_dot_value];
        batch_data.extend_from_slice(single_data);

        let single_label = &img_lbl_f64[index * 10..(index + 1) * 10];
        expect_data.extend_from_slice(single_label);

        current_value_in_chunk += 1;
        if current_value_in_chunk == chunk_size {
            mini_batch_par_chunk.push(MiniBatchParChunk {
                input: Matrix::new_from_vec(chunk_size, image_dot_value, batch_data.clone())
                    .unwrap(),
                expect: Matrix::new_from_vec(chunk_size, 10, expect_data.clone()).unwrap(),
            });
            batch_data.clear();
            expect_data.clear();
            current_value_in_chunk = 0;
        }
    }

    mini_batch_par_chunk
}

fn parallel_forward_only(
    samples: Vec<MiniBatchParChunk>,
    arc_nn: &Arc<RwLock<FullyConnectedNetwork>>,
    NN_ARC: &'static std::thread::LocalKey<UnsafeCell<Option<Arc<RwLock<FullyConnectedNetwork>>>>>,
    WORKSPACE: &'static std::thread::LocalKey<UnsafeCell<Option<NetworkWorkspace>>>,
) -> (f64, usize) {
    let result = samples
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
                            NetworkWorkspace::new_for_network(&(*cloned_arc.read().unwrap()), 1)
                                .into(),
                        );
                    });

                    // NN_ARC スロットに格納
                    *nn_slot = Some(cloned_arc);
                }
                // NN_ARC スロットからネットワーク参照を取り出して利用
                nn_slot.as_ref().unwrap()
            });

            // let now = time::Instant::now();

            // ワークスペースにアクセス (毎回)
            let mut workspace = WORKSPACE.with(|ws_cell| {
                let ws_slot = unsafe { &mut *ws_cell.get() };
                ws_slot.as_mut().unwrap()
            });

            let read_lock_nn = local_nn_ref.read().unwrap();

            read_lock_nn.forward_only(&sample.input, &sample.expect, &mut workspace);

            let mut collect_value = 0;

            let outputs = workspace.layer_outputs.last().unwrap();

            for i in 0..outputs.rows {
                let row = outputs.get_row(i).unwrap();
                let (most_index, _) = row.iter().enumerate().fold(
                    (0, f64::MIN),
                    |(idx_a, val_a), (idx_b, &val_b)| {
                        if val_b > val_a {
                            (idx_b, val_b)
                        } else {
                            (idx_a, val_a)
                        }
                    },
                );

                if sample.expect[i][most_index] == 1.0 {
                    collect_value += 1;
                }
            }

            (workspace.error, collect_value)
        })
        .reduce(
            || (0.0, 0usize),
            |current, next| (current.0 + next.0, current.1 + next.1),
        );

    result
}
