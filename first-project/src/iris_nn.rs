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
    let mut means = IrisValues::new();
    let mut averages = IrisValues::new();
    let mut modes = IrisValues::new();

    let sepal_lengths: Vec<f32> = irisdata::IRIS_DATA
        .iter()
        .map(|d| d.sepal_length)
        .collect();
    let sepal_width: Vec<f32> = irisdata::IRIS_DATA
        .iter()
        .map(|d| d.sepal_width)
        .collect();
    let petal_length: Vec<f32> = irisdata::IRIS_DATA
        .iter()
        .map(|d| d.petal_length)
        .collect();
    let petal_width: Vec<f32> = irisdata::IRIS_DATA
        .iter()
        .map(|d| d.petal_width)
        .collect();

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

    for datum in irisdata::IRIS_DATA {

        //         maximums.sepal_length = maximums.sepal_length.max(datum.sepal_length as f64);
        //         maximums.sepal_width = maximums.sepal_width.max(datum.sepal_width as f64);
        //         maximums.petal_length = maximums.petal_length.max(datum.petal_length as f64);
        //         maximums.petal_width = maximums.petal_width.max(datum.petal_width as f64);

        //         minimums.sepal_length = minimums.sepal_length.min(datum.sepal_length as f64);
        //         minimums.sepal_width = minimums.sepal_width.min(datum.sepal_width as f64);
        //         minimums.petal_length = minimums.petal_length.min(datum.petal_length as f64);
        //         minimums.petal_width = minimums.petal_width.min(datum.petal_width as f64);

        //         averages.sepal_length
        // averages.sepal_width
        // averages.petal_length
        // averages.petal_width
    }

    // 範囲、分散 (標準偏差)、歪度、尖度
}
