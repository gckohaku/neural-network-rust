use std::{arch::x86_64::_mm_sm4key4_epi32, collections::HashMap, time};

use crate::{
    iris_normalization::iris_normalization, matrix::Matrix, neural_network::*,
    output_activation_type::OutputActivationType, rand::Rand,
};

pub fn iris_nn_process() {
    let normalization_data = iris_normalization(&irisdata::IRIS_DATA);
    // println!("{:?}", &normalization_data);

    // ミニバッチ内のサンプルサイズを指定
    let mini_batch_sample_size = 10;

    // iris dataset 用ニューラルネットワーク
    let mut nn = NeuralNetwork::new(vec![4, 12, 24, 36, 24, 12, 3], mini_batch_sample_size);
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
        let mut epoch_error = 0.0;

        for indexes in shuffle_index.chunks(mini_batch_sample_size) {
            let mut batch = Vec::new();

            for index in indexes {
                batch.push(&normalization_data[*index]);
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
            nn.backward(&expects, 0.0007).unwrap();
        }

        if (epoch + 1) % 500 == 0 {
            println!(
                "epoch {:6} error: {:13.10}",
                epoch + 1,
                epoch_error / irisdata::IRIS_DATA.len() as f64
            );
            // print!("{:8.4}", nn.get_output_nodes());
        }
    }

    println!("epochs process duration: {:?}sec.", epochs_now.elapsed().as_secs_f64());

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
    let data_value = irisdata::IRIS_DATA.len();

    // 最小値、最大値、中央値、平均値
    let mut maximums = IrisValues::new();
    let mut minimums = IrisValues::new();
    let mut medians = IrisValues::new();
    let mut averages = IrisValues::new();

    // 範囲、分散 (標準偏差)、歪度、尖度
    let mut ranges = IrisValues::new();
    let mut variances = IrisValues::new();
    let mut skewnesses = IrisValues::new();
    let mut kurtosises = IrisValues::new();

    // 不偏分散
    let mut unbiased = IrisValues::new();

    let sepal_length: Vec<f32> = irisdata::IRIS_DATA.iter().map(|d| d.sepal_length).collect();
    let sepal_width: Vec<f32> = irisdata::IRIS_DATA.iter().map(|d| d.sepal_width).collect();
    let petal_length: Vec<f32> = irisdata::IRIS_DATA.iter().map(|d| d.petal_length).collect();
    let petal_width: Vec<f32> = irisdata::IRIS_DATA.iter().map(|d| d.petal_width).collect();

    maximums.sepal_length = sepal_length.iter().fold(0.0 / 0.0, |m, v| v.max(m));
    maximums.sepal_width = sepal_width.iter().fold(0.0 / 0.0, |m, v| v.max(m));
    maximums.petal_length = petal_length.iter().fold(0.0 / 0.0, |m, v| v.max(m));
    maximums.petal_width = petal_width.iter().fold(0.0 / 0.0, |m, v| v.max(m));

    minimums.sepal_length = sepal_length.iter().fold(0.0 / 0.0, |m, v| v.min(m));
    minimums.sepal_width = sepal_width.iter().fold(0.0 / 0.0, |m, v| v.min(m));
    minimums.petal_length = petal_length.iter().fold(0.0 / 0.0, |m, v| v.min(m));
    minimums.petal_width = petal_width.iter().fold(0.0 / 0.0, |m, v| v.min(m));

    (
        averages.sepal_length,
        variances.sepal_length,
        unbiased.sepal_length,
    ) = calc_average_and_variance(&sepal_length);
    (
        averages.sepal_width,
        variances.sepal_width,
        unbiased.sepal_width,
    ) = calc_average_and_variance(&sepal_width);
    (
        averages.petal_length,
        variances.petal_length,
        unbiased.petal_length,
    ) = calc_average_and_variance(&petal_length);
    (
        averages.petal_width,
        variances.petal_width,
        unbiased.petal_width,
    ) = calc_average_and_variance(&petal_width);

    let half_index = irisdata::IRIS_DATA.len() / 2;
    println!("half index: {}", half_index);

    let mut sorted_sepal_lengths = sepal_length.clone();
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

    ranges.sepal_length = maximums.sepal_length - minimums.sepal_length;
    ranges.sepal_width = maximums.sepal_width - minimums.sepal_width;
    ranges.petal_length = maximums.petal_length - minimums.petal_length;
    ranges.petal_width = maximums.petal_width - minimums.petal_width;

    skewnesses.sepal_length =
        calc_skewness(&sepal_length, averages.sepal_length, variances.sepal_length);
    skewnesses.sepal_width =
        calc_skewness(&sepal_width, averages.sepal_width, variances.sepal_width);
    skewnesses.petal_length =
        calc_skewness(&petal_length, averages.petal_length, variances.petal_length);
    skewnesses.petal_width =
        calc_skewness(&petal_width, averages.petal_width, variances.petal_width);

    // 結果の出力
    println!("sepal length:");
    println!("  maximum : {}", maximums.sepal_length);
    println!("  minimum : {}", minimums.sepal_length);
    println!("  average : {}", averages.sepal_length);
    println!("  median  : {}", medians.sepal_length);
    println!("  range   : {}", ranges.sepal_length);
    println!("  variance: {}", variances.sepal_length);
    println!("  standard: {}", variances.sepal_length.sqrt());
    println!("  skewness: {}", skewnesses.sepal_length);

    println!("sepal width:");
    println!("  maximum : {}", maximums.sepal_width);
    println!("  minimum : {}", minimums.sepal_width);
    println!("  average : {}", averages.sepal_width);
    println!("  median  : {}", medians.sepal_width);
    println!("  range   : {}", ranges.sepal_width);
    println!("  variance: {}", variances.sepal_width);
    println!("  standard: {}", variances.sepal_width.sqrt());
    println!("  skewness: {}", skewnesses.sepal_width);

    println!("petal length:");
    println!("  maximum : {}", maximums.petal_length);
    println!("  minimum : {}", minimums.petal_length);
    println!("  average : {}", averages.petal_length);
    println!("  median  : {}", medians.petal_length);
    println!("  range   : {}", ranges.petal_length);
    println!("  variance: {}", variances.petal_length);
    println!("  standard: {}", variances.petal_length.sqrt());
    println!("  skewness: {}", skewnesses.petal_length);

    println!("petal width:");
    println!("  maximum : {}", maximums.petal_width);
    println!("  minimum : {}", minimums.petal_width);
    println!("  average : {}", averages.petal_width);
    println!("  median  : {}", medians.petal_width);
    println!("  range   : {}", ranges.petal_width);
    println!("  variance: {}", variances.petal_width);
    println!("  standard: {}", variances.petal_width.sqrt());
    println!("  skewness: {}", skewnesses.petal_width);
}

fn kahan_sum_once_f32(data: f32, sum: &mut f32, c: &mut f32) {
    let y = data - *c;
    let t = *sum + y;
    *c = (t - *sum) - y;
    *sum = t;
}

// fn kahan_sum_f32(data: &Vec<f32>) -> f32 {
//     let mut sum = 0.0f32;
//     let mut c = 0.0f32;

//     for i in 0..data.len() {
//         kahan_sum_once_f32(data[i], &mut sum, &mut c);
//     }

//     sum
// }

pub fn calc_average_and_variance(data: &Vec<f32>) -> (f32, f32, f32) {
    let data_value = data.len();

    let mut average = 0.0f32;
    let mut variance = 0.0f32;
    let mut c_average = 0.0f32;
    let mut c_variance = 0.0f32;

    for i in 0..data_value {
        let delta_1 = data[i] - average;
        kahan_sum_once_f32(delta_1 / (i as f32 + 1.0), &mut average, &mut c_average);
        let delta_2 = data[i] - average;
        kahan_sum_once_f32(delta_1 * delta_2, &mut variance, &mut c_variance);
    }

    let n = data_value as f32;
    (average, variance / n, variance / (n - 1.0))
}

pub fn calc_skewness(data: &Vec<f32>, average: f32, variance: f32) -> f32 {
    let data_value = data.len();

    let mut skewness = 0.0f32;
    let mut c = 0.0f32;

    for i in 0..data_value {
        let delta_1 = data[i] - average;
        kahan_sum_once_f32((delta_1 / variance.sqrt()).powi(3), &mut skewness, &mut c);
    }

    let n = data_value as f32;
    skewness / n
}

pub fn calc_kurtosis(data: &Vec<f32>, average: f32, variance: f32) -> f32 {
    let data_value = data.len();

    let mut kurtosis = 0.0f32;
    let mut c = 0.0f32;

    for i in 0..data_value {
        let delta_1 = data[i] - average;
        kahan_sum_once_f32((delta_1 / variance.sqrt()).powi(4), &mut kurtosis, &mut c);
    }

    let n = data_value as f32;
    (((n * (n + 1.0)) / ((n - 1.0) * (n - 2.0) * (n - 3.0))) * kurtosis)
        - ((3.0 * (n - 1.0).powi(2)) / ((n - 2.0) * (n - 3.0)))
}

// https://data-viz-lab.com/stats#1-3median
// https://bellcurve.jp/statistics/course/17950.html?srsltid=AfmBOorSvnzQ2VbbF0zl1FEPcJPezuEky5RmhmDf5cK_I6S0ZMxvePNk
// https://bellcurve.jp/statistics/glossary/2113.html
// https://zenn.dev/utcarnivaldayo/articles/ffeed5ac2e62bb
