use std::{fmt::Display, ops::Add, process::Output};

use crate::{matrix::Matrix, output_activation_type::OutputActivationType, rand::Rand};

#[derive(Debug, Clone)]
pub struct NeuralNetwork {
    nodes: Vec<Matrix>,
    nodes_after_activation: Vec<Matrix>,
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    activations: Vec<fn(&f64) -> f64>,
    differential_activations: Vec<fn(&f64) -> f64>,
    error: f64,
    deltas: Vec<Matrix>,
    output_activation_type: OutputActivationType,
}

pub fn relu(x: &f64) -> f64 {
    x.max(0.0)
}

pub fn differential_relu(x: &f64) -> f64 {
    if *x > 0.0 { 1.0 } else { 0.0 }
}

pub fn leaky_relu(x: &f64) -> f64 {
    if *x > 0.0 { *x } else { 0.01 * *x }
}

pub fn differential_leaky_relu(x: &f64) -> f64 {
    if *x > 0.0 { 1.0 } else { 0.01 }
}

pub fn softmax(z: &mut Matrix) {
    let nodes_exp_sum: f64 = z.data.iter().map(|x| x.exp()).sum();
    for i in 0..z.rows {
        for j in 0..z.cols {
            let before_normalize = z.get(i, j).unwrap();
            z.set(i, j, before_normalize.exp() / nodes_exp_sum).unwrap();
        }
    }
}

pub fn cross_entropy() {}

impl NeuralNetwork {
    pub fn new(nodes_values: Vec<usize>, sample_value: usize) -> Self {
        let mut nodes = Vec::new();
        let mut nodes_after_activation = Vec::new();
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        let activations: Vec<fn(&f64) -> f64> = Vec::new();
        let differential_activations: Vec<fn(&f64) -> f64> = Vec::new();
        let error = 0.0;
        let mut deltas = Vec::new();
        let output_activation_type = OutputActivationType::Default;

        let mut r = Rand::new();

        for i in 0..nodes_values.len() {
            // ノード行列のサイズは サンプル数 x 現在の層のノードの数
            nodes.push(Matrix::new_and_fill(sample_value, nodes_values[i], 0.0));
            // 活性化関数適用後の行列のサイズは現在の層のノードのサイズと同じ
            nodes_after_activation.push(Matrix::new_and_fill(sample_value, nodes_values[i], 0.0));

            if i > 0 {
                // 重み行列のサイズは 直前の層のノードの数 x 現在の層のノードの数
                let mut layer_weights = Matrix::new(nodes_values[i - 1], nodes_values[i]);
                // 重み行列を He 初期化する
                for row in 0..layer_weights.rows {
                    for col in 0..layer_weights.cols {
                        layer_weights
                            .set(
                                row,
                                col,
                                r.normal(0.0, (2.0 / nodes_values[i - 1] as f64).sqrt()),
                            )
                            .unwrap();
                    }
                }
                weights.push(layer_weights);
                // バイアス行列のサイズは現在の層のノードのサイズと同じ
                biases.push(Matrix::new_and_fill(sample_value, nodes_values[i], 0.0));
                // デルタ行列のサイズも現在の層のノードのサイズと同じ
                deltas.push(Matrix::new_and_fill(sample_value, nodes_values[i], 0.0));
            }
        }

        Self {
            nodes,
            nodes_after_activation,
            weights,
            biases,
            activations,
            differential_activations,
            error,
            deltas,
            output_activation_type,
        }
    }

    pub fn set_activations(&mut self, activations: &mut Vec<fn(&f64) -> f64>) {
        self.activations.clear();
        self.activations.append(activations);
    }

    pub fn set_differential_activation(&mut self, differentials: &mut Vec<fn(&f64) -> f64>) {
        self.differential_activations.clear();
        self.differential_activations.append(differentials);
    }

    pub fn set_output_activation_type(&mut self, activation_type: OutputActivationType) {
        self.output_activation_type = activation_type;
    }

    pub fn view_status(&self) {
        println!("Neural Network Status:");
        println!("Nodes: {:?}", self.nodes);
        println!("After Activation: {:?}", self.nodes_after_activation);
        println!("Weights: {:?}", self.weights);
        println!("Biases: {:?}", self.biases);
        println!("Deltas: {:?}", self.deltas);
    }

    // NN の順伝搬。
    pub fn forward(&mut self, inputs: &Matrix, expects: &Matrix) -> Result<(), String> {
        if inputs.rows != self.nodes[0].rows || inputs.cols != self.nodes[0].cols {
            return Err(
                "Input dimensions do not match the first layer of the network.".to_string(),
            );
        }
        let output_index = self.nodes.len() - 1;
        if expects.rows != self.nodes[output_index].rows
            || inputs.cols != self.nodes[output_index].cols
        {
            return Err(
                "Expect dimensions do not match the output layer of the network.".to_string(),
            );
        }

        let sample_size = self.nodes[0].rows;
        self.nodes[0] = inputs.clone();
        self.nodes_after_activation[0] = inputs.clone();

        for i in 1..self.nodes.len() {
            self.nodes[i] = (&self.nodes_after_activation[i - 1] * &self.weights[i - 1]
                + &self.biases[i - 1])
                .unwrap();

            // 出力層かつ目的関数を別途指定している場合、設定に応じて処理が分かれる
            if i >= self.nodes.len() && self.output_activation_type != OutputActivationType::Default
            {
                // softmax 関数と交差エントロピー誤差を利用する場合
                if self.output_activation_type == OutputActivationType::SoftmaxAndCrossEntropy {
                    // オーバーフロー対策として、node に入れる値は max(weighted_sum) で減算する
                    // Vec<f64> の最大値はこうすることで取得できるらしい
                    let max_output_value: f64 =
                        self.nodes[i].data.iter().fold(0.0 / 0.0, |m, v| v.max(m));
                    // 減算
                    let processed_output_vec: Vec<f64> = self.nodes[i]
                        .data
                        .iter()
                        .map(|x| (x - max_output_value).exp())
                        .collect();

                    let mut exp_output =
                        Matrix::new_from_vec(expects.rows, expects.cols, processed_output_vec)
                            .unwrap();

                    // 総和行列を作成する　ブロードキャストできるようにしているので、各サンプルの要素は一つでいい
                    let mut node_sums: Vec<f64> = Vec::new();
                    for s in 0..sample_size {
                        node_sums.push(self.nodes[i].sum_row_elements(s));
                    }
                    let mut sum_matrix = Matrix::new_from_vec(sample_size, 1, node_sums).unwrap();

                    let sigmoid_result = exp_output
                        .hadamard(&sum_matrix.hadamard_function(|x| 1.0 / (x + 1e-10)))
                        .unwrap();

                    self.nodes_after_activation.push(sigmoid_result);
                }
            } else {
                self.nodes_after_activation[i] =
                    self.nodes[i].hadamard_function(self.activations[i - 1]);
            }
        }

        // 誤差を求める　求めた後にサンプル数で除算
        let mut mini_batch_error = 0.0;

        let mut output_node = self.nodes[output_index].clone();

        // softmax 関数と交差エントロピー誤差を利用する場合
        if self.output_activation_type == OutputActivationType::SoftmaxAndCrossEntropy {
            let ln_output = output_node.hadamard_function(|x| x.ln());

            mini_batch_error = expects.hadamard(&ln_output).unwrap().sum_all_elements();
        }

        self.error = mini_batch_error / self.nodes[output_index].rows as f64;

        Ok(())
    }

    pub fn backward(&mut self, expects: &Matrix, eta: f64) -> Result<(), String> {
        let node_output_index = self.nodes.len() - 1;
        let other_output_index = node_output_index - 1;
        let sample_size = self.nodes[0].rows;

        // 出力層のデルタ
        self.deltas[other_output_index] = (&self.nodes[node_output_index] - expects).unwrap();

        // 隠れ層のデルタ
        for i in (0..other_output_index - 1).rev() {
            let delta = &self.deltas[i + 1];
            let w = self.weights[i + 1].transpose();
            let u = &mut self.nodes[i + 1];
            let da = self.differential_activations[i];
            let da_u = u.hadamard_function(da);

            self.deltas[i] = (delta * &w).unwrap().hadamard(&da_u).unwrap();
        }

        // 求めたデルタを用いて勾配を計算する
        for i in (0..other_output_index).rev() {
            self.weights[i] = (self.weights[i].clone() - eta * (self.nodes_after_activation[i].transpose()) * self.deltas[i].clone()).unwrap();
            self.biases[i] = (self.biases[i].clone() - (eta * self.deltas[i].clone())).unwrap();
        }

        // サンプルサイズを変更できるようにする (重み)

        Ok(())
    }
}

impl Display for NeuralNetwork {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let width = f.width().unwrap_or(10);
        let precision = f.precision().unwrap_or(0);

        for i in 0..self.nodes.len() {
            if i > 0 {
                writeln!(f, "Layer {}:", i)?;

                writeln!(f, "\nWeights:")?;
                write!(f, "{:width$.precision$} ", self.weights[i - 1])?;

                writeln!(f, "\nBiases:")?;
                write!(f, "{:width$.precision$} ", self.biases[i - 1])?;

                writeln!(f, "\nNodes:")?;
            } else {
                writeln!(f, "Input Layer:")?;
                writeln!(f, "Inputs:")?;
            }

            write!(
                f,
                "{:width$.precision$} ",
                self.nodes[i],
                width = width,
                precision = precision
            )?;

            writeln!(f)?;
        }
        Ok(())
    }
}
