use std::fmt::Display;

use ron::ser::PrettyConfig;

use crate::{
    matrix::Matrix,
    neural_network_base::{Gradients, NetworkWorkspace, NeuralNetwork},
    output_activation_type::OutputActivationType,
    rand::Rand,
    ron_data::{LayerInfo, RonNNData},
};

#[derive(Debug, Clone)]
pub struct FullyConnectedNetwork {
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    activations: Vec<fn(&f64) -> f64>,
    differential_activations: Vec<fn(&f64) -> f64>,
    output_activation_type: OutputActivationType,
}

impl NeuralNetwork for FullyConnectedNetwork {
    fn new(nodes_values: Vec<usize>, sample_value: usize) -> Self {
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        let activations: Vec<fn(&f64) -> f64> = Vec::new();
        let differential_activations: Vec<fn(&f64) -> f64> = Vec::new();
        let output_activation_type = OutputActivationType::Default;

        let mut r = Rand::new();

        for i in 0..nodes_values.len() {
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
                // バイアス行列のサイズは 1 x 現在の層のノードの数
                biases.push(Matrix::new_and_fill(1, nodes_values[i], 0.0));
            }
        }

        Self {
            weights,
            biases,
            activations,
            differential_activations,
            output_activation_type,
        }
    }

    fn forward_and_backward(
        &self,
        inputs: &Matrix,
        expects: &Matrix,
        workspace: &mut NetworkWorkspace,
        eta: f64,
    ) {
        let sample_size = inputs.rows;
        let output_index = self.weights.len() - 1;

        // まず、順伝播を行う
        workspace.layer_outputs[0] = inputs.clone();

        for i in 0..self.weights.len() {
            workspace.layer_inputs[i] =
                (&workspace.layer_outputs[i] * &self.weights[i] + &self.biases[i]).unwrap();

            // 出力層かつ目的関数を別途指定している場合、設定に応じて処理が分かれる
            if i >= self.weights.len() - 1
                && self.output_activation_type != OutputActivationType::Default
            {
                // softmax 関数と交差エントロピー誤差を利用する場合
                if self.output_activation_type == OutputActivationType::SoftmaxAndCrossEntropy {
                    // オーバーフロー対策として、node に入れる値は max(ノードの入力値) で減算する
                    // Vec<f64> の最大値はこうすることで取得できるらしい
                    let max_input_value: f64 = workspace.layer_inputs[i]
                        .data
                        .iter()
                        .fold(0.0 / 0.0, |m, v: &f64| v.max(m));

                    if max_input_value.is_nan() {
                        println!("NAN IS APPEAR (CODE: CALC_MAX_INPUT_VALUE)");
                        println!(
                            "{},\n{},\n{:?}",
                            max_input_value, max_input_value, workspace.layer_inputs[i]
                        );
                        panic!("calc weights is NAN");
                    }

                    // 減算
                    let processed_input_vec: Vec<f64> = workspace.layer_inputs[i]
                        .data
                        .iter()
                        .map(|x| (x - max_input_value).exp())
                        .collect();

                    let mut exp_input =
                        Matrix::new_from_vec(expects.rows, expects.cols, processed_input_vec.clone())
                            .unwrap();

                    if exp_input[(0, 0)].is_nan() {
                        println!("NAN IS APPEAR (CODE: CALC_EXP_INPUT)");
                        println!(
                            "{},\n{:?},\n{:?},\n{:?}",
                            i,
                            &processed_input_vec,
                            workspace.layer_inputs[i],
                            &exp_input
                        );
                        panic!("calc weights is NAN");
                    }

                    // 総和行列を作成する　ブロードキャストできるようにしているので、各サンプルの要素は一つでいい
                    let mut node_sums: Vec<f64> = Vec::new();
                    for s in 0..sample_size {
                        node_sums.push(exp_input.sum_row_elements(s));
                    }
                    let mut sum_matrix =
                        Matrix::new_from_vec(sample_size, 1, node_sums.clone()).unwrap();

                    if sum_matrix[(0, 0)].is_nan() {
                        println!("NAN IS APPEAR (CODE: CALC_SUM_MATRIX)");
                        println!(
                            "{},\n{:?},\n{:?},\n{:?}",
                            i,
                            node_sums.clone(),
                            &sum_matrix,
                            &exp_input
                        );
                        panic!("calc weights is NAN");
                    }

                    let softmax_result = exp_input
                        .hadamard(&sum_matrix.hadamard_function(|x| 1.0 / (x)))
                        .unwrap();

                    workspace.layer_outputs[i + 1] = softmax_result.clone();

                    if workspace.layer_outputs[i + 1][(0, 0)].is_nan() {
                        println!("NAN IS APPEAR (CODE: CALC_SOFTMAX)");
                        println!(
                            "{},\n{:?},\n{:?},\n{:?},\n{:?}",
                            i,
                            &workspace.layer_outputs[i + 1],
                            &sum_matrix,
                            &exp_input,
                            &softmax_result
                        );
                        panic!("calc weights is NAN");
                    }
                }
            } else {
                workspace.layer_outputs[i + 1] =
                    workspace.layer_inputs[i].hadamard_function(self.activations[i]);
            }
        }

        // println!("\n###### debug zone #############\n");
        // println!("output node: {:?}", &workspace.layer_outputs[output_index + 1]);
        // println!("\n###### end of debug zone ######\n");

        // 誤差を求める
        let mut output_node = workspace.layer_outputs[output_index + 1].clone();

        // softmax 関数と交差エントロピー誤差を利用する場合
        if self.output_activation_type == OutputActivationType::SoftmaxAndCrossEntropy {
            let ln_output = output_node.hadamard_function(|x| (x + 1e-10).ln());
            workspace.error = -expects.hadamard(&ln_output).unwrap().sum_all_elements();
        }

        // 次に逆伝播を行い、勾配を求める
        // 出力層のデルタ
        workspace.layer_deltas[output_index] =
            (&workspace.layer_outputs[output_index + 1] - expects).unwrap();

        if workspace.layer_deltas[output_index][(0, 0)].is_nan() {
            println!("NAN IS APPEAR (CODE: CALC_OUTPUT_DELTA)");
            println!(
                "{},\n{:?},\n{:?}",
                output_index,
                &workspace.layer_outputs[output_index + 1],
                expects
            );
            panic!("calc weights is NAN");
        }

        // 隠れ層のデルタ
        for i in (0..output_index).rev() {
            let delta = &workspace.layer_deltas[i + 1];
            let w = self.weights[i + 1].transpose();
            let u = &mut workspace.layer_inputs[i];
            let da = self.differential_activations[i];
            let da_u = u.hadamard_function(da);

            workspace.layer_deltas[i] = (delta * &w).unwrap().hadamard(&da_u).unwrap();
        }

        // 求めたデルタを用いて勾配を計算する
        for i in (0..=output_index).rev() {
            workspace.next_weights[i] = (&self.weights[i]
                - &(eta
                    * (&workspace.layer_outputs[i].transpose() * &workspace.layer_deltas[i])
                        .unwrap()))
                .unwrap();
            workspace.next_biases[i] =
                (&self.biases[i] - &(eta * workspace.layer_deltas[i].mean_cols())).unwrap();

            if workspace.next_weights[i][(0, 0)].is_nan() {
                println!("NAN IS APPEAR (CODE: CALC_NEXT_WEIGHTS)");
                println!(
                    "{},\n{:?},\n{:?}",
                    i, &workspace.layer_outputs[i], &workspace.layer_deltas[i]
                );
                panic!("calc weights is NAN");
            }

            // workspace.local_gradients.weights[i] =
            //     (&workspace.layer_outputs[i].transpose() * &workspace.layer_deltas[i]).unwrap();
            // workspace.local_gradients.biases[i] = workspace.layer_deltas[i].mean_cols();
        }
    }

    fn update_weights(&mut self, next_weights: &mut Vec<Matrix>, next_biases: &mut Vec<Matrix>) {
        // for index in 0..gradients.weights.len() {
        //     self.weights[index] -= eta * &gradients.weights[index];
        //     self.biases[index] -= eta * &gradients.biases[index];
        // }

        std::mem::swap(&mut self.weights, next_weights);
        std::mem::swap(&mut self.biases, next_biases);
    }

    fn set_activations(&mut self, activations: &mut Vec<fn(&f64) -> f64>) {
        self.activations.clear();
        self.activations.append(activations);
    }

    fn set_differential_activation(&mut self, differentials: &mut Vec<fn(&f64) -> f64>) {
        self.differential_activations.clear();
        self.differential_activations.append(differentials);
    }

    fn set_output_activation_type(&mut self, activation_type: OutputActivationType) {
        self.output_activation_type = activation_type;
    }

    fn get_layer_value(&self) -> usize {
        self.weights.len()
    }

    fn get_input_node_value(&self) -> usize {
        self.weights.first().unwrap().rows
    }

    fn get_output_node_value(&self) -> usize {
        self.weights.last().unwrap().cols
    }

    fn get_weight_matrix(&self, index: usize) -> &Matrix {
        &self.weights[index]
    }

    fn export_ron(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut node_values = Vec::<usize>::new();
        let mut layer_infos = Vec::<LayerInfo>::new();

        node_values.push(self.weights[0].rows);

        for i in 0..self.weights.len() {
            let info = LayerInfo {
                weights: self.weights[i].data.clone(),
                biases: self.biases[i].data.clone(),
            };

            layer_infos.push(info);
            node_values.push(self.weights[i].cols);
        }

        let nn_ron_data = RonNNData {
            layer_value: self.weights.len(),
            node_values,
            layers: layer_infos,
        };

        // println!("{}", ron::ser::to_string_pretty(&nn_ron_data, PrettyConfig::new()).unwrap());

        std::fs::write(
            "models/data.ron",
            ron::ser::to_string_pretty(&nn_ron_data, PrettyConfig::new()).unwrap(),
        )?;

        Ok(())
    }
}

impl FullyConnectedNetwork {}
