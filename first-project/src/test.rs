pub fn anderson_darling_test_normal(mut dataset: Vec<f64>, mu: f64, sigma: f64) {
	// データを昇順に並べる
	// 浮動小数点数では partial_cmp を使う
	dataset.sort_by(|a, b| a.partial_cmp(b).unwrap());

	// データを標準化
	// まず標本平均を求める
	let dataset_mean = dataset.iter().sum::<f64>() / dataset.len() as f64;
	// それぞれのデータを標準正規分布 N(0, 1) に当てはめる
	let dataset_std: Vec<f64> = dataset.iter().map(|x| (x - dataset_mean) / sigma.sqrt()).collect();
}