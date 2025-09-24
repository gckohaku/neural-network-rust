#[derive(Debug, Clone, PartialEq)]
pub enum OutputActivationType {
	Default = 0, // set_activations などによって設定した関数を使う場合
	SoftmaxAndCrossEntropy = 1,  // 多クラス分類問題
}