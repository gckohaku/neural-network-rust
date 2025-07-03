use crate::matrix::Matrix;
use std::ops;

// https://qiita.com/k5n/items/758111b12740600cc58f

#[macro_export]
macro_rules! apply_arithmetics_base {
    ($t: ty, $u: ty, $op_trait: item, $op_func: ident, $op: tt) => {
        impl $op_trait<$u> for $t
		{
			type Output = Result<$t, String>;

			fn $op_func(self, rhs: $u) -> Self::Output {
				if self.rows == rhs.rows && self.cols == rhs.cols {
					let mut result = self.clone();
					result += rhs;
					Ok(result)
				} else {
					Err("Matrices must have the same size for addition".to_string())
				}
			}
		}

		impl $op_trait<&$u> for $t
    };
}

apply_arithmetics_base!(Matrix, Matrix, ops::Add, add, +);