use std::ops;
pub use crate::auto_apply_ops::auto_apply_arithmetic::AutoApplyArithmetic;
pub use crate::auto_apply_ops::auto_apply_arithmetic::AutoApplyArithmeticAdd;
pub use crate::auto_apply_ops::auto_apply_arithmetic::AutoApplyArithmeticSub;
pub use crate::auto_apply_ops::auto_apply_arithmetic::AutoApplyArithmeticMul;
pub use crate::auto_apply_ops::auto_apply_arithmetic::AutoApplyArithmeticDiv;
pub use crate::auto_apply_ops::auto_apply_arithmetic::AutoApplyArithmeticRem;

mod auto_apply_arithmetic;

pub trait AutoApplyOps<T> {}

impl<T: Sized + Clone + ops::AddAssign + ops::SubAssign + ops::MulAssign + ops::DivAssign + ops::RemAssign> AutoApplyOps<T> for dyn AutoApplyArithmetic<T> {}