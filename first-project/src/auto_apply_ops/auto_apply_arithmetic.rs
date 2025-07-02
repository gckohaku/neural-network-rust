use std::{clone, ops, result};

pub trait AutoApplyArithmeticAdd<T> {
    fn add(self, rhs: T) -> Result<Self, String>
    where
        Self: Sized + Clone + ops::AddAssign<T>,
    {
        let mut result = self.clone();
        result += rhs;
        Ok(result)
    }
}

// https://doc.rust-jp.rs/rust-by-example-ja/macros/dry.html

pub trait AutoApplyArithmeticSub<T> {
    fn sub(self, rhs: T) -> Result<Self, String>
    where
        Self: Sized + Clone + ops::SubAssign<T>,
    {
        let mut result = self.clone();
        result -= rhs;
        Ok(result)
    }
}

pub trait AutoApplyArithmeticMul<T> {
    fn mul(self, rhs: T) -> Result<Self, String>
    where
        Self: Sized + Clone + ops::MulAssign<T>,
    {
        let mut result = self.clone();
        result *= rhs;
        Ok(result)
    }
}

pub trait AutoApplyArithmeticDiv<T> {
    fn div(self, rhs: T) -> Result<Self, String>
    where
        Self: Sized + Clone + ops::DivAssign<T>,
    {
        let mut result = self.clone();
        result /= rhs;
        Ok(result)
    }
}

pub trait AutoApplyArithmeticRem<T> {
    fn rem(self, rhs: T) -> Result<Self, String>
    where
        Self: Sized + Clone + ops::RemAssign<T>,
    {
        let mut result = self.clone();
        result %= rhs;
        Ok(result)
    }
}

pub trait AutoApplyArithmetic<T> {}

impl<T: Sized + Clone + ops::AddAssign> AutoApplyArithmeticAdd<T> for dyn AutoApplyArithmetic<T> {}
impl<T: Sized + Clone + ops::SubAssign> AutoApplyArithmeticSub<T> for dyn AutoApplyArithmetic<T> {}
impl<T: Sized + Clone + ops::MulAssign> AutoApplyArithmeticMul<T> for dyn AutoApplyArithmetic<T> {}
impl<T: Sized + Clone + ops::DivAssign> AutoApplyArithmeticDiv<T> for dyn AutoApplyArithmetic<T> {}
impl<T: Sized + Clone + ops::RemAssign> AutoApplyArithmeticRem<T> for dyn AutoApplyArithmetic<T> {}
