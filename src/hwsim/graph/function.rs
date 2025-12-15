use std::collections::HashMap;

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub enum Value {
    UInt(u32),
    Int(i32),
    Float(f32)
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ValueType {
    UInt,
    Int,
    Float,
    View, 
    ViewView
}

impl From<usize> for Value {
    fn from(v: usize) -> Self {
        Value::UInt(v as u32)
    }
}

impl TryInto<usize> for Value {
    type Error = &'static str;

    fn try_into(self) -> Result<usize, Self::Error> {
        match self {
            Value::UInt(v) => Ok(v as usize),
            Value::Int(v) => Ok(v as usize),
            _ => Err("Value not convertible")
        }
    }
}

impl Default for Value {
    fn default() -> Self {
        Value::UInt(0)
    }
}

#[derive(Debug, Clone)]
pub struct TypelessFunction {
    pub operations: Vec<(Operation, Vec<usize>)>,
    pub n_inputs: usize,
    pub n_outputs: usize
}

#[derive(Debug, Clone, PartialEq)]
pub struct RepeatStatic {
    pub op: BaseOp,
    pub expand_rank: usize, 
    pub expand_size: usize
}

#[derive(Clone, Debug, PartialEq)]
pub enum BaseOp {
    Variable(usize, ValueType),
    Constant(Value),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Operation {
    Add(BaseOp, BaseOp),
    Sub(BaseOp, BaseOp),
    Mul(BaseOp, BaseOp),
    Div(BaseOp, BaseOp),
    Max(BaseOp, BaseOp),
    Min(BaseOp, BaseOp),
    Lt(BaseOp, BaseOp),
    Leq(BaseOp, BaseOp),
    Negate(BaseOp),
    ScaleAdd(BaseOp, BaseOp, BaseOp), // a * x + b, a being the accumulator
    Exp(BaseOp),
    Square(BaseOp),
    Pow(BaseOp, BaseOp),
    Sin(BaseOp),
    Cos(BaseOp),
    Tanh(BaseOp),
    Silu(BaseOp),
    Identity(BaseOp),
    Modulo(BaseOp, BaseOp),
    Matmul(BaseOp, BaseOp),
    MatmulT(BaseOp, BaseOp),

    // todo: We should maybe remove those later.
    RepeatStatic(RepeatStatic),
    ExpandStatic(BaseOp, HashMap<usize, usize>),
    Permute {
        op: BaseOp,
        permute: Vec<usize>,
        transpose: bool,
    },

    RetileRow(BaseOp),
    RetileCol(BaseOp),
}