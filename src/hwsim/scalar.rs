#[derive(PartialEq, Debug, Clone, Copy)]
pub enum Scalar {
    U32(u32),
    I32(i32),
    FP32(f32)
}

impl Default for Scalar {
    fn default() -> Self {
        Scalar::I32(0)
    }
}

impl Scalar {
    pub fn width(&self) -> usize {
        32
    }

    pub fn try_add(&self, other: &Scalar) -> Result<Scalar, ()> {
        match (self, other) {
            (Scalar::I32(a), Scalar::I32(b)) => Ok(Scalar::I32(a + b)),
            (Scalar::FP32(a), Scalar::FP32(b)) => Ok(Scalar::FP32(a + b)),
            (Scalar::U32(a), Scalar::U32(b)) => Ok(Scalar::U32(a + b)),
            _ => Err(()),
        }
    }
    
    pub fn try_mul(&self, other: &Scalar) -> Result<Scalar, ()> {
        match (self, other) {
            (Scalar::I32(a), Scalar::I32(b)) => Ok(Scalar::I32(a * b)),
            (Scalar::FP32(a), Scalar::FP32(b)) => Ok(Scalar::FP32(a * b)),
            (Scalar::U32(a), Scalar::U32(b)) => Ok(Scalar::U32(a * b)),
            _ => Err(()),
        }
    }
}