use super::scalar::Scalar;

#[derive(PartialEq, Debug, Clone)]
pub enum Data {
    Tile(Tile),
    Ref(usize),
    Scalar(Scalar),
}

#[derive(PartialEq, Debug, Clone)]
pub enum ChannelMessage {
    Data(Data, usize), // Latter is the stop token. Its zero when no dimension ends.
    Instruction((u16, u8)), // (instruction idx, port)
    End,
    StopSim, // Just a signal to stop the simulation. Not necessarily in actual hardware.
}

impl ChannelMessage {
    pub fn to_tile(&self) -> Result<Tile, ()> {
        match self {
            ChannelMessage::Data(data, _) => match data {
                Data::Tile(tile) => Ok(tile.clone()),
                _ => Err(())
            },
            _ => Err(())
        }
    }

    pub fn to_st(&self) -> Result<usize, ()> {
        match self {
            ChannelMessage::Data(_, st) => Ok(*st),
            _ => Err(())
        }
    }

    pub fn new_tile(tile_data: Vec<Vec<Scalar>>, st_rank: usize) -> Self {
        ChannelMessage::Data(Data::Tile(Tile::new(tile_data)), st_rank)
    }

    pub fn with_updated_rank(&self, new_rank: usize) -> Self {
        match self {
            ChannelMessage::Data(data, _) => ChannelMessage::Data(data.clone(), new_rank),
            ChannelMessage::End => ChannelMessage::End,
            ChannelMessage::Instruction(_) => self.clone(),
            ChannelMessage::StopSim => ChannelMessage::StopSim,
        }
    }

    pub fn from_scalar(scalar: Scalar, st_rank: usize) -> Self {
        ChannelMessage::Data(Data::from_scalar(scalar), st_rank)
    }
}

impl Default for ChannelMessage {
    fn default() -> Self {
        ChannelMessage::End
    }
}

impl Data {
    pub fn from_scalar(scalar: Scalar) -> Self {
        Data::Tile(Tile::from_scalar(scalar))
    }

    pub fn from_scalar_with_st(scalar: Scalar, _token_value: usize) -> Self {
        Data::Tile(Tile::from_scalar(scalar))
    }

    pub fn from_scalar_vec(scalar: Vec<Vec<Scalar>>) -> Self {
        Data::Tile(Tile::new(scalar))
    }

    pub fn from_scalar_vec_with_st(scalar: Vec<Vec<Scalar>>, _token_value: usize) -> Self {
        Data::Tile(Tile::new(scalar))
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Tile {
    pub data: Vec<Vec<Scalar>>
}

impl Tile {
    pub fn new(data: Vec<Vec<Scalar>>) -> Self {
        Self { data }
    }

    pub fn from_scalar(scalar: Scalar) -> Self {
        Self {
            data: vec![vec![scalar]]
        }
    }

    pub fn transpose(&self) -> Tile {
        let transposed_data: Vec<Vec<Scalar>> = (0..self.data[0].len())
            .map(|i| self.data.iter().map(|row| row[i].clone()).collect())
            .collect();
        Tile::new(transposed_data)
    }

    pub fn try_add(&self, other: &Tile) -> Result<Tile, ()> {
        if self.data.len() != other.data.len() || self.data[0].len() != other.data[0].len() {
            return Err(());
        }

        let mut result = Tile::new(vec![vec![Scalar::default(); self.data[0].len()]; self.data.len()]);
        for i in 0..self.data.len() {
            for j in 0..self.data[0].len() {
                result.data[i][j] = self.data[i][j].try_add(&other.data[i][j])?;
            }
        }
        Ok(result)
    }

    pub fn try_mul(&self, other: &Tile) -> Result<Tile, ()> {
        if self.data.len() != other.data.len() || self.data[0].len() != other.data[0].len() {
            return Err(());
        }
 
        let mut result = Tile::new(vec![vec![Scalar::default(); self.data[0].len()]; self.data.len()]);
        for i in 0..self.data.len() {
            for j in 0..self.data[0].len() {
                result.data[i][j] = self.data[i][j].try_mul(&other.data[i][j])?;
            }
        }
        Ok(result)
    }

    pub fn try_matmul(&self, other: &Tile) -> Result<Tile, ()> {
        if self.data[0].len() != other.data.len() {
            println!("Tile sizes are not compatible: {:?} and {:?}", self.data, other.data);
            return Err(());
        }

        let mut result = Tile::new(vec![vec![Scalar::default(); other.data[0].len()]; self.data.len()]);
        for i in 0..self.data.len() {
            for j in 0..other.data[0].len() {
                for k in 0..self.data[0].len() {
                    result.data[i][j] = result.data[i][j].try_add(&self.data[i][k].try_mul(&other.data[k][j])?)?;
                }
            }
        }
        Ok(result)
    }

    pub fn try_matmul_t(&self, other: &Tile) -> Result<Tile, ()> {
        if self.data.len() != other.data.len() {
            println!("Tile sizes are not compatible: {:?} and {:?}", self.data, other.data);
            return Err(());
        }

        let mut result: Vec<Vec<Option<Scalar>>> = vec![vec![None; other.data.len()]; self.data[0].len()];
        for i in 0..self.data[0].len() {
            for j in 0..other.data[0].len() {
                for k in 0..self.data.len() {
                    match result[i][j] {
                        Some(scalar) => {
                            result[i][j] = Some(scalar.try_add(&self.data[k][i].try_mul(&other.data[j][k])?)?);
                        }
                        None => {
                            result[i][j] = Some(self.data[k][i].try_mul(&other.data[j][k])?);
                        }
                    }
                }
            }
        }
        Ok(Tile::new(result.iter().map(|row| row.iter().map(|x| x.unwrap()).collect()).collect()))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Storage {
    pub data: Data,
    pub stop_token: usize,
}

impl Default for Storage {
    fn default() -> Self {
        Storage {
            data: Data::Tile(Tile::new(vec![])), // or Tile::default()
            stop_token: 0,
        }
    }
}