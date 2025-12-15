use ndarray::ArrayViewD;

use crate::hwsim::{scalar::Scalar, types::{ChannelMessage, Data, Tile}};

// This reshapes a normal input into our tiles.
pub fn reshape_tensor<T: Clone>(
    array: ArrayViewD<T>,
    tile_size: usize,
) -> ArrayViewD<T> {
    let shape = array.shape();
    let n_dim = shape.len();
    let (tx, ty) = (shape[n_dim - 1], shape[n_dim - 2]);

    let tile_y = tile_size;
    let tile_x = tile_size;

    assert_eq!(ty % tile_y, 0, "Y dim must be divisible by tile_y, but is {} % {}", ty, tile_y);
    assert_eq!(tx % tile_x, 0, "X dim must be divisible by tile_x, but is {} % {}", tx, tile_x);

    let tiles_y = ty / tile_y;
    let tiles_x = tx / tile_x;

    // Construct new shape as [tiles_y, tile_y, tiles_x, tile_x, ...tail_dims]
    let mut new_shape = shape[..n_dim-2].to_vec();
    new_shape.extend_from_slice(&[tiles_y, tile_y, tiles_x, tile_x]);

    // Safe because we're only reshaping the view if memory layout allows
    let mut array = array.into_shape_with_order(new_shape).expect("Invalid reshape");
    let n_dim = array.ndim();
    array.swap_axes(n_dim - 3, n_dim - 2);
    array
}

pub fn tiled_tensor_to_step(array: ArrayViewD<Scalar>, idx: String) -> Box<dyn Iterator<Item = ChannelMessage>> {
    // 1) figure out how many real values there are
    let total = array.len();
    // 2) our two mutable counters
    let mut seen = 0;
    let mut next_pct = 1;

    // 3) exactly the same recursive logic you had, pulled into a nested fn
    fn inner(a: ArrayViewD<Scalar>) -> Box<dyn Iterator<Item = ChannelMessage>> {
        assert!(a.ndim() >= 3, "tensor_to_step only supports arrays with more than 3 dimensions, since we're working on tiles.");

        let dim = a.ndim();
        if dim == 3 {
            // Extract 2D tiles from the 3D array
            let shape = a.shape();
            let num_tiles = shape[0];
            let tile_rows = shape[1];
            let tile_cols = shape[2];

            return Box::new(
                (0..num_tiles)
                    .enumerate()
                    .map(|(tile_idx, _)| {
                        // Extract a 2D slice as a tile
                        let tile_data: Vec<Vec<Scalar>> = (0..tile_rows)
                            .map(|i| {
                                (0..tile_cols)
                                    .map(|j| a[[tile_idx, i, j]])
                                    .collect()
                            })
                            .collect();
                        
                        let tile = Tile::new(tile_data);
                        
                        // If this is the last tile, use a special marker
                        if tile_idx == num_tiles - 1 {
                            ChannelMessage::Data(Data::Tile(tile), 1)
                        } else {
                            ChannelMessage::Data(Data::Tile(tile), 0)
                        }
                    })
                    .collect::<Vec<_>>()
                    .into_iter()
            );
        }
        let mut it: Box<dyn Iterator<Item = ChannelMessage>> = Box::new(std::iter::empty());
        for sub in a.axis_iter(ndarray::Axis(0)) {
            it = Box::new(it.chain(inner(sub)));
        }

        let dim_end: usize = dim.try_into().unwrap();
        let mut peekable = it.peekable();

        Box::new(std::iter::from_fn(move || {
            match peekable.next() {
                Some(msg) if peekable.peek().is_none() => {
                    // last element
                    if let ChannelMessage::Data(data, _) = msg {
                        match data {
                            Data::Tile(tile) => {
                                Some(ChannelMessage::Data(Data::Tile(Tile::new(
                                    tile.data
                                )), dim_end-2))
                            }
                            _ => Some(ChannelMessage::Data(data, 0)),
                        }
                    } else {
                        Some(msg)
                    }
                }
                Some(msg) => {
                    Some(msg)
                }
                None => None,
            }
        })) as Box<dyn Iterator<Item = ChannelMessage>>
    }

    // 4) build the raw iterator…
    let raw = inner(array);
    // 5) …then wrap it in a `.map` that prints progress whenever you cross each % point
    let with_progress = raw.map(move |item| {
        // detect a real data‐value (not a stop token)
        if let ChannelMessage::Data(_, _) = &item {
            seen += 1;
            let pct = seen * 100 / total;
            if pct >= next_pct {
                println!("Iterator {}, Progress: {}%", idx, pct);
                next_pct = pct + 1;
            }
        }
        item
    });

    // 6) return it
    Box::new(with_progress)
}

pub fn tensor_to_step(array: ArrayViewD<Scalar>, idx: String, tile_size: usize) -> Box<dyn Iterator<Item = ChannelMessage>> {
    let reshaped = reshape_tensor(array, tile_size);
    println!("Reshaped array:\n{:?}", reshaped);
    tiled_tensor_to_step(reshaped, idx)
}

#[cfg(test)]
mod tests {
    use ndarray::Array;

    use super::*;

    #[test]
    fn test_tensor_to_step() {
        let array = Array::from_shape_fn((1, 16, 32), |(_, i, j)| Scalar::FP32(j as f32 + 32.0 * (i as f32)));
        println!("Original array:\n{:?}", array);
        let binding = array.into_dyn();
        let reshaped = reshape_tensor(binding.view(), 16);
        assert_eq!(reshaped.shape(), &[1, 1, 2, 16, 16]); // dims: [tiles_y, tiles_x, tile_y, tile_x]

        let iter = tiled_tensor_to_step(reshaped.into_dyn().view(), "test".to_string()).collect::<Vec<_>>();

        let mut tile_a = Tile::new(vec![vec![Scalar::FP32(0.0); 16]; 16]);
        for i in 0..16 {
            for j in 0..16 {
                tile_a.data[i][j] = Scalar::FP32(j as f32 + 32.0 * (i as f32));
            }
        }

        let mut tile_b = Tile::new(vec![vec![Scalar::FP32(1.0); 16]; 16]);
        for i in 0..16 {
            for j in 0..16 {
                tile_b.data[i][j] = Scalar::FP32(j as f32 + 32.0 * (i as f32) + 16.0);
            }
        }

        let message_a = ChannelMessage::Data(Data::Tile(tile_a), 0);
        let message_b = ChannelMessage::Data(Data::Tile(tile_b), 3);

        assert_eq!(iter[0], message_a);
        assert_eq!(iter[1], message_b);
    }
}