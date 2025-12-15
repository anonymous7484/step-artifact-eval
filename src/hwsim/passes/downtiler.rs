use crate::hwsim::channel_id::ChannelID;

use crate::{hwsim::{graph::{function::{BaseOp, Operation, TypelessFunction, ValueType}, node_visitor::NodeVisitor, nodes::{self, next_id, Node}}, scalar::Scalar}, test::tensor_to_step::reshape_tensor};

pub struct Downtiler<'a> {
    hardware_tile_size: usize,
    nodes: &'a Vec<Box<dyn Node>>,
}

impl<'a> Downtiler<'a> {
    pub fn new(hardware_tile_size: usize, nodes: &'a Vec<Box<dyn Node>>) -> Self {
        Self { hardware_tile_size, nodes }
    }

    pub fn downtile(graph: Vec<Box<dyn Node>>, hardware_tile_size: usize) -> Vec<Box<dyn Node>> {
        let mut downtiler = Downtiler::new(hardware_tile_size, &graph);

        let retiled_graph: Vec<Box<dyn Node>> = graph.iter().flat_map(|n| {
            let result = n.accept(&mut downtiler);
    
            result.downcast::<Vec<Box<dyn Node>>>().unwrap_or_else(|e| {
                panic!("Retiler did not return a Vec<Box<dyn Node>>: {e:?}");
            }).into_iter()
        }).collect::<Vec<_>>();
    
        retiled_graph
    }

    fn scale_shape(&self, shape: &[usize]) -> Vec<usize> {
        let row_size = shape[shape.len() - 1];
        let col_size = shape[shape.len() - 2];
        assert!(row_size >= self.hardware_tile_size && row_size % self.hardware_tile_size == 0 || row_size == 1);
        assert!(col_size >= self.hardware_tile_size && col_size % self.hardware_tile_size == 0 || col_size == 1);

        let row_fold = (row_size + self.hardware_tile_size - 1) / self.hardware_tile_size;
        let col_fold = (col_size + self.hardware_tile_size - 1) / self.hardware_tile_size;

        let new_dim_end = vec![col_fold, row_fold, col_size / col_fold, row_size / row_fold];
        let mut new_dim = shape[..shape.len()-2].to_vec();
        new_dim.extend(new_dim_end);

        new_dim
    }

    fn get_shape_of_input(&self, channel_id: ChannelID) -> Option<Vec<usize>> {
        let node = self.nodes.iter().find(|n| n.get_outputs().contains(&channel_id)).unwrap();
        node.get_tile_size().map(|s| self.scale_shape(&s))
    }

    pub fn crop_shape(shape: &[usize]) -> [usize; 2] {
        let col_size = shape[shape.len() - 2];
        let row_size = shape[shape.len() - 1];
        return [col_size, row_size];
    }
}

impl<'a> NodeVisitor for Downtiler<'a> {
    fn visit_address_reader(&mut self, node: &nodes::AddressReader) -> Box<dyn std::any::Any> {
        let new_dim = self.scale_shape(&node.get_tile_size().unwrap());
        let mut my_reader = node.clone();
        my_reader.tile_size = Some(Downtiler::crop_shape(&new_dim));
        my_reader.data = my_reader.data.map(|d| {
            reshape_tensor(d.view(), self.hardware_tile_size).into_dyn().to_owned()
        });
        let mut new_shape = my_reader.shape;
        let mut tiles_down = self.scale_shape(&my_reader.tile_size.unwrap());
        tiles_down = tiles_down[..tiles_down.len() - 2].to_vec();
        new_shape.extend(tiles_down);
        my_reader.shape = Downtiler::scale_shape(&self, &new_shape);

        Box::new(vec![Box::new(my_reader) as Box<dyn Node>])
    }

    fn visit_broadcast(&mut self, node: &nodes::Broadcast) -> Box<dyn std::any::Any> {
        let new_dim = self.scale_shape(&node.get_tile_size().unwrap());
        let mut my_broadcast = node.clone();
        my_broadcast.tile_size = Some(Downtiler::crop_shape(&new_dim));

        Box::new(vec![Box::new(my_broadcast) as Box<dyn Node>])
    }

    fn visit_map(&mut self, node: &nodes::Map) -> Box<dyn std::any::Any> {
        assert!(node.func.operations.len() == 1);
        match &node.func.operations[0].0 {
            Operation::Add(_, _)
            | Operation::Sub(_, _)
            | Operation::Mul(_, _)
            | Operation::Div(_, _)
            | Operation::Max(_, _)
            | Operation::Min(_, _)
            | Operation::Lt(_, _)
            | Operation::Leq(_, _)
            | Operation::Negate(_)
            | Operation::ScaleAdd(_, _, _)
            | Operation::Exp(_)
            | Operation::Square(_)
            | Operation::Pow(_, _)
            | Operation::Sin(_)
            | Operation::Cos(_)
            | Operation::Tanh(_)
            | Operation::Silu(_)
            | Operation::Identity(_)
            | Operation::Modulo(_, _) => {
                let new_dim = self.scale_shape(&node.get_tile_size().unwrap());
                let mut my_map = node.clone();
                my_map.tile_size = Some(Downtiler::crop_shape(&new_dim));
                return Box::new(vec![Box::new(my_map) as Box<dyn Node>])
            },
            Operation::Matmul(_base_op, _base_op1) => panic!("Cannot tile a matmul without transposing things"),
            Operation::MatmulT(_base_op, _base_op1) => {
                // println!("Old dim: {:?}", node.get_tile_size().unwrap());
                let new_dim = self.scale_shape(&node.get_tile_size().unwrap());
                // println!("New dim: {new_dim:?}");

                let a_shape = self.get_shape_of_input(node.inputs[0]).unwrap();
                let b_shape = self.get_shape_of_input(node.inputs[1]).unwrap();
                println!("A shape: {a_shape:?}, B shape: {b_shape:?}");

                let a_in = node.inputs[0];
                let a_buf_tok_out = ChannelID::new();
                let a_buf_tok_rcv = ChannelID::new();
                let a_rpt_out = ChannelID::new();
                let a_buf = nodes::Bufferize::new_inlined_streamify(
                    vec![a_in],
                    vec![a_buf_tok_out],
                    vec![a_buf_tok_rcv],
                    vec![a_rpt_out],
                    1,
                    vec![],
                    false,
                    Some(a_shape[a_shape.len() - 3]),
                    next_id(),
                );

                let a_rpt = nodes::RepeatStatic::new(a_buf_tok_out, b_shape[b_shape.len() - 4], a_buf_tok_rcv);

                let b_in = node.inputs[1];
                let b_tkn_out = ChannelID::new();
                let b_tkn_in = ChannelID::new();
                let b_rpt = ChannelID::new();

                let b_buf = nodes::Bufferize::new_inlined_streamify(
                    vec![b_in],
                    vec![b_tkn_out],
                    vec![b_tkn_in],
                    vec![b_rpt],
                    2,
                    vec![],
                    false,
                    Some(b_shape[b_shape.len() - 4] * b_shape[b_shape.len() - 3]),
                    next_id(),
                );

                let b_buf_rpt = nodes::RepeatStatic::new(b_tkn_out, a_shape[a_shape.len() - 4], b_tkn_in);

                let matmul_to_acc = ChannelID::new();
                let mut matmul = node.clone();
                matmul.tile_size = Some(Downtiler::crop_shape(&new_dim));
                matmul.inputs = vec![a_rpt_out, b_rpt];
                let out = matmul.output;
                matmul.output = matmul_to_acc;

                let acc = nodes::Accum::new(vec![matmul_to_acc], out, TypelessFunction {
                    operations: vec![(Operation::Add(BaseOp::Variable(0, ValueType::Float), BaseOp::Variable(1, ValueType::Float)), vec![0])],
                    n_inputs: 2,
                    n_outputs: 1
                }, 1, Scalar::FP32(0.0));

                Box::new(vec![
                    Box::new(a_rpt) as Box<dyn Node>,
                    Box::new(a_buf) as Box<dyn Node>,
                    Box::new(b_buf_rpt) as Box<dyn Node>,
                    Box::new(b_buf) as Box<dyn Node>,
                    Box::new(matmul) as Box<dyn Node>,
                    Box::new(acc) as Box<dyn Node>,
                ])
            },
            Operation::RepeatStatic( .. ) => todo!(),
            Operation::ExpandStatic( .. ) => todo!(),
            Operation::Permute { .. } => todo!(),
            Operation::RetileRow(_base_op) => panic!("RetileRow is not supported"),
            Operation::RetileCol(_base_op) => panic!("RetileCol is not supported"),
        }
    }
    
    fn visit_accum(&mut self, node: &nodes::Accum) -> Box<dyn std::any::Any> {
        // This node actually requires state now, since we're accumulating over a matrix
        // and not just a hardware tile. 

        // This becomes a memory unit -> repeat_static scheme.
        
        let new_dim = self.scale_shape(&node.get_tile_size().unwrap());
                
        assert_eq!(new_dim.len(), node.get_tile_size().unwrap().len() + 2);

        let feedback_loop = ChannelID::new();
        let intermediate = ChannelID::new();
        let accumulator = nodes::AccumBigTile::new_with_tile_size(node.inputs[0], feedback_loop, intermediate, node.output, node.fold.clone(), node.rank + 2, Downtiler::crop_shape(&new_dim));

        let bufferize_loop = ChannelID::new();
        let partial_sum = nodes::Bufferize::new_inlined_streamify_with_tile_size(
            vec![intermediate],
            vec![bufferize_loop],
            vec![bufferize_loop],
            vec![feedback_loop],
            2,
            vec![],
            false,
            Some(new_dim[new_dim.len() - 3] * new_dim[new_dim.len() - 4]),
            next_id(),
            [new_dim[new_dim.len() - 2] * new_dim[new_dim.len() - 1], self.hardware_tile_size],
        );

        Box::new(vec![
            Box::new(accumulator) as Box<dyn Node>,
            Box::new(partial_sum) as Box<dyn Node>,
        ])
    }
    
    fn visit_promote(&mut self, node: &nodes::Promote) -> Box<dyn std::any::Any> {
        let num_old_dims = node.get_tile_size().unwrap().len();
        let new_dim = self.scale_shape(&node.get_tile_size().unwrap());
        let num_new_dims = new_dim.len();
        let num_added_dims = num_new_dims - num_old_dims;
        let mut my_promote = node.clone();
        my_promote.rank = node.rank + num_added_dims;
        my_promote.tile_size = Some(Downtiler::crop_shape(&new_dim));
        
        Box::new(vec![Box::new(my_promote) as Box<dyn Node>])
    }

    fn visit_address_writer(&mut self, node: &nodes::AddressWriter) -> Box<dyn std::any::Any> {
        let new_dim = self.scale_shape(&node.get_tile_size().unwrap());
        let mut my_writer = node.clone();
        my_writer.tile_size = Some(Downtiler::crop_shape(&new_dim));

        Box::new(vec![Box::new(my_writer) as Box<dyn Node>])
    }

    fn visit_flatten(&mut self, node: &nodes::Flatten) -> Box<dyn std::any::Any> {
        let new_dim = self.scale_shape(&node.get_tile_size().unwrap());
        let mut my_flatten = node.clone();
        my_flatten.tile_size = Some(Downtiler::crop_shape(&new_dim));

        Box::new(vec![Box::new(my_flatten) as Box<dyn Node>])
    }

    fn visit_reshape(&mut self, node: &nodes::Reshape) -> Box<dyn std::any::Any> {
        let new_dim = self.scale_shape(&node.get_tile_size().unwrap());
        let mut my_reshape = node.clone();
        let num_added_dims = new_dim.len() - node.get_tile_size().unwrap().len();
        my_reshape.tile_size = Some(Downtiler::crop_shape(&new_dim));
        my_reshape.split_dims = my_reshape.split_dims.iter().map(|d| d + num_added_dims).collect();
        Box::new(vec![Box::new(my_reshape) as Box<dyn Node>])
    }

    fn visit_partition(&mut self, node: &nodes::Partition) -> Box<dyn std::any::Any> {
        // 1. Find the number of new tiles per old tile.
        // 2. RepeatStatic the select stream for (new_tiles_per_old_tile) times.
        // 3. Partition stays similar apart from that 

        let old_tile_size = node.get_tile_size().unwrap();
        let scaled = self.scale_shape(&old_tile_size);
        let new_tile_size = Downtiler::crop_shape(&scaled);
        let new_tiles_per_old_tile = old_tile_size.iter().product::<usize>() / new_tile_size.iter().product::<usize>();
        let num_added_dims = new_tile_size.len() - old_tile_size.len();

        let repeated_select = ChannelID::new();
        let repeat = nodes::RepeatStatic::new_with_tile_size(node.sel, new_tiles_per_old_tile, repeated_select, Downtiler::crop_shape(&new_tile_size));

        let mut my_partition = node.clone();
        my_partition.tile_size = Some(new_tile_size);
        my_partition.sel = repeated_select;
        my_partition.rank = my_partition.rank + num_added_dims;

        Box::new(vec![Box::new(repeat) as Box<dyn Node>, Box::new(my_partition) as Box<dyn Node>])
    }

    fn visit_select_gen(&mut self, node: &nodes::SelectGen) -> Box<dyn std::any::Any> {
        // I don't think anything is required here.
        return Box::new(vec![Box::new(node.clone()) as Box<dyn Node>]);
    }

    fn visit_repeat_static(&mut self, node: &nodes::RepeatStatic) -> Box<dyn std::any::Any> {
        // 1. We need to bufferize the stream on rank 2 (software tile size)
        // 2. We need to repeat the bufferize stream repeat_static.count times
        // 3. Then we need to streamify the bufferize stream


        let new_dim = self.scale_shape(&node.get_tile_size().unwrap());
        let dims_without_last_two = new_dim[..new_dim.len() - 2].to_vec();

        let buf_to_rpt = ChannelID::new();
        let rpt_to_buf = ChannelID::new();

        let buf = nodes::Bufferize::new_inlined_streamify_with_tile_size(
            vec![node.input],
            vec![buf_to_rpt],
            vec![rpt_to_buf],
            vec![node.output],
            2,
            vec![],
            false,
            Some(dims_without_last_two.iter().product::<usize>()), // .. TODO: could be wrong
            next_id(),
            Downtiler::crop_shape(&new_dim),
        );
        let rpt = nodes::RepeatStatic::new_with_tile_size(buf_to_rpt, node.count, rpt_to_buf, Downtiler::crop_shape(&new_dim));

        Box::new(vec![
            Box::new(buf) as Box<dyn Node>,
            Box::new(rpt) as Box<dyn Node>,
        ])
    }

    fn visit_dyn_off_chip_load(&mut self, node: &nodes::DynOffChipLoad) -> Box<dyn std::any::Any> {
        let new_dim = self.scale_shape(&node.get_tile_size().unwrap());
        let mut my_dyn_off_chip_load = node.clone();
        

        let mut new_shape = my_dyn_off_chip_load.shape;
        let mut tiles_down = self.scale_shape(&my_dyn_off_chip_load.tile_size.unwrap());
        tiles_down = tiles_down[..tiles_down.len() - 2].to_vec();
        new_shape.extend(tiles_down);
        my_dyn_off_chip_load.tile_size = Some(Downtiler::crop_shape(&new_dim));
        my_dyn_off_chip_load.shape = new_shape;

        Box::new(vec![Box::new(my_dyn_off_chip_load) as Box<dyn Node>])
    }

    fn visit_retile_streamify(&mut self, node: &nodes::RetileStreamify) -> Box<dyn std::any::Any> {
        let new_dim = self.scale_shape(&node.get_tile_size().unwrap());
        let num_repeats = new_dim[new_dim.len()-3];

        let retile_to_buf = ChannelID::new();
        let buf_to_retile = ChannelID::new();
        let buf_loop = ChannelID::new();

        println!("Old dim: {:?}", node.get_tile_size().unwrap());
        println!("New dim: {:?}", Downtiler::crop_shape(&new_dim));
        let tiled_retile_streamify = nodes::TiledRetileStreamify::new_with_tile_size(node.input, node.output, retile_to_buf, buf_to_retile, node.split_row, node.filter_mask, num_repeats, Downtiler::crop_shape(&new_dim));
        let bufferize = nodes::Bufferize::new_inlined_streamify_with_tile_size(
            vec![retile_to_buf],
            vec![buf_loop],
            vec![buf_loop],
            vec![buf_to_retile],
            1,
            vec![],
            false,
            Some(new_dim[new_dim.len() - 3]),
            next_id(),
            Downtiler::crop_shape(&new_dim),
        );
        Box::new(vec![Box::new(tiled_retile_streamify) as Box<dyn Node>, Box::new(bufferize) as Box<dyn Node>])
    }

    fn visit_reassemble(&mut self, node: &nodes::Reassemble) -> Box<dyn std::any::Any> {
        // Same as for partition. Repeat the select stream for (new_tiles_per_old_tile) times.
        // Then reassemble the stream.

        let old_tile_size = node.get_tile_size().unwrap();
        let scaled = self.scale_shape(&old_tile_size);
        let new_tile_size = Downtiler::crop_shape(&scaled);
        let new_tiles_per_old_tile = old_tile_size.iter().product::<usize>() / new_tile_size.iter().product::<usize>();
        let num_added_dims = new_tile_size.len() - old_tile_size.len();

        let repeated_select = ChannelID::new();
        let repeat = nodes::RepeatStatic::new_with_tile_size(node.sel, new_tiles_per_old_tile, repeated_select, Downtiler::crop_shape(&new_tile_size));

        let mut my_reassemble = node.clone();
        my_reassemble.tile_size = Some(new_tile_size);
        my_reassemble.sel = repeated_select;
        my_reassemble.in_stream_rank = my_reassemble.in_stream_rank + num_added_dims;

        Box::new(vec![Box::new(repeat) as Box<dyn Node>, Box::new(my_reassemble) as Box<dyn Node>])
    }

    fn default(&mut self, _node: &dyn Node) -> Box<dyn std::any::Any> {
        panic!("Downtiler does not support node: {:?}", _node);
    }
}

#[cfg(test)]
mod tests {
    use crate::hwsim::channel_id::ChannelID;
    use ndarray::ArrayD;

    use crate::{hwsim::{dot::graph_dot::DotGenerator, graph::{function::{BaseOp, Operation, TypelessFunction, ValueType}, nodes::{Accum, AddressReader, AddressWriter, Map, Node}}, passes::{downtiler::Downtiler}, scalar::Scalar}};

    #[test]
    fn test_matmul_t() {
        let tile_size = 1;

        let arr_a = ArrayD::from_shape_fn(vec![1, 512, 1, 2*tile_size, tile_size], |indices| {
            let _i = indices[0];
            let _j = indices[1];
            Scalar::FP32(1.0)
        });
        let arr_a_shape = arr_a.shape().to_vec();
        let arr_b = ArrayD::from_shape_fn(vec![1, 512, 1, 4*tile_size, tile_size], |indices| {
            let _i = indices[0];
            let _j = indices[1];
            Scalar::FP32(1.0)
        });
        let arr_b_shape = arr_b.shape().to_vec();

        let in_a = ChannelID::new();
        let out_a = ChannelID::new();
        let a = AddressReader::new_with_data_and_tile_size(in_a, vec![out_a], arr_a, [arr_a_shape[arr_a_shape.len()-2], arr_a_shape[arr_a_shape.len()-1]], arr_a_shape[0..arr_a_shape.len()-2].to_vec());

        let in_b = ChannelID::new();
        let out_b = ChannelID::new();
        let b = AddressReader::new_with_data_and_tile_size(in_b, vec![out_b], arr_b, [arr_b_shape[arr_b_shape.len()-2], arr_b_shape[arr_b_shape.len()-1]], arr_b_shape[0..arr_b_shape.len()-2].to_vec());

        let out_c = ChannelID::new();
        let mut out_c_shape = arr_a_shape.clone();
        let out_c_shape_len = out_c_shape.len();
        out_c_shape[out_c_shape_len - 1] = arr_b_shape[arr_b_shape.len() - 1];
        let matmul = Map::new_with_tile_size(vec![out_a, out_b], out_c, TypelessFunction {
            operations: vec![(Operation::MatmulT(BaseOp::Variable(0, ValueType::Float), BaseOp::Variable(1, ValueType::Float)), vec![0])],
            n_inputs: 2,
            n_outputs: 1
        }, [arr_a_shape[arr_a_shape.len()-2], arr_a_shape[arr_a_shape.len()-1]]);

        let out = ChannelID::new();
        let output = AddressWriter::new_with_tile_size(vec![out_c], out, [tile_size, tile_size]);

        let nodes = vec![
            Box::new(a) as Box<dyn Node>,
            Box::new(b) as Box<dyn Node>,
            Box::new(matmul) as Box<dyn Node>,
            Box::new(output) as Box<dyn Node>,
        ];

        let graph = Downtiler::downtile(nodes, tile_size);

        let mut dot_gen_after = DotGenerator::new(false);
        dot_gen_after.generate_dot_graph(&graph, None);
        std::fs::write("matmul_t_downtiled.dot", dot_gen_after.to_string()).unwrap();
        
    }

    #[test]
    fn test_tile_accum() {
        let tile_size = 2;

        let shape = vec![1, 1, 10, 4, 1, 2, 1*tile_size, 2*tile_size];
        let arr_a = ArrayD::from_shape_fn(shape.clone(), |indices| {
            let _i = indices[0];
            let _j = indices[1];
            Scalar::FP32(1.0)
        });

        let in_a = ChannelID::new();
        let out_a = ChannelID::new();
        let gen_a = AddressReader::new_with_data_and_tile_size(in_a, vec![out_a], arr_a, [shape[shape.len()-2], shape[shape.len()-1]], shape[0..shape.len()-2].to_vec());

        let accum_out = ChannelID::new();
        let accum = Accum::new_with_tile_size(vec![out_a], accum_out, TypelessFunction {
            operations: vec![(Operation::Add(BaseOp::Variable(0, ValueType::Float), BaseOp::Variable(1, ValueType::Float)), vec![0])],
            n_inputs: 2,
            n_outputs: 1
        }, 1, Scalar::FP32(0.0), [shape[shape.len()-2], shape[shape.len()-1]]);

        let output = AddressWriter::new_with_tile_size(vec![accum_out], ChannelID::new(), [shape[shape.len()-2], shape[shape.len()-1]]);

        let nodes = vec![
            Box::new(gen_a) as Box<dyn Node>,
            Box::new(accum) as Box<dyn Node>,
            Box::new(output) as Box<dyn Node>,
        ];

        let graph = Downtiler::downtile(nodes, tile_size);

        let mut dot_gen_after = DotGenerator::new(false);
        dot_gen_after.generate_dot_graph(&graph, None);
        std::fs::write("tile_accum_downtiled.dot", dot_gen_after.to_string()).unwrap();

    }

    #[test]
    fn regular_accum() {
        let tile_size = 4;

        let shape = vec![1, 1, 1, 4, 1, 2, tile_size, tile_size];
        let arr_a = ArrayD::from_shape_fn(shape.clone(), |indices| {
            let _i = indices[0];
            let _j = indices[1];
            Scalar::FP32(1.0)
        });

        let in_a = ChannelID::new();
        let out_a = ChannelID::new();
        let gen_a = AddressReader::new_with_data_and_tile_size(in_a, vec![out_a], arr_a, [shape[shape.len()-2], shape[shape.len()-1]], shape[0..shape.len()-2].to_vec());

        let accum_out = ChannelID::new();
        let accum = Accum::new_with_tile_size(vec![out_a], accum_out, TypelessFunction {
            operations: vec![(Operation::Add(BaseOp::Variable(0, ValueType::Float), BaseOp::Variable(1, ValueType::Float)), vec![0])],
            n_inputs: 2,
            n_outputs: 1
        }, 1, Scalar::FP32(0.0), [shape[shape.len()-2], shape[shape.len()-1]]);

        let output = AddressWriter::new_with_tile_size(vec![accum_out], ChannelID::new(), [shape[shape.len()-2], shape[shape.len()-1]]);

        let graph = vec![
            Box::new(gen_a) as Box<dyn Node>,
            Box::new(accum) as Box<dyn Node>,
            Box::new(output) as Box<dyn Node>,
        ];

        let graph = Downtiler::downtile(graph, tile_size);

        let mut dot_gen_after = DotGenerator::new(false);
        dot_gen_after.generate_dot_graph(&graph, None);
        std::fs::write("regular_accum_downtiled.dot", dot_gen_after.to_string()).unwrap();
    }
}