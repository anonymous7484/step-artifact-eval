use std::{any, collections::HashMap};

use crate::{hwsim::{channel_id::ChannelID, graph::{function::{Operation, TypelessFunction}, node_visitor::NodeVisitor, nodes::{self, Node}}, scalar::Scalar}, test::tensor_to_step::tiled_tensor_to_step};

pub struct ConvertToStaticBluespecPass {
    hypernodes: String,
    bluespec_code: String,
    module_counter: usize,
    node_id_to_module_name: HashMap<usize, String>,
    rule_ctr: usize,
    tile_size: usize,
    name: String,
    num_tile_readers: usize
}

impl ConvertToStaticBluespecPass {
    pub fn new(tile_size: usize, name: String) -> Self {
        // Create (or empty) the "gen_bsv" folder at the start of the pass.
        let gen_bsv_path = std::path::Path::new(&name);
        if gen_bsv_path.exists() {
            // Remove all files in the directory from previous runs.
            for entry in std::fs::read_dir(gen_bsv_path).unwrap() {
                let entry = entry.unwrap();
                let path = entry.path();
                if path.is_file() {
                    std::fs::remove_file(path).unwrap();
                } else if path.is_dir() {
                    std::fs::remove_dir_all(path).unwrap();
                }
            }
        } else {
            std::fs::create_dir(gen_bsv_path).unwrap();
        }

        Self {
            bluespec_code: String::new(),
            module_counter: 0,
            node_id_to_module_name: HashMap::new(),
            rule_ctr: 0,
            tile_size: tile_size,
            name: name,
            num_tile_readers: 0,
            hypernodes: String::new(),
        }
    }

    pub fn convert_to_bluespec(&mut self, nodes: &Vec<Box<dyn Node>>) {
        for node in nodes {
            node.accept(self);
        }
        let rules_code = self.create_rules(nodes);
        let rules_comma_separated = (1..=self.rule_ctr).map(|i| format!("rule_{i}")).collect::<Vec<String>>().join(", ");
        let descending_urgency = format!("    (* descending_urgency = \"{rules_comma_separated}\" *)\n");
        self.bluespec_code.push_str(descending_urgency.as_str());
        self.bluespec_code.push_str(rules_code.as_str());
        let code = self.get_bluespec_code();
        std::fs::write("bluespec/from_hwsim/Step.bsv", code).unwrap();
    }

    fn get_bluespec_code(&self) -> String {
        let ramulator_string = if self.num_tile_readers > 0 {
            format!("    RamulatorArbiter_IFC#({}) ramulator_arbiter <- mkRamulatorArbiter({});", self.num_tile_readers, self.num_tile_readers)
        } else {
            "".to_string()
        };

        format!(
            concat!(
                "package Step;\n",
                "import Operation::*;\n",
                "import Types::*;\n",
                "import Vector::*;\n",
                "import FShow::*;\n",
                "import Debug::*;\n",
                "import PMU::*;\n",
                "import RamulatorArbiter::*;\n",
                "\n",
                "{}\n",
                "module mkStep(Empty);\n",
                "{}",
                "{}\n",
                "endmodule\n",
                "endpackage\n"
            ),
            self.hypernodes,
            ramulator_string,
            self.bluespec_code
        )
    }

    fn get_next_module_name(&mut self, node_id: usize) -> String {
        let name = format!("mod_{}", self.module_counter);
        self.module_counter += 1;
        self.node_id_to_module_name.insert(node_id, name.clone());
        name
    }

    fn create_rules(&mut self, nodes: &Vec<Box<dyn Node>>) -> String {
        self.rule_ctr = 0;
        let mut output_to_node_id: HashMap<ChannelID, (usize, usize)> = HashMap::new();
        let mut input_to_node_id: HashMap<ChannelID, (usize, usize)> = HashMap::new();

        for node in nodes {
            for (output_port_idx, cid) in node.get_outputs().iter().enumerate() {
                output_to_node_id.insert(*cid, (node.get_id(), output_port_idx));
            }
            for (input_port_idx, cid) in node.get_inputs().iter().enumerate() {
                input_to_node_id.insert(*cid, (node.get_id(), input_port_idx));
            }
        }

        // Join the two hashmaps (output_to_node_id and input_to_node_id) on the key (ChannelID)
        // and collect the pairs of ((output_node_id, output_port), (input_node_id, input_port))
        let mut connections: Vec<((usize, usize), (usize, usize))> = Vec::new();
        for (cid, &(output_node_id, output_port)) in output_to_node_id.iter() {
            if let Some(&(input_node_id, input_port)) = input_to_node_id.get(cid) {
                connections.push(((output_node_id, output_port), (input_node_id, input_port)));
            }
        }

        let mut rules_code = String::new();
        for connection in connections {
            self.rule_ctr += 1;
            rules_code.push_str(format!(
                "    rule rule_{rule_ctr};\n        ChannelMessage t;\n        t <- {output_module_name}.get({output_port});\n        {input_module_name}.put({input_port}, t);\n    endrule\n",
                output_module_name = self.node_id_to_module_name.get(&connection.0.0).unwrap_or(&"error_module".to_string()),
                output_port = connection.0.1,
                input_module_name = self.node_id_to_module_name.get(&connection.1.0).unwrap_or(&"error_module".to_string()),
                input_port = connection.1.1,
                rule_ctr = self.rule_ctr
            ).as_str());
        }
        rules_code
    }

    fn match_function_to_bluespec_function(&self, function: &TypelessFunction) -> String {
        assert!(function.operations.len() == 1, "We only support single operation accumulators for now.");
        match &function.operations[0].0 {
            Operation::Add(base_op, _base_op1) => "add_tile",
            Operation::Sub(base_op, _base_op1) => "sub_tile",
            Operation::Mul(base_op, _base_op1) => "mul_tile",
            Operation::Div(base_op, _base_op1) => "div_tile",
            Operation::Max(base_op, _base_op1) => "max_tile",
            Operation::Min(base_op, _base_op1) => "min_tile",
            Operation::Lt(base_op, _base_op1) => "lt_tile",
            Operation::Leq(base_op, _base_op1) => "leq_tile",
            Operation::ScaleAdd(base_op, base_op1, _base_op2) => "scale_add_tile",
            Operation::Negate(_base_op) => "neg_tile",
            Operation::Exp(_base_op) => "exp_tile",
            Operation::Square(_base_op) => "square_tile",
            Operation::Pow(_base_op, _base_op1) => "pow_tile",
            Operation::Sin(_base_op) => "sin_tile",
            Operation::Cos(_base_op) => "cos_tile",
            Operation::Tanh(_base_op) => "tanh_tile",
            Operation::Silu(_base_op) => "silu_tile",
            Operation::Identity(_base_op) => "identity_tile",
            Operation::Modulo(_base_op, _base_op1) => "modulo_tile",
            Operation::Matmul(_base_op, _base_op1) => "matmul_tile",
            Operation::MatmulT(_base_op, _base_op1) => "matmul_t_tile",
            Operation::RepeatStatic(_repeat_static) => "repeat_static_tile",
            Operation::ExpandStatic(_base_op, _hash_map) => "expand_static_tile",
            Operation::Permute { op: _, permute: _, transpose: _ } => "permute_tile",
            Operation::RetileRow(_base_op) => "retile_row_tile",
            Operation::RetileCol(_base_op) => "retile_col_tile",
        }.to_string()
    }

    fn handle_hypernode(&mut self, node_id: usize, node: &nodes::Hypernode) {
        // Idea: Essentially, do everything we're doing at the higher level, but also include interfaces for inputs & outputs
        let other_code = std::mem::take(&mut self.bluespec_code);

        let all_inputs = node.get_inputs();
        let all_outputs = node.get_outputs();

        self.bluespec_code.push_str(format!(
            "(* synthesize *)\nmodule mkHypernode_{id} (Operation_IFC);\n",
            id = node_id,
        ).as_str());

        for node in &node.nodes {
            node.accept(self);
        }

        let mut input_body = String::new();
        let mut output_body = String::new();

        // Problem now: Match i/o to inner module names. hmm.
        for node in &node.nodes {
            for (mod_idx, input) in node.get_inputs().iter().enumerate() {
                if let Some(matching_input_idx) = all_inputs.iter().position(|x| x == input) {
                    let module_name = self.node_id_to_module_name.get(&node.get_id()).unwrap();
                    let str = format!("        if (i == {matching_input_idx}) begin\n            {module_name}.put({mod_idx}, t);\n        end\n");
                    input_body.push_str(str.as_str());
                }
            }
            for (mod_idx, output) in node.get_outputs().iter().enumerate() {
                if let Some(matching_output_idx) = all_outputs.iter().position(|x| x == output) {
                    let module_name = self.node_id_to_module_name.get(&node.get_id()).unwrap();
                    let str = format!("        if (i == {matching_output_idx}) begin\n            t <- {module_name}.get({mod_idx});\n        end\n");
                    output_body.push_str(str.as_str());
                }
            }
        }

        let output_block = format!(
            "    method ActionValue#(ChannelMessage) get(Int#(32) i);\n        ChannelMessage t = unpack(0);\n{output_body}\n        return t;\n    endmethod\n",
            output_body = output_body,
        );

        let input_block = format!(
            "    method Action put(Int#(32) i, ChannelMessage t);\n{input_body}\n    endmethod\n",
            input_body = input_body
        );

        let rules = self.create_rules(&node.nodes);
        self.bluespec_code.push_str(rules.as_str());

        self.bluespec_code.push_str(&input_block);
        self.bluespec_code.push_str(&output_block);

        self.bluespec_code.push_str("endmodule\n");

        println!("Hypernode is {:?}", self.bluespec_code);
        self.hypernodes.push_str(std::mem::take(&mut self.bluespec_code).as_str());
        self.bluespec_code.push_str(&other_code);
    }
}

impl NodeVisitor for ConvertToStaticBluespecPass {
    fn default(&mut self, _node: &dyn Node) -> Box<dyn any::Any> {
        panic!("Not implemented for {:?}", _node);
    }

    fn visit_address_reader(&mut self, node: &nodes::AddressReader) -> Box<dyn any::Any> {
        // This here becomes (a) a binary file, (b) a context to push the file into the datapath.
        if node.data.is_none() {
            let name = self.get_next_module_name(node.get_id());
            let shape = {
                let mut s = String::from("Nil");
                for dim in node.shape.iter().rev() {
                    s = format!("Cons({}, {})", dim, s);
                }
                s
            };
            self.bluespec_code.push_str(format!(
                "    Operation_IFC {name}_inner <- mkRandomOffChipLoad({shape});\n    Operation_IFC {name} <- mkDebugOperation({name}_inner, \"{name}\");\n",
            ).as_str());
            return Box::new(());
        }

        let file_name;
        let tile_reader_id;
        let num_elements;

        tile_reader_id = self.num_tile_readers;
        self.num_tile_readers += 1;

        if node.data.is_some() {
            let data = tiled_tensor_to_step(node.data.as_ref().unwrap().view(), node.get_id().to_string());
            // Unpack each element in data into a tile, and write the tile content into a hex file in form tile_element tile_element tile_element stop_token_value

            // We'll write to a file named after the node id, e.g., "address_reader_<id>.hex"
            use std::fs::File;
            use std::io::{BufWriter, Write};

            num_elements = node.data.as_ref().unwrap().len();
            file_name = format!("{}/address_reader_{}.hex", self.name, node.get_id());
            let file = File::create(&file_name).expect("Unable to create hex file");
            let mut writer = BufWriter::new(file);

            for msg in data {
                if let crate::hwsim::types::ChannelMessage::Data(crate::hwsim::types::Data::Tile(tile), stop_token) = &msg {
                    let stop_token_value = format!("{:08X}", stop_token);

                    // Write each element in the tile (row-major order)
                    for row in &tile.data {
                        for elem in row {
                            // Convert Scalar to hex string
                            let hex_str = match elem {
                                crate::hwsim::scalar::Scalar::FP32(f) => {
                                    // Convert f32 to u32 bits, then to hex
                                    format!("{:08X}", f.to_bits())
                                }
                                crate::hwsim::scalar::Scalar::I32(i) => {
                                    format!("{:08X}", *i as u32)
                                }
                                crate::hwsim::scalar::Scalar::U32(u) => {
                                    format!("{:08X}", *u)
                                }
                            };
                            write!(writer, "{hex_str}").expect("Failed to write tile element");
                        }
                    }
                    // Write the stop token after each tile
                    writeln!(writer, "{stop_token_value}").expect("Failed to write stop token");
                }
            }
        } else {
            file_name = "TODOFILLOUTTODO".to_string();
            num_elements = 0;
        }

        let name = self.get_next_module_name(node.get_id());
        self.bluespec_code.push_str(format!(
            "    Operation_IFC {name}_inner <- mkTileReader({num_lines}, \"{file_name}\", {tile_reader_id}, ramulator_arbiter.ports);\n    Operation_IFC {name} <- mkDebugOperation({name}_inner, \"{name}\");\n",
            num_lines = num_elements / self.tile_size / self.tile_size,
            file_name = format!("from_hwsim/address_reader_{}.hex", node.get_id())
        ).as_str());
        
        return Box::new(name);
    }

    fn visit_address_writer(&mut self, node: &nodes::AddressWriter) -> Box<dyn any::Any> {
        let name = self.get_next_module_name(node.get_id());
        self.bluespec_code.push_str(format!(
            "    Operation_IFC {name}_inner <- mkPrinter(\"{name}\");\n    Operation_IFC {name} <- mkDebugOperation({name}_inner, \"{name}\");\n"
        ).as_str());
        Box::new(())
    }

    fn visit_accum(&mut self, node: &nodes::Accum) -> Box<dyn any::Any> {
        let function = self.match_function_to_bluespec_function(&node.fold);

        let name = self.get_next_module_name(node.get_id());
        self.bluespec_code.push_str(format!(
            "    Operation_IFC {name}_inner <- mkAccum({function}, {rank});\n    Operation_IFC {name} <- mkDebugOperation({name}_inner, \"{name}\");\n",
            rank = node.rank,
        ).as_str());
        Box::new(())
    }

    fn visit_accum_big_tile(&mut self, node: &nodes::AccumBigTile) -> Box<dyn any::Any> {
        let function = self.match_function_to_bluespec_function(&node.fold);

        let name = self.get_next_module_name(node.get_id());
        self.bluespec_code.push_str(format!(
            "    Operation_IFC {name}_inner <- mkAccumBigTile({function}, {rank});\n    Operation_IFC {name} <- mkDebugOperation({name}_inner, \"{name}\");\n",
            rank = node.rank,
        ).as_str());
        Box::new(())
    }

    fn visit_broadcast(&mut self, node: &nodes::Broadcast) -> Box<dyn any::Any> {
        assert!(node.outputs.len() == 2, "We only support dual output broadcasts for now.");

        let name = self.get_next_module_name(node.get_id());
        self.bluespec_code.push_str(format!(
            "    let {name}_inner <- mkBroadcast2();\n    let {name} <- mkDebugOperation({name}_inner, \"{name}\");\n"
        ).as_str());
        Box::new(())
    }

    fn visit_bufferize(&mut self, node: &nodes::Bufferize) -> Box<dyn any::Any> {
        assert!(node.data_inputs.len() == 1, "We only support single data input bufferizes for now.");
        assert!(node.token_requests.len() == 1, "We only support single token input bufferizes for now.");
        assert!(node.token_outputs.len() == 1, "We only support single token output bufferizes for now.");
        assert!(node.data_outputs.len() == 1, "We only support single data output bufferizes for now.");
        
        let name = self.get_next_module_name(node.get_id());

        self.bluespec_code.push_str(format!(
            "    PMU_IFC {name}_bufferize <- mkPMU({rank});\n    Operation_IFC {name}_inner = {name}_bufferize.operation;\n    Operation_IFC {name} <- mkDebugOperation({name}_inner, \"{name}\");\n",
            rank = node.rank,
        ).as_str());
        Box::new(())
    }

    fn visit_concatenate(&mut self, node: &nodes::Concatenate) -> Box<dyn any::Any> {
        panic!("Not implemented");
    }

    fn visit_dyn_off_chip_load(&mut self, node: &nodes::DynOffChipLoad) -> Box<dyn any::Any> {
        let name = self.get_next_module_name(node.get_id());
        let shape = {
            let mut s = String::from("Nil");
            let mut shape = node.shape.clone();
            if node.transposed_read {
                let tmp = shape[0];
                shape[0] = shape[1];
                shape[1] = tmp;
            }
            for dim in shape.iter().rev() {
                s = format!("Cons({}, {})", dim, s);
            }
            s
        };
        
        self.bluespec_code.push_str(format!(
            "    Operation_IFC {name}_inner <- mkDynamicRandomLoad({shape});\n    Operation_IFC {name} <- mkDebugOperation({name}_inner, \"{name}\");\n",
        ).as_str());
        Box::new(())
    }

    fn visit_enumerate(&mut self, node: &nodes::Enumerate) -> Box<dyn any::Any> {
        panic!("Not implemented");
    }

    fn visit_filter(&mut self, node: &nodes::Filter) -> Box<dyn any::Any> {
        panic!("Not implemented");
    }

    fn visit_flatmap(&mut self, node: &nodes::FlatMap) -> Box<dyn any::Any> {
        panic!("Not implemented");
    }

    fn visit_flatten(&mut self, node: &nodes::Flatten) -> Box<dyn any::Any> {
        assert!(node.ranks.len() == 1, "We only support single rank flatten for now, but got {} ranks. ({:?})", node.ranks.len(), node.ranks);
        let name = self.get_next_module_name(node.get_id());
        self.bluespec_code.push_str(format!(
            "    Operation_IFC {name}_inner <- mkFlatten({rank});\n    Operation_IFC {name} <- mkDebugOperation({name}_inner, \"{name}\");\n",
            rank = node.ranks[0],
        ).as_str());
        Box::new(())
    }

    fn visit_fnblock(&mut self, node: &nodes::FnBlock) -> Box<dyn any::Any> {
        panic!("Not implemented");
    }

    fn visit_hypernode(&mut self, node: &nodes::Hypernode) -> Box<dyn any::Any> {
        let name = self.get_next_module_name(node.get_id());
        self.bluespec_code.push_str(format!(
            "    Operation_IFC {name}_inner <- mkHypernode_{id};\n    Operation_IFC {name} <- mkDebugOperation({name}_inner, \"{name}\");\n",
            id = node.get_id(),
        ).as_str());
        self.handle_hypernode(node.get_id(), node);
        Box::new(())
    }

    fn visit_map(&mut self, node: &nodes::Map) -> Box<dyn any::Any> {
        let function = self.match_function_to_bluespec_function(&node.func);

        let name = self.get_next_module_name(node.get_id());
        let interface_name = match node.func.n_inputs {
            1 => "mkUnaryMap",
            2 => "mkBinaryMap",
            3 => "mkTernaryMap",
            _ => panic!("Map with {} inputs not supported.", node.func.n_inputs),
        };
        let id = node.get_id();

        
        self.bluespec_code.push_str(format!(
            "    Operation_IFC {name}_inner <- {interface_name}({id}, {function});\n    Operation_IFC {name} <- mkDebugOperation({name}_inner, \"{name}\");\n"
        ).as_str());
        Box::new(())
    }

    fn visit_matmul_t(&mut self, node: &nodes::MatmulT) -> Box<dyn any::Any> {
        panic!("Not implemented");
    }

    fn visit_parallelize(&mut self, node: &nodes::Parallelize) -> Box<dyn any::Any> {
        panic!("Not implemented");
    }

    fn visit_partition(&mut self, node: &nodes::Partition) -> Box<dyn any::Any> {
        let name = self.get_next_module_name(node.get_id());
        self.bluespec_code.push_str(format!(
            "    Partition_IFC#({num_partitions}) {name}_inner <- mkPartition({rank}, {num_partitions});\n    Operation_IFC {name} <- mkDebugOperation({name}_inner.op, \"{name}\");\n",
            rank = node.rank,
            num_partitions = node.outputs.len(),
        ).as_str());
        Box::new(())
    }

    fn visit_promote(&mut self, node: &nodes::Promote) -> Box<dyn any::Any> {
        let name = self.get_next_module_name(node.get_id());
        self.bluespec_code.push_str(format!(
            "    Operation_IFC {name}_inner <- mkPromote({rank});\n    Operation_IFC {name} <- mkDebugOperation({name}_inner, \"{name}\");\n",
            rank = node.rank,
        ).as_str());
        Box::new(())
    }

    fn visit_reassemble(&mut self, node: &nodes::Reassemble) -> Box<dyn any::Any> {
        let name = self.get_next_module_name(node.get_id());
        self.bluespec_code.push_str(format!(
            "    Reassemble_IFC#({num_inputs}) {name}_inner <- mkReassemble({num_inputs});\n    Operation_IFC {name} <- mkDebugOperation({name}_inner.op, \"{name}\");\n",
            num_inputs = node.inputs.len()
        ).as_str());
        Box::new(())
    }

    fn visit_repeat(&mut self, node: &nodes::Repeat) -> Box<dyn any::Any> {
        panic!("Not implemented");
    }

    fn visit_repeat_ref(&mut self, node: &nodes::RepeatRef) -> Box<dyn any::Any> {
        panic!("Not implemented");
    }

    fn visit_repeat_static(&mut self, node: &nodes::RepeatStatic) -> Box<dyn any::Any> {
        let name = self.get_next_module_name(node.get_id());
        self.bluespec_code.push_str(format!(
            "    Operation_IFC {name}_inner <- mkRepeatStatic({count});\n    Operation_IFC {name} <- mkDebugOperation({name}_inner, \"{name}\");\n",
            count = node.count,
        ).as_str());
        Box::new(())
    }

    fn visit_reshape(&mut self, node: &nodes::Reshape) -> Box<dyn any::Any> {
        assert!(node.split_dims.len() == 1, "We only support single split dimension reshapes for now.");
        assert!(node.chunk_sizes.len() == 1, "We only support single chunk size reshapes for now.");

        let name = self.get_next_module_name(node.get_id());
        self.bluespec_code.push_str(format!(
            "    Operation_IFC {name}_inner <- mkReshape({split_dim}, {chunk_size}, {pad_elem});\n    Operation_IFC {name} <- mkDebugOperation({name}_inner, \"{name}\");\n",
            split_dim = node.split_dims[0],
            chunk_size = node.chunk_sizes[0],
            pad_elem = {if node.elem == Some(Scalar::I32(0)) {"tagged Valid 0"} else if node.elem == None {"tagged Invalid"} else {panic!("Unexpected element type")}}
        ).as_str());
        Box::new(())
    }

    fn visit_retile_streamify(&mut self, node: &nodes::RetileStreamify) -> Box<dyn any::Any> {
        let name = self.get_next_module_name(node.get_id());
        self.bluespec_code.push_str(format!(
            "    Operation_IFC {name}_inner <- mkRetileStreamify({split_row}, {filter_mask});\n    Operation_IFC {name} <- mkDebugOperation({name}_inner, \"{name}\");\n",
            split_row = node.split_row,
            filter_mask = node.filter_mask,
        ).as_str());
        Box::new(())
    }

    fn visit_rotate(&mut self, node: &nodes::Rotate) -> Box<dyn any::Any> {
        panic!("Not implemented");
    }

    fn visit_select_gen(&mut self, node: &nodes::SelectGen) -> Box<dyn any::Any> {
        let name = self.get_next_module_name(node.get_id());
        
        self.bluespec_code.push_str(format!(
            "    Operation_IFC {name}_inner <- mkRandomSelectGen(\"{filename}\");\n    Operation_IFC {name} <- mkDebugOperation({name}_inner, \"{name}\");\n",
            filename = "TODOFILLOUTTODO"
        ).as_str());
        Box::new(())
    }

    fn visit_scan(&mut self, node: &nodes::Scan) -> Box<dyn any::Any> {
        panic!("Not implemented");
    }

    fn visit_streamify(&mut self, node: &nodes::Streamify) -> Box<dyn any::Any> {
        panic!("Not implemented");
    }

    fn visit_tiled_retile_streamify(&mut self,node: &nodes::TiledRetileStreamify) -> Box<dyn any::Any> {
        let name = self.get_next_module_name(node.get_id());
        self.bluespec_code.push_str(format!(
            "    Operation_IFC {name}_inner <- mkTiledRetileStreamify({num_repeats}, {filter_mask}, {split_row});\n    Operation_IFC {name} <- mkDebugOperation({name}_inner, \"{name}\");\n",
            num_repeats = node.num_repeats,
            filter_mask = if node.filter_mask {"True"} else {"False"},
            split_row = if node.split_row {"True"} else {"False"}
        ).as_str());
        Box::new(())
    }

    fn visit_unzip(&mut self, node: &nodes::Unzip) -> Box<dyn any::Any> {
        panic!("Not implemented");
    }

    fn visit_zip(&mut self, node: &nodes::Zip) -> Box<dyn any::Any> {
        panic!("Not implemented");
    }

}