#[cfg(test)]
mod step_paper {
    use std::{fs, io::{BufRead, BufReader, Write}, process::Command};

    use crate::hwsim::{channel_id::ChannelID, graph::{function::{BaseOp, Operation, ValueType}, nodes::{Accum, AddressReader, AddressWriter, Broadcast, Map, Node, Promote}}, passes::{convert_to_static_bluespec_pass::ConvertToStaticBluespecPass, downtiler::Downtiler}, scalar::Scalar};

    struct Result {
        cycles: usize,
        reads: usize,
        writes: usize,
    }

    impl std::process::Termination for Result {
        fn report(self) -> std::process::ExitCode {
            std::process::ExitCode::SUCCESS
        }
    }

    fn extract_numbers(line: &str) -> Vec<usize> {
        line
            .split(|c: char| !c.is_ascii_digit())
            .filter_map(|s| if s.is_empty() { None } else { s.parse::<usize>().ok() })
            .collect()
    }

    #[derive(Debug, Clone, Copy)]
    struct BuildMetrics {
        cycles: usize,
        reads: usize,
        writes: usize,
    }

    fn create_nodes(
        b: usize,
        dim: usize,
        inter_dim: usize,
        tile_b: usize,
        tile_inter: usize,
    ) -> Vec<Box<dyn Node>> {
        let offchip_load_0_data = ndarray::Array::from_shape_fn(
            (1, b / tile_b, inter_dim / tile_inter, 1, tile_b, dim),
            |(i, j, k, l, m, n)| {
                Scalar::FP32(i as f32 + j as f32 + k as f32 + l as f32 + m as f32 + n as f32)
            }
        ).into_dyn();
        let offchip_read_0_in = ChannelID::new();
        let offchip_read_0_out = ChannelID::new();
        let offchip_load_0 = AddressReader::new_with_data_and_tile_size( // this is not a transposed read
            offchip_read_0_in,
            vec![offchip_read_0_out],
            offchip_load_0_data,
            Downtiler::crop_shape(&vec![1, b / tile_b, inter_dim / tile_inter, 1, tile_b, dim]),
            vec![1, b / tile_b, inter_dim / tile_inter, 1]
            // vec![16, k, 1, 32, 4, 1]
        );

        let offchip_load_1_data = ndarray::Array::from_shape_fn(
            (1, b / tile_b, inter_dim / tile_inter, 1, tile_inter, dim),
            |(i, j, k, l, m, n)| {
                Scalar::FP32(i as f32 + j as f32 + k as f32 + l as f32 + m as f32 + n as f32)
            }
        ).into_dyn();
        let add_read_1_in = ChannelID::new();
        let add_read_1_out = ChannelID::new();
        let offchip_load_1 = AddressReader::new_with_data_and_tile_size( // this is a transposed read
            add_read_1_in,
            vec![add_read_1_out],
            offchip_load_1_data,
            // vec![16, k, 1, 32, 4, 1]
            Downtiler::crop_shape(&vec![1, b / tile_b, inter_dim / tile_inter, 1, tile_inter, dim]),
            vec![1, b / tile_b, inter_dim / tile_inter, 1 ]
        );

        let offchip_load_2_data = ndarray::Array::from_shape_fn(
            (1, b / tile_b, inter_dim / tile_inter, 1, tile_inter, dim),
            |(i, j, k, l, m, n)| {
                Scalar::FP32(i as f32 + j as f32 + k as f32 + l as f32 + m as f32 + n as f32)
            }
        ).into_dyn();
        let add_read_2_in = ChannelID::new();
        let add_read_2_out = ChannelID::new();
        let offchip_load_2 = AddressReader::new_with_data_and_tile_size( // this is a transposed read
            add_read_2_in,
            vec![add_read_2_out],
            offchip_load_2_data,
            // vec![16, k, 1, 32, 4, 1]
            Downtiler::crop_shape(&vec![1, b / tile_b, inter_dim / tile_inter, 1, tile_inter, dim]),
            vec![1, b / tile_b, inter_dim / tile_inter, 1]
        );

        let broadcast_3_out_1 = ChannelID::new();
        let broadcast_3_out_2 = ChannelID::new();
        let broadcast_3 = Broadcast::new_with_tile_size(
            offchip_read_0_out,
            vec![broadcast_3_out_1, broadcast_3_out_2],
            // vec![16, k, 1, 32, 4, 1]
            Downtiler::crop_shape(&vec![1, b / tile_b, inter_dim / tile_inter, 1, tile_b, dim])
        );

        let binary_map_3_out = ChannelID::new();
        let binary_map_3 = Map::new_with_tile_size(
            vec![broadcast_3_out_1, add_read_1_out],
            binary_map_3_out,
            crate::hwsim::graph::function::TypelessFunction { operations: vec![(Operation::MatmulT(BaseOp::Variable(0, ValueType::Float), BaseOp::Variable(1, ValueType::Float)), vec![0])], n_inputs: 2, n_outputs: 1 },
            // vec![16, 16, 32, 4, 1]
            Downtiler::crop_shape(&vec![1, b / tile_b, inter_dim / tile_inter, 1, tile_b, tile_inter])
        );
        let accum_3_out = ChannelID::new();
        let accum_3 = Accum::new_with_tile_size(
            vec![binary_map_3_out],
            accum_3_out,
            crate::hwsim::graph::function::TypelessFunction { operations: vec![(Operation::Add(BaseOp::Variable(0, ValueType::Float), BaseOp::Variable(1, ValueType::Float)), vec![0])], n_inputs: 2, n_outputs: 1 },
            1,
            Scalar::FP32(0.0),
            // vec![16, k, 1, 4, 1]
            Downtiler::crop_shape(&vec![1, b / tile_b, inter_dim / tile_inter, tile_b, tile_inter])
        );

        let binary_map_4_out = ChannelID::new();
        let binary_map_4 = Map::new_with_tile_size(
            vec![broadcast_3_out_2, add_read_2_out],
            binary_map_4_out,
            crate::hwsim::graph::function::TypelessFunction { operations: vec![(Operation::MatmulT(BaseOp::Variable(0, ValueType::Float), BaseOp::Variable(1, ValueType::Float)), vec![0])], n_inputs: 2, n_outputs: 1 },
            // vec![16, 16, 32, 4, 1]
            Downtiler::crop_shape(&vec![1, b / tile_b, inter_dim / tile_inter, 1, tile_b, tile_inter])
        );
        let accum_4_out = ChannelID::new();
        let accum_4 = Accum::new_with_tile_size(
            vec![binary_map_4_out],
            accum_4_out,
            crate::hwsim::graph::function::TypelessFunction { operations: vec![(Operation::Add(BaseOp::Variable(0, ValueType::Float), BaseOp::Variable(1, ValueType::Float)), vec![0])], n_inputs: 2, n_outputs: 1 },
            1,
            Scalar::FP32(0.0),
            // vec![16, k, 1, 4, 1]
            Downtiler::crop_shape(&vec![1, b / tile_b, inter_dim / tile_inter, tile_b, tile_inter])
        );

        let unary_map_5_out = ChannelID::new();
        let unary_map_5 = Map::new_with_tile_size(
            vec![accum_3_out],
            unary_map_5_out,
            crate::hwsim::graph::function::TypelessFunction { operations: vec![(Operation::Silu(BaseOp::Variable(0, ValueType::Float)), vec![0])], n_inputs: 1, n_outputs: 1 },
            Downtiler::crop_shape(&vec![1, b / tile_b, inter_dim / tile_inter, 1, tile_b, tile_inter])
        );

        let binary_map_6_out = ChannelID::new();
        let binary_map_6 = Map::new_with_tile_size(
            vec![unary_map_5_out, accum_4_out],
            binary_map_6_out,
            crate::hwsim::graph::function::TypelessFunction { operations: vec![(Operation::Mul(BaseOp::Variable(0, ValueType::Float), BaseOp::Variable(1, ValueType::Float)), vec![0])], n_inputs: 2, n_outputs: 1 },
            Downtiler::crop_shape(&vec![1, b / tile_b, inter_dim / tile_inter, 1, tile_b, tile_inter])
        );

        let promote_7_out = ChannelID::new();
        let promote_7 = Promote::new_with_tile_size(
            binary_map_6_out,
            promote_7_out,
            1,
            Downtiler::crop_shape(&vec![1, b / tile_b, 1, inter_dim / tile_inter, tile_b, tile_inter])
        );

        // TEMPORARY OUTPUT
        let output_8_out = ChannelID::new();
        let output_8 = AddressWriter::new_with_tile_size(
            vec![promote_7_out],
            output_8_out,
            Downtiler::crop_shape(&vec![1, b / tile_b, 1, dim, tile_b])
        );

        let offchip_load_8_data = ndarray::Array::from_shape_fn(
            (1, b / tile_b, 1, inter_dim / tile_inter, dim, tile_inter),
            |(i, j, k, l, m, n)| {
                Scalar::FP32(i as f32 + j as f32 + k as f32 + l as f32 + m as f32 + n as f32)
            }
        ).into_dyn();
        let offchip_load_8_in = ChannelID::new();
        let offchip_load_8_out = ChannelID::new();
        let offchip_load_8 = AddressReader::new_with_data_and_tile_size( // transposed read
            offchip_load_8_in,
            vec![offchip_load_8_out],
            offchip_load_8_data,
            // vec![16, k, 1, 32, 4, 1]
            Downtiler::crop_shape(&vec![1, b / tile_b, 1, inter_dim / tile_inter, dim, tile_inter]),
            vec![1, b / tile_b, 1, inter_dim / tile_inter]
        );

        let map_out = ChannelID::new();
        let map = Map::new_with_tile_size(
            vec![offchip_load_8_out, promote_7_out],
            map_out,
            crate::hwsim::graph::function::TypelessFunction { operations: vec![(Operation::MatmulT(BaseOp::Variable(0, ValueType::Float), BaseOp::Variable(1, ValueType::Float)), vec![0])], n_inputs: 2, n_outputs: 1 },
            // vec![16, k, 32, 1, 4, 1]
            Downtiler::crop_shape(&vec![1, b / tile_b, 1, inter_dim / tile_inter, tile_b, dim]),
        );

        let accum_out = ChannelID::new();
        let accum = Accum::new_with_tile_size(
            vec![map_out],
            accum_out,
            crate::hwsim::graph::function::TypelessFunction { operations: vec![(Operation::Add(BaseOp::Variable(0, ValueType::Float), BaseOp::Variable(1, ValueType::Float)), vec![0])], n_inputs: 2, n_outputs: 1 },
            1,
            Scalar::FP32(0.0),
            // vec![16, k, 1, 4, 1]
            Downtiler::crop_shape(&vec![1, b / tile_b, 1, tile_b, dim])
        );

        let output = AddressWriter::new_with_tile_size(
            vec![accum_out],
            ChannelID::new(),
            // vec![16, k, 1, 4, 1]
            Downtiler::crop_shape(&vec![1, b / tile_b, 1, tile_b, dim])
        );

        vec![
            Box::new(offchip_load_0),
            Box::new(offchip_load_1),
            Box::new(offchip_load_2),
            Box::new(broadcast_3),
            Box::new(binary_map_3),
            Box::new(accum_3),
            Box::new(binary_map_4),
            Box::new(accum_4),
            Box::new(unary_map_5),
            Box::new(binary_map_6),
            Box::new(promote_7),
            Box::new(offchip_load_8),
            Box::new(map),
            Box::new(accum),
            Box::new(output),
        ]
    }

    #[test]
    fn sweep() {
        // b, dim, inter_dim, tile_b, tile_inter
        let params = vec![
            (64, 256, 512, 16, 16), // 0
            (64, 256, 512, 16, 32), // 1
            (64, 256, 512, 16, 64), // 2
            (64, 256, 512, 16, 128), // 3
            (64, 256, 512, 16, 256), // 4
            (64, 256, 512, 32, 16), // 5
            (64, 256, 512, 32, 32), // 6
            (64, 256, 512, 32, 64), // 7
            (64, 256, 512, 32, 128), // 8
            (64, 256, 512, 32, 256), // 9
            (64, 256, 512, 64, 16), // 10
            (64, 256, 512, 64, 32), // 11
            (64, 256, 512, 64, 64), // 12
            (64, 256, 512, 64, 128), // 13
            (64, 256, 512, 64, 256), // 14
        ];

        for (idx, (b, dim, inter_dim, tile_b, tile_inter)) in params.iter().enumerate() {
            build_graph(idx, *b, *dim, *inter_dim, *tile_b, *tile_inter);
        }
    }

    fn build_graph(idx: usize, b: usize, dim: usize, inter_dim: usize, tile_b: usize, tile_inter: usize) {
        let hardware_tile_size = 16;

        let nodes = create_nodes(b, dim, inter_dim, tile_b, tile_inter);

        use crate::hwsim::dot::graph_dot::DotGenerator;

        // Print the graph before downtiling
        let mut dot_gen_before = DotGenerator::new(false);
        dot_gen_before.generate_dot_graph(&nodes, None);
        std::fs::write("step_before_downtiling.dot", dot_gen_before.to_string()).unwrap();
        let downtiled = Downtiler::downtile(nodes, hardware_tile_size);
        let mut dot_gen_after = DotGenerator::new(false);
        dot_gen_after.generate_dot_graph(&downtiled, None);
        std::fs::write("step_after_downtiling.dot", dot_gen_after.to_string()).unwrap();

        let mut bluespec_code = ConvertToStaticBluespecPass::new(hardware_tile_size, format!("../hwsim-bluespec/from_hwsim/"));
        bluespec_code.convert_to_bluespec(&downtiled);

        use std::process::Command;

        // Build the generated Bluespec code by running the shell command.
        // This assumes the current working directory is the crate root.
        let topfile = format!("step_paper_{idx}/Step.bsv");
        let topmodule = "mkStep";
        let status = Command::new("sh")
            .arg("-c")
            .arg(format!(
                "cd ../hwsim-bluespec && make b_all -j2 TOPFILE={} TOPMODULE={} > ../hwsim-bluespec/step_paper_{idx}.log",
                topfile, topmodule
            ))
            .status()
            .expect("Failed to execute make for Bluespec build");

        println!("Bluespec build exited with status: {}", status);

        // let program_builder = ProgramBuilder::default();
        // let (program_builder, instructions, senders, receivers) = to_hardware_and_simulate(program_builder, downtiled, hardware_tile_size);

        // println!("instructions: {:?}", instructions);
        // println!("senders: {:?}", senders.len());
        // println!("receivers: {:?}", receivers.len());

        // run_sim!(program_builder);
    }

    fn build_graph_for_params(tile_b: usize, tile_inter: usize) -> BuildMetrics {
        let b = 64;
        let dim = 256;
        let inter_dim = 512;
        let hardware_tile_size = 16;

        let nodes = create_nodes(b, dim, inter_dim, tile_b, tile_inter);

        use crate::hwsim::dot::graph_dot::DotGenerator;

        // Print the graph before downtiling
        let mut dot_gen_before = DotGenerator::new(false);
        dot_gen_before.generate_dot_graph(&nodes, None);
        fs::write("step_before_downtiling.dot", dot_gen_before.to_string()).unwrap();
        let downtiled = Downtiler::downtile(nodes, hardware_tile_size);
        let mut dot_gen_after = DotGenerator::new(false);
        dot_gen_after.generate_dot_graph(&downtiled, None);
        fs::write("step_after_downtiling.dot", dot_gen_after.to_string()).unwrap();

        let mut bluespec_code =
            ConvertToStaticBluespecPass::new(hardware_tile_size, "bluespec/from_hwsim".to_string());
        bluespec_code.convert_to_bluespec(&downtiled);

        let build_output_path = "bluespec/build_output.log";
        let make_output = Command::new("make")
            .current_dir("bluespec")
            .env("LC_ALL", "C")
            .env("LANG", "C")
            .arg("b_all")
            .arg("TOPFILE=from_hwsim/Step.bsv")
            .arg("TOPMODULE=mkStep")
            .output()
            .expect("failed to run make for mkStep");

        let mut combined_output = make_output.stdout;
        combined_output.extend_from_slice(&make_output.stderr);
        fs::write(build_output_path, &combined_output).expect("failed to write build output");
        assert!(make_output.status.success(), "make b_all failed, check {}", build_output_path);

        let output_file = fs::File::open(build_output_path).expect("unable to open build output");
        let lines: Vec<String> = BufReader::new(output_file)
            .lines()
            .filter_map(std::result::Result::ok)
            .collect();

        let mut reads_writes: Option<(usize, usize)> = None;
        let mut cycles: Option<usize> = None;

        for line in lines.iter().rev() {
            if cycles.is_none() && line.contains("Finished at cycle") {
                cycles = extract_numbers(line).first().copied();
            }

            if reads_writes.is_none() && line.contains("Ramulator: Num_Reads") {
                let nums = extract_numbers(line);
                if nums.len() >= 2 {
                    reads_writes = Some((nums[0], nums[1]));
                }
            }

            if reads_writes.is_some() && cycles.is_some() {
                break;
            }
        }

        let (reads, writes) = reads_writes.expect("Ramulator line missing read/write counts");
        let cycles = cycles.expect("missing cycle count in build output");

        BuildMetrics { cycles, reads, writes }
    }

    /// Run a simple design-space sweep over tile_b and tile_inter and
    /// dump the results to CSV for later plotting.
    ///
    /// CSV schema (HDL measurements):
    /// tile_b,dim,tile_inter,cycles (HDL),off_chip_mem_traffic(MB) (HDL)
    /// where off_chip_mem_traffic(MB) assumes 512 bytes per off-chip access
    /// (aligned with the Bluespec/Ramulator instrumentation used here).
    #[test]
    pub fn run_dse_sweep() {
        let tile_b_vals = [16_usize, 32, 64];
        let tile_inter_vals = [16_usize, 32, 64, 128, 256];
        let mut file = fs::File::create("dse_results.csv").expect("unable to create dse_results.csv");
        writeln!(
            &mut file,
            "tile_b,dim,tile_inter,cycles (HDL),off_chip_mem_traffic(MB) (HDL)"
        )
        .unwrap();

        for tb in tile_b_vals {
            for ti in tile_inter_vals {
                println!("Building graph for tile_b = {}, tile_inter = {}, dim = 256", tb, ti);
                let res = build_graph_for_params(tb, ti);
                let traffic_mb = ((res.reads + res.writes) as f64 * 512.0) / 1_000_000.0;
                writeln!(
                    &mut file,
                    "{},{},{},{},{}",
                    tb, 256, ti, res.cycles, traffic_mb
                )
                .expect("unable to write CSV row");
            }
        }
    }
}