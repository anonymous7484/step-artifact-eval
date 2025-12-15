use std::collections::{HashMap, HashSet};
use std::fmt::Write;
use crate::hwsim::channel_id::ChannelID;
use crate::hwsim::graph::nodes::Node;

pub struct DotGenerator {
    channel_id_to_output: HashMap<ChannelID, usize>,
    channel_id_to_input: HashMap<ChannelID, usize>,
    graph: String,
    node_counter: usize,
    node_ids: HashMap<String, usize>,
    only_struct_names: bool
}

impl DotGenerator {
    pub fn new(only_struct_names: bool) -> Self {
        Self {
            graph: String::from("digraph G {\n"),
            node_counter: 0,
            node_ids: HashMap::new(),
            channel_id_to_input: HashMap::new(),
            channel_id_to_output: HashMap::new(),
            only_struct_names
        }
    }

    fn get_node_id(&mut self, label: &str) -> usize {
        // Assign a unique ID to each node label
        if let Some(&id) = self.node_ids.get(label) {
            id
        } else {
            let id = self.node_counter;
            self.node_counter += 1;
            self.node_ids.insert(label.to_string(), id);
            id
        }
    }

    fn add_node(&mut self, label: &str) -> usize {
        let id = self.get_node_id(label);
        // Add line breaks every 30 characters
        let mut label = label.to_string();
        if self.only_struct_names {
            label = label.split(" ").next().unwrap().to_string();
        }
        let label = label.chars()
        .collect::<Vec<_>>() 
        .chunks(30) 
        .map(|chunk| chunk.iter().collect::<String>()) 
        .collect::<Vec<_>>() 
        .join("\n");
        writeln!(self.graph, "  {} [label=\"{}\"];", id, label).unwrap();
        id
    }

    fn add_edge(&mut self, from_id: usize, to_id: usize, label: &str) {
        writeln!(self.graph, "  {} -> {} [label=\"{}\";]", from_id, to_id, label).unwrap();
    }

    fn get_edge_label(&mut self, edge_annotations: Option<&HashMap<ChannelID, String>>, id: ChannelID) -> String {
        if let Some(edge_annotations) = edge_annotations.as_ref() {
            {
                let id_str = id.to_string();
                let annotation = edge_annotations.get(&id).map_or_else(|| id_str.clone(), |s| s.clone());
                format!("{} \n{}", id_str, annotation)
            }
        } else {
            id.to_string()
        }
    }

    pub fn generate_dot_graph(&mut self, nodes: &[Box<dyn Node>], edge_annotations: Option<&HashMap<ChannelID, String>>) {
        // Visit each node to add it and its connections to the DOT graph
        for node in nodes {
            let cur_node_id = self.add_node(&format!("{:?}", node));

            node.get_inputs().iter().for_each(|input| {
                self.channel_id_to_input.insert(input.clone(), cur_node_id);
            });

            node.get_outputs().iter().for_each(|output| {
                self.channel_id_to_output.insert(output.clone(), cur_node_id);
            });
        }

        let mut connections = Vec::new();
        for (output, source_node) in self.channel_id_to_output.iter() {
            if let Some(sink_node) = self.channel_id_to_input.get(output) {
                connections.push((*source_node, *sink_node, *output));
            }
        }

        for (source, sink, id) in connections.iter() {
            let label = self.get_edge_label(edge_annotations, *id);
            self.add_edge(*source, *sink, label.as_str());
        }

        let used_channels = connections.iter().map(|x| x.2).collect::<HashSet<_>>();
        let all_inputs = self.channel_id_to_input.keys().map(|x| x.clone()).collect::<HashSet<_>>();
        let remaining_inputs = all_inputs.difference(&used_channels);
        let all_outputs = self.channel_id_to_output.keys().map(|x| x.clone()).collect::<HashSet<_>>();
        let remaining_outputs = all_outputs.difference(&used_channels);
        
        for input in remaining_inputs.into_iter() {
            let input_id = self.add_node(&format!("Input {:?}", input));
            let label = self.get_edge_label(edge_annotations, *input);
            self.add_edge(input_id, *self.channel_id_to_input.get(input).unwrap(), label.as_str());
        }

        for output in remaining_outputs.into_iter() {
            let output_id = self.add_node(&format!("Output {:?}", output));
            let label = self.get_edge_label(edge_annotations, *output);
            self.add_edge(*self.channel_id_to_output.get(output).unwrap(), output_id, label.as_str());
        }
        
        self.graph.push_str("}\n");
    }

    pub fn to_string(&self) -> String {
        self.graph.clone()
    }

    pub fn write_dot_graph(nodes: &[Box<dyn Node>], edge_annotations: Option<&HashMap<ChannelID, String>>, filename: &str, only_struct_names: bool) {
        let mut dot_gen = DotGenerator::new(only_struct_names);
        dot_gen.generate_dot_graph(&nodes, edge_annotations);
        std::fs::write(filename, dot_gen.to_string()).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use crate::hwsim::{graph::{function::TypelessFunction, nodes::{Accum, Broadcast}}, scalar::Scalar};
    use crate::hwsim::graph::nodes::Node;

    use super::*;
    
    #[test]
    fn dot_test() {
        let mut dot_gen = DotGenerator::new(true);
    
        let in_id = ChannelID::new();
        let out_id_1 = ChannelID::new();
        let out_id_2 = ChannelID::new();
        let intermed_id = ChannelID::new();

        // Assuming you have a vector of nodes
        let nodes: Vec<Box<dyn Node>> = vec![
            Box::new(Accum::new(vec![in_id], intermed_id, TypelessFunction { operations: vec![], n_inputs: 1, n_outputs: 1 }, 0, Scalar::I32(0))),
            Box::new(Broadcast::new(intermed_id, vec![out_id_1, out_id_2])),
            // ... other nodes
        ];
    
        dot_gen.generate_dot_graph(&nodes, None);
    
        // Print or save the DOT representation
        // Save the file to test.dot
        std::fs::write("test.dot", dot_gen.to_string()).unwrap();

        // println!("Dot graph: {}", dot_gen.to_string());
    }
}