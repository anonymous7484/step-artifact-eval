use std::{any::Any, fmt::Debug};

use ndarray::ArrayD;

use crate::hwsim::{channel_id::ChannelID, graph::function::TypelessFunction, scalar::Scalar, types::Tile};

use super::node_visitor::NodeVisitor;
use std::sync::atomic::{AtomicUsize, Ordering};

pub type EdgeID = ChannelID;
pub type NodeID = usize;

// Each node has its ID, ID's of cloned nodes are equal.
macro_rules! impl_default_functions {
    ($visit_method:ident) => {
        fn accept(&self, visitor: &mut dyn NodeVisitor) -> Box<dyn Any> {
            visitor.$visit_method(self)
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }

        fn get_id(&self) -> NodeID {
            self.id
        }

        fn set_tile_size(&mut self, tile_size: [usize; 2]) {
            self.tile_size = Some(tile_size);
        }
    };
}

macro_rules! rewrite_vec {
    ($target:expr, $old_source:expr, $new_sources:expr) => {
        {
            let mut success = false;
            $target = $target.iter().flat_map(|&input| {
                if input == $old_source {
                    success = true;
                    $new_sources.clone()
                } else { vec![input] }
            }).collect::<Vec<_>>();
            success
        }
    };
}

macro_rules! rewrite_port {
    ($target:expr, $old_source:expr, $new_sources:expr) => {
        {
            let success = ($target == $old_source);
            if success {
                assert!($new_sources.len() == 1, "Expected exactly one new port, found ports {:?} (len {})", $new_sources, $new_sources.len());
                $target = $new_sources[0];
            }
            success
        }
    };
}

macro_rules! rewrite_optional_port {
    ($target:expr, $old_source:expr, $new_sources:expr) => {
        {
            let success = $target.map_or(false, |t| t == $old_source);
            if success {
                assert!($new_sources.len() == 1);
                $target = Some($new_sources[0]);
            }
            success
        }
    };
}

static GLOBAL_ID_GENERATOR: AtomicUsize = AtomicUsize::new(0);
pub fn next_id() -> NodeID {
    GLOBAL_ID_GENERATOR.fetch_add(1, Ordering::SeqCst)
}

pub trait Node: Any + Debug {
    fn accept(&self, visitor: &mut dyn NodeVisitor) -> Box<dyn Any>;
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn get_inputs(&self) -> Vec<EdgeID>;
    fn get_outputs(&self) -> Vec<EdgeID>;
    fn rewrite_source(&mut self, _old_source: EdgeID, _new_sources: Vec<EdgeID>) -> bool{
        panic!("Node {:?} does not support source rewrite", self);
    }
    fn rewrite_target(&mut self, _old_target: EdgeID, _new_targets: Vec<EdgeID>) -> bool {
        panic!("Node {:?} does not support target rewrite", self);
    }
    fn get_id(&self) -> usize;
    fn get_tile_size(&self) -> Option<[usize; 2]>;
    fn set_tile_size(&mut self, tile_size: [usize; 2]);
}

#[derive(Clone)]
pub struct AddressReader {
    pub input: EdgeID,
    pub outputs: Vec<EdgeID>,
    pub id: usize,
    pub data: Option<ArrayD<Scalar>>,
    pub tile_size: Option<[usize; 2]>,
    pub transposed_read: bool,
    pub shape: Vec<usize>,
}

impl AddressReader {
    pub fn new(input: EdgeID, outputs: Vec<EdgeID>) -> Self {
        Self { input, outputs, id: next_id(), data: None, tile_size: None, transposed_read: false, shape: Vec::new() }
    }

    pub fn new_with_data(input: EdgeID, outputs: Vec<EdgeID>, data: ArrayD<Scalar>) -> Self {
        Self { input, outputs, id: next_id(), data: Some(data), tile_size: None, transposed_read: false, shape: Vec::new() }
    }

    pub fn new_with_tile_size(input: EdgeID, outputs: Vec<EdgeID>, tile_size: [usize; 2], shape: Vec<usize>) -> Self {
        Self { input, outputs, id: next_id(), data: None, tile_size: Some(tile_size), transposed_read: false, shape }
    }

    pub fn new_with_data_and_tile_size(input: EdgeID, outputs: Vec<EdgeID>, data: ArrayD<Scalar>, tile_size: [usize; 2], shape: Vec<usize>) -> Self {
        Self { input, outputs, id: next_id(), data: Some(data), tile_size: Some(tile_size), transposed_read: false, shape }
    }
}

impl Node for AddressReader {
    impl_default_functions!(visit_address_reader);

    fn get_inputs(&self) -> Vec<EdgeID> {
        vec![self.input]
    }
    
    fn get_outputs(&self) -> Vec<EdgeID> {
        self.outputs.clone()
    }

    fn rewrite_source(&mut self, old_source: EdgeID, new_sources: Vec<EdgeID>) -> bool {
        rewrite_port!(self.input, old_source, new_sources)
    }

    fn rewrite_target(&mut self, old_target: EdgeID, new_targets: Vec<EdgeID>) -> bool {
        rewrite_vec!(self.outputs, old_target, new_targets)
    }

    fn get_tile_size(&self) -> Option<[usize; 2]> {
        self.tile_size.clone()
    }
}

impl Debug for AddressReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AddressReader {{ input: {:?}, outputs: {:?}, id: {}, data: {:?} }}", self.input, self.outputs, self.id, self.data.as_ref().map(|x| x.shape()))
    }
}

#[derive(Debug, Clone)]
pub struct AddressWriter {
    pub inputs: Vec<EdgeID>,
    pub output: EdgeID,
    pub id: usize,
    pub data: Option<ArrayD<Scalar>>,
    pub tile_size: Option<[usize; 2]>,
}

impl AddressWriter {
    pub fn new(inputs: Vec<EdgeID>, output: EdgeID) -> Self {
        Self { inputs, output, id: next_id(), data: None, tile_size: None }
    }

    pub fn new_with_data(inputs: Vec<EdgeID>, output: EdgeID, data: ArrayD<Scalar>) -> Self {
        Self { inputs, output, id: next_id(), data: Some(data), tile_size: None }
    }

    pub fn new_with_tile_size(inputs: Vec<EdgeID>, output: EdgeID, tile_size: [usize; 2]) -> Self {
        Self { inputs, output, id: next_id(), data: None, tile_size: Some(tile_size) }
    }

    pub fn new_with_data_and_tile_size(inputs: Vec<EdgeID>, output: EdgeID, data: ArrayD<Scalar>, tile_size: [usize; 2]) -> Self {
        Self { inputs, output, id: next_id(), data: Some(data), tile_size: Some(tile_size) }
    }
}

impl Node for AddressWriter {
    impl_default_functions!(visit_address_writer);

    fn get_inputs(&self) -> Vec<EdgeID> {
        self.inputs.clone()
    }
    
    fn get_outputs(&self) -> Vec<EdgeID> {
        vec![self.output]
    }

    fn get_tile_size(&self) -> Option<[usize; 2]> {
        self.tile_size.clone()
    }
}

// This one is used in addition to a bufferize to accumulate big tiles.
#[derive(Debug, Clone)]
pub struct AccumBigTile {
    pub input_in: EdgeID,
    pub input_partial_sum: EdgeID,
    pub output_partial_sum: EdgeID,
    pub sum: EdgeID,
    pub fold: TypelessFunction,
    pub rank: usize,
    pub id: usize,
    pub tile_size: Option<[usize; 2]>,
}

impl AccumBigTile {
    pub fn new(input_in: EdgeID, input_partial_sum: EdgeID, output_partial_sum: EdgeID, sum: EdgeID, fold: TypelessFunction, rank: usize) -> Self {
        Self { input_in, input_partial_sum, output_partial_sum, sum, fold, rank, id: next_id(), tile_size: None }
    }

    pub fn new_with_tile_size(input_in: EdgeID, input_partial_sum: EdgeID, output_partial_sum: EdgeID, sum: EdgeID, fold: TypelessFunction, rank: usize, tile_size: [usize; 2]) -> Self {
        Self { input_in, input_partial_sum, output_partial_sum, sum, fold, rank, id: next_id(), tile_size: Some(tile_size) }
    }
}

impl Node for AccumBigTile {
    impl_default_functions!(visit_accum_big_tile);

    fn get_inputs(&self) -> Vec<EdgeID> {
        vec![self.input_in, self.input_partial_sum]
    }

    fn get_outputs(&self) -> Vec<EdgeID> {
        vec![self.output_partial_sum, self.sum]
    }

    fn rewrite_source(&mut self, old_source: EdgeID, new_sources: Vec<EdgeID>) -> bool {
        rewrite_port!(self.input_in, old_source, new_sources) || 
        rewrite_port!(self.input_partial_sum, old_source, new_sources)
    }

    fn rewrite_target(&mut self, old_target: EdgeID, new_targets: Vec<EdgeID>) -> bool {
        rewrite_port!(self.output_partial_sum, old_target, new_targets) || 
        rewrite_port!(self.sum, old_target, new_targets)
    }

    fn get_tile_size(&self) -> Option<[usize; 2]> {
        self.tile_size.clone()
    }
}

#[derive(Debug, Clone)]
pub struct Accum {
    pub inputs: Vec<EdgeID>,
    pub output: EdgeID,
    pub fold: TypelessFunction,
    pub rank: usize,
    pub init: Scalar,
    pub id: usize,
    pub tile_size: Option<[usize; 2]>,
}

impl Accum {
    pub fn new(inputs: Vec<EdgeID>, output: EdgeID, fold: TypelessFunction, rank: usize, init: Scalar) -> Self {
        Self { inputs, output, fold, rank, init, id: next_id(), tile_size: None }
    }

    pub fn new_with_tile_size(inputs: Vec<EdgeID>, output: EdgeID, fold: TypelessFunction, rank: usize, init: Scalar, tile_size: [usize; 2]) -> Self {
        Self { inputs, output, fold, rank, init, id: next_id(), tile_size: Some(tile_size) }
    }
}

impl Node for Accum {
    impl_default_functions!(visit_accum);
        
    fn get_inputs(&self) -> Vec<EdgeID> {
        self.inputs.clone()
    }
    
    fn get_outputs(&self) -> Vec<EdgeID> {
        vec![self.output]
    }
    
    fn rewrite_source(&mut self, old_source: EdgeID, new_sources: Vec<EdgeID>) -> bool {
        rewrite_vec!(self.inputs, old_source, new_sources)
    }

    fn rewrite_target(&mut self, old_target: EdgeID, new_targets: Vec<EdgeID>) -> bool {
        rewrite_port!(self.output, old_target, new_targets)
    }

    fn get_tile_size(&self) -> Option<[usize; 2]> {
        self.tile_size.clone()
    }
}

#[derive(Debug, Clone)]
pub struct Broadcast {
    pub input: EdgeID,
    pub outputs: Vec<EdgeID>,
    pub id: usize,
    pub tile_size: Option<[usize; 2]>,
}

impl Broadcast {
    pub fn new(input: EdgeID, outputs: Vec<EdgeID>) -> Self {
        Self { input, outputs, id: next_id(), tile_size: None }
    }

    pub fn new_with_tile_size(input: EdgeID, outputs: Vec<EdgeID>, tile_size: [usize; 2]) -> Self {
        Self { input, outputs, id: next_id(), tile_size: Some(tile_size) }
    }
}

impl Node for Broadcast {
    impl_default_functions!(visit_broadcast);

    fn get_inputs(&self) -> Vec<EdgeID> {
        vec![self.input]
    }
    
    fn get_outputs(&self) -> Vec<EdgeID> {
        self.outputs.clone()
    }

    fn rewrite_source(&mut self, old_source: EdgeID, new_sources: Vec<EdgeID>) -> bool {
        rewrite_port!(self.input, old_source, new_sources)
    }

    fn rewrite_target(&mut self, old_target: EdgeID, new_targets: Vec<EdgeID>) -> bool {
        rewrite_vec!(self.outputs, old_target, new_targets)
    }

    fn get_tile_size(&self) -> Option<[usize; 2]> {
        self.tile_size.clone()
    }
}

#[derive(Debug, Clone)]
pub struct Bufferize {
    pub data_inputs: Vec<EdgeID>,
    pub token_outputs: Vec<EdgeID>,
    pub token_requests: Vec<EdgeID>,
    pub data_outputs: Vec<EdgeID>,
    pub rank: usize,
    pub size: Option<usize>,
    pub permute: Vec<usize>,
    pub transpose: bool,
    pub id: usize,
    pub tile_size: Option<[usize; 2]>
}

impl Bufferize {
    pub fn new(input: EdgeID, output: EdgeID, rank: usize) -> Self {
        Self { data_inputs: vec![input], token_outputs: vec![output], 
            token_requests: vec![], data_outputs: vec![], 
            rank: rank, permute: Vec::new(), size: None, id: next_id(), transpose: false, tile_size: None }
    }

    pub fn new_with_tile_size(input: EdgeID, output: EdgeID, rank: usize, tile_size: [usize; 2]) -> Self {
        Self { data_inputs: vec![input], token_outputs: vec![output], 
            token_requests: vec![], data_outputs: vec![], 
            rank: rank, permute: Vec::new(), size: None, id: next_id(), transpose: false, tile_size: Some(tile_size) }
    }

    pub fn new_inlined_streamify(
        inputs: Vec<EdgeID>, 
        token_outputs: Vec<EdgeID>, 
        token_requests: Vec<EdgeID>,
        data_outputs: Vec<EdgeID>,
        rank: usize,
        permute: Vec<usize>,
        transpose: bool,
        size: Option<usize>,
        id: usize,
    ) -> Self {
        Self { data_inputs: inputs, token_outputs: token_outputs, 
            token_requests, data_outputs, rank, permute, size, id: id, transpose, tile_size: None }
    }

    pub fn new_inlined_streamify_with_tile_size(
        inputs: Vec<EdgeID>, 
        token_outputs: Vec<EdgeID>, 
        token_requests: Vec<EdgeID>,
        data_outputs: Vec<EdgeID>,
        rank: usize,
        permute: Vec<usize>,
        transpose: bool,
        size: Option<usize>,
        id: usize,
        tile_size: [usize; 2],
    ) -> Self {
        Self { data_inputs: inputs, token_outputs: token_outputs, 
            token_requests, data_outputs, rank, permute, size, id: id, transpose, tile_size: Some(tile_size) }
    }
}

impl Node for Bufferize {
    impl_default_functions!(visit_bufferize);
    
    fn get_inputs(&self) -> Vec<EdgeID> {
        self.data_inputs.iter().cloned().chain(self.token_requests.iter().cloned()).collect()
    }
    
    fn get_outputs(&self) -> Vec<EdgeID> {
        self.token_outputs.iter().cloned().chain(self.data_outputs.iter().cloned()).collect()
    }

    fn rewrite_source(&mut self, old_source: EdgeID, new_sources: Vec<EdgeID>) -> bool {
        [
            rewrite_vec!(self.data_inputs, old_source, new_sources),
            rewrite_vec!(self.token_requests, old_source, new_sources),
        ].iter().any(|f| *f)
    }

    fn rewrite_target(&mut self, old_target: EdgeID, new_targets: Vec<EdgeID>) -> bool {
        [
            rewrite_vec!(self.data_outputs, old_target, new_targets),
            rewrite_vec!(self.token_outputs, old_target, new_targets),
        ].iter().any(|f| *f)
    }

    fn get_tile_size(&self) -> Option<[usize; 2]> {
        self.tile_size.clone()
    }
}

#[derive(Debug, Clone)]
pub struct Concatenate {
    pub inputs: Vec<EdgeID>,
    pub output: EdgeID,
    pub rank: usize,
    pub id: usize,
    pub tile_size: Option<[usize; 2]>,
}

impl Concatenate {
    pub fn new(inputs: Vec<EdgeID>, output: EdgeID, rank: usize) -> Self {
        Self { inputs, output, rank, id: next_id(), tile_size: None }
    }

    pub fn new_with_tile_size(inputs: Vec<EdgeID>, output: EdgeID, rank: usize, tile_size: [usize; 2]) -> Self {
        Self { inputs, output, rank, id: next_id(), tile_size: Some(tile_size) }
    }
}

impl Node for Concatenate {
    impl_default_functions!(visit_concatenate);
    
    fn get_inputs(&self) -> Vec<EdgeID> {
        self.inputs.clone()
    }
    
    fn get_outputs(&self) -> Vec<EdgeID> {
        vec![self.output]
    }

    fn rewrite_source(&mut self, old_source: EdgeID, new_sources: Vec<EdgeID>) -> bool {
        rewrite_vec!(self.inputs, old_source, new_sources)
    }

    fn rewrite_target(&mut self, old_target: EdgeID, new_targets: Vec<EdgeID>) -> bool {
        rewrite_port!(self.output, old_target, new_targets)
    }

    fn get_tile_size(&self) -> Option<[usize; 2]> {
        self.tile_size.clone()
    }
}

#[derive(Debug, Clone)]
pub struct DynOffChipLoad {
    pub input: EdgeID,
    pub output: EdgeID,
    pub tile_size: Option<[usize; 2]>,
    pub id: usize,
    pub transposed_read: bool,
    pub shape: Vec<usize>,
}

impl DynOffChipLoad {
    pub fn new(input: EdgeID, output: EdgeID, id: usize) -> Self {
        Self { input, output, id, tile_size: None, transposed_read: false, shape: Vec::new() }
    }

    pub fn new_with_tile_size(input: EdgeID, output: EdgeID, id: usize, tile_size: [usize; 2], shape: Vec<usize>) -> Self {
        Self { input, output, id, tile_size: Some(tile_size), transposed_read: false, shape }
    }
}

impl Node for DynOffChipLoad {
    impl_default_functions!(visit_dyn_off_chip_load);

    fn get_inputs(&self) -> Vec<EdgeID> {
        vec![self.input]
    }
    
    fn get_outputs(&self) -> Vec<EdgeID> {
        vec![self.output]
    }

    fn rewrite_source(&mut self, old_source: EdgeID, new_sources: Vec<EdgeID>) -> bool {
        rewrite_port!(self.input, old_source, new_sources)
    }

    fn rewrite_target(&mut self, old_target: EdgeID, new_targets: Vec<EdgeID>) -> bool {
        rewrite_port!(self.output, old_target, new_targets)
    }

    fn get_tile_size(&self) -> Option<[usize; 2]> {
        self.tile_size.clone()
    }
}

#[derive(Debug, Clone)]
pub struct Enumerate {
    pub inputs: Vec<EdgeID>,
    pub outputs: Vec<EdgeID>,
    pub rank: usize,
    pub id: usize,
    pub tile_size: Option<[usize; 2]>,
}

impl Enumerate {
    pub fn new(inputs: Vec<EdgeID>, output: EdgeID, rank: usize) -> Self {
        Self { inputs, outputs: vec![output], rank, id: next_id(), tile_size: None }
    }

    pub fn new_with_tile_size(inputs: Vec<EdgeID>, output: EdgeID, rank: usize, tile_size: [usize; 2]) -> Self {
        Self { inputs, outputs: vec![output], rank, id: next_id(), tile_size: Some(tile_size) }
    }
}

impl Node for Enumerate {
    impl_default_functions!(visit_enumerate);

    fn get_inputs(&self) -> Vec<EdgeID> {
        self.inputs.clone()
    }
    
    fn get_outputs(&self) -> Vec<EdgeID> {
        self.outputs.clone()
    }

    fn rewrite_source(&mut self, old_source: EdgeID, new_sources: Vec<EdgeID>) -> bool {
        let b = rewrite_vec!(self.inputs, old_source, new_sources);
        b
    }
    
    fn rewrite_target(&mut self, old_target: EdgeID, new_targets: Vec<EdgeID>) -> bool {
        let b = rewrite_vec!(self.outputs, old_target, new_targets);
        b
    }

    fn get_tile_size(&self) -> Option<[usize; 2]> {
        self.tile_size.clone()
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct FlatMap {
    pub in_stream: Option<EdgeID>,
    pub fn_send: Option<EdgeID>,
    pub fn_rcv: EdgeID,
    pub out_stream: EdgeID,
    pub rank: usize,
    pub id: usize,
    pub tile_size: Option<[usize; 2]>,
}

impl FlatMap {
    pub fn new(in_stream: EdgeID, fn_send: EdgeID, fn_rcv: EdgeID,
        out_stream: EdgeID, rank: usize) -> Self {
        Self { in_stream: Some(in_stream), fn_send: Some(fn_send), 
            fn_rcv, out_stream, rank, id: next_id(), tile_size: None }
    }

    pub fn new_no_send(fn_rcv: EdgeID, out_stream: EdgeID, rank: usize) -> Self {
        Self { in_stream: None, fn_send: None, fn_rcv, out_stream, rank, id: next_id(), tile_size: None }
    }

    pub fn new_with_tile_size(in_stream: EdgeID, fn_send: EdgeID, fn_rcv: EdgeID,
        out_stream: EdgeID, rank: usize, tile_size: [usize; 2]) -> Self {
        Self { in_stream: Some(in_stream), fn_send: Some(fn_send), 
            fn_rcv, out_stream, rank, id: next_id(), tile_size: Some(tile_size) }
    }

    pub fn new_no_send_with_tile_size(fn_rcv: EdgeID, out_stream: EdgeID, rank: usize, tile_size: [usize; 2]) -> Self {
        Self { in_stream: None, fn_send: None, fn_rcv, out_stream, rank, id: next_id(), tile_size: Some(tile_size) }
    }
}

impl Node for FlatMap {
    impl_default_functions!(visit_flatmap);
        
    fn get_inputs(&self) -> Vec<EdgeID> {
        let mut vec = Vec::new();
        if let Some(in_stream) = self.in_stream {
            vec.push(in_stream);
        }
        vec.push(self.fn_rcv);
        vec
    }
    
    fn get_outputs(&self) -> Vec<EdgeID> {
        let mut vec = Vec::new();
        if let Some(fn_send) = self.fn_send {
            vec.push(fn_send);
        }
        vec.push(self.out_stream);
        vec
    }
    
    fn rewrite_source(&mut self, old_source: EdgeID, new_sources: Vec<EdgeID>) -> bool {
        rewrite_optional_port!(self.in_stream, old_source, new_sources) || 
        rewrite_port!(self.fn_rcv, old_source, new_sources)
    }

    fn rewrite_target(&mut self, old_target: EdgeID, new_targets: Vec<EdgeID>) -> bool {
        rewrite_optional_port!(self.fn_send, old_target, new_targets) ||
        rewrite_port!(self.out_stream, old_target, new_targets)
    }

    fn get_tile_size(&self) -> Option<[usize; 2]> {
        self.tile_size.clone()
    }
}

#[derive(Debug, Clone)]
pub struct Flatten {
    pub input: EdgeID,
    pub output: EdgeID,
    pub ranks: Vec<usize>,
    pub id: usize,
    pub tile_size: Option<[usize; 2]>,
}

impl Flatten {
    pub fn new(input: EdgeID, output: EdgeID, ranks: Vec<usize>) -> Self {
        Self { input, output, ranks, id: next_id(), tile_size: None }
    }

    pub fn new_with_tile_size(input: EdgeID, output: EdgeID, ranks: Vec<usize>, tile_size: [usize; 2]) -> Self {
        Self { input, output, ranks, id: next_id(), tile_size: Some(tile_size) }
    }
}

impl Node for Flatten {
    impl_default_functions!(visit_flatten);

    fn get_inputs(&self) -> Vec<EdgeID> {
        vec![self.input]
    }
    
    fn get_outputs(&self) -> Vec<EdgeID> {
        vec![self.output]
    }

    fn rewrite_source(&mut self, old_source: EdgeID, new_sources: Vec<EdgeID>) -> bool {
        rewrite_port!(self.input, old_source, new_sources)
    }

    fn rewrite_target(&mut self, old_target: EdgeID, new_targets: Vec<EdgeID>) -> bool {
        rewrite_port!(self.output, old_target, new_targets)
    }

    fn get_tile_size(&self) -> Option<[usize; 2]> {
        self.tile_size.clone()
    }
}

#[derive(Debug, Clone)]
pub struct FnBlock {
    pub input: EdgeID,
    pub output: EdgeID,
    pub func: TypelessFunction,
    pub rank: usize,
    pub id: usize,
    pub tile_size: Option<[usize; 2]>,
}

impl FnBlock {
    pub fn new(input: EdgeID, output: EdgeID, func: TypelessFunction, rank: usize) -> Self {
        Self { input, output, func, rank, id: next_id(), tile_size: None }
    }

    pub fn new_with_tile_size(input: EdgeID, output: EdgeID, func: TypelessFunction, rank: usize, tile_size: [usize; 2]) -> Self {
        Self { input, output, func, rank, id: next_id(), tile_size: Some(tile_size) }
    }
}

impl Node for FnBlock {
    impl_default_functions!(visit_fnblock);
        
    fn get_inputs(&self) -> Vec<EdgeID> {
        vec![self.input]
    }
    
    fn get_outputs(&self) -> Vec<EdgeID> {
        vec![self.output]
    }

    fn rewrite_source(&mut self, old_source: EdgeID, new_sources: Vec<EdgeID>) -> bool {
        rewrite_port!(self.input, old_source, new_sources)
    }

    fn rewrite_target(&mut self, old_target: EdgeID, new_targets: Vec<EdgeID>) -> bool {
        rewrite_port!(self.output, old_target, new_targets)
    }

    fn get_tile_size(&self) -> Option<[usize; 2]> {
        self.tile_size.clone()
    }
}

#[derive(Debug, Clone)]
pub struct Filter {
    pub inputs: Vec<EdgeID>,
    pub output: EdgeID,
    pub func: TypelessFunction,
    pub id: usize,
    pub tile_size: Option<[usize; 2]>,
}

impl Filter {
    pub fn new(input: Vec<EdgeID>, output: EdgeID, func: TypelessFunction) -> Self {
        Self { inputs: input, output, func, id: next_id(), tile_size: None }
    }

    pub fn new_with_tile_size(input: Vec<EdgeID>, output: EdgeID, func: TypelessFunction, tile_size: [usize; 2]) -> Self {
        Self { inputs: input, output, func, id: next_id(), tile_size: Some(tile_size) }
    }
}

impl Node for Filter {

    impl_default_functions!(visit_filter);
        
    fn get_inputs(&self) -> Vec<EdgeID> {
        self.inputs.clone()
    }
    
    fn get_outputs(&self) -> Vec<EdgeID> {
        vec![self.output]
    }

    fn rewrite_source(&mut self, old_source: EdgeID, new_sources: Vec<EdgeID>) -> bool {
        let mut found = false;

        self.inputs = self.inputs.iter().flat_map(|&input| {
            if input == old_source {
                found = true;
                new_sources.clone()
            } else { vec![input] }
        }).collect();
        return found;
    }

    fn get_tile_size(&self) -> Option<[usize; 2]> {
        self.tile_size.clone()
    }
}

#[derive(Debug, Clone)]
pub struct Map {
    pub inputs: Vec<EdgeID>,
    pub output: EdgeID,
    pub func: TypelessFunction,
    pub id: usize,
    pub tile_size: Option<[usize; 2]>,
}

impl Map {
    pub fn new(input: Vec<EdgeID>, output: EdgeID, func: TypelessFunction) -> Self {
        Self { inputs: input, output, func, id: next_id(), tile_size: None }
    }

    pub fn new_with_tile_size(input: Vec<EdgeID>, output: EdgeID, func: TypelessFunction, tile_size: [usize; 2]) -> Self {
        Self { inputs: input, output, func, id: next_id(), tile_size: Some(tile_size) }
    }
}

impl Node for Map {
    impl_default_functions!(visit_map);

    fn get_inputs(&self) -> Vec<EdgeID> {
        self.inputs.clone()
    }
    
    fn get_outputs(&self) -> Vec<EdgeID> {
        vec![self.output]
    }

    fn rewrite_source(&mut self, old_source: EdgeID, new_sources: Vec<EdgeID>) -> bool {
        rewrite_vec!(self.inputs, old_source, new_sources)
    }

    fn rewrite_target(&mut self, old_target: EdgeID, new_targets: Vec<EdgeID>) -> bool {
        rewrite_port!(self.output, old_target, new_targets)
    }

    fn get_tile_size(&self) -> Option<[usize; 2]> {
        self.tile_size.clone()
    }
}

#[derive(Debug, Clone)]
pub struct MatmulT {
    pub a_in: EdgeID,
    pub a_out: Option<EdgeID>,
    pub psum_in: Option<EdgeID>,
    pub psum_out: EdgeID,
    pub b_in: EdgeID,
    pub b_out: Option<EdgeID>,
    pub row_position: usize,
    pub num_rows: usize,
    pub bottom_right: bool,
    pub id: usize,
    pub tile_size: Option<[usize; 2]>,
}

impl MatmulT {
    pub fn new(a_in: EdgeID, a_out: Option<EdgeID>, psum_in: Option<EdgeID>, psum_out: EdgeID, b_in: EdgeID, b_out: Option<EdgeID>, row_position: usize, num_rows: usize, bottom_right: bool) -> Self {
        Self { a_in, a_out, psum_in, psum_out, b_in, b_out, row_position, num_rows, bottom_right, id: next_id(), tile_size: None }
    }

    pub fn new_with_tile_size(a_in: EdgeID, a_out: Option<EdgeID>, psum_in: Option<EdgeID>, psum_out: EdgeID, b_in: EdgeID, b_out: Option<EdgeID>, row_position: usize, num_rows: usize, bottom_right: bool, tile_size: [usize; 2]) -> Self {
        Self { a_in, a_out, psum_in, psum_out, b_in, b_out, row_position, num_rows, bottom_right, id: next_id(), tile_size: Some(tile_size) }
    }
}

impl Node for MatmulT {
    impl_default_functions!(visit_matmul_t);

    fn get_inputs(&self) -> Vec<EdgeID> {
        vec![self.a_in, self.b_in]
            .into_iter()
            .chain(self.psum_in.iter().copied())
            .collect()
    }

    fn get_outputs(&self) -> Vec<EdgeID> {
        vec![].into_iter()
            .chain(self.a_out.iter().copied())
            .chain(self.b_out.iter().copied())
            .chain(std::iter::once(self.psum_out))
            .collect()
    }

    fn rewrite_source(&mut self, old_source: EdgeID, new_sources: Vec<EdgeID>) -> bool {
        rewrite_port!(self.a_in, old_source, new_sources) ||
        rewrite_port!(self.b_in, old_source, new_sources) ||
        rewrite_optional_port!(self.psum_in, old_source, new_sources)
    }

    fn rewrite_target(&mut self, old_target: EdgeID, new_targets: Vec<EdgeID>) -> bool {
        rewrite_optional_port!(self.a_out, old_target, new_targets) ||
        rewrite_optional_port!(self.b_out, old_target, new_targets) || 
        rewrite_port!(self.psum_out, old_target, new_targets)
    }

    fn get_tile_size(&self) -> Option<[usize; 2]> {
        self.tile_size.clone()
    }
}



#[derive(Debug, Clone)]
pub struct Parallelize {
    pub input: EdgeID,
    pub outputs: Vec<EdgeID>,
    pub par_factor: usize,
    pub parallelized_dim: usize,
    pub id: usize,
    pub tile_size: Option<[usize; 2]>,
}

impl Parallelize {
    pub fn new(input: EdgeID, outputs: Vec<EdgeID>, rank: usize, dim: usize) -> Self {
        Self { input, outputs, par_factor: rank, 
            parallelized_dim: dim, id: next_id(), tile_size: None }
    }

    pub fn new_with_tile_size(input: EdgeID, outputs: Vec<EdgeID>, rank: usize, dim: usize, tile_size: [usize; 2]) -> Self {
        Self { input, outputs, par_factor: rank, 
            parallelized_dim: dim, id: next_id(), tile_size: Some(tile_size) }
    }
}

impl Node for Parallelize {
    impl_default_functions!(visit_parallelize);

    fn get_inputs(&self) -> Vec<EdgeID> {
        vec![self.input]
    }
    
    fn get_outputs(&self) -> Vec<EdgeID> {
        self.outputs.clone()
    }

    fn rewrite_source(&mut self, old_source: EdgeID, new_sources: Vec<EdgeID>) -> bool {
        rewrite_port!(self.input, old_source, new_sources)
    }

    fn rewrite_target(&mut self, old_target: EdgeID, new_targets: Vec<EdgeID>) -> bool {
        rewrite_vec!(self.outputs, old_target, new_targets)
    }

    fn get_tile_size(&self) -> Option<[usize; 2]> {
        self.tile_size.clone()
    }
}

#[derive(Debug, Clone)]
pub struct Partition {
    pub input: EdgeID,
    pub sel: EdgeID,
    pub outputs: Vec<EdgeID>,
    pub rank: usize,
    pub id: usize,
    pub tile_size: Option<[usize; 2]>,
}

impl Partition {
    pub fn new(input: EdgeID, sel: EdgeID, outputs: Vec<EdgeID>, rank: usize) -> Self {
        Self { input, sel, outputs, rank, id: next_id(), tile_size: None }
    }

    pub fn new_with_tile_size(input: EdgeID, sel: EdgeID, outputs: Vec<EdgeID>, rank: usize, tile_size: [usize; 2]) -> Self {
        Self { input, sel, outputs, rank, id: next_id(), tile_size: Some(tile_size) }
    }
}

impl Node for Partition {
    impl_default_functions!(visit_partition);
    
    fn get_inputs(&self) -> Vec<EdgeID> {
        vec![self.input, self.sel]
    }
    
    fn get_outputs(&self) -> Vec<EdgeID> {
        self.outputs.clone()
    }

    fn rewrite_source(&mut self, old_source: EdgeID, new_sources: Vec<EdgeID>) -> bool {
        rewrite_port!(self.input, old_source, new_sources) || 
        rewrite_port!(self.sel, old_source, new_sources)
    }

    fn rewrite_target(&mut self, old_target: EdgeID, new_targets: Vec<EdgeID>) -> bool {
        rewrite_vec!(self.outputs, old_target, new_targets)
    }

    fn get_tile_size(&self) -> Option<[usize; 2]> {
        self.tile_size.clone()
    }
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct Promote {
    pub input: EdgeID,
    pub output: EdgeID,
    pub rank: usize,
    pub id: usize,
    pub tile_size: Option<[usize; 2]>,
}

impl Promote {
    pub fn new(input: EdgeID, output: EdgeID, rank: usize) -> Self {
        Self { input, output, rank, id: next_id(), tile_size: None }
    }
    pub fn new_with_tile_size(input: EdgeID, output: EdgeID, rank: usize, tile_size: [usize; 2]) -> Self {
        Self { input, output, rank, id: next_id(), tile_size: Some(tile_size) }
    }
}

impl Node for Promote {
    impl_default_functions!(visit_promote);
    fn get_inputs(&self) -> Vec<EdgeID> { vec![self.input] }
    fn get_outputs(&self) -> Vec<EdgeID> { vec![self.output] }
    fn rewrite_source(&mut self, old_source: EdgeID, new_sources: Vec<EdgeID>) -> bool { rewrite_port!(self.input, old_source, new_sources) }
    fn rewrite_target(&mut self, old_target: EdgeID, new_targets: Vec<EdgeID>) -> bool { rewrite_port!(self.output, old_target, new_targets) }
    fn get_tile_size(&self) -> Option<[usize; 2]> { self.tile_size.clone() }
}

#[derive(Debug, Clone)]
pub struct Reassemble {
    pub inputs: Vec<EdgeID>,
    pub sel: EdgeID,
    pub output: EdgeID,
    pub in_stream_rank: usize,
    pub id: usize,
    pub tile_size: Option<[usize; 2]>,
}

impl Reassemble {
    pub fn new(inputs: Vec<EdgeID>, sel: EdgeID, output: EdgeID, in_stream_rank: usize) -> Self {
        Self { inputs, sel, output, in_stream_rank, id: next_id(), tile_size: None }
    }
    pub fn new_with_tile_size(inputs: Vec<EdgeID>, sel: EdgeID, output: EdgeID, in_stream_rank: usize, tile_size: [usize; 2]) -> Self {
        Self { inputs, sel, output, in_stream_rank, id: next_id(), tile_size: Some(tile_size) }
    }
}

impl Node for Reassemble {
    impl_default_functions!(visit_reassemble);
    fn get_inputs(&self) -> Vec<EdgeID> { 
        let mut t = self.inputs.clone();
        t.push(self.sel);
        t
    }
    fn get_outputs(&self) -> Vec<EdgeID> { vec![self.output] }
    fn rewrite_source(&mut self, old_source: EdgeID, new_sources: Vec<EdgeID>) -> bool { rewrite_vec!(self.inputs, old_source, new_sources) || rewrite_port!(self.sel, old_source, new_sources) }
    fn rewrite_target(&mut self, old_target: EdgeID, new_targets: Vec<EdgeID>) -> bool { rewrite_port!(self.output, old_target, new_targets) }
    fn get_tile_size(&self) -> Option<[usize; 2]> { self.tile_size.clone() }
}

#[derive(Debug, Clone)]
pub struct Repeat {
    pub input: EdgeID,
    pub count: EdgeID,
    pub output: EdgeID,
    pub id: usize,
    pub tile_size: Option<[usize; 2]>,
}

impl Repeat {
    pub fn new(input: EdgeID, count: EdgeID, output: EdgeID) -> Self {
        Self { input, count, output, id: next_id(), tile_size: None }
    }
    pub fn new_with_tile_size(input: EdgeID, count: EdgeID, output: EdgeID, tile_size: [usize; 2]) -> Self {
        Self { input, count, output, id: next_id(), tile_size: Some(tile_size) }
    }
}

impl Node for Repeat {
    impl_default_functions!(visit_repeat);
    fn get_inputs(&self) -> Vec<EdgeID> { vec![self.input, self.count] }
    fn get_outputs(&self) -> Vec<EdgeID> { vec![self.output] }
    fn rewrite_source(&mut self, old_source: EdgeID, new_sources: Vec<EdgeID>) -> bool { rewrite_port!(self.input, old_source, new_sources) || rewrite_port!(self.count, old_source, new_sources) }
    fn rewrite_target(&mut self, old_target: EdgeID, new_targets: Vec<EdgeID>) -> bool { rewrite_port!(self.output, old_target, new_targets) }
    fn get_tile_size(&self) -> Option<[usize; 2]> { self.tile_size.clone() }
}

#[derive(Debug, Clone)]
pub struct RepeatRef {
    pub input: EdgeID,
    pub count: EdgeID,
    pub output: EdgeID,
    pub rank: usize,
    pub id: usize,
    pub tile_size: Option<[usize; 2]>,
}

impl RepeatRef {
    pub fn new(input: EdgeID, count: EdgeID, output: EdgeID, rank: usize) -> Self {
        Self { input, count, output, rank, id: next_id(), tile_size: None }
    }
    pub fn new_with_tile_size(input: EdgeID, count: EdgeID, output: EdgeID, rank: usize, tile_size: [usize; 2]) -> Self {
        Self { input, count, output, rank, id: next_id(), tile_size: Some(tile_size) }
    }
}

impl Node for RepeatRef {
    impl_default_functions!(visit_repeat_ref);
    fn get_inputs(&self) -> Vec<EdgeID> { vec![self.input, self.count] }
    fn get_outputs(&self) -> Vec<EdgeID> { vec![self.output] }
    fn rewrite_source(&mut self, old_source: EdgeID, new_sources: Vec<EdgeID>) -> bool { rewrite_port!(self.input, old_source, new_sources) || rewrite_port!(self.count, old_source, new_sources) }
    fn rewrite_target(&mut self, old_target: EdgeID, new_targets: Vec<EdgeID>) -> bool { rewrite_port!(self.output, old_target, new_targets) }
    fn get_tile_size(&self) -> Option<[usize; 2]> { self.tile_size.clone() }
}

#[derive(Debug, Clone)]
pub struct Reshape {
    pub input: EdgeID,
    pub output: EdgeID,
    pub split_dims: Vec<usize>,
    pub chunk_sizes: Vec<usize>,
    pub elem: Option<Scalar>,
    pub id: usize,
    pub tile_size: Option<[usize; 2]>,
}

impl Reshape {
    pub fn new(input: EdgeID, output: EdgeID, split_dims: Vec<usize>, chunk_sizes: Vec<usize>, elem: Option<Scalar>) -> Self {
        Self { input, output, split_dims, chunk_sizes, elem, id: next_id(), tile_size: None }
    }
    pub fn new_with_tile_size(input: EdgeID, output: EdgeID, split_dims: Vec<usize>, chunk_sizes: Vec<usize>, elem: Option<Scalar>, tile_size: [usize; 2]) -> Self {
        Self { input, output, split_dims, chunk_sizes, elem, id: next_id(), tile_size: Some(tile_size) }
    }
}

impl Node for Reshape {
    impl_default_functions!(visit_reshape);
    fn get_inputs(&self) -> Vec<EdgeID> { vec![self.input] }
    fn get_outputs(&self) -> Vec<EdgeID> { vec![self.output] }
    fn rewrite_source(&mut self, old_source: EdgeID, new_sources: Vec<EdgeID>) -> bool { rewrite_port!(self.input, old_source, new_sources) }
    fn rewrite_target(&mut self, old_target: EdgeID, new_targets: Vec<EdgeID>) -> bool { rewrite_port!(self.output, old_target, new_targets) }
    fn get_tile_size(&self) -> Option<[usize; 2]> { self.tile_size.clone() }
}


#[derive(Debug, Clone)]
pub struct RetileStreamify {
    pub input: EdgeID,
    pub output: EdgeID,
    pub split_row: bool,
    pub filter_mask: bool,
    pub id: usize,
    pub tile_size: Option<[usize; 2]>,
}

impl RetileStreamify {
    pub fn new(input: EdgeID, output: EdgeID, split_row: bool, filter_mask: bool) -> Self {
        Self { input, output, split_row, filter_mask, id: next_id(), tile_size: None }
    }
}

impl Node for RetileStreamify {
    impl_default_functions!(visit_retile_streamify);
    fn get_inputs(&self) -> Vec<EdgeID> { vec![self.input] }
    fn get_outputs(&self) -> Vec<EdgeID> { vec![self.output] }
    fn rewrite_source(&mut self, old_source: EdgeID, new_sources: Vec<EdgeID>) -> bool { rewrite_port!(self.input, old_source, new_sources) }
    fn rewrite_target(&mut self, old_target: EdgeID, new_targets: Vec<EdgeID>) -> bool { rewrite_port!(self.output, old_target, new_targets) }
    fn get_tile_size(&self) -> Option<[usize; 2]> { self.tile_size.clone() }
}

#[derive(Debug, Clone)]
pub struct TiledRetileStreamify {
    pub input: EdgeID,
    pub output: EdgeID,
    pub to_buf: EdgeID,
    pub from_buf: EdgeID,
    pub split_row: bool,
    pub filter_mask: bool,
    pub id: usize,
    pub num_repeats: usize,
    pub tile_size: Option<[usize; 2]>,
}

/* This node currently only exists in concept. The idea is that it reads a big tile row by row. 
 * For that, it has to bufferize the big tile (r=1), and,
 * for each row send the input of row i tile_size times. 
 */
impl TiledRetileStreamify {
    pub fn new(input: EdgeID, output: EdgeID, to_buf: EdgeID, from_buf: EdgeID, split_row: bool, filter_mask: bool, num_repeats: usize) -> Self {
        Self { input, output, to_buf, from_buf, split_row, filter_mask, id: next_id(), num_repeats, tile_size: None }
    }

    pub fn new_with_tile_size(input: EdgeID, output: EdgeID, to_buf: EdgeID, from_buf: EdgeID, split_row: bool, filter_mask: bool, num_repeats: usize, tile_size: [usize; 2]) -> Self {
        Self { input, output, to_buf, from_buf, split_row, filter_mask, id: next_id(), num_repeats, tile_size: Some(tile_size) }
    }
}

impl Node for TiledRetileStreamify {
    impl_default_functions!(visit_tiled_retile_streamify);
    fn get_inputs(&self) -> Vec<EdgeID> { vec![self.input, self.from_buf] }
    fn get_outputs(&self) -> Vec<EdgeID> { vec![self.to_buf, self.output] }
    fn rewrite_source(&mut self, old_source: EdgeID, new_sources: Vec<EdgeID>) -> bool { rewrite_port!(self.input, old_source, new_sources) }
    fn rewrite_target(&mut self, old_target: EdgeID, new_targets: Vec<EdgeID>) -> bool { rewrite_port!(self.output, old_target, new_targets) }
    fn get_tile_size(&self) -> Option<[usize; 2]> { self.tile_size.clone() }
}

#[derive(Debug, Clone)]
pub struct Rotate {
    pub input: EdgeID,
    pub outputs: Vec<EdgeID>,
    pub reset_rank: usize,
    pub id: usize,
    pub tile_size: Option<[usize; 2]>,
}

impl Rotate {
    pub fn new(input: EdgeID, outputs: Vec<EdgeID>, reset_rank: usize) -> Self {
        Self { input, outputs, reset_rank, id: next_id(), tile_size: None }
    }
    pub fn new_with_tile_size(input: EdgeID, outputs: Vec<EdgeID>, reset_rank: usize, tile_size: [usize; 2]) -> Self {
        Self { input, outputs, reset_rank, id: next_id(), tile_size: Some(tile_size) }
    }
}

impl Node for Rotate {
    impl_default_functions!(visit_rotate);
    fn get_inputs(&self) -> Vec<EdgeID> { vec![self.input] }
    fn get_outputs(&self) -> Vec<EdgeID> { self.outputs.clone() }
    fn rewrite_source(&mut self, old_source: EdgeID, new_sources: Vec<EdgeID>) -> bool { rewrite_port!(self.input, old_source, new_sources) }
    fn get_tile_size(&self) -> Option<[usize; 2]> { self.tile_size.clone() }
}

#[derive(Debug, Clone)]
pub struct RepeatStatic {
    pub input: EdgeID,
    pub output: EdgeID,
    pub count: usize,
    pub id: usize,
    pub tile_size: Option<[usize; 2]>,
}

impl RepeatStatic {
    pub fn new(input: EdgeID, count: usize, output: EdgeID) -> Self {
        Self { input, count, output, id: next_id(), tile_size: None }
    }
    pub fn new_with_tile_size(input: EdgeID, count: usize, output: EdgeID, tile_size: [usize; 2]) -> Self {
        Self { input, count, output, id: next_id(), tile_size: Some(tile_size) }
    }
}

impl Node for RepeatStatic {
    impl_default_functions!(visit_repeat_static);
    fn get_inputs(&self) -> Vec<EdgeID> { vec![self.input] }
    fn get_outputs(&self) -> Vec<EdgeID> { vec![self.output] }
    fn rewrite_source(&mut self, old_source: EdgeID, new_sources: Vec<EdgeID>) -> bool { rewrite_port!(self.input, old_source, new_sources) }
    fn rewrite_target(&mut self, old_target: EdgeID, new_targets: Vec<EdgeID>) -> bool { rewrite_port!(self.output, old_target, new_targets) }
    fn get_tile_size(&self) -> Option<[usize; 2]> { self.tile_size.clone() }
}

#[derive(Debug, Clone)]
pub struct SelectGen {
    pub input: EdgeID,
    pub output: EdgeID,
    pub id: usize,
    pub tile_size: Option<[usize; 2]>,
}

impl SelectGen {
    pub fn new(input: EdgeID, output: EdgeID) -> Self {
        Self { input, output, id: next_id(), tile_size: None }
    }
}

impl Node for SelectGen {
    impl_default_functions!(visit_select_gen);
    fn get_inputs(&self) -> Vec<EdgeID> { vec![self.input] }
    fn get_outputs(&self) -> Vec<EdgeID> { vec![self.output] }
    fn rewrite_source(&mut self, old_source: EdgeID, new_sources: Vec<EdgeID>) -> bool { rewrite_port!(self.input, old_source, new_sources) }
    fn rewrite_target(&mut self, old_target: EdgeID, new_targets: Vec<EdgeID>) -> bool { rewrite_port!(self.output, old_target, new_targets) }
    fn get_tile_size(&self) -> Option<[usize; 2]> { self.tile_size.clone() }
}

#[derive(Debug, Clone)]
pub struct Scan {
    pub input: EdgeID,
    pub output: EdgeID,
    pub scan_fn: TypelessFunction,
    pub init: Option<Tile>, // Scan with init_fn is exclusive, without it is inclusive
    pub rank: usize,
    pub id: usize,
    pub tile_size: Option<[usize; 2]>,
}

impl Scan {
    pub fn new(input: EdgeID, output: EdgeID, scan_fn: TypelessFunction, init: Option<Tile>, rank: usize) -> Self {
        Self { input, output, scan_fn, init, rank, id: next_id(), tile_size: None }
    }
    pub fn new_with_tile_size(input: EdgeID, output: EdgeID, scan_fn: TypelessFunction, init: Option<Tile>, rank: usize, tile_size: [usize; 2]) -> Self {
        Self { input, output, scan_fn, init, rank, id: next_id(), tile_size: Some(tile_size) }
    }
}

impl Node for Scan {
    impl_default_functions!(visit_scan);
    fn get_inputs(&self) -> Vec<EdgeID> { vec![self.input] }
    fn get_outputs(&self) -> Vec<EdgeID> { vec![self.output] }
    fn rewrite_source(&mut self, old_source: EdgeID, new_sources: Vec<EdgeID>) -> bool { rewrite_port!(self.input, old_source, new_sources) }
    fn rewrite_target(&mut self, old_target: EdgeID, new_targets: Vec<EdgeID>) -> bool { rewrite_port!(self.output, old_target, new_targets) }
    fn get_tile_size(&self) -> Option<[usize; 2]> { self.tile_size.clone() }
}

#[derive(Debug, Clone)]
pub struct Streamify {
    pub input: EdgeID,
    pub output: EdgeID,
    pub id: usize,
    pub tile_size: Option<[usize; 2]>,
}

impl Streamify {
    pub fn new(input: EdgeID, output: EdgeID) -> Self {
        Self { input, output, id: next_id(), tile_size: None }
    }
    pub fn new_with_tile_size(input: EdgeID, output: EdgeID, tile_size: [usize; 2]) -> Self {
        Self { input, output, id: next_id(), tile_size: Some(tile_size) }
    }
}

impl Node for Streamify {
    impl_default_functions!(visit_streamify);
    fn get_inputs(&self) -> Vec<EdgeID> { vec![self.input] }
    fn get_outputs(&self) -> Vec<EdgeID> { vec![self.output] }
    fn rewrite_source(&mut self, old_source: EdgeID, new_sources: Vec<EdgeID>) -> bool { rewrite_port!(self.input, old_source, new_sources) }
    fn rewrite_target(&mut self, old_target: EdgeID, new_targets: Vec<EdgeID>) -> bool { rewrite_port!(self.output, old_target, new_targets) }
    fn get_tile_size(&self) -> Option<[usize; 2]> { self.tile_size.clone() }
}

#[derive(Debug, Clone)]
pub struct Unzip {
    pub input: Vec<EdgeID>,
    pub output1: EdgeID,
    pub output2: EdgeID,
    pub id: usize,
    pub tile_size: Option<[usize; 2]>,
}

impl Unzip {
    pub fn new(input: Vec<EdgeID>, output1: EdgeID, output2: EdgeID) -> Self {
        Self { input, output1, output2, id: next_id(), tile_size: None }
    }
    pub fn new_with_tile_size(input: Vec<EdgeID>, output1: EdgeID, output2: EdgeID, tile_size: [usize; 2]) -> Self {
        Self { input, output1, output2, id: next_id(), tile_size: Some(tile_size) }
    }
}

impl Node for Unzip {
    impl_default_functions!(visit_unzip);
    fn get_inputs(&self) -> Vec<EdgeID> { self.input.clone() }
    fn get_outputs(&self) -> Vec<EdgeID> { vec![self.output1, self.output2] }
    fn rewrite_source(&mut self, old_source: EdgeID, new_sources: Vec<EdgeID>) -> bool { rewrite_vec!(self.input, old_source, new_sources) }
    fn rewrite_target(&mut self, old_target: EdgeID, new_targets: Vec<EdgeID>) -> bool { rewrite_port!(self.output1, old_target, new_targets) || rewrite_port!(self.output2, old_target, new_targets) }
    fn get_tile_size(&self) -> Option<[usize; 2]> { self.tile_size.clone() }
}

#[derive(Debug, Clone)]
pub struct Zip {
    pub input1: EdgeID,
    pub input2: EdgeID,
    pub output: Vec<EdgeID>,
    pub id: usize,
    pub tile_size: Option<[usize; 2]>,
}

impl Zip {
    pub fn new(input1: EdgeID, input2: EdgeID, output: Vec<EdgeID>) -> Self {
        Self { input1, input2, output, id: next_id(), tile_size: None }
    }
    pub fn new_with_tile_size(input1: EdgeID, input2: EdgeID, output: Vec<EdgeID>, tile_size: [usize; 2]) -> Self {
        Self { input1, input2, output, id: next_id(), tile_size: Some(tile_size) }
    }
}

impl Node for Zip {
    impl_default_functions!(visit_zip);
    fn get_inputs(&self) -> Vec<EdgeID> { vec![self.input1, self.input2] }
    fn get_outputs(&self) -> Vec<EdgeID> { self.output.clone() }
    fn rewrite_source(&mut self, old_source: EdgeID, new_sources: Vec<EdgeID>) -> bool { rewrite_port!(self.input1, old_source, new_sources) || rewrite_port!(self.input2, old_source, new_sources) }
    fn rewrite_target(&mut self, old_target: EdgeID, new_targets: Vec<EdgeID>) -> bool { rewrite_vec!(self.output, old_target, new_targets) }
    fn get_tile_size(&self) -> Option<[usize; 2]> { self.tile_size.clone() }
}

pub struct Hypernode {
    pub id: usize,
    pub nodes: Vec<Box<dyn Node>>,
    pub tile_size: Option<[usize; 2]>,
}

impl Hypernode {
    pub fn new(nodes: Vec<Box<dyn Node>>) -> Self {
        Self { id: next_id(), nodes, tile_size: None }
    }
}

impl Node for Hypernode {
    impl_default_functions!(visit_hypernode);

    fn get_inputs(&self) -> Vec<EdgeID> {
        let inputs: Vec<EdgeID> = self.nodes.iter().flat_map(|n| n.get_inputs()).collect();
        let outputs: std::collections::HashSet<EdgeID> = self.nodes.iter().flat_map(|n| n.get_outputs()).collect();
        inputs.into_iter().filter(|i| !outputs.contains(i)).collect()
    }
    
    fn get_outputs(&self) -> Vec<EdgeID> {
        let outputs: std::collections::HashSet<EdgeID> = self.nodes.iter().flat_map(|n| n.get_outputs()).collect();
        let inputs: std::collections::HashSet<EdgeID> = self.nodes.iter().flat_map(|n| n.get_inputs()).collect();
        outputs.difference(&inputs).cloned().collect()
    }

    fn rewrite_source(&mut self, old_source: EdgeID, new_sources: Vec<EdgeID>) -> bool {
        self.nodes.iter_mut().any(|n| n.rewrite_source(old_source, new_sources.clone()))
    }

    fn rewrite_target(&mut self, old_target: EdgeID, new_targets: Vec<EdgeID>) -> bool {
        self.nodes.iter_mut().any(|n| n.rewrite_target(old_target, new_targets.clone()))
    }

    fn get_tile_size(&self) -> Option<[usize; 2]> {
        None // Hypernodes do not have a tile size, since it might be heterogeneous inside.
    }
}

impl Debug for Hypernode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Hypernode {{ id: {}, num_nodes: {:?} inputs: {:?} outputs: {:?} }}", self.id, self.nodes.len(), self.get_inputs(), self.get_outputs())
    }
}