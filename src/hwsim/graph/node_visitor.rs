// node_visitor.rs
use super::nodes::*;
use std::any::Any;

macro_rules! define_visitors {
    ( $( $fn_name:ident => $Node:ident ),* $(,)? ) => {
        $(
            fn $fn_name(&mut self, node: &$Node) -> Box<dyn Any> {
                self.default(node)
            }
        )*
    };
}

pub trait NodeVisitor {
    // you list exactly the pairs you want:
    define_visitors!(
        visit_address_reader => AddressReader,
        visit_address_writer => AddressWriter,
        visit_accum       => Accum,
        visit_accum_big_tile => AccumBigTile,
        visit_broadcast   => Broadcast,
        visit_bufferize   => Bufferize,
        visit_concatenate => Concatenate,
        visit_dyn_off_chip_load => DynOffChipLoad,
        visit_enumerate   => Enumerate,
        visit_flatmap     => FlatMap,
        visit_flatten     => Flatten,
        visit_fnblock     => FnBlock,
        visit_map         => Map,
        visit_matmul_t    => MatmulT,
        visit_filter      => Filter,
        visit_partition   => Partition,
        visit_promote     => Promote,
        visit_reassemble  => Reassemble,
        visit_repeat      => Repeat,
        visit_repeat_ref  => RepeatRef,
        visit_repeat_static => RepeatStatic,
        visit_retile_streamify => RetileStreamify,
        visit_tiled_retile_streamify => TiledRetileStreamify,
        visit_reshape     => Reshape,
        visit_rotate      => Rotate,
        visit_scan        => Scan,
        visit_select_gen  => SelectGen,
        visit_streamify   => Streamify,
        visit_unzip       => Unzip,
        visit_zip         => Zip,
        visit_parallelize => Parallelize,
        visit_hypernode   => Hypernode,
    );

    /// fallback if you don’t override a specific visit_* method
    fn default(&mut self, _node: &dyn Node) -> Box<dyn Any> {
        return Box::new(());
    }
}
