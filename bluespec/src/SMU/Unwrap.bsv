package Unwrap;

import Types::*;

// Unwrap Data union types
function Tile unwrapTile(Data d);
    let result = case (d) matches
        tagged Tag_Tile .t : return tagged Valid t;
        default : return tagged Invalid;
    endcase;
    return fromMaybe(?, result);
endfunction

function Tuple2#(Bit#(32), Bool) unwrapRef(Data d);
    let result = case (d) matches
        tagged Tag_Ref {.r, .deallocate} : tagged Valid tuple2(r, deallocate);
        default : return tagged Invalid;
    endcase;
    return fromMaybe(?, result);
endfunction

// function Ref unwrapRef(Data d);
//     case (d) matches
//         tagged Tag_Ref .r => r;
//         default => begin
//             $display("[ERROR]: Expected Tag_Ref, got different type");
//             $finish(0);
//             return ?;
//         end
//     endcase;
// endfunction


// // Unwrap ChannelMessage union types
// function Tuple2#(Data, StopToken) unwrapData(ChannelMessage msg);
//     case (msg) matches
//         tagged Tag_Data .d => d;
//         default => begin
//             $display("[ERROR]: Expected Tag_Data, got different type");
//             $finish(0);
//             return ?;
//         end
//     endcase;
// endfunction

// function Instruction_Ptr unwrapInstruction(ChannelMessage msg);
//     case (msg) matches
//         tagged Tag_Instruction .ip => ip;
//         default => begin
//             $display("[ERROR]: Expected Tag_Instruction, got different type");
//             $finish(0);
//             return ?;
//         end
//     endcase;
// endfunction

// function EndToken unwrapEndToken(ChannelMessage msg);
//     case (msg) matches
//         tagged Tag_EndToken .et => et;
//         default => begin
//             $display("[ERROR]: Expected Tag_EndToken, got different type");
//             $finish(0);
//             return ?;
//         end
//     endcase;
// endfunction

// // Convenience functions for common patterns
// function Tile unwrapTileFromMessage(ChannelMessage msg);
//     let data = unwrapData(msg);
//     return unwrapTile(tpl_1(data));
// endfunction

// function Scalar unwrapScalarFromMessage(ChannelMessage msg);
//     let data = unwrapData(msg);
//     return unwrapScalar(tpl_1(data));
// endfunction

endpackage