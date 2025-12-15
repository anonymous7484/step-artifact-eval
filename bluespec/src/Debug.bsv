package Debug;

import Types::*;
import Operation::*;
import Parameters::*;

module mkDebugOperation#(Operation_IFC op, String name) (Operation_IFC);
    method Action put(Int#(32) input_port, ChannelMessage msg);
        let st = tpl_2(msg.Tag_Data);
        if (valueOf(PRINT_DEBUG_OPERATION) == 1) begin
            $display("DebugOperation: %s put <%d, %d>", name, input_port, st);
        end
        op.put(input_port, msg);
    endmethod

    method ActionValue#(ChannelMessage) get(Int#(32) output_port);
        let msg <- op.get(output_port);
        let st = tpl_2(msg.Tag_Data);
        if (valueOf(PRINT_DEBUG_OPERATION) == 1) begin
            $display("DebugOperation: %s get <%d, %d>", name, output_port, st);
        end
        return msg;
    endmethod
endmodule

endpackage