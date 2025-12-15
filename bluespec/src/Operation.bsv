package Operation;

import FIFOF::*;
import FIFO::*;
import Vector::*;
import Assert::*;
import Types::*;
import FileReader::*;
import RamulatorArbiter::*;
import StmtFSM::*;

interface Operation_IFC;
    method Action put(Int#(32) input_port, ChannelMessage msg);
    method ActionValue#(ChannelMessage) get(Int#(32) output_port);
endinterface

module mkAccum#(function Tile func (Tile tile, Tile tile2), Int#(32) rank) (Operation_IFC);
    FIFO#(ChannelMessage) input_fifo <- mkFIFO;
    FIFO#(ChannelMessage) output_fifo <- mkFIFO;

    Reg#(Maybe#(Tile)) acc <- mkReg(tagged Invalid);

    rule do_accum if (input_fifo.first matches tagged Tag_Data .current);
        input_fifo.deq;

        let data = tpl_1(current);
        let st = tpl_2(current);

        Tile out;
        if (acc matches tagged Valid .tile) begin
            let tile2 = data.Tag_Tile;
            out = func(tile, tile2);
        end else begin
            out = data.Tag_Tile;
        end

        if (st >= rank) begin 
            let out_rank = st - rank;
            acc <= tagged Invalid;
            output_fifo.enq(tagged Tag_Data tuple2(tagged Tag_Tile out, out_rank));
        end else begin
            acc <= tagged Valid out;
        end

    endrule

    method Action put(Int#(32) input_port, ChannelMessage msg);
        input_fifo.enq(msg);
    endmethod

    method ActionValue#(ChannelMessage) get(Int#(32) output_port);
        output_fifo.deq;
        return output_fifo.first;
    endmethod
endmodule

module mkRepeatStatic#(Int#(32) num_repeats) (Operation_IFC);
    FIFOF#(ChannelMessage) input_fifo <- mkFIFOF;
    FIFO#(ChannelMessage) output_fifo <- mkFIFO;

    Reg#(Int#(32)) count <- mkReg(0);

    rule do_repeat if (count < num_repeats);
        let cur = input_fifo.first;

        if (cur matches tagged Tag_Data .current) begin
            Data to_send = tpl_1(current);
            if (count == num_repeats - 1) begin
                if (to_send matches tagged Tag_Ref .r) begin
                    to_send = tagged Tag_Ref tuple2(tpl_1(r), True);
                end
                cur = tagged Tag_Data tuple2(to_send, tpl_2(current) + 1);
            end else begin 
                if (to_send matches tagged Tag_Ref .r) begin
                    to_send = tagged Tag_Ref tuple2(tpl_1(r), False);
                end
                cur = tagged Tag_Data tuple2(to_send, 0);
            end
        end

        output_fifo.enq(cur);

        if (cur matches tagged Tag_Data .current) begin // This if is necessary so deallocates are not repeated
            count <= count + 1;
        end
    endrule

    rule dequeue if (count == num_repeats);
        input_fifo.deq;
        count <= 0;
    endrule

    method Action put(Int#(32) input_port, ChannelMessage msg);
        input_fifo.enq(msg);
    endmethod

    method ActionValue#(ChannelMessage) get(Int#(32) output_port);
        output_fifo.deq;
        return output_fifo.first;
    endmethod
endmodule

module mkBroadcast2 (Operation_IFC);
    let output_fifo1 <- mkFIFO;
    let output_fifo2 <- mkFIFO;

    method Action put(Int#(32) input_port, ChannelMessage msg);
        output_fifo1.enq(msg);
        output_fifo2.enq(msg);
    endmethod

    method ActionValue#(ChannelMessage) get(Int#(32) output_port);
        if (output_port == 0) begin
            output_fifo1.deq;
            return output_fifo1.first;
        end else begin
            output_fifo2.deq;
            return output_fifo2.first;
        end
    endmethod
endmodule

module mkUnaryMap#(Integer id, function Tile func (Tile tile)) (Operation_IFC);
    FIFO#(ChannelMessage) input_fifo <- mkFIFO;
    FIFO#(ChannelMessage) output_fifo <- mkFIFO;

    rule do_map;
        let cur = input_fifo.first;
        input_fifo.deq;

        if (cur matches tagged Tag_Data .current) begin 
            let out = func(tpl_1(current).Tag_Tile);
            output_fifo.enq(tagged Tag_Data tuple2(tagged Tag_Tile out, tpl_2(current)));
        end
    endrule

    method Action put(Int#(32) input_port, ChannelMessage msg);
        input_fifo.enq(msg);
    endmethod

    method ActionValue#(ChannelMessage) get(Int#(32) output_port);
        output_fifo.deq;
        return output_fifo.first;
    endmethod
endmodule

module mkBinaryMap#(Int#(32) id, function Tile func (Tile tile, Tile tile2)) (Operation_IFC);
    FIFO#(ChannelMessage) input_fifo1 <- mkFIFO;
    FIFO#(ChannelMessage) input_fifo2 <- mkFIFO;
    FIFO#(ChannelMessage) output_fifo <- mkFIFO;
    Reg#(Int#(32)) count <- mkReg(0);

    rule do_map;
        let cur_1 = input_fifo1.first;
        let cur_2 = input_fifo2.first;

        input_fifo1.deq;
        input_fifo2.deq;

        if (cur_1 matches tagged Tag_Data .current_1 &&& cur_2 matches tagged Tag_Data .current_2) begin
            let out = func(tpl_1(current_1).Tag_Tile, tpl_1(current_2).Tag_Tile);
            $display("ST1: %d, ST2: %d, count: %d, id: %d", tpl_2(current_1), tpl_2(current_2), count, id);
            if (tpl_2(current_2) != tpl_2(current_1)) begin
                $error("id: %d, Stop tokens are not the same at cycle %d, left: %d, right: %d", id, count, tpl_2(current_1), tpl_2(current_2));
                $finish;
                // $finish;
            end
            // dynamicAssert(tpl_2(current_2) == tpl_2(current_1), "Stop tokens must be the same");
            output_fifo.enq(tagged Tag_Data tuple2(tagged Tag_Tile out, tpl_2(current_1)));
            count <= count + 1;
        end
    endrule

    method Action put(Int#(32) input_port, ChannelMessage msg);
        if (input_port == 0) begin
            input_fifo1.enq(msg);
        end else begin
            input_fifo2.enq(msg);
        end
    endmethod

    method ActionValue#(ChannelMessage) get(Int#(32) output_port);
        output_fifo.deq;
        return output_fifo.first;
    endmethod
endmodule

module mkPromote#(Int#(32) rank) (Operation_IFC);
    FIFO#(ChannelMessage) input_fifo <- mkFIFO;
    FIFO#(ChannelMessage) output_fifo <- mkFIFO;

    rule do_promote;
        let cur = input_fifo.first;
        input_fifo.deq;

        if (cur matches tagged Tag_Data .current) begin
            let data = tpl_1(current);
            let st = tpl_2(current);
            let st_out = (st >= rank) ? st + 1 : st;
            output_fifo.enq(tagged Tag_Data tuple2(data, st_out));
        end else begin 
            output_fifo.enq(cur);
        end
    endrule

    method Action put(Int#(32) input_port, ChannelMessage msg);
        input_fifo.enq(msg);
    endmethod

    method ActionValue#(ChannelMessage) get(Int#(32) output_port);
        output_fifo.deq;
        return output_fifo.first;
    endmethod
endmodule

typedef enum {
    AccumBigTile_Fill,
    AccumBigTile_Accum,
    AccumBigTile_Drain
} AccumBigTileState deriving (Bits, Eq, FShow);

module mkAccumBigTile#(function Tile func (Tile tile, Tile tile2), Int#(32) rank) (Operation_IFC);
    FIFO#(ChannelMessage) input_fifo <- mkFIFO;
    FIFO#(ChannelMessage) partial_input_fifo <- mkFIFO;
    FIFO#(ChannelMessage) partial_output_fifo <- mkFIFO;
    FIFO#(ChannelMessage) output_fifo <- mkFIFO;

    Reg#(AccumBigTileState) state <- mkReg(AccumBigTile_Fill);

    rule fill if (state == AccumBigTile_Fill);
        let cur = input_fifo.first;
        input_fifo.deq;
        
        let st = tpl_2(cur.Tag_Data);
        partial_output_fifo.enq(cur);

        if (st >= rank) begin
            state <= AccumBigTile_Drain;
        end else if (st == 2) begin 
            state <= AccumBigTile_Accum;
        end
    endrule

    rule accumulate if (state == AccumBigTile_Accum);
        let add = input_fifo.first;
        let acc = partial_input_fifo.first;
        input_fifo.deq;
        partial_input_fifo.deq;

        let st = tpl_2(add.Tag_Data);
        let out = func(tpl_1(add.Tag_Data).Tag_Tile, tpl_1(acc.Tag_Data).Tag_Tile);

        if (st >= rank) begin 
            state <= AccumBigTile_Drain;
        end 

        partial_output_fifo.enq(tagged Tag_Data tuple2(tagged Tag_Tile out, st));
    endrule

    rule drain if (state == AccumBigTile_Drain);
        let cur_in = partial_input_fifo.first;
        partial_input_fifo.deq;

        let data = cur_in.Tag_Data;
        let st = tpl_2(data);

        Int#(32) out_st;
        if (st >= rank) begin
            out_st = st - rank + 2;
        end else begin 
            out_st = st;
        end

        if (st >= rank) begin
            state <= AccumBigTile_Fill;
        end

        output_fifo.enq(tagged Tag_Data tuple2(tpl_1(data), out_st));
    endrule
    
    method Action put(Int#(32) input_port, ChannelMessage msg);
        if (input_port == 0) begin
            input_fifo.enq(msg);
        end else begin
            partial_input_fifo.enq(msg);
        end
    endmethod

    method ActionValue#(ChannelMessage) get(Int#(32) output_port);
        if (output_port == 0) begin
            partial_output_fifo.deq;
            return partial_output_fifo.first;
        end else begin
            output_fifo.deq;
            return output_fifo.first;
        end
    endmethod
endmodule

module mkTileReader#(Integer num_entries, String filename, Bit#(8) port_id, RamulatorArbiterIO arbiter) (Operation_IFC);
    FileReader_IFC#(TaggedTile) reader <- mkFileReader(num_entries, filename, port_id, arbiter);

    method ActionValue#(ChannelMessage) get(Int#(32) output_port);
        TaggedTile tile <- reader.readNext;
        return tagged Tag_Data tuple2(tagged Tag_Tile tile.t, tile.st);
    endmethod 

    method Action put(Int#(32) input_port, ChannelMessage msg);
        $error("TileReader does not support put");
    endmethod
endmodule

module mkTileWriter#(Bit#(8) port_id, RamulatorArbiterIO arbiter) (Operation_IFC);
    FileWriter_IFC#(Tile) writer <- mkDumpingFileWriter(port_id, arbiter);

    Reg#(Bit#(64)) cycle_count <- mkReg(0);

    rule cc;
        cycle_count <= cycle_count + 1;
    endrule

    method ActionValue#(ChannelMessage) get(Int#(32) output_port);
        $error("TileWriter does not support get");
        return ?;
    endmethod 

    method Action put(Int#(32) input_port, ChannelMessage msg);
        Tile tile = tpl_1(msg.Tag_Data).Tag_Tile;
        writer.writeNext(tile);

        if (tpl_2(msg.Tag_Data) == 5) begin // Hardcoded for Gina's application.
            $display("Finished at cycle %d", cycle_count);
            $finish;
        end
    endmethod
endmodule

module mkPrinter#(String name) (Operation_IFC);
    FIFO#(ChannelMessage) input_fifo <- mkFIFO;
    Reg#(Int#(64)) cc <- mkReg(0);

    rule print;
        let cur = input_fifo.first;
        input_fifo.deq;
        
        $display("[cycle %d] %s: %s", cc, name, fshow(cur));

        if (tpl_2(cur.Tag_Data) == 5) begin // Hardcoded for Gina's application.
            $display("Finished at cycle %d", cc);
            $finish;
        end
    endrule

    rule cc_rule;
        cc <= cc + 1;
    endrule

    method Action put(Int#(32) input_port, ChannelMessage msg);
        input_fifo.enq(msg);
    endmethod

    method ActionValue#(ChannelMessage) get(Int#(32) output_port);
        $error("Printer does not support get");
        return ?;
    endmethod
endmodule

module mkTileWriterTest (Empty);
    RamulatorArbiter_IFC#(1) arbiter <- mkRamulatorArbiter(1);
    Operation_IFC tile_writer <- mkTileWriter(0, arbiter.ports);

    mkAutoFSM (
        seq
            repeat (100)
                action 
                    Tile t = unpack(0);
                    tile_writer.put(0, tagged Tag_Data tuple2(tagged Tag_Tile t, 0));
                endaction
            action
                $display("Done.");
            endaction 
        endseq 
    );
endmodule

endpackage