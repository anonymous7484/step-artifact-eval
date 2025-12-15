import Vector::*;
import FIFO::*;
import FIFOF::*;
import Types::*;
import Operation::*;
import RegFile::*;

function Bit#(m) resize(Bit#(n) x);
    // TMax#(m,n) is a type‐level max(m,n)
    Bit#(TMax#(m,n)) x_ext = zeroExtend(x);
    Bit#(m)            y   = truncate(x_ext);
    return y;
endfunction

interface Bufferize#(numeric type num_entries, numeric type elements_per_entry);
// these methods are for uniform interface w.r.t. the operation module

    interface Operation_IFC operation;

    method Action put_data(ChannelMessage msg);
    method ActionValue#(ChannelMessage) get_token();
    method Action request_token(ChannelMessage msg);
    method ActionValue#(ChannelMessage) get_data();
endinterface

module mkSimpleBufferize#(Int#(32) rank, Integer num_entries, Integer elements_per_entry) (Bufferize#(num_entries, elements_per_entry));
    FIFOF#(ChannelMessage) input_fifo <- mkFIFOF;
    FIFOF#(ChannelMessage) output_fifo <- mkFIFOF;
    FIFOF#(ChannelMessage) token_request_fifo <- mkFIFOF;    
    FIFOF#(ChannelMessage) token_output_fifo <- mkFIFOF;

    Bit#(TLog#(elements_per_entry)) max_subindex = fromInteger(valueOf(elements_per_entry) - 1);

    RegFile#(Bit#(TAdd#(TLog#(num_entries), TLog#(elements_per_entry))), Maybe#(Tuple2#(Data, StopToken))) buffer <- mkRegFile(0, fromInteger(valueOf(num_entries) * valueOf(elements_per_entry) - 1));

    Reg#(Vector#(num_entries, Bool)) free <- mkReg(replicate(True));

    Reg#(Bit#(TLog#(num_entries))) entry_write_ptr <- mkReg(0);
    Reg#(Bit#(TLog#(elements_per_entry))) subentry_write_ptr <- mkReg(0);
    
    Reg#(Bit#(TLog#(elements_per_entry))) read_ptr <- mkReg(0);

    (* descending_urgency = "read, store" *)
    rule store if (free[entry_write_ptr] == True &&& input_fifo.first matches tagged Tag_Data { .data, .st });
        input_fifo.deq;

        //Bit#(TAdd#(TLog#(num_entries), TLog#(SIZE_PER_ENTRY))) entry_write_ptr_trunc = extend(entry_write_ptr);
        //Bit#(TAdd#(TLog#(num_entries), TLog#(SIZE_PER_ENTRY))) subentry_write_ptr_trunc = extend(subentry_write_ptr);

        //let index = entry_write_ptr_trunc << fromInteger(valueOf(TLog#(SIZE_PER_ENTRY))) | subentry_write_ptr_trunc;
        let index = { entry_write_ptr, subentry_write_ptr };
        // $display("Storing to index: ", fshow(index));
        buffer.upd(index, tagged Valid (tuple2(data, st)));
        // buffer.upd([write_ptr][write_subptr] <= tagged Valid (tuple2(data, st));
        if (st >= rank) begin 
            entry_write_ptr <= (entry_write_ptr + 1);
            subentry_write_ptr <= 0;
            Ref_Inner r = resize(entry_write_ptr);
            token_output_fifo.enq(tagged Tag_Data tuple2(tagged Tag_Ref (tuple2(r, True)), st - rank));
            free[entry_write_ptr] <= False;
        end else begin 
            subentry_write_ptr <= subentry_write_ptr + 1;
        end 
    endrule

    rule read if (token_request_fifo.first matches tagged Tag_Data { .data, .st }
    );
        if (free[tpl_1(data.Tag_Ref)] == False) begin
            let ptr = tpl_1(data.Tag_Ref);

            let index = { ptr, read_ptr };
            let el = buffer.sub(resize(index));


            // $display("Read ptr: ", read_ptr);
            // $display("Request ST: ", fshow(st));
            let v = fromMaybe(?, el);    // pick a suitable default if you ever hit Invalid
            // $display("Max subindex: ", fshow(max_subindex));
            if (read_ptr == max_subindex) begin 
                v = tuple2(tpl_1(v), st + rank);
            end
            
            output_fifo.enq(tagged Tag_Data v);

            if (read_ptr == max_subindex) begin 
                read_ptr <= 0;
                token_request_fifo.deq;

                if (data matches tagged Tag_Ref { .r, .deallocate }) begin
                    if (deallocate == True) begin
                        free[r] <= True;
                    end
                end
            end else begin 
                read_ptr <= read_ptr + 1;
            end
        end
    endrule

    method Action put_data(ChannelMessage msg);
        input_fifo.enq(msg);
    endmethod

    method ActionValue#(ChannelMessage) get_token();
        let msg = token_output_fifo.first;
        token_output_fifo.deq;
        return msg;
    endmethod

    method Action request_token(ChannelMessage msg);
        token_request_fifo.enq(msg);
    endmethod

    method ActionValue#(ChannelMessage) get_data();
        let msg = output_fifo.first;
        output_fifo.deq;
        return msg;
    endmethod

    interface Operation_IFC operation;
        method Action put(Int#(32) i, ChannelMessage msg); // i = 0 for data, 1 for token
            if (i == 0) begin
                input_fifo.enq(msg);
            end else begin
                token_request_fifo.enq(msg);
            end
        endmethod

        method ActionValue#(ChannelMessage) get(Int#(32) i); // i = 0 for token, 1 for data
            ChannelMessage msg;
            if (i == 0) begin
                token_output_fifo.deq;
                msg = token_output_fifo.first;
            end else begin
                output_fifo.deq;
                msg = output_fifo.first;
            end
            return msg;
        endmethod
    endinterface
endmodule

typedef 1 NUM_ENTRIES;
typedef 4 ENTRY_SIZE;

module mkBufferize(Empty);
    let rank = 3;
    Bufferize#(NUM_ENTRIES, ENTRY_SIZE) dut <- mkSimpleBufferize(rank, valueOf(NUM_ENTRIES), valueOf(ENTRY_SIZE));
    let rpt <- mkRepeatStatic(2);
    let drained <- mkReg(0);

    Reg#(UInt#(32)) state <- mkReg(1);

    rule push if (state % fromInteger(valueOf(ENTRY_SIZE)) != 0 && state < 10 * fromInteger(valueOf(ENTRY_SIZE)));
        let data = tagged Tag_Data (tuple2(Tag_Tile (0), 0));
        dut.put_data(data);
        state <= state + 1;
    endrule

    rule push_2 if (state % fromInteger(valueOf(ENTRY_SIZE)) == 0);
        let data = tagged Tag_Data (tuple2(Tag_Tile (1), rank));
        dut.put_data(data);
        state <= state + 1;
        $display("pushed everything.");
    endrule

    rule token_output_to_repeat_input;
        let data <- dut.get_token();
        // data = tagged Tag_Data (tuple2(tpl_1(data.Tag_Data), tpl_2(data.Tag_Data) + 1));
        $display("Forwarded to repeat.");
        rpt.put(0, data);
    endrule

    rule repeat_output_to_bufferize_input;
        let data <- rpt.get(0);
        // $display("Repeat output: ", fshow(data));
        dut.request_token(data);
    endrule

    rule drain_result;
        let data <- dut.get_data();
        $display("data: ", fshow(data), "end.");
        drained <= drained + 1;
        // if (drained == 9) begin
        //     $finish(0);
        // end
    endrule

endmodule