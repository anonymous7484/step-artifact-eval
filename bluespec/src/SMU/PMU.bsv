// TODO: Make everything work in fifo order. I think this should immediately work as long as I emit the token earlier. 
    // I already have the logic necessary to make sure i don't read data that hasn't been stored (untested)

package PMU;

import FIFO::*;
import FIFOF::*;
import Types::*;
import Vector::*;
import RegFile::*;
import ConfigReg::*;
import BankedMemory::*;
import Parameters::*;
import SetFreeList::*;
import SetUsageTracker::*;
import Unwrap::*;
import Operation::*;

// Tracking the current storage location
typedef struct {
    SET_INDEX set;
    FRAME_INDEX frame;
    Bool valid;
} StorageLocation deriving(Bits, Eq);

typedef UInt#(TAdd#(TLog#(SETS), TLog#(FRAMES_PER_SET))) StorageAddr;

// Interface that just exposes the modules existence
interface PMU_IFC;
    interface Operation_IFC operation;
    
    method Action put_data(ChannelMessage msg);
    method ActionValue#(ChannelMessage) get_token();
    method Action put_token(ChannelMessage msg);
    method ActionValue#(ChannelMessage) get_data();
endinterface


// PMU module that processes between the FIFOs
module mkPMU#(
    Int#(32) rank_in                    // Rank of the current tile
)(PMU_IFC);
    FIFO#(ChannelMessage) data_in <- mkFIFO;      // Values come in here
    FIFO#(ChannelMessage) token_out <- mkFIFO;    // Generated tokens go out here
    FIFO#(ChannelMessage) token_in <- mkFIFO;     // Tokens come back in here
    FIFO#(ChannelMessage) data_out <- mkFIFO;     // Retrieved values go out here
    
    // Only internal state needed is the storage FIFO and token counter
    BankedMemory_IFC mem <- mkBankedMemory;
    SetFreeList_IFC free_list <- mkSetFreeList;
    SetUsageTracker_IFC usage_tracker <- mkSetUsageTracker;
    Reg#(StorageLocation) curr_loc <- mkReg(StorageLocation { set: 0, frame: 0, valid: False });
    Reg#(Int#(32)) rank <- mkReg(rank_in);
    ConfigReg#(Bit#(TLog#(MAX_ENTRIES))) token_counter <- mkConfigReg(0);
    Reg#(Bit#(32)) cycle_count <- mkReg(0);

    RegFile#(Bit#(TLog#(MAX_ENTRIES)), Maybe#(Bit#(TLog#(MAX_ENTRIES)))) first_entry <- mkRegFileWCF(0, fromInteger(valueOf(MAX_ENTRIES) - 1));    
    RegFile#(Bit#(TLog#(MAX_ENTRIES)), Bool) token_complete <- mkRegFileWCF(0, fromInteger(valueOf(MAX_ENTRIES) - 1));
    Vector#(MAX_ENTRIES, ConfigReg#(Maybe#(Bit#(TLog#(MAX_ENTRIES))))) next_table <- replicateM(mkConfigReg(tagged Invalid));
    // Vector#(MAX_ENTRIES, ConfigReg#(Bit#(TLog#(MAX_ENTRIES)))) pmu_id_table <- replicateM(mkConfigReg(0)); // TODO: Unused for static
    Reg#(Bit#(TLog#(MAX_ENTRIES))) prev_stored <- mkReg(0);

    let frame_width = valueOf(TLog#(FRAMES_PER_SET));
    let set_width = valueOf(TLog#(SETS));
    Reg#(Maybe#(Tuple4#(Bit#(TLog#(MAX_ENTRIES)), Bool, StopToken, Bit#(TLog#(MAX_ENTRIES))))) load_token <- mkReg(tagged Invalid);

    Reg#(Bool) initialized <- mkReg(False);
    Reg#(Bit#(TLog#(MAX_ENTRIES))) first_entry_initialized_index <- mkReg(0);

    ConfigReg#(Maybe#(SET_INDEX)) next_free_set <- mkConfigReg(tagged Invalid);
    ConfigReg#(Maybe#(Bit#(TLog#(MAX_ENTRIES)))) deallocate_reg <- mkConfigReg(tagged Invalid);

    (* execution_order = "cycle_counter, store_tile" *)
    (* execution_order = "cycle_counter, next_free" *)
    (* execution_order = "cycle_counter, start_load_tile" *)
    (* execution_order = "cycle_counter, continue_load_tile" *)
    (* execution_order = "next_free, continue_load_tile" *)
    (* execution_order = "store_tile, next_free" *)

    (* descending_urgency = "deallocate_token, store_tile" *)

    rule initialization (!initialized);
        first_entry.upd(first_entry_initialized_index, tagged Invalid);
        token_complete.upd(first_entry_initialized_index, False);
        if (first_entry_initialized_index == fromInteger(valueOf(MAX_ENTRIES) - 1)) begin
            initialized <= True; 
        end else begin
            first_entry_initialized_index <= first_entry_initialized_index + 1;
        end
    endrule

    rule cycle_counter (initialized);
        // $display("-------------------------------------------");
        // $display("[CYCLE COUNTER]: %d", cycle_count);
        cycle_count <= cycle_count + 1;
    endrule

    rule deallocate_token (initialized &&& isValid(deallocate_reg));
        first_entry.upd(truncate(deallocate_reg.Valid), tagged Invalid);
        token_complete.upd(truncate(deallocate_reg.Valid), False);
        deallocate_reg <= tagged Invalid;
    endrule

    rule next_free (initialized && !isValid(next_free_set));
        // $display("Fired next_free");
        let mset <- free_list.allocSet();
        case (mset) matches
            tagged Valid .s: begin
                next_free_set <= tagged Valid s;
            end
            default: begin
                if (cycle_count % 1000 == 0) begin
                    // $display("***** Out of memory at cycle %d, load token validity is %d *****", cycle_count, isValid(load_token));
                end
                next_free_set <= tagged Invalid;
            end
        endcase
    endrule

    rule store_tile (initialized && (isValid(next_free_set) || curr_loc.valid));
        $display("Fired store_tile");
        let d_in = data_in.first;
        data_in.deq;


        case (d_in) matches
            tagged Tag_Data {.tt, .st}: begin
                let tile = unwrapTile(tt);
                let new_st = st;
                let emit_token = False;
                if (st >= rank) begin
                    new_st = st - rank;
                    emit_token = True;
                end

                StorageLocation new_loc = curr_loc;

                let set = 0;
                let frame = 0;

                if (!curr_loc.valid) begin
                    set = next_free_set.Valid;
                    frame = 0;
                    mem.write(set, 0, TaggedTile { t: tile, st: st });
                    usage_tracker.setFrame(set, 1);
                    new_loc = StorageLocation { set: set, frame: 1, valid: True };
                    next_free_set <= tagged Invalid;
                    // $display("[DEBUG]: !curr_loc.valid: tile in set %d, frame %d", set, frame);
                end else begin
                    // $display("[DEBUG]: Storing tile in set %d, frame %d", curr_loc.set, curr_loc.frame);
                    set = curr_loc.set;
                    frame = curr_loc.frame;

                    mem.write(set, frame, TaggedTile { t: tile, st: st });
                    let full <- usage_tracker.incFrame(set);
                    // $display("[DEBUG]: full before: %d", full);
                    full = full || emit_token;
                    // $display("[DEBUG]: full: %d", full);

                    new_loc = StorageLocation {
                        set: set,
                        frame: frame + 1,
                        valid: !full
                    };
                end

                if (first_entry.sub(token_counter) matches tagged Invalid) begin
                    let p = pack({set, frame});
                    first_entry.upd(token_counter, tagged Valid p);
                    next_table[p] <= tagged Invalid;
                    // pmu_id_table[p] <= 0; // TODO: Dynamic pmu ids
                    prev_stored <= p;
                end else begin
                    let p = pack({set, frame});
                    next_table[prev_stored] <= tagged Valid p;
                    if (prev_stored != p) begin
                        next_table[p] <= tagged Invalid;
                    end
                    prev_stored <= p;
                    // pmu_id_table[p] <= 0; // TODO: Dynamic pmu ids
                end

                if (emit_token) begin
                    Bit#(32) token_to_emit = zeroExtend(token_counter);
                    token_counter <= token_counter + 1;
                    token_complete.upd(token_counter, True);
                    // $display("Enqueuing output token: ", fshow(tagged Tag_Data tuple2(tagged Tag_Ref tuple2(token_to_emit, True), new_st)));
                    token_out.enq(tagged Tag_Data tuple2(tagged Tag_Ref tuple2(token_to_emit, True), new_st));
                end
                curr_loc <= new_loc;
            end
            tagged Tag_Instruction .ip: begin
                $display("[ERROR]: Instruction received in data input"); // TODO: Should be able to accept incoming config
                $finish(0);
            end
            tagged Tag_EndToken .et: begin
                token_out.enq(tagged Tag_EndToken et);
                $display("End token received in store");
            end
        endcase
    endrule

    // Rule to handle token retrievals: find matching value and return it
    rule start_load_tile (!isValid(load_token) && initialized);
        $display("Fired start load tile");
        let token_msg = token_in.first;
        token_in.deq;
        
        case (token_msg) matches
            tagged Tag_Data {.tt, .st}: begin
                let token_input = unwrapRef(tt);
                let maybe_packed_loc = first_entry.sub(truncate(token_input.fst));
                if (maybe_packed_loc matches tagged Valid .p) begin
                    load_token <= tagged Valid tuple4(p, token_input.snd, st, truncate(token_input.fst));
                    // $display("[DEBUG]: starting load tile at cycle %d, load_token %d", cycle_count, tpl_1(load_token.Valid));
                end else begin
                    $display("[ERROR]: No valid location found for token %d", token_input.fst);
                    $finish(0);
                end
            end
            tagged Tag_EndToken .et: begin
                // Print the state of the memory being used
                for (Integer i = 0; i < valueOf(SETS); i = i + 1) begin
                    // Check if set is valid
                    if (!free_list.isSetFree(fromInteger(i))) begin
                        $display("[MEMORY USAGE]: Set %d: %d frames used", i, usage_tracker.getCount(fromInteger(i)));
                    end else begin
                        $display("[MEMORY USAGE]: Set %d: No frames used", i);
                    end
                end
                data_out.enq(tagged Tag_EndToken et);
                $display("End token received");
            end
            default: begin
                $display("[ERROR]: Expected data message with token");
                $finish(0);
            end
        endcase
    endrule

    rule continue_load_tile (isValid(load_token) && initialized && !isValid(deallocate_reg));
        $display("Fired continue load tile");
        let loc_table_entry = tpl_1(load_token.Valid); // {set, frame}
        let deallocate = tpl_2(load_token.Valid);
        let st = tpl_3(load_token.Valid);
        let orig_token = tpl_4(load_token.Valid);

        Bool token_is_complete = token_complete.sub(orig_token);
        Bool has_next_piece = isValid(next_table[loc_table_entry]);

        if (token_is_complete || has_next_piece) begin
            SET_INDEX set   = truncate(loc_table_entry >> valueOf(TLog#(FRAMES_PER_SET)));
            FRAME_INDEX frame = truncate(loc_table_entry);
            let tile = mem.read(set, frame);
            let out_rank = tile.st;

            // Check if weve processed all entries for this token
            let next_set = set + 1;
            if (next_table[loc_table_entry] matches tagged Valid .next_loc) begin
                next_set = truncate(next_loc >> valueOf(TLog#(FRAMES_PER_SET)));
                load_token <= tagged Valid tuple4(next_loc, deallocate, st, orig_token);
                // $display("Had next, shouldn't update out_rank which is %d", out_rank);
            end else begin
                load_token <= tagged Invalid;
                // $display("Out rank is %d, st is %d, rank is %d", out_rank, st, rank);
                out_rank = rank + st;
                if (deallocate) begin
                    deallocate_reg <= tagged Valid truncate(orig_token);
                end
                // first_entry.upd(token_counter, tagged Invalid); // TODO: Need this eventually but creates conflicts
            end
            
            // $display("Enqueuing output data: %d", out_rank);
            data_out.enq(tagged Tag_Data tuple2(tagged Tag_Tile tile.t, out_rank));

            if (deallocate) begin
                if (set != next_set) begin
                    free_list.freeSet(set); 
                    // $display("[DEBUG]: Freed set %d", set);
                end 
            end
        end
    endrule

    method Action put_data(ChannelMessage msg);
        data_in.enq(msg);
    endmethod
    method ActionValue#(ChannelMessage) get_token();
        let msg = token_out.first;
        token_out.deq;
        return msg;
    endmethod
    method Action put_token(ChannelMessage msg);
        token_in.enq(msg);
    endmethod
    method ActionValue#(ChannelMessage) get_data();
        let msg = data_out.first;
        data_out.deq;
        return msg;
    endmethod

    interface Operation_IFC operation;
        method Action put(Int#(32) i, ChannelMessage msg); // i = 0 for data, 1 for token
            if (i == 0) begin
                data_in.enq(msg);
            end else begin
                token_in.enq(msg);
            end
        endmethod

        method ActionValue#(ChannelMessage) get(Int#(32) i); // i = 0 for token, 1 for data
            ChannelMessage msg;
            if (i == 0) begin
                token_out.deq;
                msg = token_out.first;
            end else begin
                data_out.deq;
                msg = data_out.first;
            end
            return msg;
        endmethod
    endinterface

endmodule

typedef 1 NUM_ENTRIES;
typedef 4 ENTRY_SIZE;

module mkTestPMUStopToken(Empty);
    let rank = 3;
    PMU_IFC dut <- mkPMU(rank);
    let rpt <- mkRepeatStatic(2);
    let drained <- mkReg(0);

    Reg#(UInt#(32)) state <- mkReg(1);

    rule push if (state % fromInteger(valueOf(ENTRY_SIZE)) != 0 && state < 10 * fromInteger(valueOf(ENTRY_SIZE)));
        let data = tagged Tag_Data (tuple2(tagged Tag_Tile (0), 0));
        dut.put_data(data);
        state <= state + 1;
    endrule

    rule push_2 if (state % fromInteger(valueOf(ENTRY_SIZE)) == 0);
        let data = tagged Tag_Data (tuple2(tagged Tag_Tile (1), rank));
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

    rule repeat_output_to_pmu_input;
        let data <- rpt.get(0);
        // $display("Repeat output: ", fshow(data));
        dut.put_token(data);
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

endpackage