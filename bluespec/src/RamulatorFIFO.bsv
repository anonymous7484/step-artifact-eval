// This module takes the responses from ramulator and returns them in FIFO request/response order. 
// This is required since ramulator2 dropped the FIFO interface.. 

import FIFO::*;
import FIFOF::*;
import SpecialFIFOs::*;
import RegFile::*;
import Vector::*;
import RamulatorInterface::*;
import Parameters::*;

interface RamulatorFIFO_IFC;
    method Action send_request(Bit#(64) addr, Bool is_write);
    method ActionValue#(Bit#(64)) get_response();
endinterface

typedef RAMULATOR_REORDER_WINDOW_SIZE REORDER_WINDOW_SIZE;
typedef Bit#(TLog#(REORDER_WINDOW_SIZE)) Tag;

module mkRamulatorFIFO(RamulatorFIFO_IFC);
    Reg#(Bit#(32)) cycle <- mkReg(0);

    FIFO#(Tuple2#(Bit#(64), Bool)) in <- mkFIFO;
    FIFOF#(Bit#(64)) out <- mkFIFOF;
    Ramulator_IFC ramulator <- mkRamulator;

    Vector#(REORDER_WINDOW_SIZE, Reg#(Maybe#(Bit#(64)))) requests <- replicateM(mkReg(tagged Invalid));
    FIFO#(Bit#(64)) in_flight <- mkSizedFIFO(valueOf(REORDER_WINDOW_SIZE));
    
    Wire#(Bit#(TLog#(REORDER_WINDOW_SIZE))) next_location <- mkWire;
    Wire#(Bit#(TLog#(REORDER_WINDOW_SIZE))) free_location <- mkWire;

    rule cc;
        cycle <= cycle + 1;
    endrule

    rule put_req;
        let addr = tpl_1(in.first);
        let is_write = tpl_2(in.first);
        ramulator.send_request(addr, is_write);
        if (!is_write) begin
            in_flight.enq(addr);
        end
        in.deq;
        // $display("Sent request to ramulator: 0x%x", addr);
    endrule

    function Bool not_full(Reg#(Maybe#(Bit#(64))) maybe_val);
        return (maybe_val matches tagged Invalid ? True : False);
    endfunction

    rule find_free;
        let position = findIndex(not_full, requests);
        if (position matches tagged Valid .p) begin
            free_location <= pack(p);
        end
    endrule 

    rule get_resp;
        let response <- ramulator.get_response();
        // $display("Ramulator response: 0x%x, enqueing to location %d. Requests rn: ", response, free_location, fshow(requests));
        requests[free_location] <= tagged Valid response;
        // $display("Enqueued the response for 0x%x", response);
    endrule

    function Bool is_next_response(Bit#(64) addr, Reg#(Maybe#(Bit#(64))) maybe_val);
        return (maybe_val matches tagged Valid .v ? (addr == v) : False);
    endfunction

    rule find_next;
        let position = findIndex(is_next_response(in_flight.first), requests);
        if (position matches tagged Valid .p) begin
            // $display("Found the response for 0x%x", in_flight.first);
            next_location <= pack(p);
        end
    endrule

    rule drain_next;
        let response = requests[next_location].Valid;
        requests[next_location] <= tagged Invalid;
        in_flight.deq;
        // $display("Drained the response for 0x%x", in_flight.first);
        out.enq(response);
    endrule

    function Fmt get_value(Reg#(Maybe#(Bit#(64))) maybe_val);
        return (maybe_val matches tagged Valid .v ? fshow(v) : fshow("Invalid"));
    endfunction

    rule reorder_buffer_full;
        Int#(32) full = 0;
        for (Integer i = 0; i < fromInteger(valueOf(REORDER_WINDOW_SIZE)); i = i + 1) begin
            if (requests[i] matches tagged Valid ._) begin
                full = full + 1;
            end
        end
        if (full == fromInteger(valueOf(REORDER_WINDOW_SIZE))) begin
            let elements = map(get_value, requests);
            $display("Reorder buffer is full at cycle %d", cycle);
            $display("The output buffer is blocking? ", !out.notFull);
            $display("in out, we currently have: ", fshow(out.first));
            $display("The reorder buffer is: ", fshow(elements));
            $finish(0);
        end
    endrule

    method Action send_request(Bit#(64) addr, Bool is_write);
        // $display("Sending request to ramulator: 0x%x", addr);
        in.enq(tuple2(addr, is_write));
    endmethod
    
    method ActionValue#(Bit#(64)) get_response();
        // $display("Sent out response for 0x%x at cycle %d", out.first, cycle);
        out.deq;
        // $display("Dequeued the response for 0x%x", out.first);
        return out.first;
    endmethod

endmodule

module mkRamulatorFIFOTest(Empty);
    Reg#(Int#(32)) sent <- mkReg(0);
    Reg#(Int#(32)) received <- mkReg(0);
    Reg#(Int#(32)) cc <- mkReg(0);
    
    Int#(32) num_elements = 256;
    RamulatorFIFO_IFC fifo_ram <- mkRamulatorFIFO;
    
    rule send if (sent < num_elements);
        fifo_ram.send_request(zeroExtend(pack(sent)) << 8, False);
        sent <= sent + 1;
    endrule

    rule print_status;
        // $display("Sent: %d, Received: %d, Cycle: %d", sent, received, cc);
        cc <= cc + 1;
    endrule

    rule receive if (received < num_elements);
        let response <- fifo_ram.get_response();
        if (response == zeroExtend(pack(received) << 8)) begin
            received <= received + 1;
        end else begin
            $error("Received the wrong response for 0x%x", response);
            $finish(0);
        end
    endrule

    rule finish if (sent == num_elements && received == num_elements);
        $display("SUCCESS");
        $finish(0);
    endrule

    // rule error;
    //     if (cc > 1000) begin
    //         $display("FAILED");
    //         $finish(0);
    //     end
    // endrule
endmodule
