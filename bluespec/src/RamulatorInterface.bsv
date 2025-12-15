import FIFO::*;
import Parameters::*;

// General ramulator functions
import "BDPI" function Action init_ramulator();
import "BDPI" function Action free_ramulator();
import "BDPI" function ActionValue#(Bool) ramulator_send(Bit#(64) addr, Bool is_write);
import "BDPI" function Action ramulator_tick();
import "BDPI" function ActionValue#(Bit#(64)) ramulator_get_cycle();
import "BDPI" function Bool ramulator_ret_available();
import "BDPI" function ActionValue#(Bit#(64)) ramulator_pop();
import "BDPI" function ActionValue#(Bool) ramulator_is_finished();

// Byte-Addressed Interface
interface Ramulator_IFC;
    method Action send_request(Bit#(64) addr, Bool is_write);
    method ActionValue#(Bit#(64)) get_response();
endinterface

module mkRamulator(Ramulator_IFC);
    Reg#(Bit#(64)) cycle_count <- mkReg(0);
    Reg#(Bit#(64)) input_bubbles <- mkReg(0);
    Reg#(Bit#(64)) output_bubbles <- mkReg(0);
    
    Reg#(Bit#(64)) num_reads <- mkReg(0);
    Reg#(Bit#(64)) num_writes <- mkReg(0);

    Reg#(Bool) initialized <- mkReg(False);
    Reg#(Bool) finished <- mkReg(False);
    FIFO#(Tuple2#(Bit#(64), Bool)) requests <- mkFIFO;
    FIFO#(Bit#(64)) responses <- mkFIFO;

    rule init_ramulator(!initialized);
        init_ramulator();
        initialized <= True;
    endrule

    (* preempts = "(send_requests, bypass), bubble_input" *)
    rule send_requests(initialized &&& valueOf(USE_RAMULATOR) == 1);
        let success <- ramulator_send(tpl_1(requests.first), tpl_2(requests.first));
        if (success) begin
            if (tpl_2(requests.first)) begin
                num_writes <= num_writes + 1;
            end else begin
                num_reads <= num_reads + 1;
            end
            $display("Ramulator: Num_Reads: %d, Num_Writes: %d", num_reads, num_writes);
            // $display("Sending request to address: 0x%x at cycle %d", tpl_1(requests.first), cycle_count);
            // $display("Sent request to address: 0x%x at cycle %d", tpl_1(requests.first), cycle_count);
            requests.deq;
        end else begin 
            // $display("Failed to send request to ramulator, trying again next cycle");
        end
    endrule

    rule bubble_input if (initialized &&& valueOf(RAMULATOR_PRINT_BUBBLES) == 1);
        input_bubbles <= input_bubbles + 1;
        $error("Ramulator could have loaded data at cycle %d", cycle_count);
        $error("But no request was input. Ramulator input idle for: %d/%d cycles", input_bubbles+1, cycle_count);
    endrule

    rule incr_cycle if (initialized);
        cycle_count <= cycle_count + 1;
        ramulator_tick();
    endrule

    (* preempts = "drain, bubble" *)
    rule drain if (initialized &&& valueOf(USE_RAMULATOR) == 1);
        if (ramulator_ret_available()) begin
            let addr <- ramulator_pop();
            // $display("Received return for address: 0x%x at cycle %d", addr, cycle_count);
            responses.enq(addr);
        end
    endrule

    rule bypass if (valueOf(USE_RAMULATOR) == 0);
        responses.enq(tpl_1(requests.first));
        requests.deq;
    endrule

    rule bubble if (initialized &&& ramulator_ret_available() &&& valueOf(RAMULATOR_PRINT_BUBBLES) == 1);
        output_bubbles <= output_bubbles + 1;
        $error("Pipeline bubble (blocked output) at cycle %d", cycle_count);
        $error("Output blocked for: %d/%d cycles", output_bubbles+1, cycle_count);
    endrule

    method Action send_request(Bit#(64) addr, Bool is_write) if (initialized);
        requests.enq(tuple2(addr, is_write));
    endmethod

    method ActionValue#(Bit#(64)) get_response() if (initialized);
        responses.deq;
        return responses.first;
    endmethod
endmodule

module mkRamulatorTest(Empty);
    Ramulator_IFC ramulator <- mkRamulator;
    Reg#(Int#(32)) sent <- mkReg(0);
    Reg#(Int#(32)) received <- mkReg(0);
    Reg#(Bit#(64)) cycle_count <- mkReg(0);

    rule send_requests if (sent < 1024);
        ramulator.send_request(zeroExtend(pack(sent) << 8), False);
        sent <= sent + 1;
    endrule

    rule get_responses if (received < 1024);
        let response <- ramulator.get_response();
        // $display("Received response: 0x%x", response);
        received <= received + 1;
    endrule

    rule incr_cycle;
        cycle_count <= cycle_count + 1;
    endrule

    rule finish if (received == 1024);
        $display("Read all data at cycle %d", cycle_count);
        $finish;
    endrule
endmodule