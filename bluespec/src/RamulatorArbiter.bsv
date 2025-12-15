import Vector::*;
import FIFO::*;
import FIFOF::*;

import RamulatorFIFO::*;

interface RamulatorArbiterIO;
    method Action send_request(Bit#(8) port_id, Bit#(56) addr, Bool is_write);
    method ActionValue#(Bit#(56)) get_response(Bit#(8) port_id);
endinterface

interface RamulatorArbiter_IFC #(numeric type num_ports);
    interface RamulatorArbiterIO ports;
endinterface

module mkRamulatorArbiter#(Integer num_ports) (RamulatorArbiter_IFC#(num_ports));

    Reg#(Bit#(32)) cycle <- mkReg(0);

    Vector#(num_ports, FIFOF#(Tuple2#(Bit#(56), Bool))) requests <- replicateM(mkFIFOF);
    
    Vector#(num_ports, FIFOF#(Bit#(56))) responses_to_world <- replicateM(mkSizedFIFOF(128)); // These two fifos are split to make detecting a fill-up threshold possible.
    Vector#(num_ports, FIFOF#(Bit#(56))) responses_buffer <- replicateM(mkSizedFIFOF(128)); // This one only is to catch in-flight responses when the output suddenly blocks. 

    FIFOF#(Tuple2#(Bit#(64), Bool)) ramulator_requests <- mkFIFOF;
    FIFO#(Bit#(64)) ramulator_responses <- mkFIFO;
    RamulatorFIFO_IFC ramulator <- mkRamulatorFIFO;

    Reg#(Bit#(8)) enqueue_idx <- mkReg(0);
    Wire#(Bit#(8)) next_enqueue_idx <- mkWire;

    for (Integer i = 0; i < valueOf(num_ports); i = i + 1) begin
        (* preempts = "enqueue_requests, nothing" *)
        rule enqueue_requests if (next_enqueue_idx == fromInteger(i));
            if (responses_to_world[i].notFull) begin
                // $display("We find it absolutely fine to enqueue request from port %d", i);
                ramulator_requests.enq(
                    tuple2(
                        {fromInteger(i), tpl_1(requests[i].first)},
                        tpl_2(requests[i].first)
                    )
                );
                requests[i].deq;
            end else begin 
                $display("responses buffer not empty, thats why we don't enqueue.", i);
            end
        endrule

        rule nothing;
            let rq = requests[i].first;
            let t = responses_to_world[i].notFull;
            let ram_empty = ramulator_requests.notFull;
            // $display("Nothing to do at cycle %d. not full? %d, ramulator empty? %d, request: %d", cycle, t, ram_empty, rq);
        endrule
    end

    rule requests_empty;
        let rq = requests[0].first;
        // $display("Request: %d at cc %d", rq, cycle);
    endrule 

    rule cc;
        cycle <= cycle + 1;
    endrule

    rule get_next_enqueue_idx;
        Bool found = False;
        for (Bit#(8) i = 0; i < fromInteger(valueOf(num_ports)); i = i + 1) begin
            let idx = (enqueue_idx + i) % fromInteger(valueOf(num_ports));
            if (requests[idx].notEmpty && !found) begin
                next_enqueue_idx <= idx;
                found = True;
                enqueue_idx <= (idx + 1) % fromInteger(valueOf(num_ports));
            end
        end
    endrule

    rule feed_ramulator;
        ramulator.send_request(tpl_1(ramulator_requests.first), tpl_2(ramulator_requests.first));
        ramulator_requests.deq;
    endrule

    rule drain_ramulator;
        let response <- ramulator.get_response;
        ramulator_responses.enq(response);
    endrule

    (* preempts = "dequeue_responses, drain_idle" *)
    rule dequeue_responses;
        let response = ramulator_responses.first;
        ramulator_responses.deq;
        // $display("Dequeued ramulator response for 0x%x at cycle %d", response, cycle);
        let port_id = response[63:56];
        let addr = response[55:0];
        responses_buffer[port_id].enq(addr);
    endrule

    rule drain_idle;
        // $display("Draining idle at cycle %d", cycle);
        // $display("Response in pipeline: ", fshow(ramulator_responses.first));
    endrule

    for (Integer i = 0; i < valueOf(num_ports); i = i + 1) begin
        rule drain_responses_buffer;
            responses_to_world[i].enq(responses_buffer[i].first);
            responses_buffer[i].deq;
        endrule
    end

    interface RamulatorArbiterIO ports;
        method Action send_request(Bit#(8) port_id, Bit#(56) addr, Bool is_write);
            requests[port_id].enq(tuple2(addr, is_write));
        endmethod 

        method ActionValue#(Bit#(56)) get_response(Bit#(8) port_id);
            responses_to_world[port_id].deq;
            return responses_to_world[port_id].first;
        endmethod

    endinterface
endmodule

module mkRamulatorArbiterTest(Empty);
    RamulatorArbiter_IFC#(2) arbiter <- mkRamulatorArbiter(2);

    let num_requests = 1000;

    Reg#(Bit#(32)) sent_requests_a <- mkReg(0);
    Reg#(Bit#(32)) sent_requests_b <- mkReg(0);

    Reg#(Bit#(32)) received_responses_a <- mkReg(0);
    Reg#(Bit#(32)) received_responses_b <- mkReg(0);

    Reg#(Bit#(32)) cycle <- mkReg(0);

    rule cc;
        cycle <= cycle + 1;
    endrule

    rule send_request_a if (sent_requests_a < num_requests);
        // $display("Sent a request (a)");
        arbiter.ports.send_request(0, extend(sent_requests_a), False);
        sent_requests_a <= sent_requests_a + 1;
    endrule

    rule send_request_b if (sent_requests_b < num_requests);
        // $display("Sent a request (b)");
        arbiter.ports.send_request(1, extend(num_requests + sent_requests_b), False);
        sent_requests_b <= sent_requests_b + 1;
    endrule

    rule get_response_a if (received_responses_a < num_requests);
        let addr <- arbiter.ports.get_response(0);
        // $display("Response: %d at cycle %d", addr, cycle);
        received_responses_a <= received_responses_a + 1;
        if (addr != extend(received_responses_a)) begin
            $display("Error: Response %d does not match expected response %d", addr, received_responses_a);
            // $finish(1);
        end
    endrule

    rule get_response_b if (received_responses_b < num_requests);
        let addr <- arbiter.ports.get_response(1);
        $display("Response: %d at cycle %d", addr, cycle);
        received_responses_b <= received_responses_b + 1;
        if (addr != extend(num_requests + received_responses_b)) begin
            $display("Error: Response %d does not match expected response %d", addr, num_requests + received_responses_b);
            // $finish(1);
        end
    endrule

    rule exit if (received_responses_a == num_requests && received_responses_b == num_requests);
        $display("SUCCESS at cycle %d", cycle);
        $finish(0);
    endrule
endmodule

module mkRamulatorArbiterSinglePortTest(Empty);
    RamulatorArbiter_IFC#(1) arbiter <- mkRamulatorArbiter(1);

    let num_requests = 32;

    Reg#(Bit#(32)) sent_requests <- mkReg(0);
    Reg#(Bit#(32)) received_responses <- mkReg(0);

    Reg#(Bit#(32)) cycle <- mkReg(0);

    rule cc;
        cycle <= cycle + 1;
    endrule
    
    rule send_request if (sent_requests < num_requests);
        arbiter.ports.send_request(0, extend(sent_requests), False);
        sent_requests <= sent_requests + 1;
    endrule

    rule get_response if (received_responses < num_requests);
        let addr <- arbiter.ports.get_response(0);
        $display("Response: %d at cycle %d", addr, cycle);
        received_responses <= received_responses + 1;
        if (addr != extend(received_responses)) begin
            $display("Error: Response %d does not match expected response %d", addr, received_responses);
            $finish(1);
        end
    endrule

    rule exit if (received_responses == num_requests);
        $display("SUCCESS at cycle %d", cycle);
        $finish(0);
    endrule
endmodule