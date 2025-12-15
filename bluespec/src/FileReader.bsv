package FileReader;

import Types::*;
import Vector::*;
import FShow::*;
import FIFOF::*;
import RegFile::*;
import RamulatorArbiter::*;
import Parameters::*;
import Assert::*;

interface FileReader_IFC#(type entry_type);
    method ActionValue#(entry_type) readNext();
    method Bool isDone();
endinterface

module mkFileReader#(
    Integer num_entries, 
    String filename,
    Bit#(8) index,
    RamulatorArbiterIO arbiter
    ) (FileReader_IFC#(entry_type)) provisos (Bits#(entry_type, entry_size));

    Reg#(Bit#(64)) read_ptr <- mkReg(0); // Iterates through all addresses in the file
    Reg#(Bit#(64)) entry_idx <- mkReg(0); // Iterates through all entries in the file

    // aggregate responses into entry_type
    FIFOF#(entry_type) output_fifo <- mkFIFOF;
    FIFOF#(Bit#(64)) ramulator_response <- mkFIFOF;

    // Actual data
    RegFile#(Bit#(64), Bit#(entry_size)) entry_regfile <- mkRegFileLoad(filename, 0, fromInteger(num_entries));

    rule send_read_request if (read_ptr + fromInteger(valueOf(RAMULATOR_BITS_PER_REQUEST)) / 8 < fromInteger(num_entries) * fromInteger(valueOf(TaggedTileSize)) / 8);
        arbiter.send_request(index, truncate(read_ptr), False);
        read_ptr <= read_ptr + fromInteger(valueOf(RAMULATOR_BITS_PER_REQUEST)) / 8;
        $display("Request!");
    endrule

    rule ramulator_output_to_fifo;
        let response <- arbiter.get_response(index);
        ramulator_response.enq(extend(response));
    endrule

    // There is a new response, but we don't have enough data for the next tile yet.
    rule dequeue_response if (ramulator_response.first * 8 + fromInteger(valueOf(RAMULATOR_BITS_PER_REQUEST)) < entry_idx * fromInteger(valueOf(TaggedTileSize)));
        ramulator_response.deq;
    endrule

    rule dequeue_and_send_tile if (ramulator_response.first * 8 + fromInteger(valueOf(RAMULATOR_BITS_PER_REQUEST)) >= entry_idx * fromInteger(valueOf(TaggedTileSize)));
        let new_entry_idx = entry_idx + 1;
        entry_idx <= new_entry_idx;
        output_fifo.enq(unpack(entry_regfile.sub(pack(entry_idx))));

        // If we don't have enought data for the tile afterwards this one, dequeue the response
        if (ramulator_response.first * 8 + fromInteger(valueOf(RAMULATOR_BITS_PER_REQUEST)) < new_entry_idx * fromInteger(valueOf(TaggedTileSize))) begin
            ramulator_response.deq;
        end
    endrule

    method Bool isDone();
        return !output_fifo.notEmpty && entry_idx >= fromInteger(num_entries);
    endmethod

    method ActionValue#(entry_type) readNext();
        output_fifo.deq;
        return output_fifo.first;
    endmethod
endmodule

interface FileWriter_IFC#(type entry_type);
    method Action writeNext(entry_type data);
endinterface

module mkDumpingFileWriter#( // dummy file writer that just dumps data
    Bit#(8) index,
    RamulatorArbiterIO arbiter
    ) (FileWriter_IFC#(entry_type)) provisos (Bits#(entry_type, entry_size));

    Reg#(Bit#(64)) write_ptr <- mkReg(0); // Iterates through all addresses in the file
    Reg#(Bit#(64)) entry_idx <- mkReg(0); // Iterates through all entries in the file
    Reg#(Bit#(64)) num_entries <- mkReg(0);
    Reg#(Bit#(64)) cycle_count <- mkReg(0);

    // aggregate responses into entry_type
    FIFOF#(entry_type) output_fifo <- mkFIFOF;
    FIFOF#(Bit#(64)) ramulator_response <- mkFIFOF;

    rule send_write_request if (write_ptr + fromInteger(valueOf(RAMULATOR_BITS_PER_REQUEST)) / 8 < num_entries * fromInteger(valueOf(TaggedTileSize)) / 8);
        arbiter.send_request(index, truncate(write_ptr), True);
        write_ptr <= write_ptr + fromInteger(valueOf(RAMULATOR_BITS_PER_REQUEST)) / 8;
        $display("Request!");
    endrule

    rule ramulator_output_to_fifo;
        let response <- arbiter.get_response(index);
        ramulator_response.enq(extend(response));
    endrule

    // There is a new response, but we don't have enough data for the next tile yet.
    rule dequeue_response if (ramulator_response.first * 8 + fromInteger(valueOf(RAMULATOR_BITS_PER_REQUEST)) < entry_idx * fromInteger(valueOf(TaggedTileSize)));
        ramulator_response.deq;
    endrule

    rule dequeue if (ramulator_response.first * 8 + fromInteger(valueOf(RAMULATOR_BITS_PER_REQUEST)) >= entry_idx * fromInteger(valueOf(TaggedTileSize)));
        let new_entry_idx = entry_idx + 1;
        entry_idx <= new_entry_idx;

        // If we don't have enought data for the tile afterwards this one, dequeue the response
        if (ramulator_response.first * 8 + fromInteger(valueOf(RAMULATOR_BITS_PER_REQUEST)) < new_entry_idx * fromInteger(valueOf(TaggedTileSize))) begin
            ramulator_response.deq;
        end
    endrule

    rule cc;
        cycle_count <= cycle_count + 1;
    endrule

    method Action writeNext(entry_type data) if (write_ptr + fromInteger(valueOf(RAMULATOR_BITS_PER_REQUEST)) / 8 >= num_entries * fromInteger(valueOf(TaggedTileSize)) / 8);
        num_entries <= num_entries + 1;
    endmethod
endmodule

module mkFileReaderTest(Empty);
    RamulatorArbiter_IFC#(1) arbiter <- mkRamulatorArbiter(1);
    FileReader_IFC#(TaggedTile) reader <- mkFileReader(32, "step_paper_0/address_reader_12.hex", 0, arbiter.ports);

    Reg#(Bit#(64)) cycle_count <- mkReg(0);

    rule tick;
        cycle_count <= cycle_count + 1;
    endrule

    rule read_tile;
        let tile <- reader.readNext;
        $display("Read tile at cycle %d:", cycle_count, fshow(tile));
        // $finish(0);
    endrule

    rule finish if (reader.isDone);
        $finish(0);
    endrule
endmodule

endpackage