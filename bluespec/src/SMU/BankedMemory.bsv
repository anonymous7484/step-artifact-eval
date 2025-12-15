package BankedMemory;

import Vector::*;
import RegFile::*;
import ConfigReg::*;
import Types::*;
import Parameters::*;

// === Configuration ===

typedef Bit#(TMul#(TILE_SIZE, SizeOf#(Scalar))) TileRow;
typedef UInt#(TAdd#(TLog#(SETS), TLog#(FRAMES_PER_SET))) BankAddr;

// === Interface ===
interface BankedMemory_IFC;
    method Action write(SET_INDEX set, FRAME_INDEX frame, TaggedTile tile);
    method TaggedTile read(SET_INDEX set, FRAME_INDEX frame);
endinterface

// === Implementation ===
module mkBankedMemory(BankedMemory_IFC);

    // Each bank holds FRAMES_PER_SET * SETS TileRows
    Vector#(TILE_SIZE, RegFile#(BankAddr, TileRow)) banks <- replicateM(mkRegFileWCF(0, fromInteger(valueOf(MAX_ENTRIES) - 1)));

    // Separate storage for StopTokens, shared across all banks
    RegFile#(BankAddr, StopToken) stop_token_mem <- mkRegFileWCF(0, fromInteger(valueOf(MAX_ENTRIES) - 1));

    // Compute flat address
    function BankAddr flatAddr(SET_INDEX set, FRAME_INDEX frame);
        return unpack(zeroExtend(pack(set)) << valueOf(TLog#(FRAMES_PER_SET)) | zeroExtend(pack(frame)));
    endfunction

    method Action write(SET_INDEX set, FRAME_INDEX frame, TaggedTile tile);
        BankAddr addr = flatAddr(set, frame);
        Vector#(TILE_SIZE, TileRow) rows = unpack(pack(tile.t));

        for (Integer i = 0; i < valueOf(TILE_SIZE); i = i + 1)
            banks[i].upd(addr, rows[i]);

        stop_token_mem.upd(addr, tile.st);
    endmethod

    method TaggedTile read(SET_INDEX set, FRAME_INDEX frame);
        BankAddr addr = flatAddr(set, frame);
        Vector#(TILE_SIZE, TileRow) rows;

        for (Integer i = 0; i < valueOf(TILE_SIZE); i = i + 1)
            rows[i] = banks[i].sub(addr);

        TaggedTile result;
        result.t = unpack(pack(rows));
        result.st = stop_token_mem.sub(addr);
        return result;
    endmethod

endmodule

endpackage
