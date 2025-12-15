package Step;
import Operation::*;
import Bufferize::*;
import Types::*;
import Vector::*;
import FShow::*;
import Debug::*;
import RamulatorArbiter::*;

module mkStep(Empty);
    RamulatorArbiter_IFC#(5) arbiter <- mkRamulatorArbiter(5);
    let mod_0_inner <- mkTileReader(2048, "gen_bsv/address_reader_0.hex", 0, arbiter.ports);
    let mod_0 <- mkDebugOperation(mod_0_inner, "mod_0");
    let mod_1_inner <- mkTileReader(2048, "gen_bsv/address_reader_1.hex", 1, arbiter.ports);
    let mod_1 <- mkDebugOperation(mod_1_inner, "mod_1");
    let mod_2_inner <- mkTileReader(2048, "gen_bsv/address_reader_2.hex", 2, arbiter.ports);
    let mod_2 <- mkDebugOperation(mod_2_inner, "mod_2");
    let mod_3_inner <- mkBroadcast2();
    let mod_3 <- mkDebugOperation(mod_3_inner, "mod_3");
    let mod_4_inner <- mkRepeatStatic(16);
    let mod_4 <- mkDebugOperation(mod_4_inner, "mod_4");
    Bufferize#(2, 1) mod_5_bufferize <- mkSimpleBufferize(1, 2, 1);
    let mod_5_inner = mod_5_bufferize.operation;
    let mod_5 <- mkDebugOperation(mod_5_inner, "mod_5");
    let mod_6_inner <- mkRepeatStatic(16);
    let mod_6 <- mkDebugOperation(mod_6_inner, "mod_6");
    Bufferize#(2, 16) mod_7_bufferize <- mkSimpleBufferize(2, 2, 16);
    let mod_7_inner = mod_7_bufferize.operation;
    let mod_7 <- mkDebugOperation(mod_7_inner, "mod_7");
    let mod_8_inner <- mkBinaryMap(8, matmul_t_tile);
    let mod_8 <- mkDebugOperation(mod_8_inner, "mod_8");
    let mod_9_inner <- mkAccum(add_tile, 1);
    let mod_9 <- mkDebugOperation(mod_9_inner, "mod_9");
    let mod_10_inner <- mkAccumBigTile(add_tile, 3);
    let mod_10 <- mkDebugOperation(mod_10_inner, "mod_10");
    Bufferize#(2, 1) mod_11_bufferize <- mkSimpleBufferize(2, 2, 1);
    let mod_11_inner = mod_11_bufferize.operation;
    let mod_11 <- mkDebugOperation(mod_11_inner, "mod_11");
    let mod_12_inner <- mkRepeatStatic(16);
    let mod_12 <- mkDebugOperation(mod_12_inner, "mod_12");
    Bufferize#(2, 1) mod_13_bufferize <- mkSimpleBufferize(1, 2, 1);
    let mod_13_inner = mod_13_bufferize.operation;
    let mod_13 <- mkDebugOperation(mod_13_inner, "mod_13");
    let mod_14_inner <- mkRepeatStatic(16);
    let mod_14 <- mkDebugOperation(mod_14_inner, "mod_14");
    Bufferize#(2, 16) mod_15_bufferize <- mkSimpleBufferize(2, 2, 16);
    let mod_15_inner = mod_15_bufferize.operation;
    let mod_15 <- mkDebugOperation(mod_15_inner, "mod_15");
    let mod_16_inner <- mkBinaryMap(16, matmul_t_tile);
    let mod_16 <- mkDebugOperation(mod_16_inner, "mod_16");
    let mod_17_inner <- mkAccum(add_tile, 1);
    let mod_17 <- mkDebugOperation(mod_17_inner, "mod_17");
    let mod_18_inner <- mkAccumBigTile(add_tile, 3);
    let mod_18 <- mkDebugOperation(mod_18_inner, "mod_18");
    Bufferize#(2, 1) mod_19_bufferize <- mkSimpleBufferize(2, 2, 1);
    let mod_19_inner = mod_19_bufferize.operation;
    let mod_19 <- mkDebugOperation(mod_19_inner, "mod_19");
    let mod_20_inner <- mkUnaryMap(20, silu_tile);
    let mod_20 <- mkDebugOperation(mod_20_inner, "mod_20");
    let mod_21_inner <- mkBinaryMap(21, mul_tile);
    let mod_21 <- mkDebugOperation(mod_21_inner, "mod_21");
    let mod_22_inner <- mkPromote(3);
    let mod_22 <- mkDebugOperation(mod_22_inner, "mod_22");
    let mod_23_inner <- mkTileReader(2048, "gen_bsv/address_reader_12.hex", 3, arbiter.ports);
    let mod_23 <- mkDebugOperation(mod_23_inner, "mod_23");
    let mod_24_inner <- mkRepeatStatic(1);
    let mod_24 <- mkDebugOperation(mod_24_inner, "mod_24");
    Bufferize#(2, 1) mod_25_bufferize <- mkSimpleBufferize(1, 2, 1);
    let mod_25_inner = mod_25_bufferize.operation;
    let mod_25 <- mkDebugOperation(mod_25_inner, "mod_25");
    let mod_26_inner <- mkRepeatStatic(16);
    let mod_26 <- mkDebugOperation(mod_26_inner, "mod_26");
    Bufferize#(2, 1) mod_27_bufferize <- mkSimpleBufferize(2, 2, 1);
    let mod_27_inner = mod_27_bufferize.operation;
    let mod_27 <- mkDebugOperation(mod_27_inner, "mod_27");
    let mod_28_inner <- mkBinaryMap(28, matmul_t_tile);
    let mod_28 <- mkDebugOperation(mod_28_inner, "mod_28");
    let mod_29_inner <- mkAccum(add_tile, 1);
    let mod_29 <- mkDebugOperation(mod_29_inner, "mod_29");
    let mod_30_inner <- mkAccumBigTile(add_tile, 3);
    let mod_30 <- mkDebugOperation(mod_30_inner, "mod_30");
    Bufferize#(2, 16) mod_31_bufferize <- mkSimpleBufferize(2, 2, 16);
    let mod_31_inner = mod_31_bufferize.operation;
    let mod_31 <- mkDebugOperation(mod_31_inner, "mod_31");
    let mod_32_inner <- mkTileWriter(4, arbiter.ports);
    let mod_32 <- mkDebugOperation(mod_32_inner, "mod_32");
    (* descending_urgency = "rule_1, rule_2, rule_3, rule_4, rule_5, rule_6, rule_7, rule_8, rule_9, rule_10, rule_11, rule_12, rule_13, rule_14, rule_15, rule_16, rule_17, rule_18, rule_19, rule_20, rule_21, rule_22, rule_23, rule_24, rule_25, rule_26, rule_27, rule_28, rule_29, rule_30, rule_31, rule_32, rule_33, rule_34, rule_35, rule_36, rule_37, rule_38, rule_39, rule_40, rule_41, rule_42, rule_43, rule_44, rule_45" *)
    rule rule_1;
        let t <- mod_14.get(0);
        mod_15.put(1, t);
    endrule
    rule rule_2;
        let t <- mod_18.get(0);
        mod_19.put(0, t);
    endrule
    rule rule_3;
        let t <- mod_17.get(0);
        mod_18.put(0, t);
    endrule
    rule rule_4;
        let t <- mod_12.get(0);
        mod_13.put(1, t);
    endrule
    rule rule_5;
        let t <- mod_15.get(1);
        mod_16.put(1, t);
    endrule
    rule rule_6;
        let t <- mod_27.get(1);
        mod_28.put(1, t);
    endrule
    rule rule_7;
        let t <- mod_5.get(1);
        mod_8.put(0, t);
    endrule
    rule rule_8;
        let t <- mod_2.get(0);
        mod_15.put(0, t);
    endrule
    rule rule_9;
        let t <- mod_31.get(0);
        mod_31.put(1, t);
    endrule
    rule rule_10;
        let t <- mod_10.get(0);
        mod_11.put(0, t);
    endrule
    rule rule_11;
        let t <- mod_13.get(1);
        mod_16.put(0, t);
    endrule
    rule rule_12;
        let t <- mod_0.get(0);
        mod_3.put(0, t);
    endrule
    rule rule_13;
        let t <- mod_3.get(0);
        mod_5.put(0, t);
    endrule
    rule rule_14;
        let t <- mod_6.get(0);
        mod_7.put(1, t);
    endrule
    rule rule_15;
        let t <- mod_19.get(0);
        mod_19.put(1, t);
    endrule
    rule rule_16;
        let t <- mod_21.get(0);
        mod_22.put(0, t);
    endrule
    rule rule_17;
        let t <- mod_28.get(0);
        mod_29.put(0, t);
    endrule
    rule rule_18;
        let t <- mod_30.get(0);
        mod_31.put(0, t);
    endrule
    rule rule_19;
        let t <- mod_7.get(0);
        mod_6.put(0, t);
    endrule
    rule rule_20;
        let t <- mod_24.get(0);
        mod_25.put(1, t);
    endrule
    rule rule_21;
        let t <- mod_15.get(0);
        mod_14.put(0, t);
    endrule
    rule rule_22;
        let t <- mod_29.get(0);
        mod_30.put(0, t);
    endrule
    rule rule_23;
        let t <- mod_13.get(0);
        mod_12.put(0, t);
    endrule
    rule rule_24;
        let t <- mod_18.get(1);
        mod_21.put(1, t);
    endrule
    rule rule_25;
        let t <- mod_23.get(0);
        mod_25.put(0, t);
    endrule
    rule rule_26;
        let t <- mod_8.get(0);
        mod_9.put(0, t);
    endrule
    rule rule_27;
        let t <- mod_7.get(1);
        mod_8.put(1, t);
    endrule
    rule rule_28;
        let t <- mod_26.get(0);
        mod_27.put(1, t);
    endrule
    rule rule_29;
        let t <- mod_31.get(1);
        mod_30.put(1, t);
    endrule
    rule rule_30;
        let t <- mod_10.get(1);
        mod_20.put(0, t);
    endrule
    rule rule_31;
        let t <- mod_20.get(0);
        mod_21.put(0, t);
    endrule
    rule rule_32;
        let t <- mod_22.get(0);
        mod_27.put(0, t);
    endrule
    rule rule_33;
        let t <- mod_27.get(0);
        mod_26.put(0, t);
    endrule
    rule rule_34;
        let t <- mod_25.get(1);
        mod_28.put(0, t);
    endrule
    rule rule_35;
        let t <- mod_16.get(0);
        mod_17.put(0, t);
    endrule
    rule rule_36;
        let t <- mod_4.get(0);
        mod_5.put(1, t);
    endrule
    rule rule_37;
        let t <- mod_25.get(0);
        mod_24.put(0, t);
    endrule
    rule rule_38;
        let t <- mod_5.get(0);
        mod_4.put(0, t);
    endrule
    rule rule_39;
        let t <- mod_3.get(1);
        mod_13.put(0, t);
    endrule
    rule rule_40;
        let t <- mod_11.get(0);
        mod_11.put(1, t);
    endrule
    rule rule_41;
        let t <- mod_11.get(1);
        mod_10.put(1, t);
    endrule
    rule rule_42;
        let t <- mod_19.get(1);
        mod_18.put(1, t);
    endrule
    rule rule_43;
        let t <- mod_1.get(0);
        mod_7.put(0, t);
    endrule
    rule rule_44;
        let t <- mod_9.get(0);
        mod_10.put(0, t);
    endrule
    rule rule_45;
        let t <- mod_30.get(1);
        mod_32.put(0, t);
    endrule

endmodule
endpackage
