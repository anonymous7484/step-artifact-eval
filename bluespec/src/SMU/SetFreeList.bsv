package SetFreeList;

import Vector::*;
import Types::*;
import Parameters::*;
import ConfigReg::*;

interface SetFreeList_IFC;
    method ActionValue#(Maybe#(SET_INDEX)) allocSet();
    method Action freeSet(SET_INDEX s);
    method Bool isSetFree(SET_INDEX s);
endinterface

module mkSetFreeList(SetFreeList_IFC);
    Vector#(SETS, ConfigReg#(Bool)) free_sets <- replicateM(mkConfigReg(True));  // All sets initially free (true = free, false = used)

    // Simple priority encoder
    function Maybe#(SET_INDEX) findFree();
        Maybe#(SET_INDEX) result = Invalid;
        for (Integer i = 0; i < valueOf(SETS); i = i + 1)
            if (free_sets[i])
                result = tagged Valid fromInteger(i);
        return result;
    endfunction

    method ActionValue#(Maybe#(SET_INDEX)) allocSet();
        actionvalue
            let result = findFree();
            if (result matches tagged Valid .idx)
                free_sets[idx] <= False;  // Mark as used 
            return result;
        endactionvalue
    endmethod

    method Action freeSet(SET_INDEX s);
        free_sets[s] <= True;  
    endmethod

    method Bool isSetFree(SET_INDEX s);
        return free_sets[s];
    endmethod

endmodule

endpackage
