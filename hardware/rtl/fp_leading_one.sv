// Copyright 2017, 2018 ETH Zurich and University of Bologna.
// Copyright and related rights are licensed under the Solderpad Hardware
// License, Version 0.51 (the "License"); you may not use this file except in
// compliance with the License.  You may obtain a copy of the License at
// http://solderpad.org/licenses/SHL-0.51. Unless required by applicable law
// or agreed to in writing, software, hardware and materials distributed under
// this License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

// src: https://github.com/pulp-platform/fpu/blob/master/hdl/fpu_v0.1/fpu_ff.sv

module fp_leading_one #(
  parameter int unsigned LEN = 32
) (
  input logic [LEN-1:0] in_i,

  output logic [$clog2(LEN)-1:0] first_one_o,
  output logic                   no_ones_o
);

  localparam int unsigned NUM_LEVELS = $clog2(LEN);

  logic [          LEN-1:0][NUM_LEVELS-1:0] index_lut;
  logic [2**NUM_LEVELS-1:0]                 sel_nodes;
  logic [2**NUM_LEVELS-1:0][NUM_LEVELS-1:0] index_nodes;

  logic [          LEN-1:0]                 in_flipped;


  //////////////////////////////////////////////////////////////////////////////
  // generate tree structure                                                  //
  //////////////////////////////////////////////////////////////////////////////

  generate
    genvar j;
    for (j = 0; j < LEN; j++) begin : gen_index_lut
      assign index_lut[j]  = (NUM_LEVELS)'($unsigned(j));
      assign in_flipped[j] = in_i[LEN-j-1];
    end
  endgenerate

  //                ┌────────────┐
  //                │sel_nodes[0]│
  //                └──────▲─────┘
  //                       │
  //        ┌────────────┬─┴─┬────────────┐
  //        │sel_nodes[1]│   │sel_nodes[2]│
  //        └──────▲─────┘   └────────────┘
  //               │
  // ┌────────────┬┴┬────────────┐
  // │sel_nodes[3]│ │sel_nodes[4]│
  // └────────────┘ └────────────┘
  //
  // all are or conditions
  // https://asciiflow.com/#/
  // same for index nodes where the first 1 position will propagate through

  generate
    genvar k;
    genvar l;
    genvar level;
    for (level = 0; level < NUM_LEVELS; level++) begin : gen_tree
      //------------------------------------------------------------
      if (level < NUM_LEVELS - 1) begin : gen_tree_lower_levels
        for (l = 0; l < 2 ** level; l++) begin : gen_tree_lower_levels_loop
          assign sel_nodes[2**level-1+l]   =
            sel_nodes[2**(level+1)-1+l*2] | sel_nodes[2**(level+1)-1+l*2+1];
          assign index_nodes[2**level-1+l] = (sel_nodes[2**(level+1)-1+l*2] == 1'b1) ?
            index_nodes[2**(level+1)-1+l*2] : index_nodes[2**(level+1)-1+l*2+1];
        end
      end
      //------------------------------------------------------------
      if (level == NUM_LEVELS - 1) begin : gen_tree_to_level
        for (k = 0; k < 2 ** level; k++) begin : gen_tree_map_inputs
          // if two successive indices are still in the vector...
          if (k * 2 < LEN - 1) begin : gen_tree_map_incoming_value_to_tree
            assign sel_nodes[2**level-1+k] = in_flipped[k*2] | in_flipped[k*2+1];
            assign index_nodes[2**level-1+k] = (in_flipped[k*2] == 1'b1) ?
              index_lut[k*2] : index_lut[k*2+1];
          end
          // if only the first index is still in the vector...
          if (k * 2 == LEN - 1) begin : gen_tree_map_first_bit
            assign sel_nodes[2**level-1+k]   = in_flipped[k*2];
            assign index_nodes[2**level-1+k] = index_lut[k*2];
          end
          // if index is out of range
          if (k * 2 > LEN - 1) begin : gen_tree_edge_of_tree
            assign sel_nodes[2**level-1+k]   = 1'b0;
            assign index_nodes[2**level-1+k] = '0;
          end
        end
      end
      //------------------------------------------------------------
    end
  endgenerate

  //////////////////////////////////////////////////////////////////////////////
  // connect output                                                           //
  //////////////////////////////////////////////////////////////////////////////

  assign first_one_o = index_nodes[0];
  assign no_ones_o   = ~sel_nodes[0];

endmodule
