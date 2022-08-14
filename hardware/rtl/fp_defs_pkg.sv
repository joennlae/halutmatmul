`define RM_NEAREST 2'h0
`define RM_TRUNC 2'h1
`define RM_PLUSINF 2'h2
`define RM_MINUSINF 2'h3

package fp_defs;

  parameter bit [2:0] C_RM_NEAREST = 3'h0;
  parameter bit [2:0] C_RM_TRUNC = 3'h1;
  parameter bit [2:0] C_RM_PLUSINF = 3'h3;
  parameter bit [2:0] C_RM_MINUSINF = 3'h2;

`ifdef HALFPREC
  parameter int unsigned C_OP = 16;
  parameter int unsigned C_MANT = 10;
  parameter int unsigned C_EXP = 5;

  parameter int unsigned C_EXP_PRENORM = 7;
  parameter int unsigned C_MANT_PRENORM = 22;
  parameter int unsigned C_MANT_ADDIN = 14;
  parameter int unsigned C_MANT_ADDOUT = 15;
  parameter int unsigned C_MANT_SHIFTIN = 13;
  parameter int unsigned C_MANT_SHIFTED = 14;
  parameter int unsigned C_MANT_PRENORM_IND = 5;
  parameter bit [4:0] C_EXP_ZERO = 5'h00;
  parameter bit [4:0] C_EXP_INF = 5'b11111;
`else
  parameter int unsigned C_MANT = 23;
  parameter int unsigned C_EXP = 8;
  parameter int unsigned C_OP = 32;

  parameter int unsigned C_EXP_PRENORM = C_EXP + 2;
  parameter int unsigned C_MANT_PRENORM = C_MANT * 2 + 2;
  parameter int unsigned C_MANT_ADDIN = C_MANT + 4;
  parameter int unsigned C_MANT_ADDOUT = C_MANT + 5;
  parameter int unsigned C_MANT_SHIFTIN = C_MANT + 3;
  parameter int unsigned C_MANT_SHIFTED = C_MANT + 4;
  parameter int unsigned C_MANT_PRENORM_IND = 6;
  parameter bit [7:0] C_EXP_ZERO = 8'h00;
  parameter bit [7:0] C_EXP_INF = 8'hff;
`endif

endpackage : fp_defs
