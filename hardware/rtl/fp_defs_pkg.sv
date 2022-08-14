`define RM_NEAREST 2'h0
`define RM_TRUNC 2'h1
`define RM_PLUSINF 2'h2
`define RM_MINUSINF 2'h3

package fp_defs;

  parameter C_RM_NEAREST = 3'h0;
  parameter C_RM_TRUNC = 3'h1;
  parameter C_RM_PLUSINF = 3'h3;
  parameter C_RM_MINUSINF = 3'h2;

`ifdef HALFPREC
  parameter C_OP = 16;
  parameter C_MANT = 10;
  parameter C_EXP = 5;

  parameter C_EXP_PRENORM = 7;
  parameter C_MANT_PRENORM = 22;
  parameter C_MANT_ADDIN = 14;
  parameter C_MANT_ADDOUT = 15;
  parameter C_MANT_SHIFTIN = 13;
  parameter C_MANT_SHIFTED = 14;
  parameter C_MANT_PRENORM_IND = 5;
  parameter C_EXP_ZERO = 5'h00;
  parameter C_EXP_INF = 5'hff;
`else
  parameter C_MANT = 23;
  parameter C_EXP = 8;
  parameter C_OP = 32;

  parameter C_EXP_PRENORM = C_EXP + 2;
  parameter C_MANT_PRENORM = C_MANT * 2 + 2;
  parameter C_MANT_ADDIN = C_MANT + 4;
  parameter C_MANT_ADDOUT = C_MANT + 5;
  parameter C_MANT_SHIFTIN = C_MANT + 3;
  parameter C_MANT_SHIFTED = C_MANT + 4;
  parameter C_MANT_PRENORM_IND = 6;
  parameter C_EXP_ZERO = 8'h00;
  parameter C_EXP_INF = 8'hff;
`endif

endpackage : fp_defs
