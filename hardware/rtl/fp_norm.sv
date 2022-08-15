// Copyright 2017, 2018 ETH Zurich and University of Bologna.
// Copyright and related rights are licensed under the Solderpad Hardware
// License, Version 0.51 (the "License"); you may not use this file except in
// compliance with the License.  You may obtain a copy of the License at
// http://solderpad.org/licenses/SHL-0.51. Unless required by applicable law
// or agreed to in writing, software, hardware and materials distributed under
// this License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

// src: https://github.com/pulp-platform/fpu/blob/master/hdl/fpu_v0.1/fpu_norm.sv
// heavily adapted

module fp_norm #(
  parameter int unsigned       C_MANT_PRENORM     = fp_defs::C_MANT_PRENORM,
  parameter int unsigned       C_EXP_PRENORM      = fp_defs::C_EXP_PRENORM,
  parameter int unsigned       C_MANT_PRENORM_IND = fp_defs::C_MANT_PRENORM_IND,
  parameter bit          [7:0] C_EXP_ZERO         = fp_defs::C_EXP_ZERO,
  // parameter C_EXP_INF          = fp_defs::C_EXP_INF,

  // parameter C_RM   = fp_defs::C_RM,
  // parameter C_CMD  = fp_defs::C_CMD,
  parameter int unsigned C_MANT = fp_defs::C_MANT,
  parameter int unsigned C_EXP  = fp_defs::C_EXP,

  // parameter C_FPU_ADD_CMD = fp_defs::C_FPU_ADD_CMD,
  // parameter C_FPU_SUB_CMD = fp_defs::C_FPU_SUB_CMD,
  // parameter C_FPU_MUL_CMD = fp_defs::C_FPU_MUL_CMD,

  parameter bit [2:0] C_RM_NEAREST  = fp_defs::C_RM_NEAREST,
  parameter bit [2:0] C_RM_TRUNC    = fp_defs::C_RM_TRUNC,
  parameter bit [2:0] C_RM_PLUSINF  = fp_defs::C_RM_PLUSINF,
  parameter bit [2:0] C_RM_MINUSINF = fp_defs::C_RM_MINUSINF,
  parameter bit [2:0] RM_SI = fp_defs::C_RM_NEAREST
) (
  //Input Operands
  input logic        [C_MANT_PRENORM-1:0] Mant_in_DI,
  input logic signed [ C_EXP_PRENORM-1:0] Exp_in_DI,
  input logic                             Sign_in_DI,

  // Rounding Mode
  // input logic [C_RM-1:0] RM_SI,
  // input logic [C_CMD-1:0] OP_SI,

  output logic [ C_MANT:0] Mant_res_DO,
  output logic [C_EXP-1:0] Exp_res_DO

  // output logic Rounded_SO,
  // output logic Exp_OF_SO,
  // output logic Exp_UF_SO
);

  /////////////////////////////////////////////////////////////////////////////
  // Normalization                                                           //
  /////////////////////////////////////////////////////////////////////////////

  logic        [C_MANT_PRENORM_IND-1:0] Mant_leadingOne_D;
  logic                                 Mant_zero_S;
  logic        [            C_MANT+4:0] Mant_norm_D;
  logic signed [     C_EXP_PRENORM-1:0] Exp_norm_D;

  //trying out stuff for denormals
  logic signed [     C_EXP_PRENORM-1:0] Mant_shAmt_D;
  logic signed [       C_EXP_PRENORM:0] Mant_shAmt2_D;

  logic        [             C_EXP-1:0] Exp_final_D;
  logic signed [     C_EXP_PRENORM-3:0] Exp_rounded_D;

  //sticky bit
  logic                                 Mant_sticky_D;

  logic                                 Denormal_S;
  logic                                 Mant_renorm_S;

  //Detect leading one
  fp_leading_one #(
    .LEN(C_MANT_PRENORM)
  ) LOD (
    .in_i       (Mant_in_DI),
    .first_one_o(Mant_leadingOne_D),
    .no_ones_o  (Mant_zero_S)
  );


  logic Denormals_shift_add_D;
  logic Denormals_exp_add_D;
  assign Denormals_shift_add_D = ~Mant_zero_S & (Exp_in_DI == (C_EXP_PRENORM)'(C_EXP_ZERO));
  assign Denormals_exp_add_D   =  Mant_in_DI[C_MANT_PRENORM-2] &
    (Exp_in_DI == (C_EXP_PRENORM)'(C_EXP_ZERO));

  assign Denormal_S = ((C_EXP_PRENORM)'($signed((Mant_leadingOne_D))) >= Exp_in_DI) || Mant_zero_S;
  assign Mant_shAmt_D = Denormal_S ?
    Exp_in_DI + (C_EXP_PRENORM)'(Denormals_shift_add_D) : Mant_leadingOne_D;
  assign Mant_shAmt2_D = {Mant_shAmt_D[$high(Mant_shAmt_D)], Mant_shAmt_D} + (C_MANT + 4 + 1);

  //Shift mantissa
  always_comb begin
    logic [C_MANT_PRENORM+C_MANT+4:0] temp;
    temp = ((C_MANT_PRENORM + C_MANT + 4 + 1)'(Mant_in_DI) << (Mant_shAmt2_D));
    Mant_norm_D = temp[C_MANT_PRENORM+C_MANT+4:C_MANT_PRENORM];
  end

  always_comb begin
    Mant_sticky_D = 1'b0;
    if (Mant_shAmt2_D <= 0) Mant_sticky_D = |Mant_in_DI;
    else if (Mant_shAmt2_D <= C_MANT_PRENORM) Mant_sticky_D = |(Mant_in_DI << (Mant_shAmt2_D));
  end

  //adjust exponent
  assign Exp_norm_D = Exp_in_DI - (C_EXP_PRENORM)'($signed(
      (Mant_leadingOne_D)
  )) + 1 + (C_EXP_PRENORM)'(Denormals_exp_add_D);
  //Explanation of the +1 since I'll probably forget:
  //we get numbers in the format xx.x...
  //but to make things easier we interpret them as
  //x.xx... and adjust the exponent accordingly

  assign Exp_rounded_D = (C_EXP)'(Exp_norm_D + (C_EXP_PRENORM)'(Mant_renorm_S));
  assign Exp_final_D = Exp_rounded_D[C_EXP-1:0];


  //detect exponent over/underflow
  // always_comb begin
  //   Exp_OF_SO = 1'b0;
  //   Exp_UF_SO = 1'b0;
  //   if (Exp_rounded_D >= signed'({2'b0,C_EXP_INF})) //overflow
  //         begin
  //     Exp_OF_SO = 1'b1;
  //   end
  //       else if (Exp_rounded_D <= signed'({2'b0,C_EXP_ZERO})) //underflow
  //         begin
  //     Exp_UF_SO = 1'b1;
  //   end
  // end

  /////////////////////////////////////////////////////////////////////////////
  // Rounding                                                                //
  /////////////////////////////////////////////////////////////////////////////

  logic [  C_MANT:0] Mant_upper_D;
  logic [       3:0] Mant_lower_D;
  logic [C_MANT+1:0] Mant_upperRounded_D;

  logic              Mant_roundUp_S;
  logic              Mant_rounded_S;

  assign Mant_lower_D   = Mant_norm_D[3:0];
  assign Mant_upper_D   = Mant_norm_D[C_MANT+4:4];


  assign Mant_rounded_S = (|(Mant_lower_D)) | Mant_sticky_D;

  //determine whether to round up or not
  always_comb begin
    Mant_roundUp_S = 1'b0;
    case (RM_SI)
      C_RM_NEAREST:
      Mant_roundUp_S = Mant_lower_D[3] &&
        (((| Mant_lower_D[2:0]) | Mant_sticky_D) || Mant_upper_D[0]);
      C_RM_TRUNC: Mant_roundUp_S = 0;
      C_RM_PLUSINF: Mant_roundUp_S = Mant_rounded_S & ~Sign_in_DI;
      C_RM_MINUSINF: Mant_roundUp_S = Mant_rounded_S & Sign_in_DI;
      default: Mant_roundUp_S = 0;
    endcase  // case (RM_DI)
  end  // always_comb begin

  assign Mant_upperRounded_D = Mant_upper_D + (C_MANT + 1)'(Mant_roundUp_S);
  assign Mant_renorm_S       = Mant_upperRounded_D[C_MANT+1];

  /////////////////////////////////////////////////////////////////////////////
  // Output Assignments                                                      //
  /////////////////////////////////////////////////////////////////////////////

  assign Mant_res_DO         = (C_MANT + 1)'(Mant_upperRounded_D >> (Mant_renorm_S & ~Denormal_S));
  assign Exp_res_DO          = Exp_final_D;
  // assign Rounded_SO          = Mant_rounded_S;

endmodule  // fp_norm
