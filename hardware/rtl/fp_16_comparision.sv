
module fp_16_comparision (
  input logic [16-1:0] operand_a_i,
  input logic [16-1:0] operand_b_i,
  output logic comparision_o
);
  logic sign_a;
  logic [4:0] exp_fp16_a;
  logic [9:0] mant_fp16_a;
  logic sign_b;
  logic [4:0] exp_fp16_b;
  logic [9:0] mant_fp16_b;

  logic sign_unequal;
  logic exp_unequal;

  logic sign_decision;
  logic exp_decision_intermediate;
  logic mant_decision_intermediate;
  logic exp_decision_corr;
  logic mant_decision_corr;
  logic exp_decision;
  logic mant_decision;
  logic all_same;

  assign sign_a = operand_a_i[15];
  assign exp_fp16_a = operand_a_i[14:10];
  assign mant_fp16_a = operand_a_i[9:0];

  assign sign_b = operand_b_i[15];
  assign exp_fp16_b = operand_b_i[14:10];
  assign mant_fp16_b = operand_b_i[9:0];

  // if a>=b
  assign all_same = !sign_unequal & !exp_unequal & (mant_fp16_a == mant_fp16_b);

  // check for a>b
  assign sign_unequal = sign_a != sign_b;
  assign sign_decision = sign_unequal & (sign_a < sign_b);

  assign exp_unequal = exp_fp16_a != exp_fp16_b;

  assign exp_decision_intermediate = (exp_fp16_a > exp_fp16_b);
  assign mant_decision_intermediate = (mant_fp16_a > mant_fp16_b);

  assign exp_decision_corr = sign_a == 1'b1 ?
    ~exp_decision_intermediate & !all_same : exp_decision_intermediate;
  assign mant_decision_corr = sign_a == 1'b1 ?
    ~mant_decision_intermediate & !all_same : mant_decision_intermediate;

  assign exp_decision = !sign_unequal & exp_unequal & exp_decision_corr;
  assign mant_decision = !sign_unequal & !exp_unequal & mant_decision_corr;

  assign comparision_o = sign_decision | exp_decision | mant_decision;  // | all_same;

endmodule
