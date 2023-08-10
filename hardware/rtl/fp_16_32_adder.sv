module fp_16_32_adder (
    input  logic signed [16-1:0] operand_fp16_i,
    input  logic signed [32-1:0] operand_fp32_i,
    output logic signed [32-1:0] result_o
  );

  logic signed [32-1:0] operand_fp16_fp32;

  fp_16_to_32_convert converter (
    .operand_fp16_i(operand_fp16_i),
    .result_o(operand_fp16_fp32)
  );

  fp_adder adder (
    .operand_a_di(operand_fp16_fp32),
    .operand_b_di(operand_fp32_i),
    .result_do(result_o)
  );

endmodule
