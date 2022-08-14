module fp_16_to_32_convert (
  input  logic [16-1:0] operand_fp16_i,
  output logic [32-1:0] result_o
);
  localparam int unsigned FP_16_EXP_BIAS = 15;
  localparam int unsigned FP_32_EXP_BIAS = 127;
  localparam int unsigned EXP_BIAS_CONVERSION = FP_32_EXP_BIAS - FP_16_EXP_BIAS;  // 112

  logic sign;
  logic [4:0] exp_fp16_in;
  logic [9:0] mant_fp16_in;
  logic [7:0] exp_fp32_converted;
  logic [31:0] operand_fp32_converted;
  logic [9:0] mant_converted;

  logic [3:0] first_one;
  logic no_ones;

  fp_leading_one #(
    .LEN(10)
  ) tree_search (
    .in_i(mant_fp16_in),
    .first_one_o(first_one),
    .no_ones_o(no_ones)
  );

  assign sign = operand_fp16_i[15];
  assign exp_fp16_in = operand_fp16_i[14:10];
  assign mant_fp16_in = operand_fp16_i[9:0];

  always_comb begin
    if (exp_fp16_in == 5'b00000) begin : denormal_subnormal_case
      if (no_ones == 1'b1) begin : zero_case
        exp_fp32_converted = 8'h0;
        mant_converted = mant_fp16_in;
      end else begin : subnormal_case
        exp_fp32_converted = 8'(EXP_BIAS_CONVERSION) - 8'(first_one);  // + 1 - 1
        mant_converted = mant_fp16_in << first_one + 1;
      end
    end else begin : default_case
      exp_fp32_converted = 8'(exp_fp16_in) + 8'(EXP_BIAS_CONVERSION);
      mant_converted = mant_fp16_in;
    end
  end
  assign operand_fp32_converted = {sign, exp_fp32_converted, mant_converted, {13'h0}};
  assign result_o = operand_fp32_converted;
endmodule
