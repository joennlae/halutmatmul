module fp_adder #(
    parameter int unsigned C_OP = fp_defs::C_OP,
    parameter int unsigned C_EXP = fp_defs::C_EXP,
    parameter int unsigned C_MANT = fp_defs::C_MANT,
    parameter int unsigned C_EXP_PRENORM = fp_defs::C_EXP_PRENORM,
    parameter int unsigned C_MANT_PRENORM = fp_defs::C_MANT_PRENORM
  ) (
    input  logic [C_OP-1:0] operand_a_di,
    input  logic [C_OP-1:0] operand_b_di,
    output logic [C_OP-1:0] result_do
  );

  //Operand components
  logic                             sign_a_d;
  logic                             sign_b_d;
  logic        [         C_EXP-1:0] exp_a_d;
  logic        [         C_EXP-1:0] exp_b_d;
  logic        [          C_MANT:0] mant_a_d;
  logic        [          C_MANT:0] mant_b_d;

  // Hidden Bits
  logic                             hb_a_d;
  logic                             hb_b_d;

  // Pre-Normalizer result
  logic                             sign_prenorm_d;
  logic signed [ C_EXP_PRENORM-1:0] exp_prenorm_d;
  logic        [C_MANT_PRENORM-1:0] mant_prenorm_d;

  // Post-Normalizer result
  logic        [         C_EXP-1:0] exp_norm_d;
  logic        [          C_MANT:0] mant_norm_d;


  assign sign_a_d = operand_a_di[C_OP-1];
  assign sign_b_d = operand_b_di[C_OP-1];
  assign exp_a_d  = operand_a_di[C_OP-2:C_MANT];
  assign exp_b_d  = operand_b_di[C_OP-2:C_MANT];
  assign mant_a_d = {hb_a_d, operand_a_di[C_MANT-1:0]};
  assign mant_b_d = {hb_b_d, operand_b_di[C_MANT-1:0]};

  assign hb_a_d   = |exp_a_d;  // hidden bit
  assign hb_b_d   = |exp_b_d;  // hidden bit

  fp_add adder (
    .Sign_a_DI(sign_a_d),
    .Sign_b_DI(sign_b_d),
    .Exp_a_DI (exp_a_d),
    .Exp_b_DI (exp_b_d),
    .Mant_a_DI(mant_a_d),
    .Mant_b_DI(mant_b_d),

    .Sign_prenorm_DO(sign_prenorm_d),
    .Exp_prenorm_DO (exp_prenorm_d),
    .Mant_prenorm_DO(mant_prenorm_d)
  );


  fp_norm norm (
    .Mant_in_DI(mant_prenorm_d),
    .Exp_in_DI (exp_prenorm_d),
    .Sign_in_DI(sign_prenorm_d),

    .Mant_res_DO(mant_norm_d),
    .Exp_res_DO (exp_norm_d)
  );

  assign result_do = {sign_prenorm_d, exp_norm_d, mant_norm_d[C_MANT-1:0]};

endmodule
