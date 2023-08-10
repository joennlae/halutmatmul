module halut_decoder #(
    parameter int unsigned K = halut_pkg::K,
    parameter int unsigned C = halut_pkg::C,
    parameter int unsigned DataTypeWidth = halut_pkg::DataTypeWidth,
    parameter int unsigned DecoderUnits = halut_pkg::DecoderUnits,
    // defaults
    parameter int unsigned TotalAddrWidth = $clog2(C * K),
    parameter int unsigned CAddrWidth = $clog2(C),
    parameter int unsigned TreeDepth = $clog2(K)
  ) (
    // Clock and Reset
    input logic clk_i,
    input logic rst_ni,

    // Write port
    input logic [TotalAddrWidth-1:0] waddr_i,
    input logic [ DataTypeWidth-1:0] wdata_i,
    input logic                      we_i,

    input logic [CAddrWidth-1:0] c_addr_i,
    input logic [TreeDepth-1:0] k_addr_i,
    input logic decoder_i,

    output logic [32-1:0] result_o,  // FP32 output
    output logic valid_o
  );

  //                                      ┌───────────────────────────────────┐
  //                                      │    32                             │
  //             ┌───────────────────┐    │                         ┌─────┐   │     ┌─────┐
  //             │                   │    │ ┌────────────────┐      │     │   │     │     │
  // c_addr_i────►                   │    └►│                │      │     │   │     │     │
  //          5  │                   │      │                │      │     │   │     │     │
  //             │        lut        ├─────►│ fp_16_32_adder ├─────►│     ├───┴────►│     ├────►result_o
  //             │                   │  16  │                │  32  │     │   32    │     │
  // k_addr_i────►  C=32, K=16, M=1  │      │                │      │     │         │     │
  //          4  │                   │      └────────────────┘      │►    │         │►    │
  //             └───────────────────┘                              └─────┘         └─────┘

  // edit: https://asciiflow.com/#/share/eJyrVspLzE1VssorzcnRUcpJrEwtUrJSqo5RqohRsrI0t9SJUaoEsowszICsktSKEiAnRkmBGPBoyh56opiYPGKdBaKMjQipQjOQQuchWU18aCHpIcbP2M1Hspx8p6M4BIMHC6%2Fk%2BMSUlKL4TBQDpu3C6yigPBZ3k2QtXJcpgTDAKk2eVahm5ZSWIImiI4gP0wriDc3ijY1AYZRahFsdDkOIUDVtV1FqcWlOSXw%2B8QnD0Ax3oIDyCGowQHINapBk44p0Z1tjIx0Fb1tDMx0FX1tDsmIBQwzZZybERDclCR6adtGcghClfhGBG%2BDXglU2RqlWqRYAVumR8A%3D%3D)
  localparam int unsigned LUTAddrWidth = $clog2(C * K);

  logic [LUTAddrWidth-1:0] raddr;
  logic [DataTypeWidth-1:0] rdata_o, rdata_o_q, rdata_o_n;

  logic [32-1:0] result_int_d, result_int_q, result_o_q, result_o_n, result_int_n;
  logic [CAddrWidth-1:0] caddr_q, caddr_n;

  logic valid_o_n;

  assign raddr = {c_addr_i, k_addr_i};

  scm #(
    .C(C),
    .K(K),
    .DataTypeWidth(DataTypeWidth)
  ) lut (
    .clk_i(clk_i),
    .rst_ni(rst_ni),
    .raddr_a_i(raddr),
    .rdata_a_o(rdata_o),
    .waddr_a_i(waddr_i),
    .wdata_a_i(wdata_i),
    .we_a_i(we_i)
  );

  fp_16_32_adder fp_adder (
    .operand_fp16_i(rdata_o_q),
    .operand_fp32_i(result_int_q),
    .result_o(result_int_d)
  );

  always_comb begin : assign_next_valid_signal
    if (!decoder_i) begin
      valid_o_n = 1'b0;
    end else if (caddr_q == CAddrWidth'(C - 1)) begin
      valid_o_n = 1'b1;
    end else if (caddr_q >= (CAddrWidth)'(DecoderUnits - 1)) begin : valid_for_DecUnit_cycles
      valid_o_n = 1'b0;
    end else begin
      valid_o_n = valid_o;
    end
  end

  always_comb begin : assing_next_signals
    if (decoder_i) begin
      rdata_o_n = rdata_o;
      caddr_n   = c_addr_i;
      if (caddr_q == CAddrWidth'(C - 1)) begin  // Attention: do net let it stay on address C - 1
        result_int_n = 0;
        result_o_n   = result_int_d;
      end else begin
        result_int_n = result_int_d;
        result_o_n   = result_o;
      end
    end else begin
      rdata_o_n = 0;
      caddr_n = 0;
      result_int_n = 0;
      result_o_n = 0;
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin : result_ffs
    if (!rst_ni) begin
      result_int_q <= 0;
      valid_o <= 0;
      caddr_q <= 0;
      rdata_o_q <= 0;
      result_o_q <= 0;
    end else begin
      result_int_q <= result_int_n;
      rdata_o_q <= rdata_o_n;
      result_o_q <= result_o_n;
      caddr_q <= caddr_n;
      valid_o <= valid_o_n;
    end
  end

  assign result_o = result_o_q;

endmodule
