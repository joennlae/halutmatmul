
module halut_matmul #(
    parameter int unsigned K = halut_pkg::K,
    parameter int unsigned C = halut_pkg::C,
    parameter int unsigned M = halut_pkg::M,
    parameter int unsigned DataTypeWidth = halut_pkg::DataTypeWidth,
    parameter int unsigned DecoderUnits = halut_pkg::DecoderUnits,
    // do not change
    parameter int unsigned EncUnits = 4,  // default
    parameter int unsigned DecUnitsX = M / DecoderUnits,
    parameter int unsigned DecAddrWidth = $clog2(DecoderUnits),
    parameter int unsigned TotalAddrWidth = $clog2(C * K),
    parameter int unsigned CAddrWidth = $clog2(C),
    parameter int unsigned TreeDepth = $clog2(K),
    parameter int unsigned CPerEncUnit = C / EncUnits,
    parameter int unsigned ThreshMemAddrWidth = $clog2(CPerEncUnit * K),
    parameter int unsigned MAddrWidth = $clog2(M)
  ) (
    // Clock and Reset
    input logic clk_i,
    input logic rst_ni,

    // Encoder
    input logic signed [     DataTypeWidth-1:0] a_input_enc_i[EncUnits][TreeDepth],
    input logic        [ThreshMemAddrWidth-1:0] waddr_enc_i  [EncUnits],
    input logic        [     DataTypeWidth-1:0] wdata_enc_i  [EncUnits],
    input logic                                 we_enc_i     [EncUnits],
    input logic                                 encoder_i,

    // Decoder
    input logic [  DecAddrWidth-1:0] m_addr_dec_i[DecUnitsX],
    input logic [TotalAddrWidth-1:0] waddr_dec_i [DecUnitsX],
    input logic [ DataTypeWidth-1:0] wdata_dec_i [DecUnitsX],
    input logic                      we_dec_i    [DecUnitsX],

    output logic [32-1:0] result_o[DecUnitsX],  // FP32 output
    output logic valid_o[DecUnitsX],
    output logic [MAddrWidth-1:0] m_addr_o[DecUnitsX]
  );
  logic unsigned [CAddrWidth-1:0] c_addr_enc_o;
  logic unsigned [TreeDepth-1:0] k_addr_enc_o;
  logic valid_enc_o;

  logic decoder_i[DecUnitsX];
  logic unsigned [CAddrWidth-1:0] c_addr_int[DecUnitsX];
  logic unsigned [TreeDepth-1:0] k_addr_int[DecUnitsX];
  logic signed [32-1:0] result_dec_o[DecoderUnits];
  logic valid_dec_o[DecoderUnits];
  logic unsigned [DecAddrWidth-1:0] m_addr_int[DecoderUnits];

  halut_encoder_4 #(
    .K(K),
    .C(C),
    .DataTypeWidth(DataTypeWidth),
    .EncUnits(EncUnits)
  ) encoder (
    .clk_i(clk_i),
    .rst_ni(rst_ni),
    .a_input_i(a_input_enc_i),
    .waddr_i(waddr_enc_i),
    .wdata_i(wdata_enc_i),
    .we_i(we_enc_i),
    .encoder_i(encoder_i),
    .c_addr_o(c_addr_enc_o),
    .k_addr_o(k_addr_enc_o),
    .valid_o(valid_enc_o)
  );


  for (genvar x = 0; x < DecUnitsX; x++) begin : gen_decoderX_units
    assign decoder_i[x]  = valid_enc_o;
    assign c_addr_int[x] = c_addr_enc_o;
    assign k_addr_int[x] = k_addr_enc_o;
    halut_decoder_x #(
      .DecoderUnits(DecoderUnits),
      .K(K),
      .C(C),
      .DataTypeWidth(DataTypeWidth)
    ) decoder (
      .clk_i(clk_i),
      .rst_ni(rst_ni),
      .m_addr_i(m_addr_dec_i[x]),
      .waddr_i(waddr_dec_i[x]),
      .wdata_i(wdata_dec_i[x]),
      .we_i(we_dec_i[x]),
      .c_addr_i(c_addr_int[x]),
      .k_addr_i(k_addr_int[x]),
      .decoder_i(decoder_i[x]),
      .result_o(result_dec_o[x]),
      .valid_o(valid_dec_o[x]),
      .m_addr_o(m_addr_int[x])
    );
    assign valid_o[x]  = valid_dec_o[x];
    assign result_o[x] = result_dec_o[x];
    assign m_addr_o[x] = (MAddrWidth)'(m_addr_int[x]) + (MAddrWidth)'(x * DecoderUnits);
  end

// https://docs.cocotb.org/en/stable/simulator_support.html#sim-icarus-waveforms
`ifdef COCOTB_SIM
  initial begin
    $dumpfile("dump.vcd");
    $dumpvars(0, halut_matmul);
    #1;
  end
`endif

endmodule
