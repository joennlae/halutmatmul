module halut_decoder_x #(
  parameter int unsigned DecoderUnits = 16,  // how many decoders, needs to be overflowing!!
  parameter int unsigned K = 16,
  parameter int unsigned C = 32,
  parameter int unsigned DataTypeWidth = 16,
  // defaults
  parameter int unsigned TotalAddrWidth = $clog2(C * K),
  parameter int unsigned CAddrWidth = $clog2(C),
  parameter int unsigned TreeDepth = $clog2(K),
  parameter int unsigned DecAddrWidth = $clog2(DecoderUnits)
) (
  // Clock and Reset
  input logic clk_i,
  input logic rst_ni,

  // Write port
  input logic [  DecAddrWidth-1:0] m_addr_i,
  input logic [TotalAddrWidth-1:0] waddr_i,
  input logic [ DataTypeWidth-1:0] wdata_i,
  input logic                      we_i,

  input logic [CAddrWidth-1:0] c_addr_i,
  input logic [TreeDepth-1:0] k_addr_i,
  input logic decoder_i,

  output logic [32-1:0] result_o,  // FP32 output
  output logic valid_o,
  output logic [DecAddrWidth-1:0] m_addr_o
);

  logic [DecoderUnits-1:0] decoder_we_i_onehot;
  logic [32-1:0] result_int[DecoderUnits];
  logic valid_int[DecoderUnits];

  logic [DecAddrWidth-1:0] m_addr_cnt;
  logic [32-1:0] result_o_q;
  logic valid_o_q;
  logic [DecAddrWidth-1:0] m_addr_o_q;

  prim_onehot_enc #(
    .OneHotWidth(DecoderUnits)
  ) wadd_onehot (
    .in_i (m_addr_i),
    .en_i (we_i),
    .out_o(decoder_we_i_onehot)
  );

  for (genvar x = 0; x < DecoderUnits; x++) begin : gen_decoders
    halut_decoder #(
      .K(K),
      .C(C),
      .DataTypeWidth(DataTypeWidth),
      .DecoderUnits(DecoderUnits)
    ) sub_unit_decoder (
      .clk_i(clk_i),
      .rst_ni(rst_ni),
      .waddr_i(waddr_i),
      .wdata_i(wdata_i),
      // .we_i(1'b1),
      .we_i(decoder_we_i_onehot[x]),
      .c_addr_i(c_addr_i),
      .k_addr_i(k_addr_i),
      .decoder_i(decoder_i),
      .result_o(result_int[x]),
      .valid_o(valid_int[x])
    );
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin : output_logic
    if (!rst_ni) begin
      m_addr_cnt <= 0;
      valid_o_q  <= 0;
      m_addr_cnt <= 0;
      result_o_q <= 0;
      m_addr_o_q <= 0;
    end else begin
      if (decoder_i) begin : decoding_activated
        if (valid_int[m_addr_cnt] == 1'b1) begin : gather_results
          result_o_q <= result_int[m_addr_cnt];
          m_addr_o_q <= m_addr_cnt;
          valid_o_q  <= 1'b1;
          m_addr_cnt <= m_addr_cnt + 1;
        end else begin
          valid_o_q  <= 1'b0;
          m_addr_cnt <= 0;
        end
      end
    end
  end

  assign result_o = result_o_q;
  assign valid_o  = valid_o_q;
  assign m_addr_o = m_addr_o_q;

endmodule
