module halut_encoder_4 #(
    parameter int unsigned K = halut_pkg::K,
    parameter int unsigned C = halut_pkg::C,
    parameter int unsigned DataTypeWidth = halut_pkg::DataTypeWidth,
    parameter int unsigned EncUnits = 4,

    // do not change
    parameter int unsigned CAddrWidth = $clog2(C),
    parameter int unsigned TreeDepth = $clog2(K),
    parameter int unsigned CPerEncUnit = C / EncUnits,
    parameter int unsigned ThreshMemAddrWidth = $clog2(CPerEncUnit * K)
  ) (
    // Clock and Reset
    input logic clk_i,
    input logic rst_ni,

    // mapping is last to first!! so first unit gets:
    // a_input_i[DataTypeWidth * (((EncUnits - 1)) * TreeDepth)+:DataTypeWidth * TreeDepth]
    input logic signed [DataTypeWidth-1:0] a_input_i[EncUnits][TreeDepth],

    // write ports for threshold memory
    // TODO: maybe one input port for all encoder units?
    input logic unsigned [ThreshMemAddrWidth-1:0] waddr_i[EncUnits],
    input logic unsigned [     DataTypeWidth-1:0] wdata_i[EncUnits],
    input logic                                   we_i   [EncUnits],

    input logic encoder_i,

    output logic unsigned [CAddrWidth-1:0] c_addr_o,
    output logic unsigned [TreeDepth-1:0] k_addr_o,
    output logic valid_o

  );
  logic [EncUnits-1:0] encoder_int, encoder_int_n;
  logic [CAddrWidth-1:0] c_addr_int[EncUnits];
  logic [TreeDepth-1:0] k_addr_int[EncUnits];
  logic valid_int[EncUnits];
  logic [$clog2(EncUnits) -1:0] valid_counter, valid_counter_n;
  logic [CAddrWidth-1:0] c_addr_o_q, c_addr_o_n;
  logic [TreeDepth-1:0] k_addr_o_q, k_addr_o_n;
  logic valid_o_q, valid_o_n;

  for (genvar x = 0; x < EncUnits; x++) begin : gen_encoder_units
    halut_encoder #(
      .K(K),
      .C(C),
      .EncUnits(EncUnits),
      .DataTypeWidth(DataTypeWidth),
      .EncUnitNumber(x)
    ) encoder_unit (
      .clk_i(clk_i),
      .rst_ni(rst_ni),
      .a_input_i(a_input_i[x]),
      .waddr_i(waddr_i[x]),
      .wdata_i(wdata_i[x]),
      .we_i(we_i[x]),
      .encoder_i(encoder_int[x]),
      .c_addr_o(c_addr_int[x]),
      .k_addr_o(k_addr_int[x]),
      .valid_o(valid_int[x])
    );
  end

  always_comb begin : assign_next_signals
    if (encoder_i) begin : encoding
      if (encoder_int < 4'b1111) begin : activate_all_encoder
        encoder_int_n = (encoder_int << 1) + 1;
        k_addr_o_n = k_addr_o;
        c_addr_o_n = c_addr_o;
        valid_o_n = valid_o;
        valid_counter_n = valid_counter;
      end else begin : capture_correct_output
        k_addr_o_n = k_addr_int[valid_counter];
        c_addr_o_n = c_addr_int[valid_counter];
        valid_o_n = valid_int[valid_counter];
        valid_counter_n = valid_counter + 1;
        encoder_int_n = encoder_int;
      end
    end else begin
      valid_counter_n = 2'b11;
      c_addr_o_n = 0;
      k_addr_o_n = 0;
      valid_o_n = 0;
      encoder_int_n = 0;
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin : output_ffs
    if (!rst_ni) begin
      valid_counter <= 2'b11;
      c_addr_o_q <= 0;
      k_addr_o_q <= 0;
      valid_o_q <= 0;
      encoder_int <= 0;
    end else begin
      valid_counter <= valid_counter_n;
      c_addr_o_q <= c_addr_o_n;
      k_addr_o_q <= k_addr_o_n;
      valid_o_q <= valid_o_n;
      encoder_int <= encoder_int_n;
    end
  end

  assign valid_o  = valid_o_q;
  assign k_addr_o = k_addr_o_q;
  assign c_addr_o = c_addr_o_q;
endmodule
