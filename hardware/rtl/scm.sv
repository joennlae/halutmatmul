
module scm #(
    parameter int unsigned C = 32,
    parameter int unsigned K = 16,
    parameter int unsigned DataTypeWidth = 16,
    // default
    parameter int unsigned SubUnitAddrWidth = 5,
    parameter int unsigned TotalAddrWidth = $clog2(C * K)
  ) (
    // Clock and Reset
    input logic clk_i,
    input logic rst_ni,

    // Read port R1
    input  logic unsigned [TotalAddrWidth-1:0] raddr_a_i,
    output logic signed   [ DataTypeWidth-1:0] rdata_a_o,

    // Write port W1
    input logic unsigned [TotalAddrWidth-1:0] waddr_a_i,
    input logic signed   [ DataTypeWidth-1:0] wdata_a_i,
    input logic                               we_a_i
  );

  localparam int unsigned UnitAddrWidth = (TotalAddrWidth - SubUnitAddrWidth);
  localparam int unsigned NumSubUnits = 2 ** UnitAddrWidth;

  logic unsigned [UnitAddrWidth-1:0] raddr_int_unit, waddr_int_unit;
  logic unsigned [SubUnitAddrWidth-1:0] raddr_int_sub, waddr_int_sub;
  logic signed [DataTypeWidth-1:0] wdata_a_q;
  logic unsigned [NumSubUnits-1:0] waddr_onehot_unit;
  logic signed [DataTypeWidth-1:0] read_outputs_subunits[NumSubUnits];

  logic clk_int;
  // Global clock gating
  tc_clk_gating cg_we_global (
    .clk_i    (clk_i),
    .en_i     (we_a_i),
    .test_en_i(1'b0),
    .clk_o    (clk_int)
  );

  // Sample input data
  // Use clk_int here, since otherwise we don't want to write anything anyway.
  always_ff @(posedge clk_i or negedge rst_ni) begin : sample_wdata
    if (!rst_ni) begin
      wdata_a_q <= 0;
    end else begin
      if (we_a_i) begin
        wdata_a_q <= wdata_a_i;
      end
    end
  end

  assign raddr_int_unit = raddr_a_i[TotalAddrWidth-1:SubUnitAddrWidth];
  assign waddr_int_unit = waddr_a_i[TotalAddrWidth-1:SubUnitAddrWidth];
  assign raddr_int_sub  = raddr_a_i[SubUnitAddrWidth-1:0];
  assign waddr_int_sub  = waddr_a_i[SubUnitAddrWidth-1:0];

  // Write address decoding subunit
  for (genvar i = 0; i < NumSubUnits; i++) begin: gen_write_enc
    assign waddr_onehot_unit[i] = (waddr_int_unit == i) & we_a_i;
  end

  for (genvar x = 0; x < NumSubUnits; x++) begin : gen_sub_units_scm
    register_file_mem_latch #(
      .AddrWidth(SubUnitAddrWidth),
      .DataWidth(DataTypeWidth)
    ) sub_unit_i (
      .clk_i(clk_int),  // clk_int?
      .raddr_a_i(raddr_int_sub),
      .rdata_a_o(read_outputs_subunits[x]),
      .waddr_a_i(waddr_int_sub),
      .wdata_a_i(wdata_a_q),
      .we_a_i(waddr_onehot_unit[x])
    );
  end

  // Mux the outputs of the subunits
  assign rdata_a_o = read_outputs_subunits[raddr_int_unit];

endmodule
