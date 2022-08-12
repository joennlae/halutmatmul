// Adapted by Jannis Schönleber 2022

module scm #(
  parameter int unsigned C = 32,
  parameter int unsigned M = 1,
  parameter int unsigned K = 16,
  parameter int unsigned DataTypeWidth = 16,
  // default
  parameter int unsigned SubUnitAddrWidth = 5,
  parameter int unsigned TotalAddrWidth = $clog2(C * K),
  parameter int unsigned TotalDataWidth = M * DataTypeWidth
) (
  // Clock and Reset
  input logic clk_i,
  input logic rst_ni,

  input logic test_en_i,

  // Read port R1
  input  logic [TotalAddrWidth-1:0] raddr_a_i,
  output logic [TotalDataWidth-1:0] rdata_a_o,

  // Write port W1
  input logic [TotalAddrWidth-1:0] waddr_a_i,
  input logic [TotalDataWidth-1:0] wdata_a_i,
  input logic                      we_a_i
);

  localparam int unsigned UnitAddrWidth = (TotalAddrWidth - SubUnitAddrWidth);
  localparam int unsigned NumSubUnits = 2 ** UnitAddrWidth;

  logic [TotalAddrWidth-1:SubUnitAddrWidth] raddr_int_unit, waddr_int_unit;
  logic [SubUnitAddrWidth-1:0] raddr_int_sub, waddr_int_sub;

  logic [DataTypeWidth-1:0] wdata_a_q;

  logic [NumSubUnits-1:0] waddr_onehot_unit, raddr_onehot_unit;

  logic [DataTypeWidth-1:0] read_outputs_subunits[NumSubUnits];
  // logic [DataTypeWidth-1:0] write_inputs_subunits[NumSubUnits];

  logic clk_int;
  // Global clock gating
  prim_clock_gating cg_we_global (
    .clk_i    (clk_i),
    .en_i     (we_a_i),
    .test_en_i(test_en_i),
    .clk_o    (clk_int)
  );

  // Sample input data
  // Use clk_int here, since otherwise we don't want to write anything anyway.
  always_ff @(posedge clk_int or negedge rst_ni) begin : sample_wdata
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
  prim_onehot_enc #(
    .OneHotWidth(NumSubUnits)
  ) wadd_onehot (
    .in_i(waddr_int_unit),
    .en_i(1'b1),  // we_a_i
    .out_o(waddr_onehot_unit)
  );

  // Read address decoding subunit
  prim_onehot_enc #(
    .OneHotWidth(NumSubUnits)
  ) radd_onehot (
    .in_i (raddr_int_unit),
    .en_i (1'b1),
    .out_o(raddr_onehot_unit)
  );

  for (genvar x = 0; x < NumSubUnits; x++) begin : gen_sub_units_scm
    register_file_mem_latch #(
      .AddrWidth(SubUnitAddrWidth),
      .DataWidth(DataTypeWidth)
    ) sub_unit_i (
      .clk_i(clk_i),  // clk_int?
      .test_en_i(test_en_i),
      .raddr_a_i(raddr_int_sub),
      .rdata_a_o(read_outputs_subunits[x]),
      .waddr_a_i(waddr_int_sub),
      .wdata_a_i(wdata_a_q),
      .we_a_i(waddr_onehot_unit[x])
    );
  end

  prim_onehot_mux #(
    .Width (DataTypeWidth),
    .Inputs(NumSubUnits)
  ) read (
    .clk_i (clk_i),
    .in_i  (read_outputs_subunits),
    .rst_ni(rst_ni),
    .sel_i (raddr_onehot_unit),
    .out_o (rdata_a_o)
  );

  // https://docs.cocotb.org/en/stable/simulator_support.html#sim-icarus-waveforms
`ifdef COCOTB_SIM
  initial begin
    $dumpfile("dump.vcd");
    $dumpvars(0, scm);
    #1;
  end
`endif
endmodule
