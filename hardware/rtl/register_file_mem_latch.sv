// Copyright lowRISC contributors.
// Copyright 2018 ETH Zurich and University of Bologna, see also CREDITS.md.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Adapted by Jannis Schönleber 2022

/**
 * RISC-V register file
 *
 * Register file with 16 x 16 bit wide registers. Register 0 is fixed to 0.
 * This register file is based on latches and is thus smaller than the flip-flop
 * based RF. It requires a target technology-specific clock gating cell. Use this
 * register file when targeting ASIC synthesis or event-based simulators.
 */
module register_file_mem_latch #(
  parameter int unsigned AddrWidth = 4,
  parameter int unsigned DataWidth = 16
) (
  // Clock and Reset
  input logic clk_i,

  // Read port R1
  input  logic [AddrWidth-1:0] raddr_a_i,
  output logic [DataWidth-1:0] rdata_a_o,

  // Write port W1
  input logic [AddrWidth-1:0] waddr_a_i,
  input logic [DataWidth-1:0] wdata_a_i,
  input logic                 we_a_i
);

  // localparam int unsigned ADDR_WIDTH = RV32E ? 4 : 5;
  localparam int unsigned NumWords = 2 ** AddrWidth;

  logic                 clk_int;
  logic [DataWidth-1:0] mem            [NumWords];

  logic [ NumWords-1:0] waddr_onehot_a;

  logic [ NumWords-1:0] mem_clocks;

  // internal addresses
  logic [AddrWidth-1:0] raddr_a_int, waddr_a_int;

  assign raddr_a_int = raddr_a_i[AddrWidth-1:0];
  assign waddr_a_int = waddr_a_i[AddrWidth-1:0];

  // assign clk_int = clk_i;

  //////////
  // READ //
  //////////
  assign rdata_a_o   = mem[raddr_a_int];

  ///////////
  // WRITE //
  ///////////
  // Global clock gating
  prim_clock_gating cg_we_global (
    .clk_i    (clk_i),
    .en_i     (we_a_i),
    .test_en_i(1'b0),
    .clk_o    (clk_int)
  );

  // Write address decoding
  always_comb begin : wad
    for (int i = 0; i < NumWords; i++) begin : wad_word_iter
      if (we_a_i && (waddr_a_int == AddrWidth'(i))) begin
        waddr_onehot_a[i] = 1'b1;
      end else begin
        waddr_onehot_a[i] = 1'b0;
      end
    end
  end

  // Individual clock gating (if integrated clock-gating cells are available)
  for (genvar x = 0; x < NumWords; x++) begin : gen_cg_word_iter
    prim_clock_gating cg_i (
      .clk_i    (clk_int),
      .en_i     (waddr_onehot_a[x]),
      .test_en_i(1'b0),
      .clk_o    (mem_clocks[x])
    );
  end

  // Actual write operation:
  // Generate the sequential process for the NUM_WORDS words of the memory.
  // The process is synchronized with the clocks mem_clocks[i], i = 1, ..., NUM_WORDS-1.
  for (genvar i = 0; i < NumWords; i++) begin : g_rf_latches
    always_latch begin
      if (mem_clocks[i]) begin
        mem[i] = wdata_a_i;
      end
    end
  end

`ifdef VERILATOR
  initial begin
    $display("Latch-based register file not supported for Verilator simulation");
    $fatal();
  end
`endif

endmodule
