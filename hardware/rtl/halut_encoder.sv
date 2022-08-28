module halut_encoder #(
  parameter int unsigned K = 16,
  parameter int unsigned C = 32,
  parameter int unsigned EncUnits = 4,
  parameter int unsigned DataTypeWidth = 16,
  parameter int unsigned EncUnitNumber = 0,
  // default
  parameter int unsigned TreeDepth = $clog2(K),
  parameter int unsigned CAddrWidth = $clog2(C),
  parameter int unsigned CPerEncUnit = C / EncUnits,
  parameter int unsigned ThreshMemAddrWidth = $clog2(CPerEncUnit * K)
) (
  input logic clk_i,
  input logic rst_ni,

  input logic [DataTypeWidth-1:0] a_input_i[TreeDepth],

  // write ports for threshold memory
  input logic [ThreshMemAddrWidth-1:0] waddr_i,
  input logic [     DataTypeWidth-1:0] wdata_i,
  input logic                          we_i,

  input logic encoder_i,

  output logic [CAddrWidth-1:0] c_addr_o,
  output logic [TreeDepth-1:0] k_addr_o,
  output logic valid_o
);

  // schematic but not with all output, input ff
  //
  //                                         │   │   │   │
  //                                         │   │   │   │
  //                                         │   │   │   │
  //                                    x────▼───▼───▼───▼────xx
  //                                     xx                  xx
  //                                      xx   input_mux    xx◄───────────────────┐
  //                                       xx              xx                     │
  //                                        x──────┬───────x                      │
  //                                               │                              │
  //                                               │                              │
  //                                               │                              │
  //                                               │                              │
  // ┌──────────────────────┐                      ▼                              │
  // │                      │             ┌──────────────────┐                    │
  // │                      │             │                  │                    │
  // │   threshold_memory   ├────────────►│ fp_16_comparision│                    │
  // │                      │             │                  │                    │
  // │                      │             └─────────┬────────┘                    │
  // └──────────────────────┘                       │                    ┌────────┴───────┐
  //           ▲                                    │                    │                │
  //           │                                    │                    │ tree_level_cnt │
  //           │                                    │                    │►               │
  //           │                                  ┌─▼─┐                  └────────────────┘
  //           │                                  │ + ├────────────────┐
  //           │                                  └─┬─┘                │
  //           │                                    │                  │
  //           │                                    │                  │
  //      ┌────┴────────────┐              ┌────────▼──────┐        ┌──┴──┐
  //      │                 │              │               │        │ <<1 │
  //      │ cal_memory_addr │◄─────┬───────┤    k_addr     │        └──▲──┘
  //      │                 │      │       │►              │           │
  //      └────────▲────────┘      │       └───────────────┘           │
  //               │               │                                   │
  //               │               │                                   │
  //               │               └───────────────────────────────────┘
  //               │
  //               │
  //               │
  //               │
  //      ┌────────┴────────┐
  //      │                 │
  //      │ c_addr_internal │
  //      │►                │
  //      └─────────────────┘

  // edit here: https://asciiflow.com/#/share/eJztVtFqgzAU%2FRXJ6%2FriYGWTfkogiGZUFqOksURK3%2FYJxf1HH0e%2Fxi%2BZtetaNdGkjY7B5CIx3nvPud5z1Q2gfoyBRzNCZoD4OWbAAxsIBATey7M7gyCvVo%2Fzp2rFseDVBQSO5lHuDt0zhPQPxIvK82LFp9l6dxBCk6cQsj3dIuvoiKYZR3EmTjvlx3uDvKFpP982c1kljmHLhD5POZqhQn5iBh3%2Bk94jqmtTYBS6JPSKsM3QHH7YSZKaLxleLRMSohjHCcsNKqlfQM5ritw5CpI49Vm0ihIV5Ynq0Us9TqdGVGtP6TdYY2LLYq%2FA1CQgfx6DkQYAnGGMCF5jggLK7QNUUu5umkOc50LZSRvd0iTiPEyJ1idgm90aKZelYe3zLfoT9LZCRl8GrnY5LhcLt1WzE%2Fjk%2B7WP%2FDBkNQuTn7ljnrdTaBfwXPjerJ7LzVIymM1QzRZeUVB1oIl6uwraAh3sTM%2FxK7mmNBkjG3t3E1IItDE6texRRDlm1Cdt%2BM4HxQ4%2FsAXbL6Awx9Q%3D)

  localparam int unsigned CntWidth = $clog2(TreeDepth);

  logic [ThreshMemAddrWidth-1:0] read_addr_thresh_mem;
  logic [DataTypeWidth-1:0] data_thresh_mem_o;

  logic [DataTypeWidth-1:0] data_input_comparision;

  logic [CntWidth-1:0] tree_level_cnt;

  logic [TreeDepth-1:0] k_addr, k_addr_o_q;
  logic [TreeDepth-1:0] prototype_addr_internal;
  logic [$clog2(CPerEncUnit)-1:0] c_addr_internal;
  logic [CAddrWidth-1:0] c_addr_o_q;
  logic [TreeDepth-1:0] input_tree_level_one_hot;

  logic fp_16_comparision_o;

  prim_onehot_enc #(
    .OneHotWidth(TreeDepth)
  ) tree_level_input (
    .in_i (tree_level_cnt),
    .en_i (1'b1),
    .out_o(input_tree_level_one_hot)
  );

  prim_onehot_mux #(
    .Width (DataTypeWidth),
    .Inputs(TreeDepth)
  ) input_mux (
    .clk_i (clk_i),
    .in_i  (a_input_i),
    .rst_ni(rst_ni),
    .sel_i (input_tree_level_one_hot),
    .out_o (data_input_comparision)
  );

  scm #(
    .C(CPerEncUnit),
    .K(K),
    .DataTypeWidth(DataTypeWidth)
  ) threshold_memory (
    .clk_i(clk_i),
    .rst_ni(rst_ni),
    .raddr_a_i(read_addr_thresh_mem),
    .rdata_a_o(data_thresh_mem_o),
    .waddr_a_i(waddr_i),
    .wdata_a_i(wdata_i),
    .we_a_i(we_i)
  );

  fp_16_comparision fp_compare (
    .operand_a_i  (data_input_comparision),
    .operand_b_i  (data_thresh_mem_o),
    .comparision_o(fp_16_comparision_o)
  );

  always_comb begin : cal_memory_addr
    // needs update when K changes!!
    unique case (tree_level_cnt)
      2'b01:   prototype_addr_internal = k_addr + 1;
      2'b10:   prototype_addr_internal = k_addr + 3;
      2'b11:   prototype_addr_internal = k_addr + 7;
      default: prototype_addr_internal = k_addr;
    endcase
    read_addr_thresh_mem = {c_addr_internal, prototype_addr_internal};
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin : all_ffs
    if (!rst_ni) begin
      tree_level_cnt <= 0;
      c_addr_internal <= 0;
      k_addr <= 0;
      k_addr_o_q <= 0;
      c_addr_o_q <= (CAddrWidth)'(EncUnitNumber - EncUnits);
      valid_o <= 0;
    end else begin
      if (encoder_i) begin
        if (tree_level_cnt < (CntWidth)'(TreeDepth - 1)) begin : increment
          tree_level_cnt <= tree_level_cnt + 1;
          k_addr <= (k_addr << 1) + (TreeDepth)'(fp_16_comparision_o);
          valid_o <= 1'b0;
        end else begin : encoding_finished
          tree_level_cnt <= 2'b0;
          c_addr_o_q <= (CAddrWidth)'(c_addr_o_q + (CAddrWidth)'(EncUnits));
          c_addr_internal <= c_addr_internal + 1;
          k_addr_o_q <= (k_addr << 1) + (TreeDepth)'(fp_16_comparision_o);
          k_addr <= 0;
          valid_o <= 1'b1;
        end
      end
    end
  end

  assign k_addr_o = k_addr_o_q;
  assign c_addr_o = c_addr_o_q;

endmodule
