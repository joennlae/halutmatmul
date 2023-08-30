module mixed_int_adder #(
    parameter int unsigned IN_WIDTH = 8,
    parameter int unsigned OUT_WIDTH = 32
  )
  (
    input logic signed  [IN_WIDTH-1:0]  int_short_i,
    input logic signed  [OUT_WIDTH-1:0] int_long_i,
    output logic signed [OUT_WIDTH-1:0] int_long_o
  );

  logic [OUT_WIDTH-1:0] int_short_converted;

  always_comb begin
    int_short_converted[IN_WIDTH-2:0] = int_short_i[IN_WIDTH-2:0];
    if (int_short_i[7] == 1'b1) begin
      int_short_converted[OUT_WIDTH-1:7] = {(OUT_WIDTH-IN_WIDTH+1){1'b1}};
    end else begin
      int_short_converted[OUT_WIDTH-1:7] = {(OUT_WIDTH-IN_WIDTH+1){1'b0}};
    end
  end

  assign int_long_o = int_short_converted + int_long_i;
endmodule
