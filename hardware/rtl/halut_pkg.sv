

package halut_pkg;

`ifndef NUM_M
  `define NUM_M 32
`endif

`ifndef NUM_DECODER_UNITS
  `define NUM_DECODER_UNITS 16
`endif

`ifndef NUM_C
  `define NUM_C 32
`endif

  localparam integer unsigned K = 16;
  localparam integer unsigned C = `NUM_C;
  localparam integer unsigned M = `NUM_M;
  localparam integer unsigned DataTypeWidth = 16;
  localparam integer unsigned DecoderUnits = `NUM_DECODER_UNITS;
endpackage : halut_pkg
