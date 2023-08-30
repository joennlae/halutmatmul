

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

`ifndef ACC_TYPE
  `define ACC_TYPE FP32
`endif

`ifndef DATA_WIDTH
  `define DATA_WIDTH 16
`endif

  localparam integer unsigned K = 16;
  localparam integer unsigned C = `NUM_C;
  localparam integer unsigned M = `NUM_M;
  localparam integer unsigned DataTypeWidth = `DATA_WIDTH;
  localparam integer unsigned DecoderUnits = `NUM_DECODER_UNITS;
  typedef enum {FP32, INT} AccumulationEnum;
  localparam AccumulationEnum AccumulationOption = `ACC_TYPE;
endpackage : halut_pkg
