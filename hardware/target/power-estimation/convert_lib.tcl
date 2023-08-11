# to be run with lc_shell


read_lib inputs/adk/stdcells.lib
write_lib $::env(LIB_NAME) -format db -output inputs/adk/stdcells.db

exit