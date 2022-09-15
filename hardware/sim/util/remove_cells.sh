#!/bin/bash

cp design.v intermediate.v

THRESHOLD="${VTH:-R}" 

sed "/TAPCELL_ASAP7_75t_$THRESHOLD/d" intermediate.v > intermediate_2.v
sed "/FILLERxp5_ASAP7_75t_$THRESHOLD/d" intermediate_2.v > removed.v

rm design.v
mv removed.v design.v
