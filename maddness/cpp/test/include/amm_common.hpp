//
//  amm_common.hpp
//  Bolt
//
//  Created by DB on 12/10/19.
//  Copyright © 2019 D Blalock. All rights reserved.
//

#ifndef amm_common_h
#define amm_common_h

// just include everything here to avoid dup includes in cpp files
#include <stdio.h>
#include <string>
#include <vector>

#include "eigen_utils.hpp"
#include "mithral.hpp"
#include "testing_utils.hpp"

static constexpr int kNreps = 5;
static constexpr int kNtrials = 10;

template <class InputT> struct input_type_traits {};
template <> struct input_type_traits<float> {
  using scales_type = float;
  using offsets_type = float;
  const char *name = "f32";
  // using output_type = float;
};
template <> struct input_type_traits<int16_t> {
  using scales_type = uint8_t;
  using offsets_type = int16_t;
  const char *name = "i16";
  // using output_type = int16_t;
};
template <> struct input_type_traits<int8_t> {
  using scales_type = uint8_t;  // doesn't matter; unused
  using offsets_type = uint8_t; // doesn't matter; unused
  const char *name = "i8";
  // using output_type = int8_t;
};

#endif /* amm_common_h */
