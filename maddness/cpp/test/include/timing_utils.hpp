//
//  timing_utils.cpp
//  Dig
//
//  Created by Davis Blalock on 2016-3-28
//  Copyright (c) 2016 Davis Blalock. All rights reserved.
//

#ifndef _TIMING_UTILS_HPP
#define _TIMING_UTILS_HPP

#include <chrono>
#include <iostream>

#ifdef _WIN32
#include <intrin.h> // for cycle counting
#endif

// cycle counting adapted from http://stackoverflow.com/a/13772771
#ifdef _WIN32 //  Windows
static inline uint64_t time_now_cycles() { return __rdtsc(); }
#else //  Linux/GCC
static inline uint64_t time_now_cycles() {
  unsigned int lo, hi;
  __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
  return ((uint64_t)hi << 32) | static_cast<uint64_t>(lo);
}
#endif

using cputime_t = std::chrono::high_resolution_clock::time_point;

static inline cputime_t timeNow() {
  return std::chrono::high_resolution_clock::now();
}

static inline int64_t durationUs(cputime_t t1, cputime_t t0) {
  auto diff = t1 >= t0 ? t1 - t0 : t0 - t1; // = abs(t1 - t0);
  return std::chrono::duration_cast<std::chrono::microseconds>(diff).count();
}

static inline double durationMs(cputime_t t1, cputime_t t0) {
  return durationUs(t1, t0) / 1000.0;
}

class EasyTimer {
public:
  using TimeT = double;
  EasyTimer(TimeT &write_to, bool add = false, bool ms = true)
      : _write_here(write_to), _tstart(timeNow()), _add(add), _ms(ms) {}
  ~EasyTimer() {
    TimeT duration = static_cast<TimeT>(durationUs(_tstart, timeNow()));
    if (_ms) {
      duration /= 1000.0;
    }
    if (_add) {
      _write_here += duration;
    } else {
      _write_here = duration;
    }
  }

private:
  TimeT &_write_here;
  cputime_t _tstart;
  bool _add;
  bool _ms;
};

#endif // _TIMING_UTILS_HPP
