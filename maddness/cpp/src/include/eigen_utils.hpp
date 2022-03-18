//
//  eigen_utils.hpp
//
//  Created By Davis Blalock on 3/2/16.
//  Copyright (c) 2016 Davis Blalock. All rights reserved.
//

#ifndef __EIGEN_UTILS_HPP
#define __EIGEN_UTILS_HPP

#define EIGEN_DONT_PARALLELIZE // ensure no multithreading

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

namespace {

using Eigen::Dynamic;
using Eigen::Map;
using Eigen::Matrix;
using Eigen::RowMajor;

// ================================================================
// typealiases
// ================================================================

template <class T, int Rows = Eigen::Dynamic, int Cols = Eigen::Dynamic>
using RowMatrix = Eigen::Matrix<T, Rows, Cols, Eigen::RowMajor>;

template <class T, int Rows = Eigen::Dynamic, int Cols = Eigen::Dynamic>
using ColMatrix = Eigen::Matrix<T, Rows, Cols, Eigen::ColMajor>;

template <class T> using ColVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <class T>
using RowVector = Eigen::Matrix<T, 1, Eigen::Dynamic, Eigen::RowMajor>;

} // namespace
#endif
