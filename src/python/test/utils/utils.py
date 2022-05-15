from __future__ import print_function
import timeit
import functools
from typing import Callable, Optional
import torch
import numpy as np

from halutmatmul.modules import ErrorTuple, error_numpy


# inspiration
# https://github.com/geohot/tinygrad/blob/58ed46963efab46bbe17439b74f82a388fa593c2/test/test_ops.py#L4
def helper_test_module(
    ts_input: torch.Tensor,
    torch_module: torch.nn.Module,
    halutmatmul_module: torch.nn.Module,
    rel_error: float = 0.2,
    scaled_error_max: float = 0.2,
) -> None:

    out = torch_module(ts_input)
    ret = halutmatmul_module(ts_input)

    print(
        "shapes in, out_pytorch, out_halutmatmul:",
        ts_input.shape,
        out.shape,
        ret.shape,
    )
    if rel_error > 0:
        error_hist_numpy(ret.detach().cpu().numpy(), out.detach().cpu().numpy())
        check_if_error_normal_dist_around_zero(
            ret.detach().cpu().numpy(),
            out.detach().cpu().numpy(),
            max_rel_error=rel_error,
        )

    scaled_error = 0.0
    if "cuda" in str(out.device):
        # pylint: disable=import-outside-toplevel
        from halutmatmul.cuda.functions import error_cupy

        res_error = error_cupy(ret, out)
        res_error /= ts_input.shape[0]
        print("SCALED_ERROR", res_error[ErrorTuple.SCALED_ERROR])
        scaled_error = res_error[ErrorTuple.SCALED_ERROR]
    else:
        res_error = error_numpy(ret.detach().cpu().numpy(), out.detach().cpu().numpy())
        res_error /= ts_input.shape[0]
        print("SCALED_ERROR", res_error[ErrorTuple.SCALED_ERROR])
        scaled_error = res_error[ErrorTuple.SCALED_ERROR]

    # scaled error check
    assert scaled_error < scaled_error_max

    torch_fp = (
        timeit.Timer(functools.partial(torch_module, ts_input)).timeit(5) * 1000 / 5
    )
    halutmatmul_fp = (
        timeit.Timer(functools.partial(halutmatmul_module, ts_input)).timeit(5)
        * 1000
        / 5
    )

    # import cProfile
    # from pstats import SortKey
    # with cProfile.Profile() as pr:
    #     halutmatmul_module(ts_input)
    #     pr.disable()
    #     pr.print_stats(sort=SortKey.CUMULATIVE)

    print(
        "torch/halutmatmul fp: %.2f / %.2f ms"
        % (
            torch_fp,
            halutmatmul_fp,
        )
    )


def error_hist_numpy(actual: np.ndarray, desired: np.ndarray) -> None:
    _abs = np.abs(actual - desired).ravel()
    rel = ((np.abs(actual - desired) / desired) * 100).ravel()

    asciihist(_abs, str_tag="Abs  ")
    asciihist(rel, str_tag="Rel %")


def check_if_error_normal_dist_around_zero(
    actual: np.ndarray, desired: np.ndarray, max_rel_error: float = 0.2
) -> None:
    rel = ((np.abs(actual - desired) / desired) * 100).ravel()
    counts, cutoffs = np.histogram(rel, bins=10)
    assert np.argmax(counts) == 0 or np.argmax(counts) == 1  # error peaks around zero
    assert cutoffs[9] < max_rel_error * 100  # check max rel error


def asciihist(
    it: np.ndarray,
    bins: int = 10,
    minmax: Optional[str] = None,
    str_tag: str = "",
    scale_output: int = 30,
    generate_only: bool = False,
    print_function: Callable = print,
) -> str:
    """Create an ASCII histogram from an interable of numbers.
    Author: Boris Gorelik boris@gorelik.net.
    based on  http://econpy.googlecode.com/svn/trunk/pytrix/pytrix.py
    Gist: https://gist.github.com/bgbg/608d9ef4fd75032731651257fe67fc81
    License: MIT
    """
    ret = []
    itarray = np.asanyarray(it)
    if minmax == "auto":
        _minmax = np.percentile(it, [5, 95])
        if _minmax[0] == _minmax[1]:
            # for very ugly distributions
            minmax = None
    if minmax is not None:
        # discard values that are outside minmax range
        mn = minmax[0]
        mx = minmax[1]
        itarray = itarray[itarray >= mn]
        itarray = itarray[itarray <= mx]
    if itarray.size:
        total = len(itarray)
        counts, cutoffs = np.histogram(itarray, bins=bins)
        cutoffs = cutoffs[1:]
        if str_tag:
            str_tag = "%s " % str_tag
        else:
            str_tag = ""
        if scale_output is not None:
            scaled_counts = counts.astype(float) / counts.sum() * scale_output
        else:
            scaled_counts = counts

        if minmax is not None:
            ret.append("Trimmed to range (%s - %s)" % (str(minmax[0]), str(minmax[1])))
        for cutoff, original_count, scaled_count in zip(cutoffs, counts, scaled_counts):
            ret.append(
                "{:s}{:>8.2f} |{:<7,d} | {:s}".format(
                    str_tag, cutoff, original_count, "*" * int(scaled_count)
                )
            )
        ret.append("{:s}{:s} |{:s} | {:s}".format(str_tag, "-" * 8, "-" * 7, "-" * 7))
        ret.append("{:s}{:>8s} |{:<7,d}".format(str_tag, "N=", total))
    else:
        ret = []
    if not generate_only:
        for line in ret:
            print_function(line)
    ret_str = "\n".join(ret)
    return ret_str
