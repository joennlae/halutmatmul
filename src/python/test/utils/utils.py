from __future__ import print_function
import timeit
import functools
from typing import Callable, Optional, Any
import torch
from torch.autograd import profiler
import numpy as np

from halutmatmul.modules import ErrorTuple, error_numpy, HalutConv2d


def getBack(var_grad_fn: Any, all_shapes: list) -> None:
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], "variable")
                print(n[0])
                # print("Tensor with grad found:", tensor)
                print(" - gradient:", tensor.grad.shape)
                all_shapes.append(tensor.grad.shape)
                print()
            except AttributeError:
                getBack(n[0], all_shapes)


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

    # backwards path check
    loss_torch = out.sum()
    loss_halut = ret.sum()

    loss_torch.backward()
    loss_halut.backward()

    all_shapes = []  # type: ignore
    getBack(loss_torch.grad_fn, [])
    getBack(loss_halut.grad_fn, all_shapes)

    state_dict = halutmatmul_module.state_dict()
    shapes_normal = []
    for k, v in state_dict.items():
        if k in (
            ["lut", "thresholds", "bias", "A"]
            if halutmatmul_module.use_A
            else ["lut", "thresholds", "bias"]
        ):
            shapes_normal.append(v.shape)
    print("all shapes:", len(all_shapes), "normal", len(shapes_normal))

    if (
        isinstance(halutmatmul_module, HalutConv2d)
        and halutmatmul_module.loop_order == "im2col"
    ):
        if not halutmatmul_module.use_prototypes:
            assert len(shapes_normal) == len(all_shapes)
            assert shapes_normal == all_shapes[::-1]
    elif (
        isinstance(halutmatmul_module, HalutConv2d)
        and halutmatmul_module.loop_order == "kn2col"
    ):
        pass
        # backprop test is not reliable for kn2col
        # sometimes one more tensor is added to the list (for whatever reason)
        # kx_x_ky = halutmatmul_module.kernel_size[0] *
        # halutmatmul_module.kernel_size[1]  # type: ignore
        # print("all", all_shapes)
        # print("normal", shapes_normal)
        # assert len(shapes_normal) * kx_x_ky == len(all_shapes)
        # assert shapes_normal == all_shapes[: len(all_shapes) // kx_x_ky][::-1]
    else:
        # linear
        if not halutmatmul_module.use_prototypes:
            pass
            # assert len(shapes_normal) == len(all_shapes)
            # assert shapes_normal == all_shapes[::-1]

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

    # pylint: disable=using-constant-test
    if False:
        # profiling
        # warmup
        # https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
        halutmatmul_module(ts_input)
        with profiler.profile(
            use_cuda=True, record_shapes=True, profile_memory=True, with_stack=True
        ) as prof:
            ret = halutmatmul_module(ts_input)
            loss = ret.sum()
            loss.backward()
        # with profiler.profile(with_stack=True, profile_memory=True) as prof:
        #     halutmatmul_module(ts_input)
        print(
            prof.key_averages(group_by_stack_n=10).table(
                sort_by="cuda_time_total", row_limit=20
            )
        )
        prof.export_chrome_trace("profiler_trace_cuda.json")


def error_hist_numpy(actual: np.ndarray, desired: np.ndarray) -> None:
    _abs = np.abs(actual - desired).ravel()  # type: ignore
    rel = ((np.abs(actual - desired) / desired) * 100).ravel()  # type: ignore

    asciihist(_abs, str_tag="Abs  ")
    asciihist(rel, str_tag="Rel %")


def check_if_error_normal_dist_around_zero(
    actual: np.ndarray, desired: np.ndarray, max_rel_error: float = 0.2
) -> None:
    rel = ((np.abs(actual - desired) / desired) * 100).ravel()  # type: ignore
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
        itarray = itarray[itarray >= mn]  # type: ignore
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
