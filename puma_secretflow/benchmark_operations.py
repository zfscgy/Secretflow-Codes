# This code is mainly from https://www.secretflow.org.cn/zh-CN/docs/secretflow/main/tutorial/gpt2_with_puma
# Instead of benchmark the GPT2 model (which requires too much RAM), here we only benchmark one transformer layer.
import numpy as np
import jax
import jax.nn as jnn
import jax.numpy as jnp

from flax_gpt2 import FlaxGPT2Block, GPT2Config  
# Notice:
# ! Must use another file to inject the jnn.gelu to the activation! Modifying the constants will not affect the code executed by SPU!!!

tensor_size = [100_000]
input_tensor = jnp.array(np.random.normal(0, 1, tensor_size))


def operator(x):
    # Layernorm
    return (x - jnp.mean(x, axis=-1)) / jnp.std(x, axis=-1)


print("Expected input:", operator(input_tensor)[:3])



import secretflow as sf
from typing import Any, Callable, Dict, Optional, Tuple, Union

import flax.linen as nn
from flax.linen.linear import Array

import argparse
import spu.utils.distributed as ppd
import spu.intrinsic as intrinsic
import spu.spu_pb2 as spu_pb2
from contextlib import contextmanager

copts = spu_pb2.CompilerOptions()
copts.enable_pretty_print = False
copts.xla_pp_kind = 2
# enable x / broadcast(y) -> x * broadcast(1/y)
copts.enable_optimize_denominator_with_broadcast = True


# In case you have a running secretflow runtime already.
sf.shutdown()


def hack_softmax(
    x: Array,
    axis: Optional[Union[int, Tuple[int, ...]]] = -1,
    where: Optional[Array] = None,
    initial: Optional[Array] = None,
) -> Array:
    x_max = jnp.max(x, axis, where=where, initial=initial, keepdims=True)
    x = x - x_max

    # exp on large negative is clipped to zero
    b = x > -14
    nexp = jnp.exp(x)

    divisor = jnp.sum(nexp, axis, where=where, keepdims=True)

    return b * (nexp / divisor)


@contextmanager
def hack_softmax_context(msg: str, enabled: bool = False):
    if not enabled:
        yield
        return
    # hijack some target functions
    raw_softmax = jnn.softmax
    jnn.softmax = hack_softmax
    yield
    # recover back
    jnn.softmax = raw_softmax


def hack_gelu(
    x: Array,
    axis: Optional[Union[int, Tuple[int, ...]]] = -1,
    where: Optional[Array] = None,
    initial: Optional[Array] = None,
) -> Array:
    b0 = x < -4.0
    b1 = x < -1.95
    b2 = x > 3.0
    b3 = b1 ^ b2 ^ True  # x in [-1.95, 3.0]
    b4 = b0 ^ b1  # x in [-4, -1.95]

    # seg1 = a[3] * x^3 + a[2] * x^2 + a[1] * x + a[0]
    # seg2 = b[6] * x^6 + b[4] * x^4 + b[2] * x^2 + b[1] * x + b[0]
    a_coeffs = jnp.array(
        [
            -0.5054031199708174,
            -0.42226581151983866,
            -0.11807612951181953,
            -0.011034134030615728,
        ]
    )
    b_coeffs = jnp.array(
        [
            0.008526321541038084,
            0.5,
            0.3603292692789629,
            0.0,
            -0.037688200365904236,
            0.0,
            0.0018067462606141187,
        ]
    )
    x2 = jnp.square(x)
    x3 = jnp.multiply(x, x2)
    x4 = jnp.square(x2)
    x6 = jnp.square(x3)

    seg1 = a_coeffs[3] * x3 + a_coeffs[2] * x2 + a_coeffs[1] * x + a_coeffs[0]
    seg2 = (
        b_coeffs[6] * x6
        + b_coeffs[4] * x4
        + b_coeffs[2] * x2
        + b_coeffs[1] * x
        + b_coeffs[0]
    )

    ret = b2 * x + b4 * seg1 + b3 * seg2

    return ret


@contextmanager
def hack_gelu_context(msg: str, enabled: bool = False):
    if not enabled:
        yield
        return
    # hijack some target functions
    raw_gelu = jnn.gelu
    jnn.gelu = hack_gelu
    yield
    # recover back
    jnn.gelu = raw_gelu


# Init the 
sf.init(['alice', 'bob', 'carol'], address='local')

alice, bob = sf.PYU('alice'), sf.PYU('bob')
conf = sf.utils.testing.cluster_def(['alice', 'bob', 'carol'])
conf['runtime_config']['protocol'] = 'ABY3'
conf['runtime_config']['field'] = 'FM64'
conf['runtime_config']['fxp_exp_mode'] = 0
conf['runtime_config']['fxp_exp_iters'] = 5

conf['runtime_config']['enable_pphlo_profile'] = True   
conf['runtime_config']['enable_hal_profile'] = True   
# This config is to record the communication and time. Please refer to 
# Refer to https://www.secretflow.org.cn/zh-CN/docs/spu/main/reference/runtime_config#runtimeconfig and https://github.com/secretflow/secretflow/issues/1037

spu = sf.SPU(conf)

def get_input_tensor():
    return input_tensor

input_tensor = bob(get_input_tensor)()

device = spu
input_tensor_ct = input_tensor.to(device)

for i in range(5):
    print("=============================================\n\n\n\n=================================")
    with hack_softmax_context("hijack jax softmax", enabled=True), hack_gelu_context("hack jax gelu", enabled=True):
        output_ct = spu(operator, copts=copts)(input_tensor_ct)

    print(sf.reveal(output_ct)[:3])
    input("Press any key to tart next...")