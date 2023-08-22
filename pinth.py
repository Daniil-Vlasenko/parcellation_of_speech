from dataclasses import dataclass
from functools import partial
from typing import Optional

from flax import struct
import jax
from jax import numpy as jnp, vmap, ops
from jax.flatten_util import ravel_pytree
from jax.tree_util import register_pytree_node
import numpy as np
import matplotlib.pyplot as plt



V_RESET = -87
V_THR = -40


@partial(jax.jit, static_argnums=(0, 3))
def integrate_ode_delta_depr(ode_fn, x0, delta_t: float, n: int):
    def body(x, _):
        dx = ode_fn(x)
        x_vector, unravel = ravel_pytree(x)
        dx_vector, _ = ravel_pytree(dx)

        x_new = unravel(x_vector + dx_vector * delta_t)

        a = x_new.Vm >= V_THR

        Vm_new = jnp.where(a, V_RESET, x_new.Vm)
        xR_new = jnp.where(a, x_new.x + a, x_new.x)

        x_new = x_new.replace(Vm=Vm_new, x=xR_new)

        return x_new, x

    return jax.lax.scan(body, x0, jnp.arange(n))[-1]


@partial(jax.jit, static_argnums=(0, 3))
def integrate_ode_delta(system, x0, delta_t: float, n: int):
    def body(x, rng):
        dx = system.ode(x)
        x_vector, unravel = ravel_pytree(x)
        dx_vector, _ = ravel_pytree(dx)

        x_new = unravel(x_vector + dx_vector * delta_t)
        x_new = system.update_delta(x_new)
        x_new = system.add_noise(x_new, rng, delta_t)

        return x_new, x

    return jax.lax.scan(body, x0, jax.random.split(jax.random.PRNGKey(0), n))[-1]


@dataclass
class LIF_:
    I_DC: float = 3.
    VL: float = -67.
    Cm: float = 1.
    gL: float = .1
    tau_D: float = 2.
    tau_R: float = .2
    std: float = 2.028

    @staticmethod
    def TE_cell():
        return LIF_(I_DC=1.25, VL=-67, gL=.0264, Cm=1, tau_D=24.3150, tau_R=4, std=2.028)

    @staticmethod
    def TI_cell(N=10):
        return LIF_(I_DC=.0851, VL=-67, gL=.1, Cm=1, tau_D=30.3575, tau_R=5, std=0.282)

    @struct.dataclass
    class Neurons:
        Vm: jnp.ndarray
        x: jnp.ndarray
        s: jnp.ndarray

    def __hash__(self):
        return hash(id(self))

    def add_noise(self, neurons, rng, dt):
        return neurons.replace(Vm=
                               neurons.Vm + jax.random.normal(rng, neurons.Vm.shape) * jnp.sqrt(dt * self.std))

    def update_delta(self, neurons):
        a = neurons.Vm >= V_THR

        Vm_new = jnp.where(a, V_RESET, neurons.Vm)
        xR_new = jnp.where(a, neurons.x + a, neurons.x)

        return neurons.replace(Vm=Vm_new, x=xR_new)

    def ode(self, neurons,
            I_syn: Optional[jnp.ndarray] = None,
            I_inp: Optional[jnp.ndarray] = None,
            noise: Optional[jnp.ndarray] = None
            ):

        if I_syn is None:
            I_syn = 0.

        if I_inp is None:
            I_inp = 0.

        if noise is None:
            noise = 0.

        dVdt = (self.gL * (self.VL - neurons.Vm) + self.I_DC + I_syn + I_inp + noise) / self.Cm

        dxdt = -neurons.x / self.tau_R
        dsdt = (neurons.x - neurons.s) / self.tau_D

        return LIF_.Neurons(dVdt, dxdt, dsdt)


@struct.dataclass
class Edges:
    frm: jnp.ndarray
    to: jnp.ndarray

    def apply(self, values_from: jnp.ndarray, size_to: int):
        values_to = jnp.zeros(size_to)
        values_to = values_to.at[self.to].add(values_from[self.frm])

        return values_to


def create_edges(N, r=1):
    R = int(N * r)

    adjacency_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(0, R + 1):
            adjacency_matrix[i, (i + j) % N] = 1
        adjacency_matrix[i, i] = 0

    to, frm = jnp.where(adjacency_matrix)

    return Edges(frm, to)


@dataclass
class Population:
    neuron_type: LIF_
    edges: Edges
    N: int

    g_syn: float
    VR: float

    @staticmethod
    def TII(N=10):
        return Population(neuron_type=LIF_.TI_cell(), edges=create_edges(N=N, r=1), N=N, g_syn=.432, VR=-80)

    @staticmethod
    def TEE(N=10):
        return Population(neuron_type=LIF_.TE_cell(), edges=create_edges(N=N, r=0), N=N, g_syn=0, VR=0)

    def __hash__(self):
        return hash(id(self))

    def add_noise(self, neurons, rng, dt):
        return self.neuron_type.add_noise(neurons, rng, dt)

    def update_delta(self, neurons: LIF_.Neurons):
        return self.neuron_type.update_delta(neurons)

    def ode(self, neurons: LIF_.Neurons, I_syn: Optional[jnp.ndarray] = None, I_inp: Optional[jnp.ndarray] = None):
        if I_syn is None:
            I_syn = jnp.zeros(self.N)

        if I_inp is None:
            I_inp = jnp.zeros(self.N)
        I_syn = I_syn.at[self.edges.to].add(
            self.g_syn * neurons.s[self.edges.frm] * (self.VR - neurons.Vm[self.edges.to]))

        deriv = self.neuron_type.ode(neurons, I_syn, I_inp)

        return deriv


@dataclass
class POPs:
    popTE: Population
    popTI: Population

    edges_TITE: Edges
    edges_TETI: Edges

    g_TITE: float = .207
    g_TETI: float = .666

    VR_E: float = 0
    VR_I: float = -80

    @struct.dataclass
    class Neurons:
        TE: LIF_.Neurons
        TI: LIF_.Neurons

    @staticmethod
    def I_syn_frm_to(popFROM, popTO, neuronsFROM, neuronsTO, edges, g_syn, VR):
        I_syn = jnp.zeros(popTO.N)
        I_syn = (g_syn * I_syn.at[edges.to].add(neuronsFROM.s[edges.frm]) * (VR - neuronsTO.Vm))
        return I_syn

    def __hash__(self):
        return hash(id(self))

    def add_noise(self, neurons, rng, dt):
        rngs = jax.random.split(rng, 4)
        return self.Neurons(self.popTE.add_noise(neurons.TE, rngs[2], dt),
                            self.popTI.add_noise(neurons.TI, rngs[3], dt))

    def update_delta(self, neurons: Neurons):
        return self.Neurons(self.popTE.update_delta(neurons.TE),
                            self.popTI.update_delta(neurons.TI))

    def ode(self, neurons: Neurons):
        I_syn_TITE = self.I_syn_frm_to(self.popTI, self.popTE, neurons.TI, neurons.TE, self.edges_TITE, self.g_TITE,
                                       self.VR_I)
        I_syn_TETI = self.I_syn_frm_to(self.popTE, self.popTI, neurons.TE, neurons.TI, self.edges_TETI, self.g_TETI,
                                       self.VR_E)
        deriv_TE = self.popTE.ode(neurons.TE, I_syn_TITE)
        deriv_TI = self.popTI.ode(neurons.TI, I_syn_TETI)

        return POPs.Neurons(deriv_TE, deriv_TI)


def create_edges_between_pops(N1, N2):
    A = jnp.ones([N1, N2], dtype=jnp.int32)
    frm, to = jnp.where(A)
    return Edges(frm, to)




