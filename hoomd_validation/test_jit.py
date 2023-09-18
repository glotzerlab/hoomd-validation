from config import CONFIG
from project_class import Project
from flow import aggregator
import util
import os
import math
import collections
import json
import pathlib

RANDOMIZE_STEPS = 20_000
EQUILIBRATE_STEPS = 100_000
RUN_STEPS = 500_000
RESTART_STEPS = RUN_STEPS // 10
TOTAL_STEPS = RANDOMIZE_STEPS + EQUILIBRATE_STEPS + RUN_STEPS

WRITE_PERIOD = 4_000
LOG_PERIOD = {'trajectory': 50_000, 'quantities': 100}
LJ_PARAMS = {'epsilon': 1.0, 'sigma': 1.0}
NUM_CPU_RANKS = min(8, CONFIG["max_cores_sim"])

WALLTIME_STOP_SECONDS = CONFIG["max_walltime"] * 3600 - 10 * 60

import hoomd
from hoomd import hpmc
import numpy
from custom_actions import ComputeDensity

# integrator
mc = hpmc.integrate.Sphere(nselect=1)
mc.shape['A'] = dict(diameter=0.0)

# pair potential
epsilon = LJ_PARAMS['epsilon']
sigma = LJ_PARAMS['sigma']
r_cut = 3
omega = 20
alpha = numpy.deg2rad(45)
# WCA goes to 0 at 2^1/6 sigma, no XPLOR needed

patchy_lj_str = """
                // WCA via shift

                float sigma;
                float sigsq;
                float rsqinv;
                float r6inv;
                float r12inv;
                float wca_energy;

                float rsq = dot(r_ij, r_ij);
                float r_cut = {r_cut_wca};
                float r_cutsq = r_cut * r_cut;
          
                if (rsq >= r_cutsq)
                {{
                    wca_energy = 0.0f;
                }}
                else
                {{
                    sigma = {wca_sigma};
                    sigsq = sigma * sigma;
                    rsqinv = sigsq / rsq;
                    r6inv = rsqinv * rsqinv * rsqinv;
                    r12inv = r6inv * r6inv;
                    wca_energy = 4 * {wca_epsilon} * (r12inv - r6inv);
                   
                    // energy shift for WCA
                    wca_energy += {wca_epsilon};
                }}

                // patchy stuff
                vec3 ni_world = rotate(q_i, n_i);
                vec3 nj_world = rotate(q_j, n_j);

                float magdr = sqrt(rsq);
                vec3 rhat = r_ij / magdr;

                float costhetai = -dot(rhat, ni_world);
                float costhetaj = dot(rhat, nj_world);

                float fi()
                {{
                    return 1.0f / (1.0f + exp(-{omega} * (costhetai - {cosalpha})) );
                }}
                float fj()
                {{
                    return 1.0f / (1.0f + exp(-{omega} * (costhetaj - {cosalpha})) );
                }}


                // loop over patches eventually
                float this_envelope = fi() * fj();

                // regular lj to be modulated
                r_cut = {r_cut};
                r_cutsq = r_cut * r_cut;
                
                if (rsq >= r_cutsq)
                {{
                    lj_energy = 0.0f;
                }}
                else
                {{
                    sigma = {sigma};
                    sigsq = sigma * sigma;
                    rsqinv = sigsq / rsq;
                    r6inv = rsqinv * rsqinv * rsqinv;
                    r12inv = r6inv * r6inv;
                    float lj_energy = 4 * {epsilon} * (r12inv - r6inv);
                    
                    // energy shift at cutoff
                    float r_cutsqinv = sigsq / r_cutsq;
                    r_cut6inv = r_cutsqinv * r_cutsqinv * r_cutsqinv;
                    r_cut12inv = r_cut6inv * r_cut6inv;
                    float cutVal = 4 * {epsilon} * (r_cut12inv - r_cut6inv);
                    lj_energy -= cutVal;
                }}
               
                return wca_energy + lj_energy * this_envelope;
                """.format(epsilon=epsilon, sigma=sigma, r_cut=r_cut,
                           wca_epsilon=epsilon, wca_sigma=sigma, r_cut_wca=sigma * 2**(1/6),
                           omega=omega, cosalpha=numpy.cos(alpha))

jit_potential = hpmc.pair.user.CPPPotential(r_cut=r_cut,
                                            code=patchy_lj_str,
                                            param_array=[])
mc.pair_potential = jit_potential


# pair force to compute virial pressure
nlist = hoomd.md.nlist.Cell(buffer=0.4)
wca = hoomd.md.pair.LJ(default_r_cut=sigma * 2**(1/6),
                      nlist=nlist)
wca.params[('A', 'A')] = dict(sigma=LJ_PARAMS['sigma'],
                             epsilon=LJ_PARAMS['epsilon']) # TODO: why not /kt
wca.mode = 'shift'

# compute the density
compute_density = ComputeDensity()

# log to gsd
logger_gsd = hoomd.logging.Logger(categories=['scalar', 'sequence'])
logger_gsd.add(jit_potential, quantities=['energy'])
logger_gsd.add(mc, quantities=['translate_moves'])
logger_gsd.add(compute_density)



# just for debugging
device = hoomd.device.CPU()
sim = hoomd.Simulation(device=device, seed=100)
sim.operations.integrator = mc

snap = hoomd.Snapshot(device.communicator)
snap.particles.N = 2
snap.particles.types = ['A']
L = 10
snap.configuration.box = [L, L, L, 0, 0, 0]
snap.particles.position[:] = [[-2,0,0],[2,0,0]]
snap.particles.typeid[:] = [0,0]
sim.create_state_from_snapshot(snap)

# end for debugging

# make simulation
# sim = util.make_simulation(job=job,
#                            device=device,
#                            initial_state=initial_state,
#                            integrator=mc,
#                            sim_mode=sim_mode,
#                            logger=logger_gsd,
#                            table_write_period=WRITE_PERIOD,
#                            trajectory_write_period=LOG_PERIOD['trajectory'],
#                            log_write_period=LOG_PERIOD['quantities'],
#                            log_start_step=RANDOMIZE_STEPS
#                            + EQUILIBRATE_STEPS)
# for loggable in extra_loggables:
#     # call attach method explicitly so we can access simulation state when
#     # computing the loggable quantity
#     if hasattr(loggable, 'attach'):
#         loggable.attach(sim)

compute_density.attach(sim)

def _compute_virial_pressure():
    virials = numpy.sum(wca.virials, 0)
    w = 0
    if virials is not None:
        w = virials[0] + virials[3] + virials[5]
    V = sim.state.box.volume
    return job.statepoint.num_particles * job.statepoint.kT / V + w / (3
                                                                       * V)

logger_gsd[('custom', 'virial_pressure')] = (_compute_virial_pressure,
                                             'scalar')

# move size tuner
mstuner = hpmc.tune.MoveSize.scale_solver(
    moves=['d'],
    target=0.2,
    max_translation_move=0.5,
    trigger=hoomd.trigger.And([
        hoomd.trigger.Periodic(100),
        hoomd.trigger.Before(RANDOMIZE_STEPS | EQUILIBRATE_STEPS // 2)
    ]))
sim.operations.add(mstuner)
sim.operations.computes.append(wca)

sim.run(10)
