import freud
import numpy as np


class LJForce:

    def __init__(self, sigma, epsilon, r_cut):
        self._sigma = sigma
        self._epsilon = epsilon
        self._r_cut = r_cut

    @property
    def r_cut(self):
        return self._r_cut

    def __call__(self, distances):
        distance_cond = distances > self._r_cut
        forces = 4 * self._epsilon * (-12 * (self._sigma**12 / distances**13) -
                                      -6 * (self._sigma**6 / distances**7))
        forces[distance_cond] = 0.0
        return forces


class MCPressureContinuous:

    def __init__(self, force_eval):
        self._force_eval = force_eval

    def compute(self, positions, box, kT):

        # do freud neighbor search
        b = freud.box.Box.from_box(box)
        aabb = freud.locality.AABBQuery.from_system((b, positions))
        nlist = aabb.query(
            positions, dict(r_max=self._force_eval.r_cut, exclude_ii=True)
        ).toNeighborList()

        # compute net force on each particle
        rijs = b.wrap(
            positions[nlist.point_indices] - positions[nlist.query_point_indices]
        )
        print(len(rijs))
        print(np.count_nonzero(rijs > 0.000001))
        forces = np.zeros_like(positions)
        np.add.at(forces, nlist.point_indices, self._force_eval(rijs))

        # compute pressure
        virial = np.average(np.einsum("ij,ij->i", forces, positions))
        num_particles = len(positions)
        density = num_particles / b.volume
        pressure = kT * density * (1 - virial / (num_particles * b.dimensions * kT))
        return pressure
