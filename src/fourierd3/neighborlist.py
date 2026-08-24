"""Verlet ("skin") neighbour list with LAMMPS-style rebuild control.

Rebuilding the coordination-number neighbour list at every step is a
non-trivial cost in an MD run, especially since it happens on the CPU while
the rest of Fourier-D3 runs on the GPU. The standard remedy is a Verlet list:
build the list with a cutoff enlarged by a ``skin`` buffer, then reuse it for
as long as no atom can have moved far enough to invalidate it.

The rebuild policy follows the LAMMPS ``neigh_modify`` semantics (``every``,
``delay``, ``check``); see https://docs.lammps.org/neigh_modify.html.

The idea of applying this to a D3 correction comes from the skin neighbour
list in Jerry Ho's fork of torch-dftd
(https://github.com/CheukHinHoJerry/torch-dftd). The implementation here is
independent: it is built on matscipy, exposed as a standalone cache rather
than calculator methods, and extends the rebuild test to cover cell
deformation so it stays valid under NPT. Only the CN list is affected, since
the Fourier-D3 dispersion sum is evaluated in reciprocal space and never
touches a neighbour list at all.
"""

import numpy as np
import torch
from matscipy.neighbours import neighbour_list


class SkinNeighborList:
    """Cached neighbour list built with a skin buffer around the true cutoff.

    The list is built at ``cutoff + skin`` and reused across calls. Pairs that
    sit in the buffer region are still returned; they are harmless because
    ``FourierD3.compute_cn_smooth`` masks out every pair beyond ``r_cut``, so
    the CN is identical to the one obtained from a freshly built list.

    Integer (unit) shifts are cached rather than Cartesian shift vectors, so
    the same list stays usable when the cell changes: the caller multiplies
    the unit shifts by the current (possibly strained) cell each step.

    Rebuild logic, evaluated on each call with ``step`` = number of previous
    calls and ``s`` = steps since the last rebuild:

      - always rebuild if there is no cached list, or if the number/identity
        of the atoms changed;
      - otherwise only consider rebuilding when ``step % every == 0`` and
        ``s >= delay``;
      - when ``check`` is True, rebuild only if the atoms could actually have
        moved out of the buffer, i.e. ``2 * max_displacement + cell_slack >
        skin``. The factor 2 accounts for two atoms moving towards each other,
        and ``cell_slack`` bounds how much the periodic image shift vectors
        moved because the cell itself deformed (relevant under NPT).
      - when ``check`` is False, rebuild unconditionally on those steps.

    Setting ``skin = 0`` disables caching entirely: every call rebuilds, which
    reproduces the original behaviour exactly.

    Args:
        cutoff:  true interaction cutoff in Ångström (the CN ``r_cut``).
        skin:    buffer width in Ångström. 0 disables the Verlet caching.
        every:   only consider rebuilding every ``every`` calls.
        delay:   never rebuild until ``delay`` calls since the last rebuild.
        check:   if True, rebuild only when the displacement criterion fires.
        device:  torch device for the returned tensors.
        dtype:   torch dtype for the returned unit shifts.
        verbose: print the configuration on construction.
    """

    def __init__(
        self,
        cutoff: float,
        skin: float = 0.0,
        every: int = 1,
        delay: int = 0,
        check: bool = True,
        device=None,
        dtype=torch.float32,
        verbose: bool = False,
    ) -> None:
        if cutoff <= 0:
            raise ValueError(f"cutoff must be positive, got {cutoff}")
        if skin < 0:
            raise ValueError(f"skin must be non-negative, got {skin}")
        if every < 1:
            raise ValueError(f"every must be >= 1, got {every}")
        if delay < 0:
            raise ValueError(f"delay must be >= 0, got {delay}")

        self.cutoff = float(cutoff)
        self.skin = float(skin)
        self.every = int(every)
        self.delay = int(delay)
        self.check = bool(check)
        self.device = device
        self.dtype = dtype

        if verbose and self.enabled:
            print(
                f"Fourier-D3 skin neighbour list: cutoff={self.cutoff:.3f} A, "
                f"skin={self.skin:.3f} A, every={self.every}, "
                f"delay={self.delay}, check={self.check}"
            )
        if self.enabled and not self.check and (self.every > 1 or self.delay > 0):
            print(
                "[WARNING] SkinNeighborList: check=False together with "
                "every>1 or delay>0 disables the displacement test, so fast-"
                "moving atoms can silently invalidate the cached list."
            )

        self.reset()

    @property
    def enabled(self) -> bool:
        """True when the list is cached across calls (i.e. skin > 0)."""
        return self.skin > 0.0

    @property
    def build_cutoff(self) -> float:
        """The cutoff the underlying list is actually built with."""
        return self.cutoff + self.skin

    def reset(self) -> None:
        """Drop the cached list and reset all counters.

        Call this whenever the calculator is pointed at a genuinely new
        structure, so that a stale list is never reused.
        """
        self.n_calls = 0
        self.n_rebuilds = 0
        self._steps_since_build = 0
        self._edge_index = None
        self._unit_shifts = None
        self._max_unit_shift = np.zeros(3, dtype=np.int64)
        self._ref_scaled = None
        self._ref_cell = None
        self._ref_numbers = None

    def get(self, atoms):
        """Return ``(edge_index, unit_shifts)`` for ``atoms``.

        Args:
            atoms: ASE Atoms object at the current step.

        Returns:
            edge_index:  (2, n_edges) long tensor of (source, target) indices.
            unit_shifts: (n_edges, 3) tensor of integer lattice shifts, in the
                         calculator dtype. Multiply by the current cell to get
                         Cartesian shift vectors.
        """
        if not self.enabled:
            self.n_calls += 1
            return self._build(atoms)

        if self._needs_rebuild(atoms):
            self._build(atoms)
            self._steps_since_build = 0
        else:
            self._steps_since_build += 1

        self.n_calls += 1
        return self._edge_index, self._unit_shifts

    def _needs_rebuild(self, atoms) -> bool:
        """Apply the LAMMPS every/delay/check policy to the current step."""
        if self._edge_index is None:
            return True

        # A different structure (or a changed composition) invalidates the list
        if len(atoms) != len(self._ref_numbers):
            return True
        if not np.array_equal(atoms.numbers, self._ref_numbers):
            return True

        # 'every': only steps that are a multiple of `every` may rebuild
        if self.n_calls % self.every != 0:
            return False

        # 'delay': hold off until enough steps have passed since the last build
        if self._steps_since_build < self.delay:
            return False

        if not self.check:
            return True

        return self._buffer_exhausted(atoms)

    def _buffer_exhausted(self, atoms) -> bool:
        """True when atoms may have moved far enough to invalidate the list.

        Two effects can bring a previously excluded pair inside the cutoff:
        the atoms moving (each by up to ``max_disp``, so a pair can close by
        ``2 * max_disp``), and the cell deforming, which moves the periodic
        image shift vectors ``S @ cell``. The latter is bounded by
        ``sum_a |S_a|_max * ||dcell_a||`` over the lattice vectors a, using the
        largest image index present in the cached list.
        """
        # calculate Cartesian displacement based on wrapped image
        scaled_positions = atoms.get_scaled_positions()
        if len(scaled_positions) == 0:
            return False

        # use minimum image for Cartesian displacements
        scaled_delta = scaled_positions - self._ref_scaled
        scaled_delta -= np.round(scaled_delta)
        delta = scaled_delta @ atoms.cell
        max_disp = float(np.sqrt(np.einsum("ij,ij->i", delta, delta)).max())

        dcell = np.asarray(atoms.cell) - self._ref_cell
        cell_slack = float(
            np.dot(self._max_unit_shift, np.linalg.norm(dcell, axis=1))
        )
        return 2.0 * max_disp + cell_slack > self.skin

    def _build(self, atoms):
        """Build the neighbour list at ``cutoff + skin`` and cache it."""
        # Use wrapped positions so that max_unit_shift in neighbor list doesn't
        # include irrelevant overall drift.  Will need to modify shifts that are
        # returned by get() to correct for this.
        positions_s = atoms.get_scaled_positions(wrap=False)
        positions_s_wrap = np.floor(positions_s)
        positions_s -= positions_s_wrap
        positions = positions_s @ atoms.cell

        sender, receiver, unit_shifts = neighbour_list(
            quantities="ijS",
            pbc=atoms.pbc,
            cell=atoms.cell,
            positions=positions,
            cutoff=self.build_cutoff,
        )

        self._edge_index = torch.as_tensor(
            np.stack((sender, receiver)),
            dtype=torch.long,
            device=self.device,
        )

        # Largest image index per lattice direction, used to bound how much a
        # cell deformation can move the shift vectors (see _buffer_exhausted)
        # add 1 to check against next closest image that is currently outside cutoff,
        # becoming close enough to matter.
        # Need to do this with raw unit_shifts, before they are corrected for wrapping
        if len(unit_shifts):
            self._max_unit_shift = np.abs(unit_shifts).max(axis=0) + 1
        else:
            self._max_unit_shift = np.zeros(3, dtype=np.int64)

        # reconstruct unit_shifts that would have been used if positions were not
        # wrapped, so that returned values are correct, namely
        #     d = p_j - p_i + S @ cell
        # as documented in https://github.com/libAtoms/matscipy/blob/770636d/matscipy/neighbours.py#L553
        # we used
        #     d = (p_j - p_j_wrap @ cell) - (p_i - p_i_wrap @ cell) + S @ cell
        #       = p_j - p_i + (S + p_i_wrap - p_j_wrap) @ cell
        unit_shifts += positions_s_wrap[sender].astype(int) # i
        unit_shifts -= positions_s_wrap[receiver].astype(int) # j

        self._unit_shifts = torch.as_tensor(
            unit_shifts,
            dtype=self.dtype,
            device=self.device,
        )

        # store wrapped position for Cartesian displacement
        self._ref_scaled = atoms.get_scaled_positions()
        self._ref_cell = np.asarray(atoms.cell).copy()
        self._ref_numbers = atoms.numbers.copy()
        self.n_rebuilds += 1

        return self._edge_index, self._unit_shifts

    @property
    def rebuild_fraction(self) -> float:
        """Fraction of calls that triggered a rebuild (1.0 without a skin)."""
        if self.n_calls == 0:
            return 0.0
        return self.n_rebuilds / self.n_calls

    def __repr__(self) -> str:
        return (
            f"SkinNeighborList(cutoff={self.cutoff:.3f}, skin={self.skin:.3f}, "
            f"every={self.every}, delay={self.delay}, check={self.check}, "
            f"rebuilds={self.n_rebuilds}/{self.n_calls})"
        )
