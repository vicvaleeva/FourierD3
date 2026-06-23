import math
from typing import Optional, List

import torch
from numpy import unique
from torchpme.lib.kvectors import get_ns_mesh, generate_kvectors_for_ewald

from fastd3.utils import decomp, load_rcov, load_cnref, safe_det_3x3
from fastd3.pair_pot import D3Potential, CNPotential
from fastd3.kspace_filter_d3 import KSpaceFilterD3, KSpaceFilterCN
from fastd3.mesh_interpolator_d3 import MeshInterpolatorD3


class FastD3(torch.nn.Module):
    """Fast DFT-D3 dispersion correction using reciprocal-space summation.

    Evaluates the DFT-D3 dispersion energy via the Fast-D3 framework: the
    environment-dependent pairwise C6 coefficients are approximated by a low-rank
    tensor decomposition, which allows the periodic lattice sum to be evaluated
    efficiently in reciprocal space using Smooth Particle-Mesh Ewald (SPME),
    Particle-Mesh Ewald (PME), or direct Ewald summation.

    The coordination numbers (CNs) that determine the C6 environment-dependence
    can be computed with two options:
      - 'smooth_cut': the modified D3 CN, which decays strictly to zero at the
        cutoff radius; requires a short real-space neighbour list (~6 Å).
      - 'd4': the D4 erfc-based CN, which is a long-range convergent sum evaluated
        entirely in reciprocal space (no neighbour list needed).

    Args:
        species:             atomic numbers for all atoms in the configuration.
        cell:                (3, 3) array of lattice vectors in Ångström.
        pbc:                 periodic boundary conditions (must be 3D periodic).
        mesh_spacing:        target mesh spacing in Å for PME/SPME (smaller = finer mesh).
        c6tol:               max relative error (in %) for the C6 rank decomposition.
        xcfunc:              XC functional name; selects BJ damping parameters.
        device:              torch device.
        dtype:               torch dtype.
        method:              reciprocal-space method: 'spme', 'pme', or 'ewald'.
        params:              BJ damping params [s6, s8, a1, a2]; overrides xcfunc defaults.
        cnfunc:              CN function: 'smooth_cut' or 'd4'.
        interpolation_nodes: B-spline order for PME/SPME interpolation.
        k_cutoff:            reciprocal-space cutoff in Å^-1 for Ewald summation.
        r_cut:               real-space cutoff in Å for the CN neighbour list.
        cncorr:              optional affine correction to CN values: CN' = a*CN + b,
                             shape (n_species, 2), with columns [b, a].
        verbose:             print rank and mesh info during initialization.
    """

    def __init__(
        self,
        species: List,
        cell,
        pbc: Optional[torch.tensor] = None,
        mesh_spacing: float = 1.2,
        c6tol: float = 1,
        xcfunc: str = 'pbe',
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dtype = torch.float32,
        method: str = 'spme',
        params = None,
        cnfunc = 'smooth_cut',
        interpolation_nodes: int = 5,
        k_cutoff: float = 10.0,
        r_cut: float = 6.0,
        cncorr: Optional[torch.tensor] = None,
        verbose = True
    ) -> None:
        super().__init__()

        # On Ampere/Hopper GPUs (e.g. H100), torch may dispatch float32 matmuls
        # (incl. the complex bmm/einsum over k-points in the reciprocal-space sum)
        # to TF32 tensor cores. TF32 keeps only a 10-bit mantissa (eps ~ 5e-4),
        # which caps the float32 energy accuracy at ~1e-2 meV/atom regardless of
        # mesh size, while true IEEE float32 reaches ~1e-5 meV/atom. Force full
        # float32 precision so accuracy does not silently depend on the global
        # cuBLAS/torch default. (No effect on the float64 path.)
        if dtype == torch.float32:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

        self.device = device
        self.dtype = dtype
        self.method = method
        self.mesh_spacing = mesh_spacing
        self.interpolation_nodes = interpolation_nodes
        self.k_cutoff = k_cutoff

        # Optional affine correction applied to coordination numbers after computation:
        # CN'_i = cncorr[species_i, 1] * CN_i + cncorr[species_i, 0]
        if cncorr is not None:
            self.cncorr = cncorr.to(device)
        else:
            self.cncorr = None

        if pbc is not None:
            assert pbc.all(), "particle-mesh only supports 3d pbc"

        if verbose:
            print("Assuming 3D PBC are satisfied")

        # Map atomic numbers to contiguous species indices 0, 1, ..., n_species-1.
        # All per-species data arrays are indexed by these compact indices.
        species_unique = unique(species)
        species_map = {species_unique[i]: i for i in range(len(species_unique))}
        converted_species = torch.tensor(
            [species_map[s] for s in species],
            dtype=torch.long,
            device=device
        )

        # Register as buffer so it moves with model
        self.register_buffer('species', converted_species)
        self.n_species = len(species_unique)

        self.xcfunc = xcfunc
        self.cnfunc = cnfunc

        # Unit conversion: all internal distances are in Bohr (atomic units)
        angstrom_to_bohr = (1 / 0.52917726)
        self.r_cut = r_cut * angstrom_to_bohr
        self.register_buffer(
            'angstrom_to_bohr',
            torch.tensor(angstrom_to_bohr, dtype=dtype, device=device)
        )

        cell = torch.tensor(cell, device=self.device, dtype=dtype)

        cell_bohr = cell * self.angstrom_to_bohr
        self.register_buffer('cell', cell_bohr)

        # Cell volume Omega (Bohr^3), used in the 1/Omega prefactor of the k-space energy
        volume = torch.abs(safe_det_3x3(cell_bohr))
        self.register_buffer('volume', volume)

        # --- Tensor decomposition of the C6^ref block matrix ---
        # decomp returns eigenvalues lambda_r and eigenvectors V of shape (n_species*7, n_rank).
        # The rank r is chosen to meet the c6tol accuracy threshold.
        eigs, eigvecs = decomp(species_unique, c6tol, verbose, dtype=self.dtype)
        eigs = eigs.to(device=device, dtype=dtype)
        eigvecs = eigvecs.to(device=device, dtype=dtype)
        self.register_buffer('eigs', eigs)
        self.register_buffer('eigvecs', eigvecs)
        self.n_rank = eigvecs.shape[1]

        # Reshape eigenvectors to (n_species, 7, n_rank) for fast per-atom access.
        # v_q_reshaped[Z, p, r] = v_{r,Z}(theta^ref_{Z,p}).
        v_q_reshaped = eigvecs.view(self.n_species, 7, self.n_rank)
        self.register_buffer('v_q_reshaped', v_q_reshaped)

        # Covalent radii R^cov_Z (Pyykko et al.) for the species present, shape (n_species,)
        rcov = load_rcov()[species_unique].to(device=device, dtype=dtype)
        # Reference coordination numbers theta^ref_{Z,p}, shape (n_species, 7);
        # entries of -1 mark unused reference environments (masked in softmax)
        cnref = load_cnref()[species_unique, :].to(device=device, dtype=dtype)
        self.register_buffer('rcov', rcov)
        self.register_buffer('cnref', cnref)

        # Constants used in the CN sigmoid formula
        self.register_buffer('factor_cn', torch.tensor(4.0/3.0, dtype=dtype, device=device))
        self.register_buffer('logit_scale', torch.tensor(-4.0, dtype=dtype, device=device))

        # --- BJ damping parameters [s6, s8, a1, a2] ---
        # Default values are for the PBE functional
        if params is not None:
            params = torch.tensor(params, device=device, dtype=dtype)
        elif xcfunc == 'pbe':
            params = torch.tensor([1.0, 0.7875, 0.4289, 4.4407], device=device, dtype=dtype)
        else:
            raise NotImplementedError

        # D3Potential computes the analytical FT G_{X,Y}(k) of the BJ-damped potential
        self.potential = D3Potential(species_unique, params, device, method, order=interpolation_nodes, dtype=dtype)

        # For the D4 CN option, set up the CN pair potential and its k-space filter
        if self.cnfunc == 'd4':
            self.cn_potential = CNPotential(species_unique, device, method, order=interpolation_nodes, dtype=dtype)
            # Self-contribution for CN: f_{XX}(0) subtracted from the k-space CN sum
            self.register_buffer('cn_selfcont', self.cn_potential.selfcont)

        # Self-interaction term V_self: phi_{X,X}(r->0) per species, shape (n_species, 1)
        self.register_buffer('selfcont', self.potential.selfcont)

        if method == 'pme' or method == 'spme':
            # Determine mesh size to match the requested mesh_spacing
            self.ns_mesh = get_ns_mesh(cell_bohr, mesh_spacing * angstrom_to_bohr)
            if verbose:
                print('Using mesh size', self.ns_mesh)

            # KSpaceFilterD3 applies G_{X,Y}(k) to the FFT of the C6-weighted mesh density
            self.kspace_filter = KSpaceFilterD3(
                cell=cell_bohr,
                ns_mesh=self.ns_mesh,
                kernel=self.potential,
                fft_norm="backward",
                ifft_norm="forward"
            )

            if self.cnfunc == 'd4':
                # KSpaceFilterCN applies G^CN_{X,Y}(k) to compute the CN potential field
                self.kspace_filter_cn = KSpaceFilterCN(
                    cell=cell_bohr,
                    ns_mesh=self.ns_mesh,
                    kernel=self.cn_potential,
                    fft_norm="backward",
                    ifft_norm="forward"
                )

        if method == 'pme':
            self.mesh_interpolator = MeshInterpolatorD3(
                cell=cell_bohr,
                ns_mesh=self.ns_mesh,
                interpolation_nodes=interpolation_nodes,
                method="Lagrange"
            )

        if method == 'spme':
            self.mesh_interpolator = MeshInterpolatorD3(
                cell=cell_bohr,
                ns_mesh=self.ns_mesh,
                interpolation_nodes=interpolation_nodes,
                method="P3M"
            )

        elif method == 'ewald':
            # For direct Ewald, precompute all k-vectors within the spherical cutoff
            # and evaluate the Green's function G once (updated on cell change)
            basis_norms = torch.linalg.norm(cell_bohr, dim=1)
            ns_float = k_cutoff * basis_norms / (2 * math.pi)
            self.ns = torch.ceil(ns_float).long()

            kvectors = generate_kvectors_for_ewald(ns=self.ns, cell=cell_bohr).to(device=device, dtype=dtype)
            knorm = torch.linalg.norm(kvectors, dim=1)
            # G[X, Y, k]: Green's function at each k-point, shape (n_species, n_species, nk)
            G = self.potential.lr_from_k_sq(knorm)
            if self.cnfunc == 'd4':
                G_cn = self.cn_potential.lr_from_k_sq(knorm)
                self.register_buffer('G_cn', G_cn)

            self.register_buffer('kvectors', kvectors)
            self.register_buffer('knorm', knorm)
            self.register_buffer('G', G)

    def _update_cell(self, cell):
        """Update the simulation cell and refresh all cell-dependent buffers.

        Must be called at each MD step if the cell changes (NPT ensemble).
        Recomputes the cell volume, mesh interpolation weights, k-space filters,
        and (for Ewald) the k-vectors and Green's function values.

        Args:
            cell: (3, 3) tensor in Ångström.
        """
        self.cell = cell * self.angstrom_to_bohr
        self.volume = torch.abs(safe_det_3x3(self.cell))
        if self.method == 'pme' or self.method == 'spme':
            self.mesh_interpolator.update(self.cell)
            self.kspace_filter.update(self.cell)
            if self.cnfunc == 'd4':
                self.kspace_filter_cn.update(self.cell)

        if self.method == 'ewald':
            self.kvectors = generate_kvectors_for_ewald(ns=self.ns, cell=self.cell).to(dtype=self.dtype)
            self.knorm = torch.linalg.norm(self.kvectors, dim=1)
            self.G = self.potential.lr_from_k_sq(self.knorm)
            if self.cnfunc == 'd4':
                self.G_cn = self.cn_potential.lr_from_k_sq(self.knorm)


    @torch.jit.export
    def compute_cn_old(self, positions: torch.Tensor, edge_index: torch.Tensor,
                   shifts: torch.Tensor, recalc=False) -> torch.Tensor:
        """Compute coordination numbers using the standard DFT-D3 sigmoid formula.

        theta_i = sum_{j != i} [1 + exp(-16*(4*R^cov_{ij}/(3*r_{ij}) - 1))]^{-1}

        Note: this formulation has a non-zero asymptotic limit as r -> inf, so the
        result depends arbitrarily on the choice of cutoff radius. See the Appendix
        of the paper for a detailed discussion. Prefer `compute_cn_smooth` instead.

        Args:
            positions:   atomic positions in Bohr, shape (n_atoms, 3).
            edge_index:  neighbour list edge indices, shape (2, n_edges).
            shifts:      periodic image shift vectors in Bohr, shape (n_edges, 3).
            recalc:      if True, skip the cncorr affine correction.

        Returns:
            cn: coordination numbers, shape (n_atoms,).
        """
        positions = positions.to(dtype=self.dtype)
        n_atoms = positions.size(0)
        source, target = edge_index

        # Displacement vectors r_{ij} including periodic image shifts
        vec_ij = positions[target] - positions[source] + shifts
        r_ab = torch.linalg.norm(vec_ij, dim=-1)

        # Exclude self-pairs (r ~ 0)
        mask = r_ab > 1e-6
        source_m = source[mask]
        target_m = target[mask]
        r_ab_m = r_ab[mask]

        # R^cov_{ij} = R^cov_i + R^cov_j
        r_cov_sum = self.rcov[self.species[source_m]] + self.rcov[self.species[target_m]]

        inv_r = 1.0 / r_ab_m
        ratio = self.factor_cn * r_cov_sum  # = 4/3 * R^cov_{ij}
        arg_r = -16.0 * (ratio * inv_r - 1.0)

        # sigmoid: [1 + exp(-arg_r)]^{-1}
        exp_r = torch.exp(arg_r)
        edge_contributions = 1.0 / (1.0 + exp_r)

        cn = torch.zeros(n_atoms, device=self.device, dtype=self.dtype)
        cn.index_add_(0, source_m, edge_contributions)

        if self.cncorr is not None and not recalc:
            cn = self.cncorr[self.species, 1] * cn + self.cncorr[self.species, 0]

        return cn

    @torch.jit.export
    def compute_cn_smooth(self, positions: torch.Tensor, edge_index: torch.Tensor,
                shifts: torch.Tensor, recalc: bool = False) -> torch.Tensor:
        """Compute coordination numbers with the modified smooth-cutoff CN function.

        Replaces the standard DFT-D3 sigmoid with a variable-exponent version that
        decays strictly to zero at r = r_cut, eliminating the divergence problem
        discussed in the Appendix. The exponent t_{ij}(r) is:

            R^mid_{ij} = (R^cov_{ij} + r_cut) / 2
            t_{ij}(r) = 16 + (r - R^mid_{ij})^2 / (r_cut - R^mid_{ij})^2    if r > R^mid_{ij}
                       = 16                                                     otherwise

        so theta_{ij}(r) = sigmoid(t_{ij}(r) * (4*R^cov_{ij}/(3*r) - 1)) decays to
        zero as r -> r_cut with a smooth sigmoidal tail (no discontinuity).

        Args:
            positions:   atomic positions in Bohr, shape (n_atoms, 3).
            edge_index:  neighbour list edge indices, shape (2, n_edges).
            shifts:      periodic image shift vectors in Bohr, shape (n_edges, 3).
            recalc:      if True, skip the cncorr affine correction.

        Returns:
            cn: coordination numbers, shape (n_atoms,).
        """
        positions = positions.to(dtype=self.dtype)
        n_atoms = positions.size(0)
        source, target = edge_index

        vec_ij = positions[target] - positions[source] + shifts
        r_ab = torch.linalg.norm(vec_ij, dim=-1)

        r_ab_safe = torch.clamp(r_ab, min=1e-6)

        # R^cov_{ij} = R^cov_i + R^cov_j; R^mid_{ij} = (R^cov_{ij} + r_cut) / 2
        r_cov_sum = self.rcov[self.species[source]] + self.rcov[self.species[target]]
        r1 = 0.5 * r_cov_sum + 0.5 * self.r_cut  # = R^mid_{ij}

        # Variable exponent: k_cn(r) = 16 for r <= R^mid, grows quadratically beyond
        r_ab_coeff = torch.where(r_ab_safe > r1, r_ab_safe, r1)
        denom = (self.r_cut - r_ab_coeff).pow(2) + 1e-6
        k_cn = 16.0 + (r_ab_coeff - r1)**2 / denom

        inv_r = 1.0 / r_ab_safe
        ratio = self.factor_cn * r_cov_sum  # = 4/3 * R^cov_{ij}

        # sigmoid(-arg_r) = sigmoid(k_cn * (1 - 4*R^cov_{ij}/(3*r)))
        arg_r = -k_cn * (ratio * inv_r - 1.0)
        edge_contributions = torch.sigmoid(-arg_r)

        # Zero out self-pair contributions (r ~ 0)
        edge_contributions = torch.where(
            r_ab > 1e-6,
            edge_contributions,
            torch.tensor(0.0, dtype=self.dtype, device=self.device)
        )

        cn = torch.zeros(n_atoms, device=self.device, dtype=self.dtype)
        cn.index_add_(0, source, edge_contributions)

        if self.cncorr is not None and not recalc:
            cn = self.cncorr[self.species, 1] * cn + self.cncorr[self.species, 0]

        return cn

    @torch.jit.export
    def compute_cn_d4(self, positions: torch.Tensor, ex: torch.Tensor = None, nk = None) -> torch.Tensor:
        """Compute D4 coordination numbers entirely in reciprocal space.

        The D4 CN function theta_{ij}(r) = (delta_{ij}/2)*erfc(k0*(r - r0_{ij})/r0_{ij})
        is a convergent pairwise sum (decays exponentially at large r). Its Fourier
        transform is precomputed in CNPotential, so the CN at atom i is obtained by:

            CN_i = [sum_Y G^CN_{X_i,Y}(k) * rho_Y(k)] evaluated at r_i / Omega
                   - f_{X_i,X_i}(0)    (subtract the self-interaction term)

        For PME/SPME, this is done via: (1) spreading a one-hot species density onto
        the mesh, (2) convolving with the CN Green's function in k-space, (3) back-
        transforming and interpolating back to atom positions.
        For Ewald, it is done via explicit structure factors and matrix-vector products.

        No real-space neighbour list is needed for this CN variant.

        Args:
            positions: atomic positions in Bohr, shape (n_atoms, 3).
            ex:        trig exponentials [cos(k.r), sin(k.r)], shape (2, nk, n_atoms);
                       only needed for the Ewald path.
            nk:        number of k-vectors; only needed for the Ewald path.

        Returns:
            cn: coordination numbers, shape (n_atoms,).
        """
        n_atoms = positions.shape[0]

        if self.method in ('pme', 'spme'):
            # One-hot species density: cn_charges[i, X, 0] = 1 if species_i == X
            cn_charges = torch.zeros(n_atoms, self.n_species, 1, device=self.device, dtype=self.dtype)
            cn_charges[torch.arange(n_atoms, device=self.device), self.species, 0] = 1.0

            # Spread to mesh; squeeze rank dim since CN uses n_rank=1
            rho_mesh = self.mesh_interpolator.points_to_mesh(cn_charges, dtype=self.dtype).squeeze(1)
            # rho_mesh shape: (n_species, nx, ny, nz)

            # Convolve with CN Green's function in k-space, back-transform to real space
            phi_mesh = self.kspace_filter_cn.forward(rho_mesh).unsqueeze(1)
            # phi_mesh shape: (n_species, 1, nx, ny, nz)

            # Interpolate the potential field back to atom positions
            phi_atoms = self.mesh_interpolator.mesh_to_points(phi_mesh, dtype=self.dtype).squeeze(-1)
            # phi_atoms[i, X] = phi_X(r_i), shape (n_atoms, n_species)

        else:  # Ewald path
            # Structure factor per species: S_X(k) = sum_{i: Z_i=X} exp(ik.r_i)
            S = torch.mm(ex.reshape(2 * nk, n_atoms), self.one_hot_species).view(2, nk, self.n_species)
            # Apply Green's function: phi_X(k) = sum_Y G^CN_{XY}(k) * S_Y(k)
            F = torch.einsum('tkj,kij->tki', S, self.G_cn.permute(2, 0, 1))  # (2, nk, n_species)
            # Back-transform to atom positions via dot with exp(ik.r_i)
            phi_atoms = torch.einsum('tka,tki->ai', ex, F)  # (n_atoms, n_species)

        # Read off the potential at each atom for its own species,
        # divide by Omega, and subtract the diagonal self-interaction
        batch = torch.arange(n_atoms, device=self.device)
        cn_raw = phi_atoms[batch, self.species]
        cn = cn_raw / self.volume - self.cn_selfcont[self.species]
        return cn

    @torch.jit.export
    def compute_c6_weights(self, cn: torch.Tensor) -> torch.Tensor:
        """Compute the softmax interpolation weights L^p_i for each reference environment.

        For atom i with coordination number theta_i, the weight for reference
        environment p of species Z_i is:

            L^p_i(theta_i) = exp(-4 * (theta_i - theta^ref_{Z_i, p})^2)

        The normalized weights (softmax over logits = -4*(diff)^2) are used to
        interpolate the per-rank eigenvector components in `forward`.

        Args:
            cn: coordination numbers for all atoms, shape (n_atoms,).

        Returns:
            weights: softmax-normalized Gaussian weights over reference environments,
                     shape (n_atoms, 7). Entries for unused reference environments
                     (cnref == -1) are effectively zeroed by the -inf masking.
        """
        atom_refs = self.cnref[self.species]           # theta^ref_{Z_i, p}, shape (n_atoms, 7)
        diff = cn.unsqueeze(1) - atom_refs             # theta_i - theta^ref_{Z_i, p}
        logits = self.logit_scale * torch.square(diff) # = -4 * diff^2

        # Mask out unused reference environments (marked by -1 in cnref)
        mask = atom_refs == -1
        logits = logits.masked_fill(mask, float('-inf'))

        # softmax gives L^p_i / sum_q L^q_i
        return torch.softmax(logits, dim=1)

    def forward(self, positions: torch.Tensor,
                edge_index: torch.Tensor = None,
                shifts: torch.Tensor = None) -> torch.Tensor:
        """Compute the DFT-D3 dispersion energy for the current configuration.

        Pipeline:
          1. Compute coordination numbers theta_i (real-space or k-space).
          2. Compute interpolation weights L^p_i (softmax over Gaussian logits).
          3. Compute per-atom rank-r C6 coefficients C^6_{r,i}:
                C^6_{r,i} = sum_p v_{r,Z_i}(theta^ref_{Z_i,p}) * L^p_i / sum_q L^q_i
          4. Compute the self-interaction correction V_self.
          5. Compute the reciprocal-space energy sum via PME/SPME or Ewald.
          6. Return E_D3 = E_k / (-2*Omega) + V_self / 2  (in Hartree).

        Args:
            positions:   atomic positions in Ångström, shape (n_atoms, 3).
            edge_index:  neighbour list [source, target] indices, shape (2, n_edges).
                         Required for cnfunc='smooth_cut'; not used for cnfunc='d4'.
            shifts:      lattice-vector shift vectors in Ångström, shape (n_edges, 3).
                         Required for cnfunc='smooth_cut'; not used for cnfunc='d4'.

        Returns:
            energy: DFT-D3 dispersion energy in Hartree (scalar).
        """
        # Convert positions to Bohr for internal calculations
        positions = positions * self.angstrom_to_bohr
        if self.cnfunc == 'smooth_cut':
            shifts = shifts * self.angstrom_to_bohr

        n_atoms = positions.size(0)

        # --- Step 1: Coordination numbers ---
        if self.cnfunc == 'smooth_cut':
            # Real-space CN with smooth cutoff; requires neighbour list
            cn = self.compute_cn_smooth(positions, edge_index, shifts)
        if self.cnfunc == 'd4' and self.method in ('pme', 'spme'):
            # D4 CN via SPME mesh convolution; no neighbour list needed
            cn = self.compute_cn_d4(positions)
        if self.cnfunc == 'd4' and self.method == 'ewald':
            # D4 CN via explicit Ewald structure factors
            trig_args = torch.matmul(self.kvectors, positions.T)  # (nk, n_atoms)
            nk = trig_args.shape[0]
            ex = torch.stack([torch.cos(trig_args), torch.sin(trig_args)], dim=0)
            cn = self.compute_cn_d4(positions, ex, nk)

        # --- Step 2: Interpolation weights L^p_i (softmax) ---
        weights = self.compute_c6_weights(cn)

        # --- Step 3: Per-atom C6 decomposition coefficients ---
        # atom_v_qs[i, p, r] = v_{r,Z_i}(theta^ref_{Z_i, p})
        atom_v_qs = self.v_q_reshaped[self.species]
        # c6[i, r] = sum_p weights[i, p] * v_{r,Z_i}(theta^ref_{Z_i, p})
        c6 = torch.einsum('nk,nkr->nr', weights, atom_v_qs)

        # Build one-hot-encoded charge array: onehot[i, X, r] = c6[i, r] if species_i == X
        # This is the particle weight array that will be spread onto the mesh.
        # This is induced by the double-sum from the species-dependent damping function
        batch_idx = torch.arange(n_atoms, device=self.device)
        onehot = torch.zeros(n_atoms, self.n_species, self.n_rank,
                            device=c6.device,
                            dtype=self.dtype).index_put(
                                (batch_idx, self.species),
                                c6
                            )

        # --- Step 4: Self-interaction correction V_self ---
        # V_self = (1/2) * sum_r lambda_r * sum_{X} sum_{i in X} (C^6_{r,i})^2 * phi_{X,X}(0)
        c6_sq = torch.square(c6)
        selfcont_atoms = self.selfcont[self.species]  # phi_{X_i,X_i}(0), shape (n_atoms, 1)
        tmp = torch.sum(c6_sq * selfcont_atoms, dim=0)  # sum over atoms, shape (n_rank,)
        vself = torch.dot(self.eigs, tmp)               # dot with eigenvalues lambda_r

        # --- Step 5: Reciprocal-space energy ---
        if self.method == 'pme' or self.method == 'spme':
            # Compute B-spline interpolation weights for current atom positions
            self.mesh_interpolator.compute_weights(positions)
            # Spread C6 weights onto the mesh: rho_mesh[X, r, ix, iy, iz]
            rho_mesh = self.mesh_interpolator.points_to_mesh(onehot, dtype=self.dtype)
            # Apply Green's function in k-space; returns sum_k G_{XY}(k)*S^r_X*conj(S^r_Y)
            # per rank r, then dot with eigenvalues gives the k-space energy
            filtered_hat = self.kspace_filter.forward(rho_mesh)
            energy = torch.dot(self.eigs, filtered_hat)

        elif self.method == 'ewald':
            if self.cnfunc == 'smooth_cut':
                # Compute complex exponentials exp(ik.r_i) for the Ewald structure factors
                trig_args = torch.matmul(self.kvectors, positions.T)
                c = torch.cos(trig_args)
                s = torch.sin(trig_args)
                ex = torch.stack([c, s], dim=0)  # (2, nk, n_atoms)
            # sc[r, z, p, k]: structure factor components S^r_X(k) decomposed into cos/sin
            sc = torch.einsum('izr,pki->rzpk', onehot, ex)
            # Use sqrt(|lambda_r|) weighting to factor the bilinear energy expression
            sqrt_eigs = torch.sqrt(torch.abs(self.eigs))
            sc_weighted = sc * sqrt_eigs.view(-1, 1, 1, 1)
            # sum_{X,Y,k} G_{XY}(k) * S^r_X(k) * conj(S^r_Y(k)), summed over r
            convolved = torch.einsum('ijk,ripk,rjpk->jk', self.G, sc_weighted, sc_weighted)
            energy = torch.sum(convolved)

        # --- Step 6: Combine k-space and self terms ---
        # E_D3 = -(1/(2*Omega)) * E_k + (1/2) * V_self
        energy = energy / (-2 * self.volume) + vself / 2

        return energy
