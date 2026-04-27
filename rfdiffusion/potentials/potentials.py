import torch
import numpy as np 
from rfdiffusion.util import generate_Cbeta

class Potential:
    '''
        Interface class that defines the functions a potential must implement
    '''

    def compute(self, xyz):
        '''
            Given the current structure of the model prediction, return the current
            potential as a PyTorch tensor with a single entry

            Args:
                xyz (torch.tensor, size: [L,27,3]: The current coordinates of the sample
            
            Returns:
                potential (torch.tensor, size: [1]): A potential whose value will be MAXIMIZED
                                                     by taking a step along it's gradient
        '''
        raise NotImplementedError('Potential compute function was not overwritten')

class monomer_ROG(Potential):
    '''
        Radius of Gyration potential for encouraging monomer compactness

        Written by DJ and refactored into a class by NRB
    '''

    def __init__(self, weight=1, min_dist=15):

        self.weight   = weight
        self.min_dist = min_dist

    def compute(self, xyz):
        Ca = xyz[:,1] # [L,3]

        centroid = torch.mean(Ca, dim=0, keepdim=True) # [1,3]

        dgram = torch.cdist(Ca[None,...].contiguous(), centroid[None,...].contiguous(), p=2) # [1,L,1,3]

        dgram = torch.maximum(self.min_dist * torch.ones_like(dgram.squeeze(0)), dgram.squeeze(0)) # [L,1,3]

        rad_of_gyration = torch.sqrt( torch.sum(torch.square(dgram)) / Ca.shape[0] ) # [1]

        return -1 * self.weight * rad_of_gyration

class binder_ROG(Potential):
    '''
        Radius of Gyration potential for encouraging binder compactness

        Author: NRB
    '''

    def __init__(self, binderlen, weight=1, min_dist=15):

        self.binderlen = binderlen
        self.min_dist  = min_dist
        self.weight    = weight

    def compute(self, xyz):
        
        # Only look at binder residues
        Ca = xyz[:self.binderlen,1] # [Lb,3]

        centroid = torch.mean(Ca, dim=0, keepdim=True) # [1,3]

        # cdist needs a batch dimension - NRB
        dgram = torch.cdist(Ca[None,...].contiguous(), centroid[None,...].contiguous(), p=2) # [1,Lb,1,3]

        dgram = torch.maximum(self.min_dist * torch.ones_like(dgram.squeeze(0)), dgram.squeeze(0)) # [Lb,1,3]

        rad_of_gyration = torch.sqrt( torch.sum(torch.square(dgram)) / Ca.shape[0] ) # [1]

        return -1 * self.weight * rad_of_gyration


class dimer_ROG(Potential):
    '''
        Radius of Gyration potential for encouraging compactness of both monomers when designing dimers

        Author: PV
    '''

    def __init__(self, binderlen, weight=1, min_dist=15):

        self.binderlen = binderlen
        self.min_dist  = min_dist
        self.weight    = weight

    def compute(self, xyz):

        # Only look at monomer 1 residues
        Ca_m1 = xyz[:self.binderlen,1] # [Lb,3]
        
        # Only look at monomer 2 residues
        Ca_m2 = xyz[self.binderlen:,1] # [Lb,3]

        centroid_m1 = torch.mean(Ca_m1, dim=0, keepdim=True) # [1,3]
        centroid_m2 = torch.mean(Ca_m1, dim=0, keepdim=True) # [1,3]

        # cdist needs a batch dimension - NRB
        #This calculates RoG for Monomer 1
        dgram_m1 = torch.cdist(Ca_m1[None,...].contiguous(), centroid_m1[None,...].contiguous(), p=2) # [1,Lb,1,3]
        dgram_m1 = torch.maximum(self.min_dist * torch.ones_like(dgram_m1.squeeze(0)), dgram_m1.squeeze(0)) # [Lb,1,3]
        rad_of_gyration_m1 = torch.sqrt( torch.sum(torch.square(dgram_m1)) / Ca_m1.shape[0] ) # [1]

        # cdist needs a batch dimension - NRB
        #This calculates RoG for Monomer 2
        dgram_m2 = torch.cdist(Ca_m2[None,...].contiguous(), centroid_m2[None,...].contiguous(), p=2) # [1,Lb,1,3]
        dgram_m2 = torch.maximum(self.min_dist * torch.ones_like(dgram_m2.squeeze(0)), dgram_m2.squeeze(0)) # [Lb,1,3]
        rad_of_gyration_m2 = torch.sqrt( torch.sum(torch.square(dgram_m2)) / Ca_m2.shape[0] ) # [1]

        #Potential value is the average of both radii of gyration (is avg. the best way to do this?)
        return -1 * self.weight * (rad_of_gyration_m1 + rad_of_gyration_m2)/2

class binder_ncontacts(Potential):
    '''
        Differentiable way to maximise number of contacts within a protein
        
        Motivation is given here: https://www.plumed.org/doc-v2.7/user-doc/html/_c_o_o_r_d_i_n_a_t_i_o_n.html

    '''

    def __init__(self, binderlen, weight=1, r_0=8, d_0=4):

        self.binderlen = binderlen
        self.r_0       = r_0
        self.weight    = weight
        self.d_0       = d_0

    def compute(self, xyz):

        # Only look at binder Ca residues
        Ca = xyz[:self.binderlen,1] # [Lb,3]
        
        #cdist needs a batch dimension - NRB
        dgram = torch.cdist(Ca[None,...].contiguous(), Ca[None,...].contiguous(), p=2) # [1,Lb,Lb]
        divide_by_r_0 = (dgram - self.d_0) / self.r_0
        numerator = torch.pow(divide_by_r_0,6)
        denominator = torch.pow(divide_by_r_0,12)
        binder_ncontacts = (1 - numerator) / (1 - denominator)
        
        print("BINDER CONTACTS:", binder_ncontacts.sum())
        #Potential value is the average of both radii of gyration (is avg. the best way to do this?)
        return self.weight * binder_ncontacts.sum()

class interface_ncontacts(Potential):

    '''
        Differentiable way to maximise number of contacts between binder and target
        
        Motivation is given here: https://www.plumed.org/doc-v2.7/user-doc/html/_c_o_o_r_d_i_n_a_t_i_o_n.html

        Author: PV
    '''


    def __init__(self, binderlen, weight=1, r_0=8, d_0=6):

        self.binderlen = binderlen
        self.r_0       = r_0
        self.weight    = weight
        self.d_0       = d_0

    def compute(self, xyz):

        # Extract binder Ca residues
        Ca_b = xyz[:self.binderlen,1] # [Lb,3]

        # Extract target Ca residues
        Ca_t = xyz[self.binderlen:,1] # [Lt,3]

        #cdist needs a batch dimension - NRB
        dgram = torch.cdist(Ca_b[None,...].contiguous(), Ca_t[None,...].contiguous(), p=2) # [1,Lb,Lt]
        divide_by_r_0 = (dgram - self.d_0) / self.r_0
        numerator = torch.pow(divide_by_r_0,6)
        denominator = torch.pow(divide_by_r_0,12)
        interface_ncontacts = (1 - numerator) / (1 - denominator)
        #Potential is the sum of values in the tensor
        interface_ncontacts = interface_ncontacts.sum()

        print("INTERFACE CONTACTS:", interface_ncontacts.sum())

        return self.weight * interface_ncontacts


class monomer_contacts(Potential):
    '''
        Differentiable way to maximise number of contacts within a protein

        Motivation is given here: https://www.plumed.org/doc-v2.7/user-doc/html/_c_o_o_r_d_i_n_a_t_i_o_n.html
        Author: PV

        NOTE: This function sometimes produces NaN's -- added check in reverse diffusion for nan grads
    '''

    def __init__(self, weight=1, r_0=8, d_0=2, eps=1e-6):

        self.r_0       = r_0
        self.weight    = weight
        self.d_0       = d_0
        self.eps       = eps

    def compute(self, xyz):

        Ca = xyz[:,1] # [L,3]

        #cdist needs a batch dimension - NRB
        dgram = torch.cdist(Ca[None,...].contiguous(), Ca[None,...].contiguous(), p=2) # [1,Lb,Lb]
        divide_by_r_0 = (dgram - self.d_0) / self.r_0
        numerator = torch.pow(divide_by_r_0,6)
        denominator = torch.pow(divide_by_r_0,12)

        ncontacts = (1 - numerator) / ((1 - denominator))


        #Potential value is the average of both radii of gyration (is avg. the best way to do this?)
        return self.weight * ncontacts.sum()


class olig_contacts(Potential):
    """
    Applies PV's num contacts potential within/between chains in symmetric oligomers 

    Author: DJ 
    """

    def __init__(self, 
                 contact_matrix, 
                 weight_intra=1, 
                 weight_inter=1,
                 r_0=8, d_0=2):
        """
        Parameters:
            chain_lengths (list, required): List of chain lengths, length is (Nchains)

            contact_matrix (torch.tensor/np.array, required): 
                square matrix of shape (Nchains,Nchains) whose (i,j) enry represents 
                attractive (1), repulsive (-1), or non-existent (0) contact potentials 
                between chains in the complex

            weight (int/float, optional): Scaling/weighting factor
        """
        self.contact_matrix = contact_matrix
        self.weight_intra = weight_intra 
        self.weight_inter = weight_inter 
        self.r_0 = r_0
        self.d_0 = d_0

        # check contact matrix only contains valid entries 
        assert all([i in [-1,0,1] for i in contact_matrix.flatten()]), 'Contact matrix must contain only 0, 1, or -1 in entries'
        # assert the matrix is square and symmetric 
        shape = contact_matrix.shape 
        assert len(shape) == 2 
        assert shape[0] == shape[1]
        for i in range(shape[0]):
            for j in range(shape[1]):
                assert contact_matrix[i,j] == contact_matrix[j,i]
        self.nchain=shape[0]

         
    def _get_idx(self,i,L):
        """
        Returns the zero-indexed indices of the residues in chain i
        """
        assert L%self.nchain == 0
        Lchain = L//self.nchain
        return i*Lchain + torch.arange(Lchain)


    def compute(self, xyz):
        """
        Iterate through the contact matrix, compute contact potentials between chains that need it,
        and negate contacts for any 
        """
        L = xyz.shape[0]

        all_contacts = 0
        start = 0
        for i in range(self.nchain):
            for j in range(self.nchain):
                # only compute for upper triangle, disregard zeros in contact matrix 
                if (i <= j) and (self.contact_matrix[i,j] != 0):

                    # get the indices for these two chains 
                    idx_i = self._get_idx(i,L)
                    idx_j = self._get_idx(j,L)

                    Ca_i = xyz[idx_i,1]  # slice out crds for this chain 
                    Ca_j = xyz[idx_j,1]  # slice out crds for that chain 
                    dgram           = torch.cdist(Ca_i[None,...].contiguous(), Ca_j[None,...].contiguous(), p=2) # [1,Lb,Lb]

                    divide_by_r_0   = (dgram - self.d_0) / self.r_0
                    numerator       = torch.pow(divide_by_r_0,6)
                    denominator     = torch.pow(divide_by_r_0,12)
                    ncontacts       = (1 - numerator) / (1 - denominator)

                    # weight, don't double count intra 
                    scalar = (i==j)*self.weight_intra/2 + (i!=j)*self.weight_inter

                    #                 contacts              attr/repuls          relative weights 
                    all_contacts += ncontacts.sum() * self.contact_matrix[i,j] * scalar 

        return all_contacts 
                    
def get_damped_lj(r_min, r_lin,p1=6,p2=12):
    
    y_at_r_lin = lj(r_lin, r_min, p1, p2)
    ydot_at_r_lin = lj_grad(r_lin, r_min,p1,p2)
    
    def inner(dgram):
        return (dgram < r_lin) * (ydot_at_r_lin * (dgram - r_lin) + y_at_r_lin) + (dgram >= r_lin) * lj(dgram, r_min, p1, p2)
    return inner

def lj(dgram, r_min,p1=6, p2=12):
    return 4 * ((r_min / (2**(1/p1) * dgram))**p2 - (r_min / (2**(1/p1) * dgram))**p1)

def lj_grad(dgram, r_min,p1=6,p2=12):
    return -p2 * r_min**p1*(r_min**p1-dgram**p1) / (dgram**(p2+1))

def mask_expand(mask, n=1):
    mask_out = mask.clone()
    assert mask.ndim == 1
    for i in torch.where(mask)[0]:
        for j in range(i-n, i+n+1):
            if j >= 0 and j < len(mask):
                mask_out[j] = True
    return mask_out

def contact_energy(dgram, d_0, r_0):
    divide_by_r_0 = (dgram - d_0) / r_0
    numerator = torch.pow(divide_by_r_0,6)
    denominator = torch.pow(divide_by_r_0,12)
    
    ncontacts = (1 - numerator) / ((1 - denominator)).float()
    return - ncontacts

def poly_repulse(dgram, r, slope, p=1):
    a = slope / (p * r**(p-1))

    return (dgram < r) * a * torch.abs(r - dgram)**p * slope

#def only_top_n(dgram


class substrate_contacts(Potential):
    '''
    Implicitly models a ligand with an attractive-repulsive potential.
    '''

    def __init__(self, weight=1, r_0=8, d_0=2, s=1, eps=1e-6, rep_r_0=5, rep_s=2, rep_r_min=1):

        self.r_0       = r_0
        self.weight    = weight
        self.d_0       = d_0
        self.eps       = eps
        
        # motif frame coordinates
        # NOTE: these probably need to be set after sample_init() call, because the motif sequence position in design must be known
        self.motif_frame = None # [4,3] xyz coordinates from 4 atoms of input motif
        self.motif_mapping = None # list of tuples giving positions of above atoms in design [(resi, atom_idx)]
        self.motif_substrate_atoms = None # xyz coordinates of substrate from input motif
        r_min = 2
        self.energies = []
        self.energies.append(lambda dgram: s * contact_energy(torch.min(dgram, dim=-1)[0], d_0, r_0))
        if rep_r_min:
            self.energies.append(lambda dgram: poly_repulse(torch.min(dgram, dim=-1)[0], rep_r_0, rep_s, p=1.5))
        else:
            self.energies.append(lambda dgram: poly_repulse(dgram, rep_r_0, rep_s, p=1.5))


    def compute(self, xyz):
        
        # First, get random set of atoms
        # This operates on self.xyz_motif, which is assigned to this class in the model runner (for horrible plumbing reasons)
        self._grab_motif_residues(self.xyz_motif)
        
        # for checking affine transformation is corect
        first_distance = torch.sqrt(torch.sqrt(torch.sum(torch.square(self.motif_substrate_atoms[0] - self.motif_frame[0]), dim=-1))) 

        # grab the coordinates of the corresponding atoms in the new frame using mapping
        res = torch.tensor([k[0] for k in self.motif_mapping])
        atoms = torch.tensor([k[1] for k in self.motif_mapping])
        new_frame = xyz[self.diffusion_mask][res,atoms,:]
        # calculate affine transformation matrix and translation vector b/w new frame and motif frame
        A, t = self._recover_affine(self.motif_frame, new_frame)
        # apply affine transformation to substrate atoms
        substrate_atoms = torch.mm(A, self.motif_substrate_atoms.transpose(0,1)).transpose(0,1) + t
        second_distance = torch.sqrt(torch.sqrt(torch.sum(torch.square(new_frame[0] - substrate_atoms[0]), dim=-1)))
        assert abs(first_distance - second_distance) < 0.01, "Alignment seems to be bad" 
        diffusion_mask = mask_expand(self.diffusion_mask, 1)
        Ca = xyz[~diffusion_mask, 1]

        #cdist needs a batch dimension - NRB
        dgram = torch.cdist(Ca[None,...].contiguous(), substrate_atoms.float()[None], p=2)[0] # [Lb,Lb]

        all_energies = []
        for i, energy_fn in enumerate(self.energies):
            energy = energy_fn(dgram)
            all_energies.append(energy.sum())
        return - self.weight * sum(all_energies)

        #Potential value is the average of both radii of gyration (is avg. the best way to do this?)
        return self.weight * ncontacts.sum()

    def _recover_affine(self,frame1, frame2):
        """
        Uses Simplex Affine Matrix (SAM) formula to recover affine transform between two sets of 4 xyz coordinates
        See: https://www.researchgate.net/publication/332410209_Beginner%27s_guide_to_mapping_simplexes_affinely

        Args: 
        frame1 - 4 coordinates from starting frame [4,3]
        frame2 - 4 coordinates from ending frame [4,3]
        
        Outputs:
        A - affine transformation matrix from frame1->frame2
        t - affine translation vector from frame1->frame2
        """

        l = len(frame1)
        # construct SAM denominator matrix
        B = torch.vstack([frame1.T, torch.ones(l)])
        D = 1.0 / torch.linalg.det(B) # SAM denominator

        M = torch.zeros((3,4), dtype=torch.float64)
        for i, R in enumerate(frame2.T):
            for j in range(l):
                num = torch.vstack([R, B])
                # make SAM numerator matrix
                num = torch.cat((num[:j+1],num[j+2:])) # make numerator matrix
                # calculate SAM entry
                M[i][j] = (-1)**j * D * torch.linalg.det(num)

        A, t = torch.hsplit(M, [l-1])
        t = t.transpose(0,1)
        return A, t

    def _grab_motif_residues(self, xyz) -> None:
        """
        Grabs 4 atoms in the motif.
        Currently random subset of Ca atoms if the motif is >= 4 residues, or else 4 random atoms from a single residue
        """
        idx = torch.arange(self.diffusion_mask.shape[0])
        idx = idx[self.diffusion_mask].float()
        if torch.sum(self.diffusion_mask) >= 4:
            rand_idx = torch.multinomial(idx, 4).long()
            # get Ca atoms
            self.motif_frame = xyz[rand_idx, 1]
            self.motif_mapping = [(i,1) for i in rand_idx]
        else:
            rand_idx = torch.multinomial(idx, 1).long()
            self.motif_frame = xyz[rand_idx[0],:4]
            self.motif_mapping = [(rand_idx, i) for i in range(4)]


class motif_rigid(Potential):
    '''
    Penalize changes in a motif's internal geometry while allowing free rigid-body motion.

    Motifs are specified using input-PDB residue IDs (e.g. 'A10-25/A30-31'). This class
    is "contig-aware" in the sense that it resolves those residue IDs through the active
    `ContigMap` (attached at runtime as `self.contig_map`).

    Required runtime attributes (set by the model runner):
      - contig_map: ContigMap
      - xyz_ref: torch.Tensor [L,27,3] reference coordinates (pre-noise template frame)

    Notes:
      - This potential is maximized, so we return the negative of the distance-matrix MSE.
      - Uses a single atom per residue (default: CA) for the internal distance matrix.
    '''

    _ATOM_NAME_TO_IDX = {
        'N': 0,
        'CA': 1,
        'C': 2,
        'O': 3,
        'CB': 4,
    }

    def __init__(
        self,
        weight=1.0,
        k=1.0,
        motif=None,
        motif1=None,
        motif2=None,
        atom='CA',
        eps=1e-6,
    ):
        self.weight = float(weight)
        self.k = float(k)
        self.motif_spec = motif
        self.motif1_spec = motif1
        self.motif2_spec = motif2
        self.atom = str(atom).upper()
        self.eps = float(eps)

        self._motif_idx_cache = None  # dict[str, torch.LongTensor]
        self._ref_dmat_cache = None   # dict[str, torch.Tensor]
        self._cache_device = None
        self._cache_dtype = None

    def _parse_residue_spec(self, spec):
        if spec is None:
            return []
        spec = str(spec).strip()
        if not spec:
            return []

        res = []
        parts = [p for p in spec.split('/') if p]
        for part in parts:
            part = part.strip()
            if len(part) < 2 or not part[0].isalpha():
                raise ValueError(f"Invalid residue spec '{part}'. Expected like A10-25")
            chain = part[0]
            rng = part[1:]
            if '-' in rng:
                start_s, end_s = rng.split('-', 1)
                start = int(start_s)
                end = int(end_s)
                if end < start:
                    raise ValueError(f"Invalid residue range '{part}'")
                res.extend([(chain, i) for i in range(start, end + 1)])
            else:
                res.append((chain, int(rng)))
        return res

    def _resolve_indices(self, residue_ids):
        # contig_map.ref is a list like [('A',10), ..., ('_', '_'), ...]
        ref = getattr(self.contig_map, 'ref', None)
        if ref is None:
            raise ValueError('motif_rigid requires `contig_map.ref` at runtime')

        residue_set = set(residue_ids)
        idx = [i for i, r in enumerate(ref) if r in residue_set]
        if not idx:
            raise ValueError(
                f"motif_rigid: none of the requested residues were found in contig_map.ref. "
                f"Requested: {sorted(residue_set)[:5]}{'...' if len(residue_set) > 5 else ''}"
            )
        # ensure stable order and uniqueness
        idx = sorted(set(idx))
        return torch.tensor(idx, dtype=torch.long)

    def _get_groups(self):
        groups = {}
        if self.motif_spec is not None:
            groups['motif'] = self.motif_spec
        if self.motif1_spec is not None:
            groups['motif1'] = self.motif1_spec
        if self.motif2_spec is not None:
            groups['motif2'] = self.motif2_spec
        if not groups:
            raise ValueError('motif_rigid requires `motif`, `motif1`, and/or `motif2` to be set')
        return groups

    def _ensure_cache(self, xyz):
        if not hasattr(self, 'contig_map') or self.contig_map is None:
            raise ValueError('motif_rigid requires `contig_map` to be attached at runtime')
        if not hasattr(self, 'xyz_ref') or self.xyz_ref is None:
            raise ValueError('motif_rigid requires `xyz_ref` to be attached at runtime')

        device = xyz.device
        dtype = xyz.dtype

        if (
            self._motif_idx_cache is not None
            and self._ref_dmat_cache is not None
            and self._cache_device == device
            and self._cache_dtype == dtype
        ):
            return

        atom_names = [a.strip().upper() for a in self.atom.replace('_', ',').split(',') if a.strip()]
        atom_idx = []
        for a in atom_names:
            idx_val = self._ATOM_NAME_TO_IDX.get(a, None)
            if idx_val is None:
                raise ValueError(f"motif_rigid: unsupported atom '{a}'. Use one of {sorted(self._ATOM_NAME_TO_IDX)}")
            atom_idx.append(idx_val)
        self._motif_atom_idx = atom_idx

        xyz_ref = self.xyz_ref
        if not torch.is_tensor(xyz_ref):
            xyz_ref = torch.as_tensor(xyz_ref)
        xyz_ref = xyz_ref.to(device=device, dtype=dtype)

        idx_cache = {}
        dmat_cache = {}
        for name, spec in self._get_groups().items():
            residue_ids = self._parse_residue_spec(spec)
            idx = self._resolve_indices(residue_ids).to(device=device)
            if idx.numel() < 2 and len(atom_idx) < 2:
                raise ValueError(f"motif_rigid: group '{name}' must include at least 2 residues or 2 atoms")

            ref_coords = xyz_ref[idx][:, atom_idx, :].reshape(-1, 3).contiguous()
            if torch.isnan(ref_coords).any():
                raise ValueError(f"motif_rigid: reference coords contain NaNs for group '{name}'")

            dref = torch.cdist(ref_coords[None, ...], ref_coords[None, ...], p=2)[0]
            idx_cache[name] = idx
            dmat_cache[name] = dref

        self._motif_idx_cache = idx_cache
        self._ref_dmat_cache = dmat_cache
        self._cache_device = device
        self._cache_dtype = dtype

    def compute(self, xyz):
        self._ensure_cache(xyz)

        atom_idx = self._motif_atom_idx
        # total_mse = xyz.new_tensor(0.0)
        pseudo_huber_loss = xyz.new_tensor(0.0)

        for name, idx in self._motif_idx_cache.items():
            coords = xyz[idx][:, atom_idx, :].reshape(-1, 3).contiguous()
            dcur = torch.cdist(coords[None, ...], coords[None, ...], p=2)[0]
            dref = self._ref_dmat_cache[name]
            # total_mse = total_mse + torch.mean((dcur - dref) ** 2)
            pseudo_huber_loss = pseudo_huber_loss + torch.mean(torch.sqrt(((dcur - dref)/self.k) ** 2 + 1) - 1)

        return -self.weight * self.k ** 2 * pseudo_huber_loss


class motif_distance(Potential):
    """A potential that encourages two motifs to stay a set distance apart.

    This is intended for *unfrozen* motifs (i.e. positions where `diffusion_mask` is False)
    and operates on motif centroids (CA by default), so relative rigid-body orientation is
    unconstrained.

    Motifs can be specified either as:
      - explicit index mappings: `motif1_mapping`, `motif2_mapping` (0-indexed positions in the
        full design length), OR
      - contig/PDB residue specs: `motif1`, `motif2` like 'A10-25/A30-31'. These require
        `contig_map` to be attached at runtime.
    """

    _ATOM_NAME_TO_IDX = {
        # RFdiffusion atom14/27 conventions: CA is index 1
        'CA': 1,
    }

    def __init__(
        self,
        weight=1,
        target_distance=10,
        motif1=None,
        motif2=None,
        motif1_mapping=None,
        motif2_mapping=None,
        atom='CA',
        loss='l2',
    ):
        self.weight = weight
        self.target_distance = target_distance

        self.motif1 = motif1
        self.motif2 = motif2
        self.motif1_mapping = motif1_mapping
        self.motif2_mapping = motif2_mapping

        self.atom = atom
        self.loss = loss

        # caches (populated on first call)
        self._idx1_cache = None
        self._idx2_cache = None
        self._cache_device = None

    def _parse_residue_spec(self, spec):
        if spec is None:
            return []
        spec = str(spec).strip()
        if not spec:
            return []

        parts = [p.strip() for p in spec.split('/') if p.strip()]
        residue_ids = []
        for part in parts:
            # Accept the common format used throughout RFdiffusion configs:
            #   A25-109   (chain letter specified once)
            #   A10-25/A30-31
            if len(part) < 2 or not part[0].isalpha():
                raise ValueError(f"motif_distance: invalid residue spec '{part}'. Expected like A10-25")

            chain = part[0]
            rng = part[1:]
            if '-' in rng:
                start_s, end_s = rng.split('-', 1)
                start = int(start_s)
                end = int(end_s)
                if end < start:
                    raise ValueError(f"motif_distance: invalid residue range '{part}'")
                residue_ids.extend([(chain, r) for r in range(start, end + 1)])
            else:
                residue_ids.append((chain, int(rng)))

        return residue_ids

    def _resolve_indices(self, residue_ids):
        if not hasattr(self, 'contig_map') or self.contig_map is None:
            raise ValueError('motif_distance: contig residue specs require `contig_map` to be attached at runtime')

        ref = self.contig_map.ref
        ref_to_idx = {res: i for i, res in enumerate(ref) if res != ("_", "_")}

        idx = []
        for res in residue_ids:
            if res not in ref_to_idx:
                raise ValueError(f"motif_distance: residue {res} not found in contig_map.ref")
            idx.append(ref_to_idx[res])

        return torch.as_tensor(idx, dtype=torch.long)

    def _ensure_idx_cache(self, xyz):
        device = xyz.device
        if self._idx1_cache is not None and self._idx2_cache is not None and self._cache_device == device:
            return

        # Resolve motif1 indices
        if self.motif1_mapping is not None:
            idx1 = torch.as_tensor(self.motif1_mapping, dtype=torch.long)
        elif self.motif1 is not None:
            idx1 = self._resolve_indices(self._parse_residue_spec(self.motif1))
        else:
            raise ValueError('motif_distance: provide either motif1_mapping or motif1')

        # Resolve motif2 indices
        if self.motif2_mapping is not None:
            idx2 = torch.as_tensor(self.motif2_mapping, dtype=torch.long)
        elif self.motif2 is not None:
            idx2 = self._resolve_indices(self._parse_residue_spec(self.motif2))
        else:
            raise ValueError('motif_distance: provide either motif2_mapping or motif2')

        if idx1.numel() == 0 or idx2.numel() == 0:
            raise ValueError('motif_distance: motifs must each include at least one residue')

        self._idx1_cache = idx1.to(device=device)
        self._idx2_cache = idx2.to(device=device)
        self._cache_device = device

    def _coords_for_idx(self, xyz, idx, atom_idx):
        coords = xyz[idx, atom_idx, :]

        # Prefer unfrozen residues if a diffusion mask is attached (diffusion_mask==True => frozen)
        if hasattr(self, 'diffusion_mask') and self.diffusion_mask is not None:
            dm = torch.as_tensor(self.diffusion_mask, device=coords.device, dtype=torch.bool)
            try:
                unfrozen = idx[~dm[idx]]
                if unfrozen.numel() > 0:
                    coords = xyz[unfrozen, atom_idx, :]
            except Exception:
                # If mask/indexing shapes don't align, just fall back to using all idx
                pass

        # Drop any NaN coordinates (can occur for missing atoms)
        keep = ~torch.isnan(coords).any(dim=-1)
        coords = coords[keep]

        if coords.numel() == 0:
            return None

        return coords

    def compute(self, xyz):
        self._ensure_idx_cache(xyz)

        atom_idx = self._ATOM_NAME_TO_IDX.get(self.atom, None)
        if atom_idx is None:
            raise ValueError(f"motif_distance: unsupported atom '{self.atom}'. Only 'CA' is supported")

        coords1 = self._coords_for_idx(xyz, self._idx1_cache, atom_idx)
        coords2 = self._coords_for_idx(xyz, self._idx2_cache, atom_idx)

        # If coordinates are undefined at this timestep (all-NaN), don't crash the run.
        # Returning zero means "no guidance" for this potential on this step.
        if coords1 is None or coords2 is None:
            return xyz.new_tensor(0.0)

        c1 = torch.mean(coords1, dim=0)
        c2 = torch.mean(coords2, dim=0)
        distance = torch.sqrt(torch.sum((c1 - c2) ** 2))

        delta = distance - xyz.new_tensor(self.target_distance)
        if self.loss == 'l1':
            err = torch.abs(delta)
        elif self.loss == 'l2':
            err = delta ** 2
        else:
            raise ValueError("motif_distance: loss must be 'l1' or 'l2'")

        return -self.weight * err


# Dictionary of types of potentials indexed by name of potential. Used by PotentialManager.
# If you implement a new potential you must add it to this dictionary for it to be used by
# the PotentialManager
implemented_potentials = { 'monomer_ROG':          monomer_ROG,
                           'binder_ROG':           binder_ROG,
                           'dimer_ROG':            dimer_ROG,
                           'binder_ncontacts':     binder_ncontacts,
                           'interface_ncontacts':  interface_ncontacts,
                           'monomer_contacts':     monomer_contacts,
                           'olig_contacts':        olig_contacts,
                           'substrate_contacts':    substrate_contacts,
                           'motif_rigid':          motif_rigid,
                           'motif_distance':       motif_distance}

require_binderlen      = { 'binder_ROG',
                           'binder_distance_ReLU',
                           'binder_any_ReLU',
                           'dimer_ROG',
                           'binder_ncontacts',
                           'interface_ncontacts'}

