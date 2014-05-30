import numpy as np
from sfepy.base.goptions import goptions
goptions['verbose'] = False
from sfepy.base.base import IndexedStruct
from sfepy.discrete.fem import Field
try:
    from sfepy.discrete.fem import FEDomain as Domain
except ImportError:
    from sfepy.discrete.fem import Domain
from sfepy.discrete import (FieldVariable, Material, Integral, Function,
                            Equation, Equations, Problem)
from sfepy.terms import Term
from sfepy.discrete.conditions import Conditions, EssentialBC, PeriodicBC
from sfepy.solvers.ls import ScipyDirect
from sfepy.solvers.nls import Newton
# from sfepy.postprocess import Viewer
import sfepy.discrete.fem.periodic as per
from sfepy.discrete import Functions
from sfepy.mesh.mesh_generators import gen_block_mesh


class ElasticFEModel(object):
    """

    Use SfePy to solve a linear strain problem in 2D with a varying
    microstructure on a rectangular grid. The rectangle is held at the
    left edge and displaced by 1 on the right edge. Periodic boundary
    conditions are applied to the upper and lower plains.

    The microstructurs is of shape (Nsample, Nx, Ny, Nproperty) where
    Nproperty is 2 for the elastic modulus and Poisson's ratio.

    >>> X = np.ones((1, 5, 5, 2))

    >>> model = ElasticFEModel(dx=0.2)
    >>> y = model.predict(X)

    y is the strain with compontents as follows

    >>> exx = y[..., 0]
    >>> eyy = y[..., 1]
    >>> exy = y[..., 2]

    Since there is no contrast in the microstructe the strain is only
    in the x-direction and has a uniform value of 1 since the
    displacement is always 1 and the size of the domain is 1.

    >>> assert np.allclose(exx, 1)
    >>> assert np.allclose(eyy, 0)
    >>> assert np.allclose(exy, 0)
    
    """
    def __init__(self, dx=1.):
        """
        Args:
          dx: the grid spacing

        """
        self.dx = dx

    def convert_properties(self, X):
        """
        Convert from elastic modulus and Poisson's ratio to the Lame
        parameter and shear modulus

        Args:
           X: array of material properties, X[...,0] is the elastic
              modulus and X[...,1] is the Poisson's ratio

        Returns:
          returns a new array with the Lame parameter and the shear modulus
        """
        E = X[...,0]
        nu = X[...,1]
        lame = E * nu / (1 + nu) / (1 - 2 * nu)
        K = E / 3 / (1 - 2 * nu)
        mu = K - lame
        return np.concatenate((lame[...,None], mu[...,None]), axis=-1)

    def cell_to_node(self, X):
        """
        Maps the cell values given by X to node values. The values are
        not interpolated, but mapped out from the center cell.

        Args:
          X: cell values with shape (Nsample, Nx, Ny, Nproperty)

        Returns:
          The nodal values with shape (Nsample, Nx + 1, Ny + 1, Nproperty)
          
        >>> X = np.arange(15).reshape((1, 5, 3))
        >>> X_ = np.array([[[  0.,  1.,  1.,  2.],
        ...                 [  3,   4,   4,   5.],
        ...                 [  6,   7,   7,   8.],
        ...                 [  6,   7,   7,   8.],
        ...                 [  9,  10,  10,  11.],
        ...                 [ 12,  13,  13,  14.]]])
        >>> model = ElasticFEModel(1)
        >>> assert np.all(np.equal(X_, model.cell_to_node(X)))
        
        """
        Nsample, Nx, Ny = X.shape[:3]
        X_ = np.zeros((Nsample, Nx + 1, Ny + 1) + X.shape[3:])
        Mx = (Nx + 1) / 2
        My = (Ny + 1) / 2
        X_[:, Mx:, My:] = X[:, Mx - 1:, My - 1:]
        X_[:, :Mx, My:] = X[:, :Mx, My - 1:]
        X_[:, Mx:, :My] = X[:, Mx - 1:, :My]
        X_[:, :Mx, :My] = X[:, :Mx, :My]
        return X_
            
    def predict(self, X):
        """
        Predict the displacement field give an initial microstructure
        and a strain in the x direction.

        Args:
          X: microstructure with shape (Nsample, Nx, Ny, Nproperty)
             with len(Nproperty) = 2. X[..., 0] represents the elastic
             modulus and X[..., 1] is the Poisson's ratio

        Returns:
          the strain field over each cell
          
        """
        Nsample, Nx, Ny, Nproperty = X.shape
        if (Nx % 2 == 0) or (Ny % 2 == 0) or (Nproperty != 2):
            raise RuntimeError, 'the shape of X is incorrect'
        
        Xnode = self.cell_to_node(X)
        
        X_ = self.convert_properties(Xnode)

        y_disp = np.array([self.solve(x) for x in X_])

        y_strain = self.get_strain(y_disp)
        
        return y_strain

    def get_strain(self, y_disp):
        """
        Calculate the strain field from the displacement field.

        Args:
        y_disp: the displacement field with shape (Nsample, Nx, Ny, 2)
                where the last index represents the x and y
                displacement

        Returns:
          y_strain: the strain field with shape (Nsample, Nx, Ny, 3)
          where the last index represents the e_xx, e_yy, e_xy strain
          fields, respectively

        """
        
        xx = (y_disp[:, 1:, :, 0] - y_disp[:, :-1, :, 0]) / self.dx
        exx = (xx[:, :, 1:] + xx[:, :, :-1]) / 2
        
        yy = (y_disp[:, :, 1:, 1] - y_disp[:, :, :-1, 1]) / self.dx
        eyy = (yy[:, 1:] + yy[:, :-1]) / 2

        xy_ = (y_disp[:, 1:, :, 1] - y_disp[:, :-1, :, 1]) / self.dx
        xy = (xy_[:, :, 1:] + xy_[:, :, :-1]) / 2.
        yx_ = (y_disp[:, :, 1:, 0] - y_disp[:, :, :-1, 0]) / self.dx
        yx = (yx_[:, 1:] + yx_[:, :-1]) / 2
        exy = (xy + yx) / 2

        return np.concatenate((exx[..., None], eyy[..., None], exy[..., None]), axis=-1)
        
    
    def get_material(self, property_array, domain):
        """
        Creates an SfePy material from the material property fields

        Args:
          property_array: array of the properties with shape (Nx, Ny, 2)

        Returns:
          an SfePy material
          
        """
        minx, maxx = domain.get_mesh_bounding_box()[:, 0]
        miny, maxy = domain.get_mesh_bounding_box()[:, 1]

        def material_func_(ts, coors, mode=None, **kwargs):
            if mode != 'qp':
                return
            else:
                x, y = coors[:, 0], coors[:, 1]
                i_out = np.empty_like(x, dtype=np.int64)
                j_out = np.empty_like(y, dtype=np.int64)
                i = np.floor((x + self.dx / 2 - minx) / self.dx, i_out)
                j = np.floor((y + self.dx / 2 - miny) / self.dx, j_out)
                
                property_array_ = property_array[i, j]
                lam = property_array_[..., 0]
                mu = property_array_[..., 1]
                lam = np.ascontiguousarray(lam.reshape((lam.shape[0], 1, 1)))
                mu = np.ascontiguousarray(mu.reshape((mu.shape[0], 1, 1)))
                return {'lam' : lam, 'mu' : mu}

        material_func = Function('material_func', material_func_)
        return Material('m', function=material_func)

    def subdomain_func(self, x=(), y=()):
        """
        Creates a function to mask subdomains in Sfepy.

        Args:
          x: tuple of lines or points to be masked in the x-plane
          y: tuple of lines or points to be masked in the y-plane

        Returns:
          array of masked location indices
          
        """
        eps = 1e-3 * self.dx

        def func(coords, domain=None):
            flag_x = len(x) == 0
            flag_y = len(y) == 0

            for x_ in x:
                flag = (coords[:, 0] < (x_ + eps)) & (coords[:, 0] > (x_ - eps))
                flag_x = flag_x | flag

            for y_ in y:
                flag = (coords[:, 1] < (y_ + eps)) & (coords[:, 1] > (y_ - eps))
                flag_y = flag_y | flag

            return np.where(flag_x & flag_y)[0]

        return func

    def get_periodicBCs(self, domain):
        """
        Creates periodic boundary conditions with the top and bottom y-planes.

        Args:
          domain: an Sfepy domain

        Returns:
          a tuple of Sfepy boundary condition and associated matching functions

        """
        miny, maxy = domain.get_mesh_bounding_box()[:, 1]
        yup_ = self.subdomain_func(y=(maxy,))
        ydown_ = self.subdomain_func(y=(miny,))
        yup = Function('yup', yup_)
        ydown = Function('ydown', ydown_)
        region_up = domain.create_region('region_up',
                                         'vertices by yup',
                                         'facet',
                                         functions=Functions([yup]))
        region_down = domain.create_region('region_down',
                                           'vertices by ydown',
                                           'facet',
                                           functions=Functions([ydown]))
        match_x_line = Function('match_x_line', per.match_x_line)
        periodic_y = PeriodicBC('periodic_y', [region_up, region_down], {'u.all' : 'u.all'}, match='match_x_line')
        return Conditions([periodic_y]), Functions([match_x_line])

    def get_displacementBCs(self, domain):
        """
        Fix the left plane in x, displace the right plane by 1 and fix
        the y-direction with the top and bottom points on the left x
        plane.

        Args:
          domain: an Sfepy domain

        Returns:
          the Sfepy boundary conditions

        """
        minx, maxx = domain.get_mesh_bounding_box()[:, 0]
        miny, maxy = domain.get_mesh_bounding_box()[:, 1]
        xright_ = self.subdomain_func(x=(maxx,))
        xleft_ = self.subdomain_func(x=(minx,))
        yfix_ = self.subdomain_func(x=(minx,), y=(maxy, miny))
        xright = Function('xright', xright_)
        xleft = Function('xleft', xleft_)
        yfix = Function('yfix', yfix_)
        region_right = domain.create_region('region_right',
                                            'vertices by xright',
                                            'facet',
                                            functions=Functions([xright]))
        region_left = domain.create_region('region_left',
                                           'vertices by xleft',
                                           'facet',
                                           functions=Functions([xleft]))
        region_fix = domain.create_region('region_fix',
                                          'vertices by yfix',
                                          'vertex',
                                          functions=Functions([yfix]))
        fixed_BC = EssentialBC('fixed_BC', region_left, {'u.0' : 0.0})
        displaced_BC = EssentialBC('displaced_BC', region_right, {'u.0' : 1.0})
        fixy_BC = EssentialBC('fixy_BC', region_fix, {'u.1' : 0.0})

        return Conditions([fixed_BC, displaced_BC, fixy_BC])

    def get_mesh(self, shape):
        """
        Generate an Sfepy rectangular mesh

        Args:
          shape: proposed shape of domain (vertex shape) (Nx, Ny)

        Returns:
          Sfepy mesh
          
        """
        Lx = (shape[0] - 1) * self.dx
        Ly = (shape[1] - 1) * self.dx
        center = (0., 0.)
        return gen_block_mesh((Lx, Ly), shape, center, verbose=False)
    
    def solve(self, property_array):
        """
        Solve the Sfepy problem for one sample.

        Args:
          property_array: array of shape (Nx, Ny, 2) where the last
          index is for Lame's parameter and shear modulus,
          respectively.

        Returns:
          the displacement field of shape (Nx, Ny, 2) where the last
          index represents the x and y displacements
          
        """
        shape = property_array.shape[:-1]
        mesh = self.get_mesh(shape)
        domain = Domain('domain', mesh)

        region_all = domain.create_region('region_all', 'all')
        
        field = Field.from_args('fu', np.float64, 'vector', region_all, approx_order=2)

        u = FieldVariable('u', 'unknown', field)
        v = FieldVariable('v', 'test', field, primary_var_name='u')

        m = self.get_material(property_array, domain)
        f = Material('f', val=[[0.0], [0.0]])

        integral = Integral('i', order=3)
    
        t1 = Term.new('dw_lin_elastic_iso(m.lam, m.mu, v, u)',
                      integral, region_all, m=m, v=v, u=u)
        t2 = Term.new('dw_volume_lvf(f.val, v)', integral, region_all, f=f, v=v)
        eq = Equation('balance', t1 + t2)
        eqs = Equations([eq])

        ls = ScipyDirect({})
        nls_status = IndexedStruct()
        nls = Newton({}, lin_solver=ls, status=nls_status)

        pb = Problem('elasticity', equations=eqs, nls=nls, ls=ls)
        pb.save_regions_as_groups('regions')

        epbcs, functions = self.get_periodicBCs(domain)
        
        ebcs = self.get_displacementBCs(domain)

        pb.time_update(ebcs=ebcs,
                       epbcs=epbcs,
                       functions=functions)

        vec = pb.solve()

        return vec.create_output_dict()['u'].data.reshape(shape + (2,))


    
