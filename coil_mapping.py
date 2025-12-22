"""
Helper classes for PF coil optimization.

Provides flexible parameterization mappings for coil positions.
"""

import numpy as np
import copy


class CoilMapping:
    """
    Base class for mapping optimization parameters to coil geometry.
    Users can subclass this to define custom parameterizations.
    """
    
    def __init__(self, position_space=None, coil_dx=0.08, coil_dy=0.08):
        """
        Parameters
        ----------
        position_space : CoilPositionSpace, optional
            Search space (for theta/radial parameterization)
        coil_dx : float
            Half-width of coil in R direction
        coil_dy : float
            Half-height of coil in Z direction
        """
        self.position_space = position_space
        self.coil_dx = coil_dx
        self.coil_dy = coil_dy
    
    def params_to_positions(self, params, ncoils):
        """
        Convert optimization parameters to (R, Z) positions.
        
        Parameters
        ----------
        params : array-like
            Optimization parameters
        ncoils : int
            Number of coil pairs
            
        Returns
        -------
        positions : list of (R, Z) tuples
            Coil center positions (top-side only)
        """
        raise NotImplementedError("Subclasses must implement params_to_positions")
    
    def get_bounds(self, ncoils):
        """
        Get optimization parameter bounds.
        
        Parameters
        ----------
        ncoils : int
            Number of coil pairs
            
        Returns
        -------
        bounds : list of (min, max) tuples
        """
        raise NotImplementedError("Subclasses must implement get_bounds")
    
    def positions_to_geometry(self, positions):
        """
        Convert (R, Z) positions to coil geometry.

        Users can override this to customize coil shapes, sizes, or
        remove top/bottom mirroring.

        Parameters
        ----------
        positions : list of (R, Z) tuples
            Coil center positions for top-side coils

        Returns
        -------
        coil_geometry : dict
            Coil geometry dictionary with 'coils' key
        """
        coil_geometry = {"coils": {}}

        for i, (R, Z) in enumerate(positions):
            # Create rectangular coil corners for top coil
            pts_top = np.array([
                [R - self.coil_dx, Z + self.coil_dy],
                [R + self.coil_dx, Z + self.coil_dy],
                [R + self.coil_dx, Z - self.coil_dy],
                [R - self.coil_dx, Z - self.coil_dy]
            ])

            # Mirror for bottom coil (Z -> -Z)
            pts_bot = pts_top * np.array([1, -1])

            # Add to geometry
            coil_geometry["coils"][f'F{i}A'] = {
                'pts': copy.deepcopy(pts_top),
                'nturns': 1.0
            }
            coil_geometry["coils"][f'F{i}B'] = {
                'pts': copy.deepcopy(pts_bot),
                'nturns': 1.0
            }

        return coil_geometry

    def make_coils_from_params(self, params, ncoils):
        """
        Generate coil geometry from optimization parameters.

        This is the main method called by the optimizer. It combines
        params_to_positions and positions_to_geometry.

        Parameters
        ----------
        params : array-like
            Optimization parameters
        ncoils : int
            Number of coil pairs

        Returns
        -------
        coil_geometry : dict
            Coil geometry dictionary with 'coils' key
        """
        # Step 1: params → positions (user-defined)
        positions = self.params_to_positions(params, ncoils)

        # Step 2: positions → geometry (can be overridden)
        return self.positions_to_geometry(positions)

    def make_filaments(self, coil_centers, Rfil):
        """
        Convert coil centers to filament positions for thick coil model.

        Default implementation uses 3x3 filament arrangement.
        Users can override for custom arrangements (5x5, circular, etc.).

        Parameters
        ----------
        coil_centers : list of arrays
            Coil center positions, each element is array of shape (1, 2) with [R, Z]
        Rfil : float
            Filament radius (spacing between filaments)

        Returns
        -------
        filament_sets : list of lists
            For each coil, a list of [R, Z] filament positions
        """
        filament_sets = []
        for center in coil_centers:
            filaments = self._make_3x3_thick(center[0], Rfil)
            filament_sets.append(filaments)
        return filament_sets

    def _make_3x3_thick(self, center, R):
        """
        Generate centers of 9 filaments in 3×3 arrangement.

        Parameters
        ----------
        center : array-like of shape (2,)
            Coil center position [R0, Z0]
        R : float
            Filament radius (spacing between filaments)

        Returns
        -------
        fil_centers : list of [R, Z] positions
        """
        R0, Z0 = center
        offsets = [-1, 0, 1]
        fil_centers = []
        for dx in offsets:
            for dy in offsets:
                fil_centers.append([R0 + 2 * R * dx, Z0 + 2 * R * dy])
        return fil_centers


class ThetaRadialMapping(CoilMapping):
    """
    Default parameterization using (theta, radial) coordinates.

    This is the original parameterization where coils are positioned
    along curves defined by CoilPositionSpace.

    Parameters are structured as:
    params = [theta1, theta2, ..., thetaN, radial1, radial2, ..., radialN]

    where:
    - theta: Poloidal angle in degrees (typically 0-180)
    - radial: Radial position between inner/outer curves (0=inner, 1=outer)
    """

    def __init__(self, position_space, coil_dx=0.08, coil_dy=0.08):
        """
        Initialize ThetaRadialMapping.

        Parameters
        ----------
        position_space : CoilPositionSpace or PerCoilPositionSpace
            Search space defining inner/outer boundary curves
        coil_dx : float
            Half-width of coil in R direction
        coil_dy : float
            Half-height of coil in Z direction
        """
        if position_space is None:
            raise ValueError("ThetaRadialMapping requires a position_space")
        super().__init__(position_space, coil_dx, coil_dy)

    def params_to_positions(self, params, ncoils):
        """
        Convert theta/radial parameters to (R, Z) positions.

        Parameters
        ----------
        params : array-like of length 2*ncoils
            [theta1, ..., thetaN, radial1, ..., radialN]
        ncoils : int
            Number of coil pairs

        Returns
        -------
        positions : list of (R, Z) tuples
        """
        thetas = params[:ncoils]
        radials = params[ncoils:2*ncoils]

        positions = []
        for i, (theta, radial) in enumerate(zip(thetas, radials)):
            # Check if we have per-coil or global position space
            if hasattr(self.position_space, 'interpolate_for_coil'):
                # PerCoilPositionSpace
                R, Z = self.position_space.interpolate_for_coil(i, theta, radial)
            else:
                # CoilPositionSpace
                R, Z = self.position_space.interpolate(theta, radial)

            positions.append((R, Z))

        return positions

    def get_bounds(self, ncoils):
        """
        Get bounds for theta and radial parameters.

        Parameters
        ----------
        ncoils : int
            Number of coil pairs

        Returns
        -------
        bounds : list of (min, max) tuples
            First ncoils entries are theta bounds,
            next ncoils entries are radial bounds
        """
        bounds = []

        # Theta bounds for each coil
        for i in range(ncoils):
            if hasattr(self.position_space, 'get_bounds_for_coil'):
                # PerCoilPositionSpace
                theta_bounds, _ = self.position_space.get_bounds_for_coil(i)
            else:
                # CoilPositionSpace
                theta_bounds, _ = self.position_space.get_bounds()
            bounds.append(theta_bounds)

        # Radial bounds for each coil
        for i in range(ncoils):
            if hasattr(self.position_space, 'get_bounds_for_coil'):
                # PerCoilPositionSpace
                _, radial_bounds = self.position_space.get_bounds_for_coil(i)
            else:
                # CoilPositionSpace
                _, radial_bounds = self.position_space.get_bounds()
            bounds.append(radial_bounds)

        return bounds


class DirectRZMapping(CoilMapping):
    """
    Direct (R, Z) coordinate parameterization.

    This is the simplest parameterization where optimization parameters
    are directly the physical (R, Z) coordinates of coil centers.

    Parameters are structured as:
    params = [R1, Z1, R2, Z2, ..., RN, ZN]

    This mapping does not require a CoilPositionSpace.
    """

    def __init__(self, R_bounds=(0.5, 2.0), Z_bounds=(-1.5, 1.5),
                 coil_dx=0.08, coil_dy=0.08):
        """
        Initialize DirectRZMapping.

        Parameters
        ----------
        R_bounds : tuple (float, float)
            (min, max) bounds for R coordinate (meters)
        Z_bounds : tuple (float, float)
            (min, max) bounds for Z coordinate (meters)
        coil_dx : float
            Half-width of coil in R direction
        coil_dy : float
            Half-height of coil in Z direction
        """
        super().__init__(position_space=None, coil_dx=coil_dx, coil_dy=coil_dy)
        self.R_bounds = R_bounds
        self.Z_bounds = Z_bounds

    def params_to_positions(self, params, ncoils):
        """
        Extract (R, Z) positions directly from parameters.

        Parameters
        ----------
        params : array-like of length 2*ncoils
            [R1, Z1, R2, Z2, ..., RN, ZN]
        ncoils : int
            Number of coil pairs

        Returns
        -------
        positions : list of (R, Z) tuples
        """
        positions = []
        for i in range(ncoils):
            R = params[2*i]
            Z = params[2*i + 1]
            positions.append((R, Z))
        return positions

    def get_bounds(self, ncoils):
        """
        Get R, Z bounds for each coil.

        Parameters
        ----------
        ncoils : int
            Number of coil pairs

        Returns
        -------
        bounds : list of (min, max) tuples
            Alternating R bounds, Z bounds for each coil
        """
        bounds = []
        for i in range(ncoils):
            bounds.append(self.R_bounds)  # R bound for coil i
            bounds.append(self.Z_bounds)  # Z bound for coil i
        return bounds


class PolarCoordinateMapping(CoilMapping):
    """
    Polar coordinate parameterization relative to a center point.

    This maps coils using polar coordinates (radius, angle) relative
    to a specified center point in the R-Z plane.

    Parameters are structured as:
    params = [r1, angle1, r2, angle2, ..., rN, angleN]

    where:
    - r: Radial distance from center point (meters)
    - angle: Poloidal angle in degrees (0° = outboard midplane)

    This is an example custom mapping that users can use as a template.
    """

    def __init__(self, center_R=1.5, center_Z=0.0,
                 radius_bounds=(0.3, 1.0),
                 angle_bounds=(0, 180),
                 coil_dx=0.08, coil_dy=0.08):
        """
        Initialize PolarCoordinateMapping.

        Parameters
        ----------
        center_R : float
            R coordinate of center point (meters)
        center_Z : float
            Z coordinate of center point (meters)
        radius_bounds : tuple (float, float)
            (min, max) radial distance from center (meters)
        angle_bounds : tuple (float, float)
            (min, max) poloidal angle in degrees
        coil_dx : float
            Half-width of coil in R direction
        coil_dy : float
            Half-height of coil in Z direction
        """
        super().__init__(position_space=None, coil_dx=coil_dx, coil_dy=coil_dy)
        self.center_R = center_R
        self.center_Z = center_Z
        self.radius_bounds = radius_bounds
        self.angle_bounds = angle_bounds

    def params_to_positions(self, params, ncoils):
        """
        Convert polar coordinates to (R, Z) positions.

        Parameters
        ----------
        params : array-like of length 2*ncoils
            [r1, angle1, r2, angle2, ..., rN, angleN]
        ncoils : int
            Number of coil pairs

        Returns
        -------
        positions : list of (R, Z) tuples
        """
        positions = []
        for i in range(ncoils):
            radius = params[2*i]
            angle_deg = params[2*i + 1]
            angle_rad = angle_deg * np.pi / 180.0  # Convert to radians

            # Polar to Cartesian conversion
            R = self.center_R + radius * np.cos(angle_rad)
            Z = self.center_Z + radius * np.sin(angle_rad)

            positions.append((R, Z))

        return positions

    def get_bounds(self, ncoils):
        """
        Get radius and angle bounds for each coil.

        Parameters
        ----------
        ncoils : int
            Number of coil pairs

        Returns
        -------
        bounds : list of (min, max) tuples
            Alternating radius bounds, angle bounds for each coil
        """
        bounds = []
        for i in range(ncoils):
            bounds.append(self.radius_bounds)  # Radius bound for coil i
            bounds.append(self.angle_bounds)    # Angle bound for coil i
        return bounds