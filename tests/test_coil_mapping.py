#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for coil_mapping module.

Tests all CoilMapping classes: DirectRZMapping, PolarCoordinateMapping,
ThetaRadialMapping, and base class methods.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coil_mapping import (
    CoilMapping,
    DirectRZMapping,
    PolarCoordinateMapping,
    ThetaRadialMapping
)


# ==============================================================================
# Mock CoilPositionSpace for testing ThetaRadialMapping
# ==============================================================================

class MockCoilPositionSpace:
    """Mock CoilPositionSpace for testing."""

    def __init__(self, inner_curve=None, outer_curve=None):
        """Initialize mock space with simple linear boundaries."""
        # Default: simple straight lines
        if inner_curve is None:
            # Inner boundary: R from 1.0 to 1.5, Z from 0 to 1.0
            thetas = np.linspace(0, 180, 100)
            self.inner_curve = np.column_stack([
                np.linspace(1.0, 1.5, 100),
                np.linspace(0, 1.0, 100)
            ])
        else:
            self.inner_curve = inner_curve

        if outer_curve is None:
            # Outer boundary: R from 1.5 to 2.0, Z from 0 to 1.2
            self.outer_curve = np.column_stack([
                np.linspace(1.5, 2.0, 100),
                np.linspace(0, 1.2, 100)
            ])
        else:
            self.outer_curve = outer_curve

    def interpolate(self, theta, radial):
        """Simple linear interpolation."""
        # theta: 0-180 degrees
        # radial: 0-1 (0=inner, 1=outer)

        # Map theta to index
        idx = int(theta / 180.0 * (len(self.inner_curve) - 1))
        idx = np.clip(idx, 0, len(self.inner_curve) - 1)

        # Get positions at theta
        R_inner, Z_inner = self.inner_curve[idx]
        R_outer, Z_outer = self.outer_curve[idx]

        # Interpolate radially
        R = (1 - radial) * R_inner + radial * R_outer
        Z = (1 - radial) * Z_inner + radial * Z_outer

        return R, Z

    def get_bounds(self):
        """Return default bounds."""
        return (0, 180), (0, 1)


# ==============================================================================
# Test DirectRZMapping
# ==============================================================================

class TestDirectRZMapping:
    """Test DirectRZMapping class."""

    def test_initialization_default(self):
        """Test default initialization."""
        mapping = DirectRZMapping()
        assert mapping.R_bounds == (0.5, 2.0)
        assert mapping.Z_bounds == (-1.5, 1.5)
        assert mapping.coil_dx == 0.08
        assert mapping.coil_dy == 0.08
        assert mapping.position_space is None

    def test_initialization_custom(self):
        """Test custom initialization."""
        mapping = DirectRZMapping(
            R_bounds=(1.0, 3.0),
            Z_bounds=(-2.0, 2.0),
            coil_dx=0.1,
            coil_dy=0.12
        )
        assert mapping.R_bounds == (1.0, 3.0)
        assert mapping.Z_bounds == (-2.0, 2.0)
        assert mapping.coil_dx == 0.1
        assert mapping.coil_dy == 0.12

    def test_params_to_positions_single_coil(self):
        """Test converting params to positions for single coil."""
        mapping = DirectRZMapping()
        params = [1.5, 0.5]  # [R1, Z1]
        ncoils = 1

        positions = mapping.params_to_positions(params, ncoils)

        assert len(positions) == 1
        assert positions[0] == (1.5, 0.5)

    def test_params_to_positions_multiple_coils(self):
        """Test converting params to positions for multiple coils."""
        mapping = DirectRZMapping()
        params = [1.0, 0.2, 1.5, 0.5, 2.0, 0.8]  # [R1, Z1, R2, Z2, R3, Z3]
        ncoils = 3

        positions = mapping.params_to_positions(params, ncoils)

        assert len(positions) == 3
        assert positions[0] == (1.0, 0.2)
        assert positions[1] == (1.5, 0.5)
        assert positions[2] == (2.0, 0.8)

    def test_get_bounds_single_coil(self):
        """Test bounds generation for single coil."""
        mapping = DirectRZMapping(R_bounds=(1.0, 2.5), Z_bounds=(-1.0, 1.0))
        bounds = mapping.get_bounds(ncoils=1)

        assert len(bounds) == 2
        assert bounds[0] == (1.0, 2.5)  # R bound
        assert bounds[1] == (-1.0, 1.0)  # Z bound

    def test_get_bounds_multiple_coils(self):
        """Test bounds generation for multiple coils."""
        mapping = DirectRZMapping(R_bounds=(1.0, 2.5), Z_bounds=(-1.0, 1.0))
        bounds = mapping.get_bounds(ncoils=3)

        assert len(bounds) == 6  # 2 params per coil * 3 coils
        # First coil
        assert bounds[0] == (1.0, 2.5)
        assert bounds[1] == (-1.0, 1.0)
        # Second coil
        assert bounds[2] == (1.0, 2.5)
        assert bounds[3] == (-1.0, 1.0)
        # Third coil
        assert bounds[4] == (1.0, 2.5)
        assert bounds[5] == (-1.0, 1.0)

    def test_positions_to_geometry(self):
        """Test conversion from positions to coil geometry."""
        mapping = DirectRZMapping(coil_dx=0.1, coil_dy=0.15)
        positions = [(1.5, 0.5), (2.0, 0.8)]

        geometry = mapping.positions_to_geometry(positions)

        # Check structure
        assert 'coils' in geometry
        assert 'F0A' in geometry['coils']  # Top coil 0
        assert 'F0B' in geometry['coils']  # Bottom coil 0
        assert 'F1A' in geometry['coils']  # Top coil 1
        assert 'F1B' in geometry['coils']  # Bottom coil 1

        # Check first coil geometry (top)
        pts_top = geometry['coils']['F0A']['pts']
        assert pts_top.shape == (4, 2)

        # Expected corners for R=1.5, Z=0.5, dx=0.1, dy=0.15
        expected_top = np.array([
            [1.4, 0.65],   # [R-dx, Z+dy]
            [1.6, 0.65],   # [R+dx, Z+dy]
            [1.6, 0.35],   # [R+dx, Z-dy]
            [1.4, 0.35]    # [R-dx, Z-dy]
        ])
        assert_allclose(pts_top, expected_top)

        # Check mirroring for bottom coil
        pts_bot = geometry['coils']['F0B']['pts']
        expected_bot = expected_top * np.array([1, -1])
        assert_allclose(pts_bot, expected_bot)

    def test_make_coils_from_params(self):
        """Test end-to-end coil generation from params."""
        mapping = DirectRZMapping(coil_dx=0.08, coil_dy=0.08)
        params = [1.5, 0.5, 2.0, 0.8]
        ncoils = 2

        geometry = mapping.make_coils_from_params(params, ncoils)

        assert 'coils' in geometry
        assert len(geometry['coils']) == 4  # 2 pairs


# ==============================================================================
# Test PolarCoordinateMapping
# ==============================================================================

class TestPolarCoordinateMapping:
    """Test PolarCoordinateMapping class."""

    def test_initialization_default(self):
        """Test default initialization."""
        mapping = PolarCoordinateMapping()
        assert mapping.center_R == 1.5
        assert mapping.center_Z == 0.0
        assert mapping.radius_bounds == (0.3, 1.0)
        assert mapping.angle_bounds == (0, 180)
        assert mapping.coil_dx == 0.08
        assert mapping.coil_dy == 0.08

    def test_initialization_custom(self):
        """Test custom initialization."""
        mapping = PolarCoordinateMapping(
            center_R=2.0,
            center_Z=0.5,
            radius_bounds=(0.5, 1.5),
            angle_bounds=(30, 150),
            coil_dx=0.1,
            coil_dy=0.12
        )
        assert mapping.center_R == 2.0
        assert mapping.center_Z == 0.5
        assert mapping.radius_bounds == (0.5, 1.5)
        assert mapping.angle_bounds == (30, 150)

    def test_params_to_positions_zero_angle(self):
        """Test polar conversion at 0 degrees (outboard midplane)."""
        mapping = PolarCoordinateMapping(center_R=1.5, center_Z=0.0)
        params = [0.5, 0.0]  # radius=0.5, angle=0�
        ncoils = 1

        positions = mapping.params_to_positions(params, ncoils)

        # At 0�: R = center_R + radius, Z = center_Z
        expected_R = 1.5 + 0.5
        expected_Z = 0.0
        assert_allclose(positions[0], (expected_R, expected_Z), rtol=1e-10)

    def test_params_to_positions_90_degrees(self):
        """Test polar conversion at 90 degrees (top)."""
        mapping = PolarCoordinateMapping(center_R=1.5, center_Z=0.0)
        params = [0.5, 90.0]  # radius=0.5, angle=90�
        ncoils = 1

        positions = mapping.params_to_positions(params, ncoils)

        # At 90�: R = center_R, Z = center_Z + radius
        expected_R = 1.5
        expected_Z = 0.0 + 0.5
        assert_allclose(positions[0], (expected_R, expected_Z), rtol=1e-10)

    def test_params_to_positions_180_degrees(self):
        """Test polar conversion at 180 degrees (inboard midplane)."""
        mapping = PolarCoordinateMapping(center_R=1.5, center_Z=0.0)
        params = [0.5, 180.0]  # radius=0.5, angle=180�
        ncoils = 1

        positions = mapping.params_to_positions(params, ncoils)

        # At 180�: R = center_R - radius, Z = center_Z
        expected_R = 1.5 - 0.5
        expected_Z = 0.0
        assert_allclose(positions[0], (expected_R, expected_Z), atol=1e-15)

    def test_params_to_positions_multiple_coils(self):
        """Test polar conversion for multiple coils."""
        mapping = PolarCoordinateMapping(center_R=1.5, center_Z=0.0)
        params = [0.5, 0.0, 0.6, 45.0, 0.7, 90.0]  # 3 coils
        ncoils = 3

        positions = mapping.params_to_positions(params, ncoils)

        assert len(positions) == 3

        # Coil 0: radius=0.5, angle=0�
        R0 = 1.5 + 0.5 * np.cos(0)
        Z0 = 0.0 + 0.5 * np.sin(0)
        assert_allclose(positions[0], (R0, Z0), rtol=1e-10)

        # Coil 1: radius=0.6, angle=45�
        R1 = 1.5 + 0.6 * np.cos(45 * np.pi / 180)
        Z1 = 0.0 + 0.6 * np.sin(45 * np.pi / 180)
        assert_allclose(positions[1], (R1, Z1), rtol=1e-10)

        # Coil 2: radius=0.7, angle=90�
        R2 = 1.5 + 0.7 * np.cos(90 * np.pi / 180)
        Z2 = 0.0 + 0.7 * np.sin(90 * np.pi / 180)
        assert_allclose(positions[2], (R2, Z2), rtol=1e-10)

    def test_get_bounds(self):
        """Test bounds generation."""
        mapping = PolarCoordinateMapping(
            radius_bounds=(0.4, 1.2),
            angle_bounds=(20, 160)
        )
        bounds = mapping.get_bounds(ncoils=3)

        assert len(bounds) == 6  # 2 params per coil
        # Alternating radius and angle bounds
        assert bounds[0] == (0.4, 1.2)  # radius coil 0
        assert bounds[1] == (20, 160)   # angle coil 0
        assert bounds[2] == (0.4, 1.2)  # radius coil 1
        assert bounds[3] == (20, 160)   # angle coil 1
        assert bounds[4] == (0.4, 1.2)  # radius coil 2
        assert bounds[5] == (20, 160)   # angle coil 2


# ==============================================================================
# Test ThetaRadialMapping
# ==============================================================================

class TestThetaRadialMapping:
    """Test ThetaRadialMapping class."""

    def test_initialization_requires_position_space(self):
        """Test that initialization requires a position_space."""
        with pytest.raises(ValueError, match="requires a position_space"):
            ThetaRadialMapping(position_space=None)

    def test_initialization_with_position_space(self):
        """Test initialization with mock position space."""
        space = MockCoilPositionSpace()
        mapping = ThetaRadialMapping(position_space=space, coil_dx=0.1, coil_dy=0.12)

        assert mapping.position_space is space
        assert mapping.coil_dx == 0.1
        assert mapping.coil_dy == 0.12

    def test_params_to_positions_single_coil(self):
        """Test converting theta/radial params to positions."""
        space = MockCoilPositionSpace()
        mapping = ThetaRadialMapping(position_space=space)

        params = [90.0, 0.5]  # [theta, radial]
        ncoils = 1

        positions = mapping.params_to_positions(params, ncoils)

        assert len(positions) == 1
        # Should match MockCoilPositionSpace.interpolate(90, 0.5)
        expected = space.interpolate(90.0, 0.5)
        assert_allclose(positions[0], expected)

    def test_params_to_positions_multiple_coils(self):
        """Test converting params for multiple coils."""
        space = MockCoilPositionSpace()
        mapping = ThetaRadialMapping(position_space=space)

        thetas = [30.0, 90.0, 150.0]
        radials = [0.2, 0.5, 0.8]
        params = thetas + radials  # First all thetas, then all radials
        ncoils = 3

        positions = mapping.params_to_positions(params, ncoils)

        assert len(positions) == 3
        for i in range(3):
            expected = space.interpolate(thetas[i], radials[i])
            assert_allclose(positions[i], expected)

    def test_get_bounds_from_position_space(self):
        """Test that bounds come from position space."""
        space = MockCoilPositionSpace()
        mapping = ThetaRadialMapping(position_space=space)

        bounds = mapping.get_bounds(ncoils=2)

        # Should be 4 bounds total: [theta0, theta1, radial0, radial1]
        assert len(bounds) == 4

        # First two should be theta bounds
        theta_bounds, radial_bounds = space.get_bounds()
        assert bounds[0] == theta_bounds
        assert bounds[1] == theta_bounds

        # Last two should be radial bounds
        assert bounds[2] == radial_bounds
        assert bounds[3] == radial_bounds

    def test_params_to_positions_at_inner_boundary(self):
        """Test that radial=0 gives inner boundary."""
        space = MockCoilPositionSpace()
        mapping = ThetaRadialMapping(position_space=space)

        params = [90.0, 0.0]  # theta=90, radial=0 (inner)
        positions = mapping.params_to_positions(params, ncoils=1)

        # Should match inner curve at theta=90
        expected_R, expected_Z = space.interpolate(90.0, 0.0)
        assert_allclose(positions[0], (expected_R, expected_Z))

    def test_params_to_positions_at_outer_boundary(self):
        """Test that radial=1 gives outer boundary."""
        space = MockCoilPositionSpace()
        mapping = ThetaRadialMapping(position_space=space)

        params = [90.0, 1.0]  # theta=90, radial=1 (outer)
        positions = mapping.params_to_positions(params, ncoils=1)

        # Should match outer curve at theta=90
        expected_R, expected_Z = space.interpolate(90.0, 1.0)
        assert_allclose(positions[0], (expected_R, expected_Z))


# ==============================================================================
# Test CoilMapping Base Class Methods
# ==============================================================================

class TestCoilMappingBaseMethods:
    """Test base class methods shared by all mappings."""

    def test_positions_to_geometry_creates_top_bottom_pairs(self):
        """Test that positions_to_geometry creates mirrored top/bottom coils."""
        mapping = DirectRZMapping(coil_dx=0.1, coil_dy=0.1)
        positions = [(1.5, 0.5)]

        geometry = mapping.positions_to_geometry(positions)

        # Should have both top and bottom
        assert 'F0A' in geometry['coils']
        assert 'F0B' in geometry['coils']

        # Bottom should be mirror of top
        pts_top = geometry['coils']['F0A']['pts']
        pts_bot = geometry['coils']['F0B']['pts']

        expected_bot = pts_top * np.array([1, -1])
        assert_allclose(pts_bot, expected_bot)

    def test_positions_to_geometry_coil_names(self):
        """Test coil naming convention."""
        mapping = DirectRZMapping()
        positions = [(1.0, 0.2), (1.5, 0.5), (2.0, 0.8)]

        geometry = mapping.positions_to_geometry(positions)

        expected_names = ['F0A', 'F0B', 'F1A', 'F1B', 'F2A', 'F2B']
        assert sorted(geometry['coils'].keys()) == sorted(expected_names)

    def test_positions_to_geometry_nturns(self):
        """Test that all coils have nturns=1.0."""
        mapping = DirectRZMapping()
        positions = [(1.5, 0.5), (2.0, 0.8)]

        geometry = mapping.positions_to_geometry(positions)

        for coil_name, coil_data in geometry['coils'].items():
            assert coil_data['nturns'] == 1.0

    def test_make_filaments_3x3_single_coil(self):
        """Test 3x3 filament generation for single coil."""
        mapping = DirectRZMapping()
        coil_centers = [np.array([[1.5, 0.5]])]  # Shape (1, 2)
        Rfil = 0.01

        filaments = mapping.make_filaments(coil_centers, Rfil)

        assert len(filaments) == 1  # One coil
        assert len(filaments[0]) == 9  # 3x3 = 9 filaments

        # Check central filament is at coil center
        central_filament = filaments[0][4]  # Middle of 3x3
        assert_allclose(central_filament, [1.5, 0.5])

    def test_make_filaments_3x3_offsets(self):
        """Test that filaments are correctly offset."""
        mapping = DirectRZMapping()
        coil_centers = [np.array([[1.5, 0.5]])]
        Rfil = 0.01

        filaments = mapping.make_filaments(coil_centers, Rfil)

        # Expected offsets: [-1, 0, 1] * 2 * Rfil
        expected_filaments = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                R = 1.5 + 2 * Rfil * dx
                Z = 0.5 + 2 * Rfil * dy
                expected_filaments.append([R, Z])

        assert_allclose(filaments[0], expected_filaments)

    def test_make_filaments_multiple_coils(self):
        """Test filament generation for multiple coils."""
        mapping = DirectRZMapping()
        coil_centers = [
            np.array([[1.0, 0.2]]),
            np.array([[1.5, 0.5]]),
            np.array([[2.0, 0.8]])
        ]
        Rfil = 0.01

        filaments = mapping.make_filaments(coil_centers, Rfil)

        assert len(filaments) == 3
        for fil_set in filaments:
            assert len(fil_set) == 9

    def test_make_coils_from_params_integration(self):
        """Test full pipeline: params -> positions -> geometry."""
        mapping = DirectRZMapping()
        params = [1.5, 0.5]
        ncoils = 1

        geometry = mapping.make_coils_from_params(params, ncoils)

        # Verify it produces valid geometry
        assert 'coils' in geometry
        assert 'F0A' in geometry['coils']
        assert 'F0B' in geometry['coils']
        assert geometry['coils']['F0A']['pts'].shape == (4, 2)


# ==============================================================================
# Integration Tests
# ==============================================================================

class TestIntegration:
    """Integration tests across different mapping types."""

    def test_all_mappings_produce_same_structure(self):
        """Test that all mapping types produce compatible geometry structure."""
        ncoils = 2

        # DirectRZ
        direct_mapping = DirectRZMapping()
        direct_params = [1.5, 0.5, 2.0, 0.8]
        direct_geom = direct_mapping.make_coils_from_params(direct_params, ncoils)

        # Polar
        polar_mapping = PolarCoordinateMapping()
        polar_params = [0.5, 45.0, 0.6, 90.0]
        polar_geom = polar_mapping.make_coils_from_params(polar_params, ncoils)

        # ThetaRadial
        space = MockCoilPositionSpace()
        theta_mapping = ThetaRadialMapping(position_space=space)
        theta_params = [45.0, 90.0, 0.3, 0.6]
        theta_geom = theta_mapping.make_coils_from_params(theta_params, ncoils)

        # All should have same structure
        for geom in [direct_geom, polar_geom, theta_geom]:
            assert 'coils' in geom
            assert len(geom['coils']) == 4  # 2 pairs
            for coil_name in geom['coils']:
                assert 'pts' in geom['coils'][coil_name]
                assert 'nturns' in geom['coils'][coil_name]
                assert geom['coils'][coil_name]['pts'].shape == (4, 2)

    def test_different_coil_sizes(self):
        """Test that coil size parameters work correctly."""
        mapping1 = DirectRZMapping(coil_dx=0.05, coil_dy=0.05)
        mapping2 = DirectRZMapping(coil_dx=0.15, coil_dy=0.20)

        params = [1.5, 0.5]
        ncoils = 1

        geom1 = mapping1.make_coils_from_params(params, ncoils)
        geom2 = mapping2.make_coils_from_params(params, ncoils)

        pts1 = geom1['coils']['F0A']['pts']
        pts2 = geom2['coils']['F0A']['pts']

        # Width of coil 1: 2*0.05 = 0.1
        width1 = pts1[1, 0] - pts1[0, 0]
        assert_allclose(width1, 0.1)

        # Width of coil 2: 2*0.15 = 0.3
        width2 = pts2[1, 0] - pts2[0, 0]
        assert_allclose(width2, 0.3)

        # Height of coil 1: 2*0.05 = 0.1
        height1 = pts1[0, 1] - pts1[2, 1]
        assert_allclose(height1, 0.1)

        # Height of coil 2: 2*0.20 = 0.4
        height2 = pts2[0, 1] - pts2[2, 1]
        assert_allclose(height2, 0.4)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])