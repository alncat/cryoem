/***************************************************************************
 *
 * Author: "Sjors H.W. Scheres"
 * MRC Laboratory of Molecular Biology
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/
/***************************************************************************
 *
 * Authors:     Carlos Oscar S. Sorzano (coss@cnb.csic.es)
 *
 * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307  USA
 *
 *  All comments concerning this program package may be sent to the
 *  e-mail address 'xmipp@cnb.csic.es'
 ***************************************************************************/

#ifndef _CTF_HH
#define _CTF_HH

#include "src/multidim_array.h"
#include "src/metadata_table.h"
#include <map>

class CTF
{
protected:

    // Different constants
    RFLOAT K1;
    RFLOAT K2;
    RFLOAT K3;
    RFLOAT K4;
    RFLOAT K5;

    // Azimuthal angle in radians
    RFLOAT rad_azimuth;

    // defocus_average = -(defocus_u + defocus_v)/2
    RFLOAT defocus_average;

    // defocus_deviation = -(defocus_u - defocus_v)/2
    RFLOAT defocus_deviation;

public:

    /// Accelerating Voltage (in KiloVolts)
    RFLOAT kV;

    /// Defocus in U (in Angstroms). Positive values are underfocused
    RFLOAT DeltafU;

    /// Defocus in V (in Angstroms). Postive values are underfocused
    RFLOAT DeltafV;

    /// Azimuthal angle (between X and U) in degrees
    RFLOAT azimuthal_angle;

    // Electron wavelength (Amstrongs)
    RFLOAT lambda;

    // Radius of the aperture (in micras)
    // RFLOAT aperture;
    /// Spherical aberration (in milimeters). Typical value 5.6
    RFLOAT Cs;

    /// Chromatic aberration (in milimeters). Typical value 2
    RFLOAT Ca;

    /** Mean energy loss (eV) due to interaction with sample.
        Typical value 1*/
    RFLOAT espr;

    /// Objective lens stability (deltaI/I) (ppm). Typical value 1
    RFLOAT ispr;

    /// Convergence cone semiangle (in mrad). Typical value 0.5
    RFLOAT alpha;

    /// Longitudinal mechanical displacement (Angstrom). Typical value 100
    RFLOAT DeltaF;

    /// Transversal mechanical displacement (Angstrom). Typical value 3
    RFLOAT DeltaR;

    /// Amplitude contrast. Typical values 0.07 for cryo, 0.2 for negative stain
    RFLOAT Q0;

    // B-factor fall-off
    RFLOAT Bfac;

    // Overall scale-factor of CTF
    RFLOAT scale;

    // Phase-shift from a phase-plate (in rad)
    RFLOAT phase_shift;

    /** Empty constructor. */
    CTF() { clear(); }

	/** get Ks and Q0 */
	void getKsQ0(std::vector<float>& Ks, float& Q0_);

    /** Read CTF parameters from MetaDataTables MD1 and MD2
     * If a parameter is not found in MD1 it is tried to be read from MD2
     * Only if it is also not present in the second then a default value is used
     * This is useful if micrograph-specific parameters are stored in a separate MD from the image-specific parameters
     */
    void read(MetaDataTable &MD1, MetaDataTable &MD2, long int objectID = -1);

    /** Just set all values explicitly */
    void setValues(RFLOAT _defU, RFLOAT _defV, RFLOAT _defAng,
    		RFLOAT _voltage, RFLOAT _Cs, RFLOAT _Q0, RFLOAT _Bfac, RFLOAT _scale = 1., RFLOAT _phase_shift = 0.);

    /** Read from a single MetaDataTable */
    void read(MetaDataTable &MD);

    /** Write to MetaDataTable. */
    void write(MetaDataTable &MD);

    /** Write to output. */
    void write(std::ostream &out);

    /// Clear.
    void clear();

    /// Set up the CTF object, read parameters from MetaDataTables with micrograph and particle information
    /// If no MDmic is provided or it does not contain certain parameters, these parameters are tried to be read from MDimg
    void initialise();

    /// Compute CTF at (U,V). Continuous frequencies
    inline RFLOAT getCTF(RFLOAT X, RFLOAT Y,
    		bool do_abs = false, bool do_only_flip_phases = false, bool do_intact_until_first_peak = false, bool do_damping = true) const
    {
        RFLOAT u2 = X * X + Y * Y;
        RFLOAT u = sqrt(u2);
        RFLOAT u4 = u2 * u2;
        // if (u2>=ua2) return 0;
        RFLOAT deltaf = getDeltaF(X, Y);
        RFLOAT argument = K1 * deltaf * u2 + K2 * u4 - K5;
        RFLOAT retval;
        if (do_intact_until_first_peak && ABS(argument) < PI/2.)
        {
        	retval = 1.;
        }
        else
        {
            retval = -(K3*sin(argument) - Q0*cos(argument)); // Q0 should be positive
        }
        if (do_damping)
        {
        	RFLOAT E = exp(K4 * u2); // B-factor decay (K4 = -Bfac/4);
        	retval *= E;
        }
        if (do_abs)
        {
        	retval = ABS(retval);
        }
        else if (do_only_flip_phases)
        {
        	retval = (retval < 0.) ? -1. : 1.;
        }
        return scale * retval;
    }

    /// Compute CTF at (U,V). Continuous frequencies
    inline RFLOAT getCTFandGrads(RFLOAT X, RFLOAT Y, RFLOAT& grad_u, RFLOAT& grad_v, RFLOAT& grad_t, RFLOAT& hessian_u, RFLOAT& hessian_v, RFLOAT& hessian_uv,
    		RFLOAT& hessian_t, RFLOAT& hessian_tu, RFLOAT& hessian_tv, bool do_abs = false, bool do_only_flip_phases = false, bool do_intact_until_first_peak = false, bool do_damping = true) const
    {
        RFLOAT u2 = X * X + Y * Y;
        RFLOAT u = sqrt(u2);
        RFLOAT u4 = u2 * u2;
        // if (u2>=ua2) return 0;
        RFLOAT hess_t, hess_tu;//, hessian_t, hessian_tu, hessian_tv;
        RFLOAT deltaf = getDeltaFandGrads(X, Y, grad_u, grad_v, grad_t, hess_t, hess_tu);
        RFLOAT argument = K1 * deltaf * u2 + K2 * u4 - K5;
        grad_u *= K1*u2;
        grad_v *= K1*u2;
        grad_t *= K1*u2;
        RFLOAT retval;
        if (do_intact_until_first_peak && ABS(argument) < PI/2.)
        {
        	retval = 1.;
            grad_u = 0.;
            grad_v = 0.;
            grad_t = 0.;
            hessian_u = 0.;
            hessian_v = 0.;
            hessian_uv = 0.;
            hessian_t = 0.;
            hessian_tu = 0.;
            hessian_tv = 0.;
        }
        else
        {
            RFLOAT grad;
            retval = -(K3*sin(argument) - Q0*cos(argument)); // Q0 should be positive
            grad = -(K3*cos(argument) + Q0*sin(argument));
            RFLOAT hessian = -retval;//K3*sin(argument) - Q0*cos(argument);
            hessian_u = hessian*grad_u*grad_u;
            hessian_v = hessian*grad_v*grad_v;
            hessian_uv = hessian*grad_u*grad_v;
            hessian_t = hessian*grad_t*grad_t + grad*hess_t*K1*u2;
            hessian_tu = hessian*grad_u*grad_t + grad*K1*u2*hess_tu;
            hessian_tv = hessian*grad_v*grad_t - grad*K1*u2*hess_tu;
            grad_u = grad*grad_u;
            grad_v = grad*grad_v;
            grad_t = grad*grad_t;
        }
        if (do_damping)
        {
        	RFLOAT E = exp(K4 * u2); // B-factor decay (K4 = -Bfac/4);
        	retval *= E;
            hessian_u *= E;
            hessian_v *= E;
            hessian_uv *= E;
            hessian_t *= E;
            hessian_tu *=E;
            hessian_tv *=E;
            grad_u *= E;
            grad_v *= E;
            grad_t *= E;
        }
        if (do_abs)
        {
            if(retval < 0.) {
                grad_u *= -1.;
                grad_v *= -1.;
                grad_t *= -1.;
                hessian_u *= -1.;
                hessian_v *= -1.;
                hessian_uv *= -1.;
                hessian_t *= -1;
                hessian_tu *= -1;
                hessian_tv *= -1;
            }
        	retval = ABS(retval);
        }
        else if (do_only_flip_phases)
        {
        	retval = (retval < 0.) ? -1. : 1.;
            grad_u = 0.;
            grad_v = 0.;
            grad_t = 0.;
            hessian_u = 0.;
            hessian_v = 0.;
            hessian_uv = 0.;
            hessian_t = 0.;
            hessian_tu = 0.;
            hessian_tv = 0.;
        }
        grad_u *= scale;
        grad_v *= scale;
        grad_t *= scale;
        hessian_u *= scale;
        hessian_v *= scale;
        hessian_uv *= scale;
        hessian_t *= scale;
        hessian_tu *= scale;
        hessian_tv *= scale;
        return scale * retval;
    }
    /// Compute Deltaf at a given direction
    inline RFLOAT getDeltaF(RFLOAT X, RFLOAT Y) const
    {
        if (ABS(X) < XMIPP_EQUAL_ACCURACY &&
            ABS(Y) < XMIPP_EQUAL_ACCURACY)
            return 0;

        RFLOAT ellipsoid_ang = atan2(Y, X) - rad_azimuth;
        /*
        * For a derivation of this formulae confer
        * Principles of Electron Optics page 1380
        * in particular term defocus and twofold axial astigmatism
        * take into account that a1 and a2 are the coefficient
        * of the zernike polynomials difference of defocus at 0
        * and at 45 degrees. In this case a2=0
        */
        RFLOAT cos_ellipsoid_ang_2 = cos(2*ellipsoid_ang);
        return (defocus_average + defocus_deviation*cos_ellipsoid_ang_2);

    }
    /// Compute Deltaf at a given direction
    inline RFLOAT getDeltaFandGrads(RFLOAT X, RFLOAT Y, RFLOAT& grad_u, RFLOAT& grad_v, RFLOAT& grad_t, RFLOAT& hess_t, RFLOAT & hess_tu) const
    {
        grad_u = 0., grad_v = 0., grad_t = 0., hess_t = 0., hess_tu = 0.;
        if (ABS(X) < XMIPP_EQUAL_ACCURACY &&
            ABS(Y) < XMIPP_EQUAL_ACCURACY)
            return 0;

        RFLOAT ellipsoid_ang = atan2(Y, X) - rad_azimuth;
        /*
        * For a derivation of this formulae confer
        * Principles of Electron Optics page 1380
        * in particular term defocus and twofold axial astigmatism
        * take into account that a1 and a2 are the coefficient
        * of the zernike polynomials difference of defocus at 0
        * and at 45 degrees. In this case a2=0
        */
        RFLOAT cos_ellipsoid_ang_2 = cos(2*ellipsoid_ang);
        grad_u = -0.5 - 0.5*cos_ellipsoid_ang_2;
        grad_v = -0.5 + 0.5*cos_ellipsoid_ang_2;
        grad_t = defocus_deviation*sin(2.*ellipsoid_ang)*2.;
        hess_t = -4.*defocus_deviation*cos_ellipsoid_ang_2;
        hess_tu = -sin(2.*ellipsoid_ang);
        return (defocus_average + defocus_deviation*cos_ellipsoid_ang_2);

    }

    /// Generate (Fourier-space, i.e. FFTW format) image with all CTF values.
    /// The dimensions of the result array should have been set correctly already
    void getFftwImage(MultidimArray < RFLOAT > &result, int orixdim, int oriydim, RFLOAT angpix,
    		bool do_abs = false, bool do_only_flip_phases = false, bool do_intact_until_first_peak = false, bool do_damping = true);
    /// Generate (Fourier-space, i.e. FFTW format) image with all CTF values.
    /// The dimensions of the result array should have been set correctly already
    void getFftwImageandGrads(MultidimArray < RFLOAT > &result, MultidimArray<RFLOAT>& grad_u, MultidimArray<RFLOAT>& grad_v, MultidimArray<RFLOAT>& grad_t, MultidimArray<RFLOAT>& hessian_u, MultidimArray<RFLOAT>& hessian_v, MultidimArray<RFLOAT>& hessian_uv, MultidimArray<RFLOAT>& hessian_t, MultidimArray<RFLOAT>& hessian_tu, MultidimArray<RFLOAT>& hessian_tv, int orixdim, int oriydim, RFLOAT angpix,
    		bool do_abs = false, bool do_only_flip_phases = false, bool do_intact_until_first_peak = false, bool do_damping = true);

    /// Generate a centered image (with hermitian symmetry)
    /// The dimensions of the result array should have been set correctly already
    void getCenteredImage(MultidimArray < RFLOAT > &result, RFLOAT angpix,
    		bool do_abs = false, bool do_only_flip_phases = false, bool do_intact_until_first_peak = false, bool do_damping = true);

    /// Generate a 1D profile along defocusAngle
    /// The dimensions of the result array should have been set correctly already, i.e. at the image size!
    void get1DProfile(MultidimArray < RFLOAT > &result, RFLOAT angle, RFLOAT angpix,
    		bool do_abs = false, bool do_only_flip_phases = false, bool do_intact_until_first_peak = false, bool do_damping = true);


};
//@}
#endif
