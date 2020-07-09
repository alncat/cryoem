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
 * Authors:    Roberto Marabini                 (roberto@cnb.csic.es)
 *             Carlos Oscar S. Sorzano          (coss@cnb.csic.es)
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

#ifndef __RELIONFFTW_H
#define __RELIONFFTW_H

#include <algorithm>
#include <fftw3.h>
#include "src/multidim_array.h"
#include "src/funcs.h"
#include "src/tabfuncs.h"
#include "src/complex.h"
#include "src/CPlot2D.h"

/** @defgroup FourierW FFTW Fourier transforms
  * @ingroup DataLibrary
  */

/** For all direct elements in the complex array in FFTW format.
 *
 * This macro is used to generate loops for the volume in an easy way. It
 * defines internal indexes 'k','i' and 'j' which ranges the volume using its
 * physical definition. It also defines 'kp', 'ip' and 'jp', which are the logical coordinates
 * It also works for 1D or 2D FFTW transforms
 *
 * @code
 * FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(V)
 * {
 *     int r2 = jp*jp + ip*ip + kp*kp;
 *
 *     std::cout << "element at physical coords: "<< i<<" "<<j<<" "<<k<<" has value: "<<DIRECT_A3D_ELEM(m, k, i, j) << std::endl;
 *     std::cout << "its logical coords are: "<< ip<<" "<<jp<<" "<<kp<<std::endl;
 *     std::cout << "its distance from the origin = "<<sqrt(r2)<<std::endl;
 *
 * }
 * @endcode
 */
#define FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(V) \
    for (long int k = 0, kp = 0; k<ZSIZE(V); k++, kp = (k < XSIZE(V)) ? k : k - ZSIZE(V)) \
    	for (long int i = 0, ip = 0 ; i<YSIZE(V); i++, ip = (i < XSIZE(V)) ? i : i - YSIZE(V)) \
    		for (long int j = 0, jp = 0; j<XSIZE(V); j++, jp = j)

/** For all direct elements in the complex array in FFTW format.
 *  The same as above, but now only for 2D images (this saves some time as k is not sampled
 */
#define FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM2D(V) \
	for (long int i = 0, ip = 0 ; i<YSIZE(V); i++, ip = (i < XSIZE(V)) ? i : i - YSIZE(V)) \
		for (long int j = 0, jp = 0; j<XSIZE(V); j++, jp = j)

/** FFTW volume element: Logical access.
 *
 * @code
 *
 * FFTW_ELEM(V, -1, -2, 1) = 1;
 * val = FFTW_ELEM(V, -1, -2, 1);
 * @endcode
 */
#define FFTW_ELEM(V, kp, ip, jp) \
    (DIRECT_A3D_ELEM((V),((kp < 0) ? (kp + ZSIZE(V)) : (kp)), ((ip < 0) ? (ip + YSIZE(V)) : (ip)), (jp)))

/** FFTW 2D image element: Logical access.
 *
 * @code
 *
 * FFTW2D_ELEM(V, --2, 1) = 1;
 * val = FFTW2D_ELEM(V, -2, 1);
 * @endcode
 */
#define FFTW2D_ELEM(V, ip, jp) \
    (DIRECT_A2D_ELEM((V), ((ip < 0) ? (ip + YSIZE(V)) : (ip)), (jp)))

/** Fourier Transformer class.
 * @ingroup FourierW
 *
 * The memory for the Fourier transform is handled by this object.
 * However, the memory for the real space image is handled externally
 * and this object only has a pointer to it.
 *
 * Here you have an example of use
 * @code
 * FourierTransformer transformer;
 * MultidimArray< Complex > Vfft;
 * transformer.FourierTransform(V(),Vfft,false);
 * MultidimArray<RFLOAT> Vmag;
 * Vmag.resize(Vfft);
 * FOR_ALL_ELEMENTS_IN_ARRAY3D(Vmag)
 *     Vmag(k,i,j)=20*log10(abs(Vfft(k,i,j)));
 * @endcode
 */
class FourierTransformer
{
public:
    /** Real array, in fact a pointer to the user array is stored. */
    MultidimArray<RFLOAT> *fReal;

     /** Complex array, in fact a pointer to the user array is stored. */
    MultidimArray<Complex > *fComplex;

    /** Fourier array  */
    MultidimArray< Complex > fFourier;

#ifdef RELION_SINGLE_PRECISION
    /* fftw Forward plan */
    fftwf_plan fPlanForward;

    /* fftw Backward plan */
    fftwf_plan fPlanBackward;
#else
    /* fftw Forward plan */
    fftw_plan fPlanForward;

    /* fftw Backward plan */
    fftw_plan fPlanBackward;
#endif

    bool plans_are_set;

// Public methods
public:
    /** Default constructor */
    FourierTransformer();

    /** Destructor */
    ~FourierTransformer();

    /** Copy constructor
     *
     * The created FourierTransformer is a perfect copy of the input array but with a
     * different memory assignment.
     *
     */
    FourierTransformer(const FourierTransformer& op);

    /** Compute the Fourier transform of a MultidimArray, 2D and 3D.
        If getCopy is false, an alias to the transformed data is returned.
        This is a faster option since a copy of all the data is avoided,
        but you need to be careful that an inverse Fourier transform may
        change the data.
        */
    template <typename T, typename T1>
        void FourierTransform(T& v, T1& V, bool getCopy=true)
        {
            setReal(v);
            Transform(FFTW_FORWARD);
            if (getCopy) getFourierCopy(V);
            else         getFourierAlias(V);
        }
    
    /** Compute the Fourier transform of a MultidimArray, 2D and 3D.
        If getCopy is false, an alias to the transformed data is returned.
        This is a faster option since a copy of all the data is avoided,
        but you need to be careful that an inverse Fourier transform may
        change the data.
        */
    template <typename T, typename T1>
        void FourierTransform(T& v, T1& V, int nr_threads, bool getCopy=true)
        {
            setReal(v, nr_threads);
            Transform(FFTW_FORWARD);
            if (getCopy) getFourierCopy(V);
            else         getFourierAlias(V);
        }

    /** Compute the Fourier transform.
        The data is taken from the matrix with which the object was
        created. */
    void FourierTransform();

    /** Inforce Hermitian symmetry.
        If the Fourier transform risks of losing Hermitian symmetry,
        use this function to renforce it. */
    void enforceHermitianSymmetry();

    /** Compute the inverse Fourier transform.
        The result is stored in the same real data that was passed for
        the forward transform. The Fourier coefficients are taken from
        the internal Fourier coefficients */
    void inverseFourierTransform();

    /** Compute the inverse Fourier transform.
        New data is provided for the Fourier coefficients and the output
        can be any matrix1D, 2D or 3D. It is important that the output
        matrix is already resized to the right size before entering
        in this function. */
    template <typename T, typename T1>
        void inverseFourierTransform(T& V, T1& v)
        {
            setReal(v);
            setFourier(V);
            Transform(FFTW_BACKWARD);
        }

    template <typename T, typename T1>
        void inverseFourierTransform(T& V, T1& v, int nr_threads)
        {
            setReal(v, nr_threads);
            setFourier(V);
            Transform(FFTW_BACKWARD);
        }

    /** Get Fourier coefficients. */
    template <typename T>
        void getFourierAlias(T& V) {V.alias(fFourier); return;}

    /** Get Fourier coefficients. */
    MultidimArray< Complex>& getFourierReference() {return fFourier;}

    /** Get Fourier coefficients. */
    template <typename T>
        void getFourierCopy(T& V) {
            V.reshape(fFourier);
            memcpy(MULTIDIM_ARRAY(V),MULTIDIM_ARRAY(fFourier),
                MULTIDIM_SIZE(fFourier)*2*sizeof(RFLOAT));
        }

    /** Return a complete Fourier transform (two halves).
    */
    template <typename T>
        void getCompleteFourier(T& V) {
            V.reshape(*fReal);
            int ndim=3;
            if (ZSIZE(*fReal)==1)
            {
                ndim=2;
                if (YSIZE(*fReal)==1)
                    ndim=1;
            }
            switch (ndim)
            {
                case 1:
                    FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY1D(V)
                        if (i<XSIZE(fFourier))
                            DIRECT_A1D_ELEM(V,i)=DIRECT_A1D_ELEM(fFourier,i);
                        else
                            DIRECT_A1D_ELEM(V,i)=
                                conj(DIRECT_A1D_ELEM(fFourier,
                                    XSIZE(*fReal)-i));
                    break;
                case 2:
                    FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(V)
                        if (j<XSIZE(fFourier))
                            DIRECT_A2D_ELEM(V,i,j)=
                                DIRECT_A2D_ELEM(fFourier,i,j);
                        else
                            DIRECT_A2D_ELEM(V,i,j)=
                                conj(DIRECT_A2D_ELEM(fFourier,
                                    (YSIZE(*fReal)-i)%YSIZE(*fReal),
                                     XSIZE(*fReal)-j));
                    break;
                case 3:
                    FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(V)
                        if (j<XSIZE(fFourier))
                            DIRECT_A3D_ELEM(V,k,i,j)=
                                DIRECT_A3D_ELEM(fFourier,k,i,j);
                        else
                            DIRECT_A3D_ELEM(V,k,i,j)=
                                conj(DIRECT_A3D_ELEM(fFourier,
                                    (ZSIZE(*fReal)-k)%ZSIZE(*fReal),
                                    (YSIZE(*fReal)-i)%YSIZE(*fReal),
                                     XSIZE(*fReal)-j));
                    break;
            }
        }

    /** Set one half of the FT in fFourier from the input complete Fourier transform (two halves).
        The fReal and fFourier already should have the right sizes
    */
    template <typename T>
        void setFromCompleteFourier(T& V) {
        int ndim=3;
        if (ZSIZE(*fReal)==1)
        {
            ndim=2;
            if (YSIZE(*fReal)==1)
                ndim=1;
        }
        switch (ndim)
        {
        case 1:
            FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY1D(fFourier)
                DIRECT_A1D_ELEM(fFourier,i)=DIRECT_A1D_ELEM(V,i);
            break;
        case 2:
            FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(fFourier)
                DIRECT_A2D_ELEM(fFourier,i,j) = DIRECT_A2D_ELEM(V,i,j);
            break;
        case 3:
            FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(fFourier)
                DIRECT_A3D_ELEM(fFourier,k,i,j) = DIRECT_A3D_ELEM(V,k,i,j);
            break;
        }
    }

// Internal methods
public:
    /* Pointer to the array of RFLOATs with which the plan was computed */
    RFLOAT * dataPtr;

    /* Pointer to the array of complex<RFLOAT> with which the plan was computed */
    Complex * complexDataPtr;

    /* Initialise all pointers to NULL */
    void init();

    /** Clear object */
    void clear();

    /** This calls fftw_cleanup.
    */
    void cleanup();

    void cleanup_threads();

    /** Destroy both forward and backward fftw planes (mutex locked */
    void destroyPlans();

    /** Computes the transform, specified in Init() function
        If normalization=true the forward transform is normalized
        (no normalization is made in the inverse transform)
        If normalize=false no normalization is performed and therefore
        the image is scaled by the number of pixels.
    */
    void Transform(int sign);

    /** Get the Multidimarray that is being used as input. */
    const MultidimArray<RFLOAT> &getReal() const;
    const MultidimArray<Complex > &getComplex() const;

    /** Set a Multidimarray for input.
        The data of img will be the one of fReal. In forward
        transforms it is not modified, but in backward transforms,
        the result will be stored in img. This means that the size
        of img cannot change between calls. */
    void setReal(MultidimArray<RFLOAT> &img);

    void setReal(MultidimArray<RFLOAT> &img, int nr_threads);

    /** Set a Multidimarray for input.
        The data of img will be the one of fComplex. In forward
        transforms it is not modified, but in backward transforms,
        the result will be stored in img. This means that the size
        of img cannot change between calls. */
    void setReal(MultidimArray<Complex > &img);

    /** Set a Multidimarray for the Fourier transform.
        The values of the input array are copied in the internal array.
        It is assumed that the container for the real image as well as
        the one for the Fourier array are already resized.
        No plan is updated. */
    void setFourier(MultidimArray<Complex > &imgFourier);
};

// Randomize phases beyond the given shell (index)
void randomizePhasesBeyond(MultidimArray<RFLOAT> &I, int index);

/** Center an array, to have its origin at the origin of the FFTW
 *
 */
template <typename T>
void CenterFFT(MultidimArray< T >& v, bool forward)
{
    if ( v.getDim() == 1 )
    {
        // 1D
        MultidimArray< T > aux;
        int l, shift;

        l = XSIZE(v);
        aux.reshape(l);
        shift = (int)(l / 2);

        if (!forward)
            shift = -shift;

        // Shift the input in an auxiliar vector
        for (int i = 0; i < l; i++)
        {
            int ip = i + shift;

            if (ip < 0)
                ip += l;
            else if (ip >= l)
                ip -= l;

            DIRECT_A1D_ELEM(aux, ip) = DIRECT_A1D_ELEM(v, i);
        }

        // Copy the vector
        for (int i = 0; i < l; i++)
            DIRECT_A1D_ELEM(v, i) = DIRECT_A1D_ELEM(aux, i);
    }
    else if ( v.getDim() == 2 )
    {
        // 2D
        MultidimArray< T > aux;
        int l, shift, sshift;

        // Shift in the X direction
        l = XSIZE(v);
        aux.reshape(l);
        sshift = shift = (int)(l / 2);

        if (!forward)
            sshift = -shift;

        for (int i = 0; i < YSIZE(v); i++)
        {
            // Shift the input in an auxiliar vector
            for (int j = 0; j < l; j++)
            {
                int jp = j + sshift;

                if (jp < 0)
                    jp += l;
                else if (jp >= l)
                    jp -= l;

                DIRECT_A1D_ELEM(aux, jp) = DIRECT_A2D_ELEM(v, i, j);
            }

            // Copy the vector
            for (int j = 0; j < l; j++)
                DIRECT_A2D_ELEM(v, i, j) = DIRECT_A1D_ELEM(aux, j);
        }

        // Shift in the Y direction
        l = YSIZE(v);
        //aux.reshape(l);
        sshift = shift = (int)(l / 2);

        if (!forward)
            sshift = -shift;
        for (int i = 0; i < shift; i++){
            int ip = i + sshift;
            if(ip < 0) ip += l;
            else if(ip >= l) ip -= l;
            //swap v[i, :] and v[ip, :]
            //std::cout << i << ", " << ip << std::endl;
            std::copy(v.data + ip*XSIZE(v), v.data + (ip+1)*XSIZE(v), aux.data);
            std::copy(v.data + i*XSIZE(v), v.data + (i+1)*XSIZE(v), v.data + ip*XSIZE(v));
            std::copy(aux.data, aux.data + XSIZE(v), v.data + i*XSIZE(v));
        }
        if(l % 2){
            if(sshift > 0){
                //copy v[shift, :]
                std::copy(v.data, v.data + XSIZE(v), aux.data);
                //shift the vectors in v[1:shift-1, :] left
                for(int i = 1; i < shift; i++){
                    std::copy(v.data + i*XSIZE(v), v.data + (i+1)*XSIZE(v), v.data + (i-1)*XSIZE(v));
                }
                //copy v[l-1, :] to v[shift-1, :]
                std::copy(v.data + (l-1)*XSIZE(v), v.data + l*XSIZE(v), v.data + (shift-1)*XSIZE(v));
                //copy aux to v[l-1, :]
                std::copy(aux.data, aux.data + XSIZE(v), v.data + (l-1)*XSIZE(v));
            } else {
                //backup v[shift, :]
                std::copy(v.data + shift*XSIZE(v), v.data + (shift+1)*XSIZE(v), aux.data);
                //shift the vectors in v[k, 0:shift-1, :] right
                for(int i = shift-1; i >= 0; i--){
                    std::copy(v.data + i*XSIZE(v), v.data + (i+1)*XSIZE(v), v.data + (i+1)*XSIZE(v));
                }
                //copy aux to v[0, :]
                std::copy(aux.data, aux.data + XSIZE(v), v.data);
            }

        }

        //for (int j = 0; j < XSIZE(v); j++)
        //{
        //    // Shift the input in an auxiliar vector
        //    for (int i = 0; i < l; i++)
        //    {
        //        int ip = i + shift;

        //        if (ip < 0)
        //            ip += l;
        //        else if (ip >= l)
        //            ip -= l;

        //        aux(ip) = DIRECT_A2D_ELEM(v, i, j);
        //    }

        //    // Copy the vector
        //    for (int i = 0; i < l; i++)
        //        DIRECT_A2D_ELEM(v, i, j) = DIRECT_A1D_ELEM(aux, i);
        //}
    }
    else if ( v.getDim() == 3 )
    {
        // 3D
        MultidimArray< T > aux;
        int l, shift, sshift;

        // Shift in the X direction
        l = XSIZE(v);
        aux.reshape(l);
        sshift = shift = (int)(l / 2);

        if (!forward)
            sshift = -shift;

        for (int k = 0; k < ZSIZE(v); k++)
            for (int i = 0; i < YSIZE(v); i++)
            {
                // Shift the input in an auxiliar vector
                for (int j = 0; j < l; j++)
                {
                    int jp = j + sshift;

                    if (jp < 0)
                        jp += l;
                    else if (jp >= l)
                        jp -= l;

                    DIRECT_A1D_ELEM(aux, jp) = DIRECT_A3D_ELEM(v, k, i, j);
                }

                // Copy the vector
                std::copy(aux.data, aux.data + l, v.data + k*YXSIZE(v)+i*XSIZE(v));
                //for (int j = 0; j < l; j++)
                //    DIRECT_A3D_ELEM(v, k, i, j) = DIRECT_A1D_ELEM(aux, j);
            }

        // Shift in the Y direction
        l = YSIZE(v);
        //aux.reshape(l);
        sshift = shift = (int)(l / 2);

        if (!forward)
            sshift = -shift;

        for (int k = 0; k < ZSIZE(v); k++)
            for (int i = 0; i < shift; i++){
                int ip = i + sshift;
                if(ip < 0) ip += l;
                else if(ip >= l) ip -= l;
                //swap v[k, i, :] and v[k, ip, :]
                std::copy(v.data + k*YXSIZE(v)+ip*XSIZE(v), v.data + k*YXSIZE(v)+(ip+1)*XSIZE(v), aux.data);
                std::copy(v.data + k*YXSIZE(v)+i*XSIZE(v), v.data + k*YXSIZE(v)+(i+1)*XSIZE(v), v.data + k*YXSIZE(v) + ip*XSIZE(v));
                std::copy(aux.data, aux.data + XSIZE(v), v.data + k*YXSIZE(v)+i*XSIZE(v));
            }
        if(l % 2){
            if(sshift > 0){
                for(int k = 0; k < ZSIZE(v); k++){
                    //copy v[k, 0, :]
                    std::copy(v.data + k*YXSIZE(v), v.data + k*YXSIZE(v) + XSIZE(v), aux.data);
                    //shift the vectors in v[k, 1:shift-1, :] left
                    for(int i = 1; i < shift; i++){
                        std::copy(v.data + k*YXSIZE(v) + i*XSIZE(v), v.data + k*YXSIZE(v)+(i+1)*XSIZE(v), v.data + k*YXSIZE(v) + (i-1)*XSIZE(v));
                    }
                    //copy v[k, l-1, :] to v[k, shift-1, :]
                    std::copy(v.data + k*YXSIZE(v) + (l-1)*XSIZE(v), v.data + k*YXSIZE(v)+l*XSIZE(v), v.data + k*YXSIZE(v) + (shift-1)*XSIZE(v));
                    //copy aux to v[k, l-1, :]
                    std::copy(aux.data, aux.data + XSIZE(v), v.data + k*YXSIZE(v) + (l-1)*XSIZE(v));
                }
            } else {
                for(int k = 0; k < ZSIZE(v); k++){
                    //backup v[k, shift, :]
                    std::copy(v.data + k*YXSIZE(v) + shift*XSIZE(v), v.data + k*YXSIZE(v) + (shift+1)*XSIZE(v), aux.data);
                    //shift the vectors in v[k, 0:shift-1, :] right
                    for(int i = shift-1; i >= 0; i--){
                        std::copy(v.data + k*YXSIZE(v) + i*XSIZE(v), v.data + k*YXSIZE(v)+(i+1)*XSIZE(v), v.data + k*YXSIZE(v) + (i+1)*XSIZE(v));
                    }
                    //copy aux to v[k, 0, :]
                    std::copy(aux.data, aux.data + XSIZE(v), v.data + k*YXSIZE(v));
                }

            }

        }

        //for (int k = 0; k < ZSIZE(v); k++)
        //    for (int j = 0; j < XSIZE(v); j++)
        //    {
        //        // Shift the input in an auxiliar vector
        //        for (int i = 0; i < l; i++)
        //        {
        //            int ip = i + shift;

        //            if (ip < 0)
        //                ip += l;
        //            else if (ip >= l)
        //                ip -= l;

        //            aux(ip) = DIRECT_A3D_ELEM(v, k, i, j);
        //        }

        //        // Copy the vector
        //        for (int i = 0; i < l; i++)
        //            DIRECT_A3D_ELEM(v, k, i, j) = DIRECT_A1D_ELEM(aux, i);
        //    }

        // Shift in the Z direction
        l = ZSIZE(v);
        //aux.reshape(XSIZE(v));
        sshift = shift = (int)(l / 2);

        if (!forward)
            sshift = -shift;
        
        for (int k = 0; k < shift; k++) {
            int kp = k + sshift;
            if(kp < 0) kp += l;
            else if(kp >= l) kp -= l;
            for (int i = 0; i < YSIZE(v); i++){
                //swap v[k, i, :] and v[kp, i, :]
                std::copy(v.data + k*YXSIZE(v)+i*XSIZE(v), v.data + k*YXSIZE(v)+(i+1)*XSIZE(v), aux.data);
                std::copy(v.data + kp*YXSIZE(v)+i*XSIZE(v), v.data + kp*YXSIZE(v)+(i+1)*XSIZE(v), v.data + k*YXSIZE(v) + i*XSIZE(v));
                std::copy(aux.data, aux.data + XSIZE(v), v.data + kp*YXSIZE(v)+i*XSIZE(v));
            }
        }

        if(l % 2){
            if(sshift > 0){
                for(int i = 0; i < YSIZE(v); i++){
                    //copy v[0, i, :]
                    std::copy(v.data + i*XSIZE(v), v.data + (i+1)*XSIZE(v), aux.data);
                    //shift the vectors in v[1:shift-1, i, :] left
                    for(int k = 1; k < shift; k++){
                        std::copy(v.data + k*YXSIZE(v) + i*XSIZE(v), v.data + k*YXSIZE(v)+(i+1)*XSIZE(v), v.data + (k-1)*YXSIZE(v) + i*XSIZE(v));
                    }
                    //copy v[l-1, i, :] to v[shift-1, i, :]
                    std::copy(v.data + (l-1)*YXSIZE(v) + i*XSIZE(v), v.data + (l-1)*YXSIZE(v)+(i+1)*XSIZE(v), v.data + (shift-1)*YXSIZE(v) + i*XSIZE(v));
                    //copy aux to v[l-1, i, :]
                    std::copy(aux.data, aux.data + XSIZE(v), v.data + (l-1)*YXSIZE(v) + i*XSIZE(v));
                }
            } else {
                for(int i = 0; i < YSIZE(v); i++){
                    //backup v[shift, i, :]
                    std::copy(v.data + shift*YXSIZE(v) + i*XSIZE(v), v.data + shift*YXSIZE(v) + (i+1)*XSIZE(v), aux.data);
                    //shift the vectors in v[0:shift-1, i, :] right
                    for(int k = shift-1; k >= 0; k--){
                        std::copy(v.data + k*YXSIZE(v) + i*XSIZE(v), v.data + k*YXSIZE(v)+(i+1)*XSIZE(v), v.data + (k+1)*YXSIZE(v) + i*XSIZE(v));
                    }
                    //copy aux to v[0, i, :]
                    std::copy(aux.data, aux.data + XSIZE(v), v.data + i*XSIZE(v));
                }

            }

        }


        //for (int i = 0; i < YSIZE(v); i++)
        //    for (int j = 0; j < XSIZE(v); j++)
        //    {
        //        // Shift the input in an auxiliar vector
        //        for (int k = 0; k < l; k++)
        //        {
        //            int kp = k + shift;
        //            if (kp < 0)
        //                kp += l;
        //            else if (kp >= l)
        //                kp -= l;

        //            aux(kp) = DIRECT_A3D_ELEM(v, k, i, j);
        //        }

        //        // Copy the vector
        //        for (int k = 0; k < l; k++)
        //            DIRECT_A3D_ELEM(v, k, i, j) = DIRECT_A1D_ELEM(aux, k);
        //    }
    }
    else
    {
    	v.printShape();
    	REPORT_ERROR("CenterFFT ERROR: Dimension should be 1, 2 or 3");
    }
}


// Window an FFTW-centered Fourier-transform to a given size
template<class T>
void windowFourierTransform(MultidimArray<T > &in,
                              MultidimArray<T > &out,
                              long int newdim)
{
    // Check size of the input array
    if (YSIZE(in) > 1 && YSIZE(in)/2 + 1 != XSIZE(in))
        REPORT_ERROR("windowFourierTransform ERROR: the Fourier transform should be of an image with equal sizes in all dimensions!");
    long int newhdim = newdim/2 + 1;

    // If same size, just return input
    if (newhdim == XSIZE(in))
    {
        out = in;
        return;
    }

    // Otherwise apply a windowing operation
    // Initialise output array
    switch (in.getDim())
    {
    case 1:
        out.initZeros(newhdim);
        break;
    case 2:
        out.initZeros(newdim, newhdim);
        break;
    case 3:
        out.initZeros(newdim, newdim, newhdim);
        break;
    default:
        REPORT_ERROR("windowFourierTransform ERROR: dimension should be 1, 2 or 3!");
    }
    if (newhdim > XSIZE(in))
    {
        long int max_r2 = (XSIZE(in) -1) * (XSIZE(in) - 1);
        FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(in)
        {
            // Make sure windowed FT has nothing in the corners, otherwise we end up with an asymmetric FT!
            if (kp*kp + ip*ip + jp*jp <= max_r2)
                FFTW_ELEM(out, kp, ip, jp) = FFTW_ELEM(in, kp, ip, jp);
        }
    }
    else
    {
        FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(out)
        {
            FFTW_ELEM(out, kp, ip, jp) = FFTW_ELEM(in, kp, ip, jp);
        }
    }
}

// Same as above, acts on the input array directly
template<class T>
void windowFourierTransform(MultidimArray<T > &V,
                              long int newdim)
{
    // Check size of the input array
    if (YSIZE(V) > 1 && YSIZE(V)/2 + 1 != XSIZE(V))
        REPORT_ERROR("windowFourierTransform ERROR: the Fourier transform should be of an image with equal sizes in all dimensions!");
    long int newhdim = newdim/2 + 1;

    // If same size, just return input
    if (newhdim == XSIZE(V))
    {
        return;
    }

    MultidimArray<T> tmp;
    windowFourierTransform<T>(V, tmp, newdim);
    V.moveFrom(tmp);

}

// A resize operation in Fourier-space (i.e. changing the sampling of the Fourier Transform) by windowing in real-space
// If recenter=true, the real-space array will be recentered to have its origin at the origin of the FT
template<class T>
void resizeFourierTransform(MultidimArray<T > &in,
			  			    MultidimArray<T > &out,
			  			    long int newdim, bool do_recenter=true)
{
	// Check size of the input array
	if (YSIZE(in) > 1 && YSIZE(in)/2 + 1 != XSIZE(in))
		REPORT_ERROR("windowFourierTransform ERROR: the Fourier transform should be of an image with equal sizes in all dimensions!");
	long int newhdim = newdim/2 + 1;
	long int olddim = 2* (XSIZE(in) - 1);

	// If same size, just return input
	if (newhdim == XSIZE(in))
	{
		out = in;
		return;
	}

	// Otherwise apply a windowing operation
	MultidimArray<Complex > Fin;
	MultidimArray<RFLOAT> Min;
	FourierTransformer transformer;
	long int x0, y0, z0, xF, yF, zF;
	x0 = y0 = z0 = FIRST_XMIPP_INDEX(newdim);
	xF = yF = zF = LAST_XMIPP_INDEX(newdim);

	// Initialise output array
	switch (in.getDim())
	{
	case 1:
		Min.reshape(olddim);
		y0=yF=z0=zF=0;
		break;
	case 2:
		Min.reshape(olddim, olddim);
		z0=zF=0;
		break;
	case 3:
		Min.reshape(olddim, olddim, olddim);
		break;
	default:
    	REPORT_ERROR("resizeFourierTransform ERROR: dimension should be 1, 2 or 3!");
    }

	// This is to handle RFLOAT-valued input arrays
	Fin.reshape(ZSIZE(in), YSIZE(in), XSIZE(in));
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(in)
	{
		DIRECT_MULTIDIM_ELEM(Fin, n) = DIRECT_MULTIDIM_ELEM(in, n);
	}
	transformer.inverseFourierTransform(Fin, Min);
	Min.setXmippOrigin();
	if (do_recenter)
		CenterFFT(Min, false);

	// Now do the actual windowing in real-space
	Min.window(z0, y0, x0, zF, yF, xF);
	Min.setXmippOrigin();

	// If upsizing: mask the corners to prevent aliasing artefacts
	if (newdim > olddim)
	{
		FOR_ALL_ELEMENTS_IN_ARRAY3D(Min)
		{
			if (k*k + i*i + j*j > olddim*olddim/4)
			{
				A3D_ELEM(Min, k, i, j) = 0.;
			}
		}
	}

	// Recenter FFT back again
	if (do_recenter)
		CenterFFT(Min, true);

	// And do the inverse Fourier transform
	transformer.clear();
	transformer.FourierTransform(Min, out);
}

/** Fourier-Ring-Correlation between two multidimArrays using FFT
 * From precalculated Fourier Transforms
 * Simpler I/O than above.
 */
void getFSC(MultidimArray< Complex > &FT1,
		    MultidimArray< Complex > &FT2,
		    MultidimArray< RFLOAT > &fsc);

/** Fourier-Ring-Correlation between two multidimArrays using FFT
 * @ingroup FourierOperations
 * Simpler I/O than above.
 */
void getFSC(MultidimArray< RFLOAT > & m1,
		    MultidimArray< RFLOAT > & m2,
		    MultidimArray< RFLOAT > &fsc);

void getAmplitudeCorrelationAndDifferentialPhaseResidual(MultidimArray< Complex > &FT1,
		    MultidimArray< Complex > &FT2,
		    MultidimArray< RFLOAT > &acorr,
		    MultidimArray< RFLOAT > &dpr);

void getAmplitudeCorrelationAndDifferentialPhaseResidual(MultidimArray< RFLOAT > &m1,
		    MultidimArray< RFLOAT > &m2,
		    MultidimArray< RFLOAT > &acorr,
		    MultidimArray< RFLOAT > &dpr);

// Get precalculated AB-matrices for on-the-fly shift calculations (without tabulated sine and cosine)
void getAbMatricesForShiftImageInFourierTransform(MultidimArray<Complex > &in,
									MultidimArray<Complex > &out,
									RFLOAT oridim, RFLOAT shift_x, RFLOAT shift_y, RFLOAT shift_z = 0.);

void shiftImageInFourierTransformWithTabSincos(MultidimArray<Complex > &in,
									MultidimArray<Complex > &out,
									RFLOAT oridim, long int newdim,
									TabSine& tabsin, TabCosine& tabcos,
									RFLOAT xshift, RFLOAT yshift, RFLOAT zshift = 0.);

// Shift an image through phase-shifts in its Fourier Transform (without tabulated sine and cosine)
// Note that in and out may be the same array, in that case in is overwritten with the result
// if oridim is in pixels, xshift, yshift and zshift should be in pixels as well!
// or both can be in Angstroms
void shiftImageInFourierTransform(MultidimArray<Complex > &in,
						          MultidimArray<Complex > &out,
								  RFLOAT oridim, RFLOAT shift_x, RFLOAT shift_y, RFLOAT shift_z = 0.);

#define POWER_SPECTRUM 0
#define AMPLITUDE_SPECTRUM 1
#define AMPLITUDE_MAP 0
#define PHASE_MAP 1

/** Get the amplitude or power_class spectrum of the map in Fourier space.
 * @ingroup FourierOperations
    i.e. the radial average of the (squared) amplitudes of all Fourier components
*/
void getSpectrum(MultidimArray<RFLOAT> &Min,
                 MultidimArray<RFLOAT> &spectrum,
                 int spectrum_type=POWER_SPECTRUM);

/** Divide the input map in Fourier-space by the spectrum provided.
 * @ingroup FourierOperations
    If leave_origin_intact==true, the origin pixel will remain untouched
*/
void divideBySpectrum(MultidimArray<RFLOAT> &Min,
                      MultidimArray<RFLOAT> &spectrum,
                      bool leave_origin_intact=false);

/** Multiply the input map in Fourier-space by the spectrum provided.
 * @ingroup FourierOperations
    If leave_origin_intact==true, the origin pixel will remain untouched
*/
void multiplyBySpectrum(MultidimArray<RFLOAT> &Min,
                        MultidimArray<RFLOAT> &spectrum,
                        bool leave_origin_intact=false);

/** Perform a whitening of the amplitude/power_class spectrum of a 3D map
 * @ingroup FourierOperations
    If leave_origin_intact==true, the origin pixel will remain untouched
*/
void whitenSpectrum(MultidimArray<RFLOAT> &Min,
                    MultidimArray<RFLOAT> &Mout,
                    int spectrum_type=AMPLITUDE_SPECTRUM,
                    bool leave_origin_intact=false);

/** Adapts Min to have the same spectrum as spectrum_ref
 * @ingroup FourierOperations
    If only_amplitudes==true, the amplitude rather than the power_class spectrum will be equalized
*/
void adaptSpectrum(MultidimArray<RFLOAT> &Min,
                   MultidimArray<RFLOAT> &Mout,
                   const MultidimArray<RFLOAT> &spectrum_ref,
                   int spectrum_type=AMPLITUDE_SPECTRUM,
                   bool leave_origin_intact=false);

/** Kullback-Leibner divergence */
RFLOAT getKullbackLeibnerDivergence(MultidimArray<Complex > &Fimg,
		MultidimArray<Complex > &Fref, MultidimArray<RFLOAT> &sigma2,
		MultidimArray<RFLOAT> &p_i, MultidimArray<RFLOAT> &q_i,
		int highshell = -1, int lowshell = -1);


// Resize a map by windowing it's Fourier Transform
void resizeMap(MultidimArray<RFLOAT > &img, int newsize);

// Apply a B-factor to a map (given it's Fourier transform)
void applyBFactorToMap(MultidimArray<Complex > &FT, int ori_size, RFLOAT bfactor, RFLOAT angpix);

// Apply a B-factor to a map (given it's real-space array)
void applyBFactorToMap(MultidimArray<RFLOAT > &img, RFLOAT bfactor, RFLOAT angpix);

// Low-pass filter a map (given it's Fourier transform)
void lowPassFilterMap(MultidimArray<Complex > &FT, int ori_size,
		RFLOAT low_pass, RFLOAT angpix, int filter_edge_width = 2, bool do_highpass_instead = false);

// Low-pass and high-pass filter a map (given it's real-space array)
void lowPassFilterMap(MultidimArray<RFLOAT > &img, RFLOAT low_pass, RFLOAT angpix, int filter_edge_width = 2);
void highPassFilterMap(MultidimArray<RFLOAT > &img, RFLOAT low_pass, RFLOAT angpix, int filter_edge_width = 2);

/*
 *  Beamtilt x and y are given in mradians
 *  Wavelength in Angstrom, Cs in mm
 *  Phase shifts caused by the beamtilt will be calculated and applied to Fimg
 */
void selfApplyBeamTilt(MultidimArray<Complex > &Fimg, RFLOAT beamtilt_x, RFLOAT beamtilt_y,
		RFLOAT wavelength, RFLOAT Cs, RFLOAT angpix, int ori_size);

void applyBeamTilt(const MultidimArray<Complex > &Fin, MultidimArray<Complex > &Fout, RFLOAT beamtilt_x, RFLOAT beamtilt_y,
		RFLOAT wavelength, RFLOAT Cs, RFLOAT angpix, int ori_size);

void padAndFloat2DMap(const MultidimArray<RFLOAT > &v, MultidimArray<RFLOAT> &out, int factor = 2);

void amplitudeOrPhaseMap(const MultidimArray<RFLOAT > &v, MultidimArray<RFLOAT > &amp, int output_map_type);

void helicalLayerLineProfile(const MultidimArray<RFLOAT > &v, std::string title, std::string fn_eps);

#endif // __RELIONFFTW_H
