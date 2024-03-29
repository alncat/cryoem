# CMake helper to locate the needed libraries and headers
# for compilation of RELION binaries.
#

set(FFTW_EXTERNAL_PATH "${CMAKE_SOURCE_DIR}/external/fftw")
message(STATUS "${FFTW_EXTERNAL_PATH}")

if(DoublePrec_CPU)
   # set fftw lib to use double precision
	set(libfft "fftw3")
    set(libfft_t "fftw3_threads")
else(DoublePrec_CPU)
	# set fftw lib to use single precision
	set(libfft "fftw3f")
    set(libfft_t "fftw3f_threads")
endif(DoublePrec_CPU)	

set(LIB_PATHFFT $ENV{FFTW_LIB})
set(INC_PATHFFT $ENV{FFTW_INCLUDE})

unset(FFTW_PATH CACHE)
unset(FFTW_INCLUDES CACHE)
unset(FFTW_LIBRARIES CACHE)
unset(FFTW_LIBRARIES_THREADS CACHE)

find_library(FFTW_LIBRARIES  NAMES ${libfft} PATHS ${LIB_PATHFFT} $ENV{FFTW_LIB} $ENV{FFTW_HOME} ) 
find_library(FFTW_LIBRARIES_THREADS  NAMES ${libfft_t} PATHS ${LIB_PATHFFT} $ENV{FFTW_LIB} $ENV{FFTW_HOME} ) 

if(FFTW_LIBRARIES_THREADS)
    set(FFTW_LIBRARIES ${FFTW_LIBRARIES} ${FFTW_LIBRARIES_THREADS})
endif()

if(DEFINED ENV{FFTW_INCLUDE})
    find_path(FFTW_PATH     NAMES fftw3.h PATHS ${INC_PATHFFT} )
    find_path(FFTW_INCLUDES NAMES fftw3.h PATHS ${INC_PATHFFT} )
else()
    find_path(FFTW_PATH     NAMES fftw3.h )
    find_path(FFTW_INCLUDES NAMES fftw3.h )
endif()


if(FFTW_PATH AND FFTW_INCLUDES AND FFTW_LIBRARIES)
   set(FFTW_FOUND TRUE)
endif(FFTW_PATH AND FFTW_INCLUDES AND FFTW_LIBRARIES)

if (FFTW_FOUND)
	message(STATUS "Found FFTW: ${libfft} ${libfft_t}")
	message(STATUS "FFTW_LIBRARIES: ${FFTW_LIBRARIES}")
else(FFTW_FOUND)
	if(DoublePrec_CPU)
		message(STATUS "Double-precision FFTW was NOT found")
	else(DoublePrec_CPU)
		message(STATUS "Single-precision FFTW was NOT found")
	endif(DoublePrec_CPU)	
endif(FFTW_FOUND)

if(FFTW_FIND_REQUIRED AND NOT FFTW_FOUND)
	message( FATAL_ERROR "FFTW is required." )
endif(FFTW_FIND_REQUIRED AND NOT FFTW_FOUND)
