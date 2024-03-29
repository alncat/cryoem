include_directories("${CMAKE_SOURCE_DIR}")

include_directories("${FFTW_INCLUDES}")
#include_directories(${CMAKE_BINARY_DIR}/include)

find_path (X11_INCLUDES Xdbe.h)

message(STATUS "CMAKE_BINARY_DIR:" ${CMAKE_BINARY_DIR})

#paddle
#if(NOT DEFINED PADDLE_LIB)
#  message(FATAL_ERROR "please set PADDLE_LIB with -DPADDLE_LIB=/paddle/lib/dir")
#endif()
#
#option(WITH_MKLDNN     "Compile PaddlePaddle with MKLDNN"                                   OFF)
#option(WITH_MKL        "Compile PaddlePaddle with MKL support, default use openblas."       OFF)
#
#include_directories("${PADDLE_LIB}")
#include_directories("${PADDLE_LIB}/third_party/install/protobuf/include")
#include_directories("${PADDLE_LIB}/third_party/install/glog/include")
#include_directories("${PADDLE_LIB}/third_party/install/gflags/include")
#include_directories("${PADDLE_LIB}/third_party/install/xxhash/include")
#include_directories("${PADDLE_LIB}/third_party/install/zlib/include")
#
#include_directories("${PADDLE_LIB}/third_party/boost")
#include_directories("${PADDLE_LIB}/third_party/eigen3")
#include_directories("${PADDLE_LIB}/third_party/threadpool")
#include_directories("${PADDLE_LIB}/third_party/dlpack")
#
#link_directories("${PADDLE_LIB}/third_party/install/protobuf/lib")
#link_directories("${PADDLE_LIB}/third_party/install/glog/lib")
#link_directories("${PADDLE_LIB}/third_party/install/gflags/lib")
#link_directories("${PADDLE_LIB}/third_party/install/xxhash/lib")
#link_directories("${PADDLE_LIB}/third_party/install/zlib/lib")
#
#if(WITH_MKLDNN)
#  include_directories("${PADDLE_LIB}/third_party/install/mkldnn/include")
#  if(WIN32)
#    set(MKLDNN_LIB ${PADDLE_LIB}/third_party/install/mkldnn/lib/mkldnn.lib)
#  else(WIN32)
#    set(MKLDNN_LIB ${PADDLE_LIB}/third_party/install/mkldnn/lib/libmkldnn.so.0)
#  endif(WIN32)
#endif(WITH_MKLDNN)
#
#if(WITH_MKL)
#  include_directories("${PADDLE_LIB}/third_party/install/mklml/include")
#  if(WIN32)
#    set(MATH_LIB ${PADDLE_LIB}/third_party/install/mklml/lib/mklml.lib)
#  else(WIN32)
#    set(MATH_LIB ${PADDLE_LIB}/third_party/install/mklml/lib/libmklml_intel.so)
#  endif(WIN32)
#else()
#  if(APPLE)
#    set(MATH_LIB cblas)
#  elseif(WIN32)
#    set(MATH_LIB ${PADDLE_LIB}/third_party/install/openblas/lib/libopenblas.lib)
#  else()
#    set(MATH_LIB ${PADDLE_LIB}/third_party/install/openblas/lib/libopenblas.a)
#  endif(APPLE)
#endif()
#
#if(APPLE)
#  set(MACOS_LD_FLAGS "-undefined dynamic_lookup -Wl,-all_load -framework CoreFoundation -framework Security")
#else(APPLE)
#  set(ARCHIVE_START "-Wl,--whole-archive")
#  set(ARCHIVE_END "-Wl,--no-whole-archive")
#  set(EXTERNAL_LIB "-lrt -ldl -lpthread")
#endif(APPLE)
#
##paddle

file(GLOB REL_SRC "${CMAKE_SOURCE_DIR}/src/*.cpp" "${CMAKE_SOURCE_DIR}/src/*.c" "${CMAKE_SOURCE_DIR}/src/gpu_utils/*.cpp")
file(GLOB REL_SRC_H "${CMAKE_SOURCE_DIR}/src/*.h" "${CMAKE_SOURCE_DIR}/src/gpu_utils/*.h")
file(GLOB REL_HP "${CMAKE_SOURCE_DIR}/src/Healpix_2.15a/*.cc")
#file(GLOB REL_VAE_H "${CMAKE_SOURCE_DIR}/src/vae/*.h")
#file(GLOB REL_VAE "${CMAKE_SOURCE_DIR}/src/vae/*.cpp")

file(GLOB RELION_TARGETS "${CMAKE_SOURCE_DIR}/src/apps/*.cpp")

#--Remove apps using X11 if no GUI--
if(NOT GUI)
    list(REMOVE_ITEM REL_SRC "${CMAKE_SOURCE_DIR}/src/manualpicker.cpp")
    list(REMOVE_ITEM REL_SRC "${CMAKE_SOURCE_DIR}/src/gui_entries.cpp")
    list(REMOVE_ITEM REL_SRC "${CMAKE_SOURCE_DIR}/src/gui_jobwindow.cpp")
    list(REMOVE_ITEM REL_SRC "${CMAKE_SOURCE_DIR}/src/gui_mainwindow.cpp")
    list(REMOVE_ITEM REL_SRC "${CMAKE_SOURCE_DIR}/src/displayer.cpp")
    list(REMOVE_ITEM RELION_TARGETS "${CMAKE_SOURCE_DIR}/src/apps/display.cpp")
    list(REMOVE_ITEM RELION_TARGETS "${CMAKE_SOURCE_DIR}/src/apps/maingui.cpp")
    list(REMOVE_ITEM RELION_TARGETS "${CMAKE_SOURCE_DIR}/src/apps/manualpick.cpp")
    list(REMOVE_ITEM RELION_TARGETS "${CMAKE_SOURCE_DIR}/src/apps/pipeliner.cpp")
endif(NOT GUI)


# relion_lib is STATIC or SHARED type based on BUILD_SHARED_LIBS=ON/OFF
if(BUILD_SHARED_LIBS)
    #add_library(vae_lib SHARED ${REL_VAE} ${REL_VAE_H})# ${tiny_dnn_headers})
    add_library(relion_lib SHARED ${REL_SRC} ${REL_SRC_H} ${REL_HP})
    install(TARGETS relion_lib LIBRARY DESTINATION lib)
    #install(TARGETS vae_lib LIBRARY DESTINATION lib)
else()
    add_library(relion_lib STATIC ${REL_SRC} ${REL_SRC_H} ${REL_HP})
    #add_library(vae_lib STATIC ${REL_VAE} ${REL_VAE_H})
endif()

#target_include_directories(vae_lib PUBLIC ${TinyDNN_INCLUDE_DIRS})
#target_include_directories(relion_lib PUBLIC ${TinyDNN_INCLUDE_DIRS})
#message("${TinyDNN_INCLUDE_DIRS}")
#target_link_libraries(vae_lib ${TORCH_LIBRARIES})
#target_link_libraries(vae_lib ${TinyDNN_LIBRARY_DIRS})
#target_link_libraries(vae_lib ${MACOS_LD_FLAGS}
#	${ARCHIVE_START}
#	${PADDLE_LIB}/paddle/fluid/inference/libpaddle_fluid.so
#	${ARCHIVE_END}
#    /ssd/lzw/paddle/build/paddle/fluid/platform/libflags.a
#	${MATH_LIB}
#	${MKLDNN_LIB}
#	glog gflags protobuf z xxhash
#	${EXTERNAL_LIB})
#
target_link_libraries(relion_lib ${FFTW_LIBRARIES})# vae_lib)# ${EXTERNAL_LIB})
target_link_directories(relion_lib PUBLIC ${MPI_LIBRARIES})
#target_link_libraries(relion_lib ${MACOS_LD_FLAGS}
#	${ARCHIVE_START}
#	${PADDLE_LIB}/paddle/fluid/inference/libpaddle_fluid.so
#	${ARCHIVE_END}
#    /ssd/lzw/paddle/build/paddle/fluid/platform/libflags.a
#	${MATH_LIB}
#	${MKLDNN_LIB}
#	glog gflags protobuf z xxhash
#	${EXTERNAL_LIB})

if(BUILD_OWN_FFTW)
    add_dependencies(relion_lib OWN_FFTW)
endif()

if(GUI)
    target_link_libraries(relion_lib ${FLTK_LIBRARIES})
    if(BUILD_OWN_FLTK)
        add_dependencies(relion_lib OWN_FLTK)
    endif()
endif(GUI)



if (CUDA_FOUND)

    file(GLOB REL_CUDA_SRC "${CMAKE_SOURCE_DIR}/src/gpu_utils/*.cu" "${CMAKE_SOURCE_DIR}/src/gpu_utils/cuda_kernels/*.cu")

    add_library(relion_gpu_util ${REL_CUDA_SRC})
    target_link_libraries(relion_gpu_util ${CUDA_LIBRARIES})

    if (${CMAKE_BUILD_TYPE_LOWER} STREQUAL "profiling")
        find_library(NV_TOOLS_LIBRARIES   NAMES nvToolsExt  PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
        list(APPEND EXTRA_LIBS "${NV_TOOLS_LIBRARIES}")
        target_link_libraries(relion_gpu_util ${NV_TOOLS_LIBRARIES})
        message(STATUS "Adding extra library for NVIDIA profiling: ${NV_TOOLS_LIBRARIES}")
    endif()

    #target_link_libraries(relion_gpu_util ${MACOS_LD_FLAGS}
	#${ARCHIVE_START}
	#${PADDLE_LIB}/paddle/fluid/inference/libpaddle_fluid.so
    #/ssd/lzw/paddle/build/paddle/fluid/platform/libflags.a
	#${ARCHIVE_END}
	#${MATH_LIB}
	#${MKLDNN_LIB}
	#glog gflags protobuf z xxhash
	#${EXTERNAL_LIB})

	# Presently we have a number of (bad) circular dependencies between the gpu util
	# and relion libraries, which cause errors at least on OS X with clang. Tell the
	# compiler to ignore them.
    if(APPLE)
        set(new_link_flags "-undefined suppress -flat_namespace")
        get_target_property(existing_link_flags relion_gpu_util LINK_FLAGS)
        if(existing_link_flags)
            set(new_link_flags "${existing_link_flags} ${new_link_flags}")
        endif()
        set_target_properties(relion_gpu_util PROPERTIES LINK_FLAGS "${new_link_flags}")
    endif()

    list(APPEND EXTRA_LIBS "${CUDA_CUFFT_LIBRARIES}")
    if(BUILD_SHARED_LIBS)
        install (TARGETS relion_gpu_util LIBRARY DESTINATION lib)
    else()
	    target_link_libraries(relion_gpu_util relion_lib)
	    target_link_libraries(relion_gpu_util ${CUDA_CUFFT_LIBRARIES})
    endif()

    #target_include_directories(relion_lib PUBLIC ${TORCH_INCLUDE_DIRS})
    #target_include_directories(relion_gpu_util PUBLIC ${TORCH_INCLUDE_DIRS})
    #target_include_directories(relion_gpu_util PUBLIC ${TinyDNN_INCLUDE_DIRS})
    target_include_directories(relion_gpu_util PUBLIC ${CUDA_INCLUDE_DIRS})
    target_link_libraries(relion_lib relion_gpu_util ${CUDA_CUFFT_LIBRARIES})# ${TORCH_LIBRARIES})

endif(CUDA_FOUND)

foreach (_target ${RELION_TARGETS})
	GET_FILENAME_COMPONENT(_target "relion_${_target}" NAME_WE)  #specify target name WE=WithoutExtension

    add_executable(${_target} ${_target}.cpp )
    set(LIB relion_lib)

 	add_dependencies(${_target} relion_lib)

	set_target_properties(${_target} PROPERTIES PREFIX "relion_")

    #target_include_directories(${_target} PUBLIC ${TORCH_INCLUDE_DIRS})

    target_link_libraries(${_target} ${LIB} ${FFTW_LIBRARIES} ${EXTRA_LIBS} ${MPI_LIBRARIES} ${CMAKE_DL_LIBS})# ${TORCH_LIBRARIES})
    #target_link_libraries(${_target} ${LIB} ${FFTW_LIBRARIES} ${EXTRA_LIBS} ${CMAKE_DL_LIBS})# ${TORCH_LIBRARIES})
    #target_link_directories(${_target} PUBLIC ${MPI_LIBRARIES})

	if(CUDA_FOUND)
	    target_link_libraries(${_target} relion_gpu_util)
	endif(CUDA_FOUND)

	if(GUI)
        include_directories("${FLTK_INCLUDE_DIR}")
		target_link_libraries(${_target} ${FLTK_LIBRARIES} ${X11})
	endif(GUI)

	message(STATUS "added ${_target}...")

	install (TARGETS ${_target} RUNTIME DESTINATION bin)
endforeach()

if(GUI)
    ADD_CUSTOM_COMMAND(TARGET maingui
              POST_BUILD
              COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_BINARY_DIR}/bin/relion_maingui ${PROJECT_BINARY_DIR}/bin/relion
	      COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_BINARY_DIR}/bin/relion_qsub.csh ${PROJECT_BINARY_DIR}/bin/qsub.csh
    )
endif(GUI)


