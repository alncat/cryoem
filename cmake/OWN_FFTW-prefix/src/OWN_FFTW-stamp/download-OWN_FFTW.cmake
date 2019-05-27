if(EXISTS "/home/zluo/work/relion/external/fftw/fftw-3.3.4.tar.gz")
  file("MD5" "/home/zluo/work/relion/external/fftw/fftw-3.3.4.tar.gz" hash_value)
  if("x${hash_value}" STREQUAL "x2edab8c06b24feeb3b82bbb3ebf3e7b3")
    return()
  endif()
endif()
message(STATUS "downloading...
     src='ftp://ftp.fftw.org/pub/fftw/fftw-3.3.4.tar.gz'
     dst='/home/zluo/work/relion/external/fftw/fftw-3.3.4.tar.gz'
     timeout='none'")




file(DOWNLOAD
  "ftp://ftp.fftw.org/pub/fftw/fftw-3.3.4.tar.gz"
  "/home/zluo/work/relion/external/fftw/fftw-3.3.4.tar.gz"
  SHOW_PROGRESS
  # no TIMEOUT
  STATUS status
  LOG log)

list(GET status 0 status_code)
list(GET status 1 status_string)

if(NOT status_code EQUAL 0)
  message(FATAL_ERROR "error: downloading 'ftp://ftp.fftw.org/pub/fftw/fftw-3.3.4.tar.gz' failed
  status_code: ${status_code}
  status_string: ${status_string}
  log: ${log}
")
endif()

message(STATUS "downloading... done")
