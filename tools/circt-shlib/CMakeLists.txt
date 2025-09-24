if (MSVC)
  return()
endif()

get_property(circt_libs GLOBAL PROPERTY CIRCT_ALL_LIBS)
list(REMOVE_DUPLICATES circt_libs)

list(FILTER circt_libs EXCLUDE REGEX "obj\.*")

foreach (lib ${circt_libs})
  if(XCODE)
    # Xcode doesn't support object libraries, so we have to trick it into
    # linking the static libraries instead.
    list(APPEND _DEPS "-force_load" ${lib})
  else()
    list(APPEND _OBJECTS $<TARGET_OBJECTS:obj.${lib}>)
  endif()
endforeach ()

if(CIRCT_LINK_CIRCT_DYLIB)
  set(INSTALL_WITH_TOOLCHAIN INSTALL_WITH_TOOLCHAIN)
endif()

set(ORTOOL_LIB "")
if(ortools_FOUND)
set(ORTOOLS_LIB ortools::ortools)
endif()

set(SLANG_LIB "")
if(CIRCT_SLANG_FRONTEND_ENABLED)
set(SLANG_LIB slang_slang)
endif()

add_circt_library(
  CIRCT
  SHARED
  ${INSTALL_WITH_TOOLCHAIN}
  circt-shlib.cpp
  ${_OBJECTS}
  LINK_LIBS
  ${_DEPS}
  MLIR
  ${Z3_LIBRARIES}
  ${ORTOOLS_LIB}
  ${SLANG_LIB}
)
target_link_libraries(CIRCT PRIVATE ${LLVM_PTHREAD_LIB})

#message("Libraries included in libCIRCT.so: ${circt_libs}")
