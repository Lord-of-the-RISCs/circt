if (MSVC)
  return()
endif()

get_property(circt_libs GLOBAL PROPERTY CIRCT_ALL_LIBS)
get_property(mlir_llvm_link_components GLOBAL PROPERTY MLIR_LLVM_LINK_COMPONENTS)
list(REMOVE_DUPLICATES circt_libs)
list(REMOVE_DUPLICATES mlir_llvm_link_components)

if(CIRCT_LINK_CIRCT_DYLIB)
  set(INSTALL_WITH_TOOLCHAIN INSTALL_WITH_TOOLCHAIN)
endif()

add_circt_library(
  CIRCT
  SHARED
  ${INSTALL_WITH_TOOLCHAIN}
  circt-shlib.cpp
  LINK_LIBS
  ${circt_libs}
  MLIR
  ${Z3_LIBRARIES}
)
target_link_libraries(CIRCT PRIVATE ${LLVM_PTHREAD_LIB})

#message("Libraries included in libCIRCT.so: ${circt_libs}")
