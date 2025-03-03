if (MSVC)
  return()
endif()

get_property(mlir_libs GLOBAL PROPERTY MLIR_STATIC_LIBS)
get_property(mlir_llvm_link_components GLOBAL PROPERTY MLIR_LLVM_LINK_COMPONENTS)
list(REMOVE_DUPLICATES mlir_libs)
list(REMOVE_DUPLICATES mlir_llvm_link_components)

foreach (lib ${mlir_libs})
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

  LINK_COMPONENTS
  ${mlir_llvm_link_components}
)
target_link_libraries(CIRCT PRIVATE ${LLVM_PTHREAD_LIB})

#message("Libraries included in libCIRCT.so: ${mlir_libs}")
#message("LLVM Components included in libCIRCT.so: ${mlir_llvm_link_components}")
