add_circt_dialect_library(CIRCTSystemCTransforms
  SystemCLowerInstanceInterop.cpp

  SHARED
  
  DEPENDS
  CIRCTSystemCTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTSystemC
  CIRCTInteropDialect
  CIRCTSupport
  MLIRIR
  MLIRFuncDialect
  MLIRPass
  MLIRTransformUtils
)
