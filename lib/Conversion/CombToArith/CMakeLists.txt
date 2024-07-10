add_circt_conversion_library(CIRCTCombToArith
  CombToArith.cpp
  SHARED
  

  DEPENDS
  CIRCTConversionPassIncGen
  MLIRArithDialect

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTHW
  MLIRArithDialect
  MLIRTransforms
)
