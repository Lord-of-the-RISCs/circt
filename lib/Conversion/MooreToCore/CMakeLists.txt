add_circt_conversion_library(CIRCTMooreToCore
  MooreToCore.cpp
  SHARED
  

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTMoore
  CIRCTLLHD
  CIRCTHW
  CIRCTComb
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRTransforms
)
