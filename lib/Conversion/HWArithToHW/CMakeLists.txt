add_circt_conversion_library(CIRCTHWArithToHW
  HWArithToHW.cpp
  SHARED
  

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTSV
  CIRCTSeq
  CIRCTMSFT
  CIRCTHW
  CIRCTHWArith
  MLIRTransforms
)
