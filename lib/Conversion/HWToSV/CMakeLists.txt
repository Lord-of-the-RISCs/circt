add_circt_conversion_library(CIRCTHWToSV
  HWToSV.cpp
  SHARED
  

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTSV
  MLIREmitCDialect
  MLIRTransforms
)
