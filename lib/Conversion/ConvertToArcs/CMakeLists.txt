add_circt_conversion_library(CIRCTConvertToArcs
  ConvertToArcs.cpp
  SHARED
  

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTArc
  CIRCTHW
  CIRCTSeq
  MLIRTransforms
)
