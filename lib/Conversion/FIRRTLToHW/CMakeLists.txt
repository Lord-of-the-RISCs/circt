add_circt_conversion_library(CIRCTFIRRTLToHW
  LowerToHW.cpp
  SHARED
  

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTFIRRTL
  CIRCTHW
  CIRCTLTL
  CIRCTSeq
  CIRCTSim
  CIRCTSV
  CIRCTVerif
  MLIRTransforms
)
