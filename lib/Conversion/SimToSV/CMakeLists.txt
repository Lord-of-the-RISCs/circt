add_circt_conversion_library(CIRCTSimToSV
  SimToSV.cpp
  SHARED
  

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTSim
)
