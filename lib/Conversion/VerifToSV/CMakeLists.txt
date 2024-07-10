add_circt_conversion_library(CIRCTVerifToSV
  VerifToSV.cpp
  SHARED
  

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTSV
  CIRCTVerif
  MLIRTransforms
)
