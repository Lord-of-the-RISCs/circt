add_circt_conversion_library(CIRCTCombToLLVM
  CombToLLVM.cpp

  SHARED
  

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTComb
  MLIRLLVMCommonConversion
  MLIRTransforms
)
