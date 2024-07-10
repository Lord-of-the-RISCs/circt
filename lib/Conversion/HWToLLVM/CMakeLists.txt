add_circt_conversion_library(CIRCTHWToLLVM
  HWToLLVM.cpp
  SHARED
  

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTSupport
  MLIRLLVMCommonConversion
  MLIRTransforms
)
