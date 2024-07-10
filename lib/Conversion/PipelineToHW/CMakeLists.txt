add_circt_library(CIRCTPipelineToHW
  PipelineToHW.cpp

  SHARED

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  CIRCTSeq
  CIRCTHW
  CIRCTComb
  CIRCTPipelineOps
  )
