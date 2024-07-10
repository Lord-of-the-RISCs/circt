add_circt_conversion_library(CIRCTFSMToSV
    FSMToSV.cpp

  SHARED
  
    DEPENDS
    CIRCTConversionPassIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    CIRCTComb
    CIRCTHW
    CIRCTFSM
    CIRCTSeq
    CIRCTSV
    CIRCTSupport
    MLIRTransforms
)
