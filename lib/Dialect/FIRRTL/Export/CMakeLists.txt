add_circt_translation_library(CIRCTExportFIRRTL
  FIREmitter.cpp

  SHARED

  LINK_LIBS PUBLIC
  CIRCTFIRRTL
  MLIRTranslateLib
)
