add_circt_translation_library(CIRCTExportArc
  ModelInfoExport.cpp

  SHARED

  LINK_LIBS PUBLIC
  CIRCTArc
  MLIRTranslateLib
)
