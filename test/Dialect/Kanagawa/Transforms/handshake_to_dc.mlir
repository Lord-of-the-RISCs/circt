// RUN: circt-opt --pass-pipeline='builtin.module(kanagawa.design(kanagawa.class(kanagawa-convert-handshake-to-dc)))' %s | FileCheck %s

// Actual handshake-to-dc conversion is tested in the respective pass tests.
// This file just tests the kanagawa-specific hooks.

// CHECK-LABEL:   kanagawa.class sym @ToDC {
// CHECK:           kanagawa.method.df @foo(%[[VAL_1:.*]]: !dc.value<i32>) -> !dc.value<i32> {
// CHECK:             %[[VAL_2:.*]], %[[VAL_3:.*]] = dc.unpack %[[VAL_1]] : !dc.value<i32>
// CHECK:             %[[VAL_4:.*]]:2 = dc.fork [2] %[[VAL_2]]
// CHECK:             %[[VAL_5:.*]] = dc.pack %[[VAL_4]]#0, %[[VAL_3]] : i32
// CHECK:             %[[VAL_6:.*]] = dc.pack %[[VAL_4]]#1, %[[VAL_3]] : i32
// CHECK:             %[[VAL_7:.*]] = kanagawa.sblock.dc (%[[VAL_8:.*]] : !dc.value<i32> = %[[VAL_5]], %[[VAL_9:.*]] : !dc.value<i32> = %[[VAL_6]]) -> !dc.value<i32> {
// CHECK:               %[[VAL_10:.*]] = arith.addi %[[VAL_8]], %[[VAL_9]] : i32
// CHECK:               kanagawa.sblock.return %[[VAL_10]] : i32
// CHECK:             }
// CHECK:             kanagawa.return %[[VAL_7]] : !dc.value<i32>
// CHECK:           }
// CHECK:         }

kanagawa.design @foo {
kanagawa.class sym @ToDC {
  kanagawa.method.df @foo(%arg0: i32) -> (i32) {
    %o0, %o1 = handshake.fork [2] %arg0 : i32
    %1 = kanagawa.sblock.isolated(%a0 : i32 = %o0, %a1 : i32 = %o1) -> i32 {
      %res = arith.addi %a0, %a1 : i32
      kanagawa.sblock.return %res : i32
    }
    kanagawa.return %1 : i32
  }
}
}
