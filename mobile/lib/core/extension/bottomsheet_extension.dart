import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

import '../constants/dimens.dart';
import '../cubit/bottom_sheet_cubit.dart';

extension BottomSheetExtension on BuildContext {
  Future openBottomSheet(
    Widget bottomSheetView, {
    bool isScrollControlled = false,
    bool isDismissible = true,
    bool isDragEnabled = true,
    double borderRadius = Dimens.large,
    bool useSafeArea = false,
    bool showDragHandle = false,
    Color? backgroundColor,
    Color? barrierColor,
  }) {
    return showModalBottomSheet(
      isScrollControlled: isScrollControlled,
      isDismissible: isDismissible,
      enableDrag: isDragEnabled,
      context: this,
      backgroundColor: backgroundColor,
      useSafeArea: useSafeArea,
      barrierColor: barrierColor,
      showDragHandle: showDragHandle,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.only(
          topRight: Radius.circular(borderRadius),
          topLeft: Radius.circular(borderRadius),
        ),
      ),
      builder: (_) => bottomSheetView,
    ).whenComplete(() {
      if (mounted) {
        read<BottomSheetCubit>().closed();
      }
    });
  }
}
