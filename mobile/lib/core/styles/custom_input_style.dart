import 'package:flutter/material.dart';
import '../constants/dimens.dart';

class CustomInputStyle {
  OutlineInputBorder getOutlineInputBorder(Color color, double width) {
    return OutlineInputBorder(
      borderSide: BorderSide(color: color, width: width),
      borderRadius: const BorderRadius.all(
        Radius.circular(Dimens.radiusMedium),
      ),
    );
  }
}
