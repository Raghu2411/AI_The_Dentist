import 'package:flutter/material.dart';

/// Usage Example:
///   Theme.of(context).primaryColor
///

abstract class AppColors {
  static const MaterialColor primarySwatch = MaterialColor(
    0xFF0b1922,
    <int, Color>{
      50: Color(0xFF0a171f), //10%
      100: Color(0xFF09141b), //20%
      200: Color(0xFF081218), //30%
      300: Color(0xFF070f14), //40%
      400: Color(0xFF060d11), //50%
      500: Color(0xFF040a0e), //60%
      600: Color(0xFF03070a), //70%
      700: Color(0xFF020507), //80%
      800: Color(0xFF010203), //90%
      900: Color(0xFF000000), //100%
    },
  );

  static const MaterialColor errorSwatch = MaterialColor(
    0xFFF64333,
    <int, Color>{
      50: Color(0xFFdd3c2e), //10%
      100: Color(0xFFc53629), //20%
      200: Color(0xFFac2f24), //30%
      300: Color(0xFF94281f), //40%
      400: Color(0xFF7b221a), //50%
      500: Color(0xFF621b14), //60%
      600: Color(0xFF4a140f), //70%
      700: Color(0xFF310d0a), //80%
      800: Color(0xFF190705), //90%
      900: Color(0xFF000000), //100%
    },
  );

  // Basic Colors
  static const Color primary = Color(0xFF0B1922);
  static const Color secondary = Color(0xFFFF991C);
  static const Color labelColor = Color(0xFF667085);

  // Background Colors
  static const Color background = Colors.white;

  // Button Colors
  static const Color disabledButton = Color(0xFFF2F4F7);

  // TextField & Common Colors
  static const Color error = Color(0xFFF04438);
  static const Color white = Colors.white;
  static const Color textFieldBorder = Color(0xFFD0D5DD);
  static const Color hintColor = Color(0xFF667085);
  static const Color textFieldTextColor = Color(0xFF101828);
  static const Color textFieldErrorBorder = Color(0xFFFDA29B);
  static const Color black = Colors.black;
  static const Color transparent = Colors.transparent;
  static const Color textFieldLabelColor = Color(0xFF344054);
  static const Color transparentColor = Colors.transparent;
  static const Color gray200 = Color(0xFFDDDDDD);
  static const Color borderColor = Color(0xFF9ABDBA);
  static const Color green = Colors.green;
}
