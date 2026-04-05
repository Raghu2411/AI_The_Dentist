import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:kernel/kernel.dart';

import '../constants/app_colors.dart';
import '../constants/dimens.dart';
import '../constants/font_sizes.dart';
import 'custom_input_style.dart';

class ThemeBuilder {
  ThemeData getDefault() {
    final borderBuilder = getIt.get<CustomInputStyle>();
    return ThemeData(
      useMaterial3: true,
      fontFamily: "SFPro",
      primarySwatch: AppColors.primarySwatch,
      scaffoldBackgroundColor: AppColors.white,
      primaryColor: AppColors.primary,
      splashColor: Colors.transparent,
      highlightColor: Colors.transparent,
      dividerColor: AppColors.primary,
      colorScheme: const ColorScheme.light(
        primary: AppColors.primarySwatch,
        onPrimary: Colors.white,
        secondary: Colors.white,
        onSecondary: Colors.white,
        surface: Colors.white,
        // Light background for surfaces
        onSurface: Colors.black,
        // Dark text/icons on surfaces
        error: AppColors.errorSwatch,
        onError: Colors.white,
        // ignore: deprecated_member_use
        background: AppColors.white,
        // ignore: deprecated_member_use
        onBackground: Colors.black,
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: AppColors.white,
        hintStyle: TextStyle(color: AppColors.hintColor),
        enabledBorder: borderBuilder.getOutlineInputBorder(
          AppColors.textFieldBorder,
          Dimens.borderXSmall,
        ),
        disabledBorder: borderBuilder.getOutlineInputBorder(
          AppColors.textFieldBorder,
          Dimens.borderXSmall,
        ),
        focusedBorder: borderBuilder.getOutlineInputBorder(
          AppColors.secondary.withValues(alpha: 0.3),
          Dimens.borderSmall,
        ),
        contentPadding: EdgeInsets.symmetric(
          vertical: Dimens.large,
          horizontal: Dimens.marginMedium,
        ), // Adjust height
      ),
      textSelectionTheme: TextSelectionThemeData(cursorColor: Colors.black),
      appBarTheme: const AppBarTheme(
        elevation: 0,
        iconTheme: IconThemeData(color: Colors.black),
        backgroundColor: Colors.transparent,
        systemOverlayStyle: SystemUiOverlayStyle(
          // Status bar color
          statusBarColor: Colors.white,
          statusBarIconBrightness: Brightness.dark, // For Android (dark icons)
          statusBarBrightness: Brightness.light, // For iOS (dark icons)
        ),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ButtonStyle(
          elevation: WidgetStateProperty.all<double>(0),
          backgroundColor: WidgetStateProperty.resolveWith<Color>((
            Set<WidgetState> states,
          ) {
            if (states.contains(WidgetState.disabled)) {
              return AppColors.disabledButton;
            }
            return AppColors.primary;
          }),
          foregroundColor: WidgetStateProperty.all(AppColors.white),
          minimumSize: WidgetStateProperty.all<Size>(
            const Size(double.infinity, Dimens.button),
          ),
          maximumSize: WidgetStateProperty.all<Size>(
            const Size(double.infinity, Dimens.button),
          ),
          shape: WidgetStateProperty.all<RoundedRectangleBorder>(
            const RoundedRectangleBorder(
              borderRadius: BorderRadius.all(
                Radius.circular(Dimens.radiusLarge),
              ),
            ),
          ),
        ),
      ),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: ButtonStyle(
          elevation: WidgetStateProperty.all<double>(0),
          backgroundColor: WidgetStateProperty.resolveWith<Color>((
            Set<WidgetState> states,
          ) {
            if (states.contains(WidgetState.disabled)) {
              return AppColors.disabledButton;
            }
            return AppColors.white;
          }),
          foregroundColor: WidgetStateProperty.all(AppColors.primary),
          minimumSize: WidgetStateProperty.all<Size>(
            const Size(double.infinity, Dimens.button),
          ),
          maximumSize: WidgetStateProperty.all<Size>(
            const Size(double.infinity, Dimens.button),
          ),
          shape: WidgetStateProperty.all<RoundedRectangleBorder>(
            const RoundedRectangleBorder(
              borderRadius: BorderRadius.all(
                Radius.circular(Dimens.radiusLarge),
              ),
            ),
          ),
          side: WidgetStateProperty.all<BorderSide>(
            BorderSide(color: AppColors.primary, width: Dimens.borderXSmall),
          ),
        ),
      ),
      textTheme: const TextTheme(
        headlineMedium: TextStyle(
          fontSize: FontSizes.xxLarge,
          fontWeight: FontWeight.w600,
          letterSpacing: -1,
          color: AppColors.primary,
        ),
        headlineSmall: TextStyle(
          fontSize: FontSizes.large,
          fontWeight: FontWeight.w700,
          letterSpacing: 0,
          color: AppColors.primary,
        ),
        titleLarge: TextStyle(
          fontSize: FontSizes.xxMedium,
          fontWeight: FontWeight.w600,
          letterSpacing: 0.15,
        ),
        titleMedium: TextStyle(
          fontSize: FontSizes.xMedium,
          fontWeight: FontWeight.w500,
          letterSpacing: 0,
          color: AppColors.primary,
        ),
        titleSmall: TextStyle(
          fontSize: FontSizes.xMedium,
          fontWeight: FontWeight.w400,
          letterSpacing: -0.24,
        ),
        bodyLarge: TextStyle(
          fontSize: FontSizes.xMedium,
          fontWeight: FontWeight.w500,
          letterSpacing: -0.3,
          color: AppColors.primary,
        ),
        bodyMedium: TextStyle(
          fontSize: FontSizes.medium,
          fontWeight: FontWeight.w500,
          letterSpacing: -0.3,
          color: AppColors.primary,
        ),
        labelLarge: TextStyle(
          fontSize: FontSizes.xMedium,
          fontWeight: FontWeight.w600,
          letterSpacing: 0.1,
          color: AppColors.white,
        ),
        bodySmall: TextStyle(
          fontSize: FontSizes.xSmall,
          fontWeight: FontWeight.w400,
          letterSpacing: 0.4,
        ),
        labelSmall: TextStyle(
          fontSize: FontSizes.medium,
          fontWeight: FontWeight.w500,
          letterSpacing: 0,
        ),
      ),
    );
  }
}
