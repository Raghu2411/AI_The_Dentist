import 'package:injectable/injectable.dart';

import '../core/styles/custom_input_style.dart';
import '../core/styles/theme.dart';

// required by code generation
@microPackageInit
void initAppModule() {}

@module
abstract class AppModule {
  @Singleton()
  ThemeBuilder get themeBuilder => ThemeBuilder();

  @Singleton()
  CustomInputStyle get customInputStyle => CustomInputStyle();
}
