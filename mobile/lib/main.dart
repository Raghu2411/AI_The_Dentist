import 'dart:async';

import 'package:flutter/material.dart';
import 'package:kernel/kernel.dart';
import 'package:logger/logger.dart';

import 'core/app.dart';
import 'di/injector.dart';
import 'navigation/app_router.dart';

///
/// Main entry app point. Only modify this file if is something about
/// main configuration. Any runtime stuff should be handled in the [App]
///
void main() async {
  // Ensure that plugin services are initialized
  await runZonedGuarded(
    () async {
      WidgetsFlutterBinding.ensureInitialized();
      await injectDependencies();

      runApp(
        Builder(
          builder: (context) {
            final mediaQuery = MediaQuery.of(context);
            final textScalar = mediaQuery.textScaler;
            final mediaQueryData = mediaQuery.copyWith(
              textScaler: textScalar.clamp(
                minScaleFactor: 1,
                maxScaleFactor: 1,
              ),
            );
            return MediaQuery(
              data: mediaQueryData,
              child: App(appRouter: getIt<AppRouter>()),
            );
          },
        ),
      );
    },
    (Object error, StackTrace stackTrace) {
      final logger = getIt<Logger>();
      logger.e(stackTrace);
      logger.e(error);
    },
  );
}
