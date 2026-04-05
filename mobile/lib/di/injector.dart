import 'package:ai_dental_studio/di/injector.config.dart';
import 'package:data/data.dart';
import 'package:domain/domain.dart';
import 'package:injectable/injectable.dart';
import 'package:kernel/kernel.dart';

@InjectableInit(
  includeMicroPackages: false,
  externalPackageModulesBefore: [
    ExternalModule(DataPackageModule),
    ExternalModule(DomainPackageModule),
  ],
)
Future<void> injectDependencies() async => getIt.init();
