import 'package:equatable/equatable.dart';

class AppConfig extends Equatable {
  final String minSupportedVersion;

  const AppConfig({
    required this.minSupportedVersion,
  });

  @override
  List<Object?> get props => [
    minSupportedVersion,
  ];
}
