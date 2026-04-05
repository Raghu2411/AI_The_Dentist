import 'package:equatable/equatable.dart';

class Prediction extends Equatable {
  final double confidence;
  final String label;
  final List<int> color;

  const Prediction({
    required this.confidence,
    required this.label,
    required this.color,
  });

  @override
  List<Object> get props => [confidence, label, color];
}
