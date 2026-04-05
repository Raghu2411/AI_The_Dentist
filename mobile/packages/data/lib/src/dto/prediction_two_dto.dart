import 'package:json_annotation/json_annotation.dart';

part 'prediction_two_dto.g.dart';

@JsonSerializable(includeIfNull: false)
class PredictionTwoDto {
  final double? confidence;
  final String? label;
  final List<int>? color;

  const PredictionTwoDto({this.confidence, this.label, this.color});

  factory PredictionTwoDto.fromJson(Map<String, dynamic> json) =>
      _$PredictionTwoDtoFromJson(json);

  Map<String, dynamic> toJson() => _$PredictionTwoDtoToJson(this);
}
