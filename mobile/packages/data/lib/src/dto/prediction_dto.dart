import 'package:json_annotation/json_annotation.dart';

part 'prediction_dto.g.dart';

@JsonSerializable(includeIfNull: false)
class PredictionDto {
  final double? confidence;
  final String? label;
  final List<int>? color;

  const PredictionDto({this.confidence, this.label, this.color});

  factory PredictionDto.fromJson(Map<String, dynamic> json) =>
      _$PredictionDtoFromJson(json);

  Map<String, dynamic> toJson() => _$PredictionDtoToJson(this);
}
