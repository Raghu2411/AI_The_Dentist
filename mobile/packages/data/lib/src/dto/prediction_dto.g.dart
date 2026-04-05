// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'prediction_dto.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

PredictionDto _$PredictionDtoFromJson(Map<String, dynamic> json) =>
    PredictionDto(
      confidence: (json['confidence'] as num?)?.toDouble(),
      label: json['label'] as String?,
      color: (json['color'] as List<dynamic>?)
          ?.map((e) => (e as num).toInt())
          .toList(),
    );

Map<String, dynamic> _$PredictionDtoToJson(PredictionDto instance) =>
    <String, dynamic>{
      'confidence': ?instance.confidence,
      'label': ?instance.label,
      'color': ?instance.color,
    };
