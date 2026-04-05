// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'prediction_two_dto.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

PredictionTwoDto _$PredictionTwoDtoFromJson(Map<String, dynamic> json) =>
    PredictionTwoDto(
      confidence: (json['confidence'] as num?)?.toDouble(),
      label: json['label'] as String?,
      color: (json['color'] as List<dynamic>?)
          ?.map((e) => (e as num).toInt())
          .toList(),
    );

Map<String, dynamic> _$PredictionTwoDtoToJson(PredictionTwoDto instance) =>
    <String, dynamic>{
      'confidence': ?instance.confidence,
      'label': ?instance.label,
      'color': ?instance.color,
    };
