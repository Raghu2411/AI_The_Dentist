// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'analyses_dto.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

AnalysesDto _$AnalysesDtoFromJson(Map<String, dynamic> json) => AnalysesDto(
  api_urls: json['api_urls'] == null
      ? null
      : ApiUrlsDto.fromJson(json['api_urls'] as Map<String, dynamic>),
  predictions: (json['predictions'] as List<dynamic>?)
      ?.map((e) => PredictionDto.fromJson(e as Map<String, dynamic>))
      .toList(),
  job_id: json['job_id'] as String?,
);

Map<String, dynamic> _$AnalysesDtoToJson(AnalysesDto instance) =>
    <String, dynamic>{
      'api_urls': ?instance.api_urls?.toJson(),
      'predictions': ?instance.predictions?.map((e) => e.toJson()).toList(),
      'job_id': ?instance.job_id,
    };
