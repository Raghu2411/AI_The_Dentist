import 'package:data/src/dto/prediction_dto.dart';
import 'package:json_annotation/json_annotation.dart';

import 'api_urls_dto.dart';

part 'analyses_dto.g.dart';

@JsonSerializable(explicitToJson: true, includeIfNull: false)
class AnalysesDto {
  final ApiUrlsDto? api_urls;
  final List<PredictionDto>? predictions;
  final String? job_id;

  const AnalysesDto({this.api_urls, this.predictions, this.job_id});

  factory AnalysesDto.fromJson(Map<String, dynamic> json) =>
      _$AnalysesDtoFromJson(json);

  Map<String, dynamic> toJson() => _$AnalysesDtoToJson(this);
}
