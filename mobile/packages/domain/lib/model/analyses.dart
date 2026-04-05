import 'package:domain/model/prediction.dart';
import 'package:equatable/equatable.dart';

import 'api_urls.dart';

class Analyses extends Equatable {
  final ApiUrls apiUrls;
  final List<Prediction> predictions;
  final String jobId;

  const Analyses({
    required this.apiUrls,
    required this.predictions,
    required this.jobId,
  });

  @override
  List<Object> get props => [apiUrls, predictions];
}
