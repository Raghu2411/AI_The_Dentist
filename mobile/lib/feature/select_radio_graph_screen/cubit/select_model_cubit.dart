import 'package:freezed_annotation/freezed_annotation.dart';
import 'package:injectable/injectable.dart';

import '../../../../core/cubit/app_cubit.dart';
import '../../../core/constants/strings.dart';

part 'select_model_cubit.freezed.dart';

@injectable
class SelectModelCubit extends AppCubit<SelectModelState> {
  SelectModelCubit() : super(const _Initial());

  void setSelectedModel(List<String> selectedModels) async {
    emit(_SelectedModels(selectedModels: selectedModels));
  }

  List<String> get selectedModelValue {
    return state.maybeWhen(
      selectedModels: (selectedModels) => selectedModels,
      orElse: () => [Strings.defaultModel],
    );
  }
}

@freezed
class SelectModelState with _$SelectModelState {
  const factory SelectModelState.initial() = _Initial;

  const factory SelectModelState.selectedModels({
    @Default([Strings.defaultModel]) List<String> selectedModels,
  }) = _SelectedModels;
}
