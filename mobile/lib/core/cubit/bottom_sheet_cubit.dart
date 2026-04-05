import 'package:freezed_annotation/freezed_annotation.dart';
import 'package:injectable/injectable.dart';

import 'app_cubit.dart';

part 'bottom_sheet_cubit.freezed.dart';

@injectable
class BottomSheetCubit extends AppCubit<BottomSheetState> {
  BottomSheetCubit() : super(const _Initial());

  void openSelectImageSheet() => emit(const _OpenSelectImageSheet());

  void closed() => emit(const _Initial());
}

@freezed
sealed class BottomSheetState with _$BottomSheetState {
  const factory BottomSheetState.initial() = _Initial;

  const factory BottomSheetState.openSelectImageSheet() = _OpenSelectImageSheet;
}
